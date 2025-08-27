import torch
from typing import Optional
from genesis.utils.geom import (
    quat_to_xyz,
    transform_by_quat,
    inv_quat,
    transform_quat_by_quat,
)

class TorchPIDController:
    """支持多环境并行的Torch PID控制器"""
    
    def __init__(self, kp: float, ki: float, kd: float, n_envs: int, device: torch.device):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.n_envs = n_envs
        self.device = device
        
        # 多环境状态 [n_envs]
        self.integral = torch.zeros(n_envs, device=device, dtype=torch.float32)
        self.prev_error = torch.zeros(n_envs, device=device, dtype=torch.float32)

    def update(self, error: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Args:
            error: [n_envs] 误差张量
            dt: 时间步长
        Returns:
            [n_envs] 控制输出
        """
        with torch.no_grad():
            # 积分项
            self.integral += error * dt
            
            # 微分项
            derivative = (error - self.prev_error) / dt
            self.prev_error = error.clone()
            
            # PID输出
            output = self.kp * error + self.ki * self.integral + self.kd * derivative
            
            return output
    
    def reset(self, env_ids: Optional[torch.Tensor] = None):
        """重置指定环境的PID状态"""
        with torch.no_grad():
            if env_ids is None:
                # 重置所有环境
                self.integral.zero_()
                self.prev_error.zero_()
            else:
                # 重置指定环境
                self.integral[env_ids] = 0.0
                self.prev_error[env_ids] = 0.0


class MultiEnvDronePIDController:
    """多环境并行无人机PID控制器"""
    
    def __init__(
        self, 
        drone, 
        dt: float, 
        base_rpm: float, 
        pid_params: list, 
        n_envs: int,
        device: torch.device
    ):
        self.drone = drone
        self.dt = dt
        self.base_rpm = base_rpm
        self.n_envs = n_envs
        self.device = device
        
        # 创建姿态PID控制器
        self.pid_roll = TorchPIDController(
            kp=pid_params[0][0], ki=pid_params[0][1], kd=pid_params[0][2],
            n_envs=n_envs, device=device
        )
        self.pid_pitch = TorchPIDController(
            kp=pid_params[1][0], ki=pid_params[1][1], kd=pid_params[1][2],
            n_envs=n_envs, device=device
        )
        self.pid_yaw = TorchPIDController(
            kp=pid_params[2][0], ki=pid_params[2][1], kd=pid_params[2][2],
            n_envs=n_envs, device=device
        )
        self.drone = drone
    
    def get_drone_attitude(self) -> torch.Tensor:
        """获取无人机姿态"""
        with torch.no_grad():
            quat = self.drone.get_quat()  # [n_envs, 4]
            euler = quat_to_xyz(quat, rpy=True, degrees=True)  # [n_envs, 3]
            return euler
        
    def get_drone_ang_vel(self) -> torch.Tensor:
        """获取无人机姿态"""
        with torch.no_grad():
            quat = self.drone.get_ang()  # [n_envs, 3]
            return quat
    
    def mixer(self, thrust: torch.Tensor, roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        """
        四旋翼混控器
        Args:
            thrust: [n_envs] 推力指令
            roll, pitch, yaw: [n_envs] 姿态力矩指令
        Returns:
            [n_envs, 4] 四个电机的RPM
        """
        with torch.no_grad():
            # 混控矩阵计算
            M1 = self.base_rpm + (thrust - roll - pitch - yaw)
            M2 = self.base_rpm + (thrust - roll + pitch + yaw)  
            M3 = self.base_rpm + (thrust + roll + pitch - yaw)
            M4 = self.base_rpm + (thrust + roll - pitch + yaw)
            
            # 堆叠为 [n_envs, 4]
            motor_rpms = torch.stack([M1, M2, M3, M4], dim=1)
            
            # 限制RPM范围
            motor_rpms = torch.clamp(motor_rpms, min=0.0, max=self.base_rpm * 4.0)
            
            return motor_rpms
    
    def update(self, target: torch.Tensor) -> torch.Tensor:
        """
        PID控制更新
        Args:
            target: [n_envs, 4] 目标姿态和推力 [roll_rate, pitch_rate, yaw_rate, thrust]
        Returns:
            [n_envs, 4] 四个电机的RPM
        """
        with torch.no_grad():
            # 解析目标指令
            target_roll_rate = target[:, 0]    # [n_envs]
            target_pitch_rate = target[:, 1]   # [n_envs]
            target_yaw_rate = target[:, 2]     # [n_envs]
            thrust_cmd = target[:, 3]          # [n_envs]
            
            # 获取当前姿态
            current_ang_vel = self.get_drone_ang_vel()  # [n_envs, 3]
            current_roll_vel = current_ang_vel[:, 0]
            current_pitch_vel = current_ang_vel[:, 1] 
            current_yaw_vel = current_ang_vel[:, 2]
            
            roll_error = target_roll_rate - current_roll_vel
            pitch_error = target_pitch_rate - current_pitch_vel
            yaw_error = target_yaw_rate - current_yaw_vel
            
            # PID控制计算
            roll_output = self.pid_roll.update(roll_error, self.dt)
            pitch_output = self.pid_pitch.update(pitch_error, self.dt)
            yaw_output = self.pid_yaw.update(yaw_error, self.dt)
            
            # 混控器计算电机RPM
            motor_rpms = self.mixer(thrust_cmd, roll_output, pitch_output, yaw_output)
            
            return motor_rpms
    
    def reset(self, env_ids: Optional[torch.Tensor] = None):
        """重置PID控制器状态"""
        self.pid_roll.reset(env_ids)
        self.pid_pitch.reset(env_ids)
        self.pid_yaw.reset(env_ids)


class AngularRatePIDController(MultiEnvDronePIDController):
    """角速度PID控制器版本"""
    
    def __init__(self, drone, dt: float, base_rpm: float, pid_params: list, n_envs: int, device: torch.device):
        super().__init__(drone, dt, base_rpm, pid_params, n_envs, device)
    
    def get_drone_angular_velocity(self) -> torch.Tensor:
        """获取无人机角速度"""
        with torch.no_grad():
            base_quat: torch.Tensor = self.drone.get_quat()
            inv_base_quat: torch.Tensor = inv_quat(base_quat)
            base_ang_vel: torch.Tensor = transform_by_quat(self.drone.get_ang(), inv_base_quat)
            return base_ang_vel
    
    def update(self, target: torch.Tensor) -> torch.Tensor:
        """
        角速度PID控制
        Args:
            target: [n_envs, 4] [target_roll_rate, target_pitch_rate, target_yaw_rate, thrust]
        Returns:
            [n_envs, 4] 四个电机的RPM
        """
        with torch.no_grad():
            # 解析目标指令
            target_roll_rate = target[:, 0]
            target_pitch_rate = target[:, 1] 
            target_yaw_rate = target[:, 2]
            thrust_cmd = target[:, 3]
            # if torch.rand(1) < 0.01:
            #     print(f"Target Rates: Roll {target_roll_rate}, Pitch {target_pitch_rate}, Yaw {target_yaw_rate}, Thrust {thrust_cmd}")
            
            # 获取当前角速度
            current_angular_vel = self.get_drone_angular_velocity()  # [n_envs, 3]
            current_roll_rate = current_angular_vel[:, 0]
            current_pitch_rate = current_angular_vel[:, 1]
            current_yaw_rate = current_angular_vel[:, 2]
            
            # 计算角速度误差
            roll_rate_error = target_roll_rate - current_roll_rate
            pitch_rate_error = target_pitch_rate - current_pitch_rate
            yaw_rate_error = target_yaw_rate - current_yaw_rate
            
            # PID控制计算
            roll_output = self.pid_roll.update(roll_rate_error, self.dt)
            pitch_output = self.pid_pitch.update(pitch_rate_error, self.dt)
            yaw_output = self.pid_yaw.update(yaw_rate_error, self.dt)
            
            # 混控器计算电机RPM
            motor_rpms = self.mixer(thrust_cmd, roll_output, pitch_output, yaw_output)
            
            return motor_rpms

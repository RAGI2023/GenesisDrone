import time
import threading

from pynput import keyboard
import numpy as np
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver

import genesis as gs

link_indices = {
    "base_link": 0,           # 主体
    "prop0_link": 1,          # 右前螺旋桨
    "prop1_link": 2,          # 左前螺旋桨  
    "prop2_link": 3,          # 左后螺旋桨
    "prop3_link": 4,          # 右后螺旋桨
    "center_of_mass_link": 5  # 质心参考点
}

class DroneController:
    def __init__(self):
        self.thrust = 14690.2  # Base hover RPM - constant hover
        self.rotation_delta = 200.0  # Differential RPM for rotation
        self.thrust_delta = 10.0  # Amount to change thrust by when accelerating/decelerating
        self.running = True
        self.rpms = [self.thrust] * 4
        self.pressed_keys = set()

    def on_press(self, key):
        try:
            if key == keyboard.Key.esc:
                self.running = False
                return False
            self.pressed_keys.add(key)
            print(f"Key pressed: {key}")
        except AttributeError:
            pass

    def on_release(self, key):
        try:
            self.pressed_keys.discard(key)
        except KeyError:
            pass

    def update_thrust(self):
        # Store previous RPMs for debugging
        prev_rpms = self.rpms.copy()

        # Reset RPMs to hover thrust
        self.rpms = [self.thrust] * 4

        # Acceleration (Spacebar) - All rotors spin faster
        if keyboard.Key.space in self.pressed_keys:
            self.thrust += self.thrust_delta
            self.rpms = [self.thrust] * 4
            print("Accelerating")

        # Deceleration (Left Shift) - All rotors spin slower
        if keyboard.Key.shift in self.pressed_keys:
            self.thrust -= self.thrust_delta
            self.rpms = [self.thrust] * 4
            print("Decelerating")

        # Forward (North) - Front rotors spin faster
        if keyboard.Key.up in self.pressed_keys:
            self.rpms[0] += self.rotation_delta  # Front left
            self.rpms[1] += self.rotation_delta  # Front right
            self.rpms[2] -= self.rotation_delta  # Back left
            self.rpms[3] -= self.rotation_delta  # Back right
            print("Moving Forward")

        # Backward (South) - Back rotors spin faster
        if keyboard.Key.down in self.pressed_keys:
            self.rpms[0] -= self.rotation_delta  # Front left
            self.rpms[1] -= self.rotation_delta  # Front right
            self.rpms[2] += self.rotation_delta  # Back left
            self.rpms[3] += self.rotation_delta  # Back right
            print("Moving Backward")

        # Left (West) - Left rotors spin faster
        if keyboard.Key.left in self.pressed_keys:
            self.rpms[0] -= self.rotation_delta  # Front left
            self.rpms[2] -= self.rotation_delta  # Back left
            self.rpms[1] += self.rotation_delta  # Front right
            self.rpms[3] += self.rotation_delta  # Back right
            print("Moving Left")

        # Right (East) - Right rotors spin faster
        if keyboard.Key.right in self.pressed_keys:
            self.rpms[0] += self.rotation_delta  # Front left
            self.rpms[2] += self.rotation_delta  # Back left
            self.rpms[1] -= self.rotation_delta  # Front right
            self.rpms[3] -= self.rotation_delta  # Back right
            print("Moving Right")

        self.rpms = np.clip(self.rpms, 0, 25000)

        # Debug print if any RPMs changed
        if not np.array_equal(prev_rpms, self.rpms):
            print(f"RPMs changed from {prev_rpms} to {self.rpms}")

        return self.rpms


def run_sim(scene: gs.Scene, drone, controller):
    rotation_direction = 1
    step_count = 0
    
    while controller.running:
        rpms = controller.update_thrust()
        # drone.set_propellels_rpm(rpms)

        # 每200步改变方向
        if step_count % 200 == 0:
            rotation_direction *= -1

        # 创建正确的3D结构
        torque_value = [0, rotation_direction * 5e-3, 0]
        disturbance_torque = [[torque_value] for _ in range(scene.n_envs)]
        # 形状: [10, 1, 3] = [n_envs, n_links, xyz]
        
        scene.sim.rigid_solver.apply_links_external_torque(
            torque=disturbance_torque,
            links_idx=[1]
        )

        scene.step(refresh_visualizer=False)
        time.sleep(1.0 / scene.viewer.max_FPS)
        step_count += 1


def main():
    # Initialize Genesis
    gs.init(backend=gs.gpu)

    # Create scene
    scene: gs.Scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            gravity=(0, 0, -9.81),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -2.0, 1.0),
            camera_lookat=(0.0, 0.0, 0.3),
            camera_fov=45,
            max_FPS=60,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=False,
        ),
        show_viewer=True,
    )

    # Add entities
    plane = scene.add_entity(gs.morphs.Plane())
    drone = scene.add_entity(
        morph=gs.morphs.Drone(
            file="urdf/custom.urdf",
            # pos=(0.0, 0, 0.5),  # Start a bit higher
        ),
    )

    # scene.viewer.follow_entity(drone)

    # Initialize controller
    controller = DroneController()

    # Start keyboard listener.
    # Note that instantiating the listener after building the scene causes segfault on MacOS.
    listener = keyboard.Listener(on_press=controller.on_press, on_release=controller.on_release)
    listener.start()

    # Build scene
    scene.build(n_envs=10)

    # Print control instructions
    print("\nDrone Controls:")
    print("↑ - Move Forward (North)")
    print("↓ - Move Backward (South)")
    print("← - Move Left (West)")
    print("→ - Move Right (East)")
    print("space - Increase RPM")
    print("shift - Decrease RPM")
    print("ESC - Quit\n")
    print("Initial hover RPM:", controller.thrust)

    num_links = drone.n_links
    print("Number of drone links:", num_links)
    # for i in range(1, num_links+1):
    #     print(f"link {i} name", drone.get_link(uid=i).name)
    base_link = drone.get_link(name="base_link")
    prop0_link = drone.get_link(name="prop0_link")

    # 查看链接属性
    print("Link name:", base_link.name)
    print("Link UID:", base_link.uid)


    # Run simulation in another thread
    threading.Thread(target=run_sim, args=(scene, drone, controller)).start()
    # scene.viewer.run()

    listener.stop()


if __name__ == "__main__":
    main()

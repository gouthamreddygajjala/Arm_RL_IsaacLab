import argparse
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ------------------------------------------------------------

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationContext
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from pxr import UsdPhysics

from svaya_arm.robots.darm import DARM_CONFIG


def main():

    # Simulation
    sim_cfg = sim_utils.SimulationCfg(
        dt=1.0 / 120.0,
        device=args_cli.device,
    )
    sim = SimulationContext(sim_cfg)

    # --------------------------------------------------------
    # Ground
    # --------------------------------------------------------

    spawn_ground_plane(
        prim_path="/World/ground",
        cfg=GroundPlaneCfg(),
    )

    # Make ground a kinematic rigid body
    ground_prim = sim.stage.GetPrimAtPath("/World/ground")

    rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(ground_prim)
    rigid_body_api.CreateRigidBodyEnabledAttr(True)
    rigid_body_api.CreateKinematicEnabledAttr(True)

    print("Ground valid:", ground_prim.IsValid())
    print("Has RigidBodyAPI:", ground_prim.HasAPI(UsdPhysics.RigidBodyAPI))
    print(
        "Kinematic:",
        rigid_body_api.GetKinematicEnabledAttr().Get(),
    )

    # --------------------------------------------------------
    # Robot
    # --------------------------------------------------------

    robot_cfg = DARM_CONFIG.replace(
        prim_path="/World/Robot"
    )

    robot = Articulation(robot_cfg)

    # --------------------------------------------------------
    # Ground contact sensor
    # --------------------------------------------------------

    contact_sensor_cfg = ContactSensorCfg(
        prim_path="/World/Robot/l[1-7]_left_arm",
        update_period=0.0,
        history_length=3,
        debug_vis=True,
        filter_prim_paths_expr=["/World/ground"],
    )

    contact_sensor = ContactSensor(contact_sensor_cfg)
    
    SELF_COLLISION_BODIES = (
    "/World/Robot/(l[1-7]_left_arm|torso|torso_base|head|neck)"
)

    self_collision_sensor_cfg = ContactSensorCfg(
    prim_path=SELF_COLLISION_BODIES,
    update_period=0.0,
    history_length=3,
    debug_vis=True,
    filter_prim_paths_expr=[SELF_COLLISION_BODIES],
)

    self_collision_sensor = ContactSensor(self_collision_sensor_cfg)

    # --------------------------------------------------------
    # Light
    # --------------------------------------------------------

    light_cfg = sim_utils.DomeLightCfg(
        intensity=2000.0,
        color=(0.75, 0.75, 0.75),
    )
    light_cfg.func("/World/Light", light_cfg)

    # --------------------------------------------------------
    # Start simulation
    # --------------------------------------------------------

    sim.reset()

    print("Robot spawned.")
    print("Sensor body names:", contact_sensor.body_names)
    
    print("Self-collision sensor body names:")
    print(self_collision_sensor.body_names)

    num_bodies = len(self_collision_sensor.body_names)

    # Exclude self-pairs: body i against itself
    sc_exclude_mask = torch.eye(
        num_bodies,
        dtype=torch.bool,
        device=sim.device,
    )

    self_collision_threshold = 1.0  # Newtons

    count = 0

    while simulation_app.is_running():

        # No actions
        # No reset
        # Just run physics

        sim.step()

        robot.update(sim.get_physics_dt())
        contact_sensor.update(sim.get_physics_dt())
        self_collision_sensor.update(sim.get_physics_dt())


        count += 1

        # Print every 60 physics steps
        if count % 60 == 0:

            print("\n============================")

            print("net_forces_w:")
            print(contact_sensor.data.net_forces_w)

            print("force_matrix_w:")
            print(contact_sensor.data.force_matrix_w)

            net_mag = torch.norm(
                contact_sensor.data.net_forces_w,
                dim=-1,
            )

            print("Net force magnitudes:")
            print(net_mag)

            if contact_sensor.data.force_matrix_w is not None:

                filtered_mag = torch.norm(
                    contact_sensor.data.force_matrix_w,
                    dim=-1,
                )

                print("Filtered ground force magnitudes:")
                print(filtered_mag)

            print("============================")
            
            
        print("\n========================================")

        print("BODY NAMES:")
        for i, name in enumerate(self_collision_sensor.body_names):
            print(f"{i}: {name}")

        self_forces = self_collision_sensor.data.force_matrix_w

        print("\nforce_matrix_w:")
        print(self_forces)

        if self_forces is not None:

            print("\nforce_matrix_w shape:")
            print(self_forces.shape)

            # Expected:
            # (1, num_sensor_bodies, num_filtered_bodies, 3)
            #
            # Example with 11 matched bodies:
            # (1, 11, 11, 3)

            self_force_mag = torch.norm(
                self_forces,
                dim=-1,
            )

            print("\nForce magnitude matrix:")
            print(self_force_mag)

            collision_pairs = (
                self_force_mag > self_collision_threshold
            ) & ~sc_exclude_mask.unsqueeze(0)

            print("\nCollision pair matrix:")
            print(collision_pairs)

            self_collision = torch.any(
                collision_pairs,
                dim=(1, 2),
            )

            print("\nSELF COLLISION:")
            print(self_collision)

            # Print actual colliding body pairs
            collision_indices = torch.nonzero(
                collision_pairs[0],
                as_tuple=False,
            )

            if len(collision_indices) > 0:
                print("\nDetected collision pairs:")

                for i, j in collision_indices.tolist():
                    print(
                        f"{self_collision_sensor.body_names[i]}"
                        f" <-> "
                        f"{self_collision_sensor.body_names[j]}"
                        f" | force = "
                        f"{self_force_mag[0, i, j].item():.4f} N"
                    )
            else:
                print("\nNo self-collision detected.")

        else:
            print("force_matrix_w is None")

        print("========================================")            


if __name__ == "__main__":
    main()
    simulation_app.close()
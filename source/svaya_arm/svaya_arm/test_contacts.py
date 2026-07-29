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

    count = 0

    while simulation_app.is_running():

        # No actions
        # No reset
        # Just run physics

        sim.step()

        robot.update(sim.get_physics_dt())
        contact_sensor.update(sim.get_physics_dt())

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


if __name__ == "__main__":
    main()
    simulation_app.close()
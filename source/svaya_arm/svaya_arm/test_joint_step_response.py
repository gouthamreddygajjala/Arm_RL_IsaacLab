"""
Standalone PD-gain step-response test for the DARM left arm.

WHY THIS EXISTS
----------------
The stiffness/damping values currently on the USD (darm_fixed.usd) are not
trusted (likely leftover importer defaults). This script lets you derive a
starting Kp/Kd per joint from real physics — reflected inertia + a target
closed-loop bandwidth — and then verify the result with a direct step
response, completely decoupled from RL, reward shaping, or policy noise.

WHAT IT DOES
------------
1. Loads the DARM articulation in a single-env scene (no RL env, no rewards).
2. Reads per-link masses from the USD and estimates the *reflected inertia*
   at each left-arm joint (crude point-mass approximation: sum of
   mass_i * dist(joint, link_i_COM)^2 for every link distal to that joint).
3. Proposes Kp, Kd per joint from:
       Kp = I_eff * omega_n^2
       Kd = 2 * zeta * I_eff * omega_n
   (standard 2nd-order closed-loop PD tuning)
4. Commands a step target on ONE joint (others held at their current pose)
   and logs joint_pos / joint_vel every control step so you can see the
   actual settling time, overshoot, and steady-state velocity.

USAGE
-----
    # Isaac Lab / Isaac Sim ships its own python wrapper - use it, not system python
    ./isaaclab.sh -p test_joint_step_response.py --joint left_arm_joint7 \
        --omega_n 20 --zeta 1.0 --step_deg 15 --steps 300

    # Or force explicit gains instead of the inertia-derived estimate:
    ./isaaclab.sh -p test_joint_step_response.py --joint left_arm_joint7 \
        --stiffness 230 --damping 0.3 --step_deg 15 --steps 300

    # Sweep all 7 left-arm joints in one run, one after another:
    ./isaaclab.sh -p test_joint_step_response.py --joint all \
        --omega_n 20 --zeta 1.0 --step_deg 15 --steps 300

NOTE ON API CALLS
------------------
Mass/COM access (`get_masses`, body world positions) uses the Isaac Lab
Articulation data API as of recent releases. If `robot.root_physx_view.get_masses()`
or `robot.data.body_pos_w` don't match your installed version's signature,
check `isaaclab.assets.Articulation` / `ArticulationData` for the equivalent
call in your release - the rest of the script (inertia math, gain formulas,
step-response logging) does not depend on that detail.
"""

from __future__ import annotations

import argparse

parser = argparse.ArgumentParser(description="DARM joint PD gain step-response test.")
parser.add_argument("--joint", type=str, default="left_arm_joint7",
                     help="Joint name to test, or 'all' to sweep all 7 left-arm joints.")
parser.add_argument("--stiffness", type=float, default=None,
                     help="Explicit Kp override. If omitted, Kp is derived from reflected inertia.")
parser.add_argument("--damping", type=float, default=None,
                     help="Explicit Kd override. If omitted, Kd is derived from reflected inertia.")
parser.add_argument("--omega_n", type=float, default=20.0,
                     help="Target closed-loop natural frequency [rad/s] used to derive Kp/Kd.")
parser.add_argument("--zeta", type=float, default=1.0,
                     help="Target damping ratio (1.0 = critically damped, no overshoot).")
parser.add_argument("--step_deg", type=float, default=15.0,
                     help="Step command size in degrees.")
parser.add_argument("--steps", type=int, default=300,
                     help="Number of control steps to log after the step command.")
parser.add_argument("--headless", action="store_true", default=True)
args, _ = parser.parse_known_args()

# ---------------------------------------------------------------------------
# Isaac Lab app must be launched before importing anything from isaaclab.*
# ---------------------------------------------------------------------------
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

import torch
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

from robots.darm_fixed import DARM_CONFIG, left_dof_names


# ---------------------------------------------------------------------------
# Scene: just the robot, one env
# ---------------------------------------------------------------------------

@configclass
class TestSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = DARM_CONFIG.replace(prim_path="/World/envs/env_.*/Robot")


def estimate_reflected_inertia(robot: Articulation, joint_names: list[str]) -> dict[str, float]:
    """Crude reflected-inertia estimate per joint: sum over all links distal
    to the joint of mass_i * dist(joint_pos, link_i_com)^2, evaluated at the
    robot's current pose. This is a point-mass approximation (ignores each
    link's own rotational inertia tensor) - good enough for an initial Kp/Kd
    guess, not a substitute for the step-response test itself.
    """
    body_names = robot.data.body_names
    body_pos_w = robot.data.body_pos_w[0]  # (num_bodies, 3), env 0

    # masses - adjust this call if your Isaac Lab version exposes it differently
    try:
        masses = robot.root_physx_view.get_masses()[0]  # (num_bodies,)
    except AttributeError:
        raise RuntimeError(
            "robot.root_physx_view.get_masses() not found - check your Isaac Lab "
            "version's Articulation/ArticulationData API for the equivalent mass "
            "accessor and swap it in here."
        )

    name_to_idx = {n: i for i, n in enumerate(body_names)}

    # left-arm chain, distal-order: l1..l7, ee is captured by l7's mass onward
    # if there are separate wrist/tool bodies, extend this list to match your USD
    chain = [f"l{i+1}_left_arm" for i in range(7)]
    chain = [n for n in chain if n in name_to_idx]  # keep only bodies that exist

    i_eff = {}
    for jname in joint_names:
        # joint index k -> approx joint position = position of link l{k+1}_left_arm
        # (the link the joint directly actuates)
        k = int(jname.replace("left_arm_joint", "")) - 1
        if k >= len(chain):
            continue
        joint_body = chain[k]
        joint_pos = body_pos_w[name_to_idx[joint_body]]

        total = 0.0
        for link_name in chain[k:]:  # this link and everything distal to it
            idx = name_to_idx[link_name]
            m = masses[idx].item()
            d = torch.norm(body_pos_w[idx] - joint_pos).item()
            total += m * d * d

        # floor to avoid divide-by-zero on a zero-mass/zero-distance edge case
        i_eff[jname] = max(total, 1e-5)

    return i_eff


def derive_gains(i_eff: float, omega_n: float, zeta: float) -> tuple[float, float]:
    kp = i_eff * omega_n ** 2
    kd = 2.0 * zeta * i_eff * omega_n
    return kp, kd


def run_step_response(scene: InteractiveScene, sim: sim_utils.SimulationContext,
                       joint_name: str, kp: float, kd: float,
                       step_deg: float, num_steps: int) -> None:
    robot: Articulation = scene["robot"]
    joint_idx, _ = robot.find_joints([joint_name])
    joint_idx = joint_idx[0]

    # apply gains for this joint only
    stiffness = robot.data.joint_stiffness.clone()
    damping = robot.data.joint_damping.clone()
    stiffness[:, joint_idx] = kp
    damping[:, joint_idx] = kd
    robot.write_joint_stiffness_to_sim(stiffness)
    robot.write_joint_damping_to_sim(damping)

    start_pos = robot.data.joint_pos[:, joint_idx].clone()
    target_pos = start_pos + math.radians(step_deg)

    pos_target = robot.data.joint_pos_target.clone()
    pos_target[:, joint_idx] = target_pos
    robot.set_joint_position_target(target_pos, joint_ids=[joint_idx])

    print(f"\n=== {joint_name}: Kp={kp:.3f} Kd={kd:.3f} step={step_deg:.1f} deg ===")
    print(f"{'step':>5} {'t[s]':>7} {'pos[deg]':>10} {'vel[deg/s]':>12} {'err[deg]':>10}")

    target_deg = math.degrees(target_pos[0].item())
    for step in range(num_steps):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())

        pos_deg = math.degrees(robot.data.joint_pos[0, joint_idx].item())
        vel_deg = math.degrees(robot.data.joint_vel[0, joint_idx].item())
        err_deg = target_deg - pos_deg

        if step % 10 == 0 or step == num_steps - 1:
            t = step * sim.get_physics_dt()
            print(f"{step:5d} {t:7.3f} {pos_deg:10.3f} {vel_deg:12.3f} {err_deg:10.3f}")

    final_vel = abs(math.degrees(robot.data.joint_vel[0, joint_idx].item()))
    final_err = abs(target_deg - math.degrees(robot.data.joint_pos[0, joint_idx].item()))
    print(f"--- final: |err|={final_err:.4f} deg, |vel|={final_vel:.4f} deg/s ---")


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1 / 120, device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(2.5, 2.5, 2.5), target=(0.0, 0.0, 0.5))

    scene_cfg = TestSceneCfg(num_envs=1, env_spacing=4.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    robot: Articulation = scene["robot"]

    joint_names = left_dof_names if args.joint == "all" else [args.joint]

    print("\nEstimating reflected inertia from USD link masses (point-mass approx)...")
    i_eff_map = estimate_reflected_inertia(robot, joint_names)
    for jn, ie in i_eff_map.items():
        print(f"  {jn}: I_eff ~= {ie:.6f} kg*m^2")

    for jname in joint_names:
        if args.stiffness is not None and args.damping is not None:
            kp, kd = args.stiffness, args.damping
        else:
            i_eff = i_eff_map[jname]
            kp, kd = derive_gains(i_eff, args.omega_n, args.zeta)
            print(f"\nDerived gains for {jname} (omega_n={args.omega_n}, zeta={args.zeta}): "
                  f"Kp={kp:.3f}, Kd={kd:.3f}")

        run_step_response(scene, sim, jname, kp, kd, args.step_deg, args.steps)

    simulation_app.close()


if __name__ == "__main__":
    main()
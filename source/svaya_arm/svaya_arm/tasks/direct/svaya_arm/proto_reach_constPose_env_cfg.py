# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from svaya_arm.robots.darm_fixed import DARM_CONFIG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
import isaaclab.sim as sim_utils

from isaaclab.sensors import ContactSensorCfg


@configclass
class ReachConstTargetEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    # - spaces definition
    action_space = 7
    observation_space = 35 # 14 (joint pos+vel) + 7 (EE pose) + 7 (target pose) + 7 (Past actions)
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot(s)
    robot_cfg: ArticulationCfg = DARM_CONFIG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=15, env_spacing=4.0, replicate_physics=True
    )

    # custom parameters/scales
    # - controllable joint
    left_dof_names = [f"left_arm_joint{i+1}" for i in range(7)]
    right_dof_names = [f"right_arm_joint{i+1}" for i in range(7)]

    dof_names = left_dof_names + right_dof_names

    ee_link_name = "toolio_left_arm"

    # - action scale
    action_scale = 1.0  # [N]

    # target sampling: sphere around robot base (in robot's local frame)
    # (unused by the current reach-constant-target reset logic, kept for reference)
    target_pos_min = [0.2, -0.1, 0.3]  # [m] min xyz offset from base
    target_pos_max = [0.6, 0.5, 0.7]  # [m] max xyz offset from base

    # --- fixed target joint configuration ---
    # the EE pose reached by FK from these 7 left-arm joint angles becomes the
    # single constant target used every episode. If None, the target is FK'd
    # from the robot's default_joint_pos instead.

    # target_joint_pos: list | None = None

    # target_joint_pos = [0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.5]  # [rad] left-arm joint angles for the constant target
    target_joint_pos = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]  # [rad] left-arm joint angles for the constant target

    max_joint_delta = 0.02 #0.05 # [rad] per control step — tune this, see note below

    # reward scales
    rew_scale_alive = 0.5
    rew_scale_pos_error = -5.0  # penalise EE-to-target distance
    rew_scale_rot_error = -5.0  #-1.0 # penalise orientation error
    rew_scale_reach_bonus = 15.0  #10.0 # sparse bonus for being close enough
    rew_scale_joint_limit = -2.0  # penalty for approaching/hitting joint limits
    rew_scale_termination = -75.0  # penalty for termination (collision, joint limit, etc.)
    rew_scale_stability = 30.0  # bonus for keeping EE stable (low velocity) when close to target
    rew_scale_action_rate = -0.5
    stability_vel_scale = 3.0  #3000
    rew_scale_centering = -0.1 

    # thresholds
    pos_success_threshold = 0.01 #0.05  # [m] EE within 5 cm → success
    rot_success_threshold = 0.1  # [rad] EE orientation within 0.1 rad → success
    joint_limit_margin = 0.05
    
    
    # # --- visualization markers ---
    # goal_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
    #     prim_path="/Visuals/goal_marker",
    #     markers={
    #         "sphere": sim_utils.SphereCfg(
    #             radius=0.05,
    #             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),  # green
    #         )
    #     },
    # )

    # ee_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
    #     prim_path="/Visuals/ee_marker",
    #     markers={
    #         "sphere": sim_utils.SphereCfg(
    #             radius=0.05,
    #             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),  # red
    #         )
    #     },
    # )

    # sensor 1: arm links vs ground
    ground_contact_sensor_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/l[1-7]_left_arm",
        update_period=0.0,
        history_length=3,
        debug_vis=False,
        filter_prim_paths_expr=["/World/ground"],
    )

    # sensor 2: self collision — arm links vs everything except ground
    # self_collision_sensor_cfg: ContactSensorCfg = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/Robot/l[1-7]_left_arm",
    #     update_period=0.0,
    #     history_length=3,
    #     debug_vis=False,
    #     # no filter — we handle adjacent exclusion in _get_dones
    # )

    # one regex covering every body you care about for self-collision purposes
    SELF_COLLISION_BODIES = (
        "/World/envs/env_.*/Robot/(l[1-7]_left_arm|torso|torso_base|head|neck)"
    )

    self_collision_sensor_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path=SELF_COLLISION_BODIES,
        update_period=0.0,
        history_length=3,
        debug_vis=False,
        filter_prim_paths_expr=[SELF_COLLISION_BODIES],
    )

    # # --- sensor 2: arm vs torso/head/neck ---
    # arm_body_collision_sensor_cfg: ContactSensorCfg = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/Robot/l[1-7]_left_arm",
    #     update_period=0.0,
    #     history_length=3,
    #     debug_vis=False,
    #     filter_prim_paths_expr=[
    #         "/World/envs/env_.*/Robot/torso",
    #         "/World/envs/env_.*/Robot/torso_base",
    #         "/World/envs/env_.*/Robot/head",
    #         "/World/envs/env_.*/Robot/neck",
    #     ],
    # )

    # collision termination threshold [N] — contacts below this are ignored
    # (avoids terminating on tiny numerical contacts)
    # contact_force_threshold = 10.0  # [N]

    ground_contact_threshold = 5.0  # [N] any arm link touching ground
    self_collision_threshold = 20.0  # [N] self collision above this force → terminate (should be higher than contact_force_threshold to avoid false positives)

    goal_marker_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/goal_marker"
    )
    ee_marker_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/ee_marker"
    )

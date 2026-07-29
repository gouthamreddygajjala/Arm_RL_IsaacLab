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
class ReachTargetEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 7.0
    # - spaces definition
    action_space = 7
    observation_space = 35 # 14 (joint pos+vel) + 7 (EE pose) + 7 (target pose) + 7 (Past actions)
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot(s)
    robot_cfg: ArticulationCfg = DARM_CONFIG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=15, env_spacing=4.0, replicate_physics=True)

    # custom parameters/scales
    # - controllable joint
    left_dof_names = [f"left_arm_joint{i+1}" for i in range(7)]
    right_dof_names = [f"right_arm_joint{i+1}" for i in range(7)]

    dof_names = left_dof_names + right_dof_names

    ee_link_name = "toolio_left_arm"

    # - action scale
    action_scale = 1.0  # [N]

    # target sampling: sphere around robot base (in robot's local frame)
    target_pos_min = [0.2, -0.1, 0.3]   # [m] min xyz offset from base
    target_pos_max = [0.6,  0.5, 0.7]   # [m] max xyz offset from base

    # reward scales
    rew_scale_alive = 0.5
    rew_scale_pos_error = -5.0          # penalise EE-to-target distance
    rew_scale_rot_error = -1.0          # penalise orientation error
    rew_scale_reach_bonus = 10.0        # sparse bonus for being close enough
    rew_scale_joint_limit = -2.0        # penalty for approaching/hitting joint limits
    rew_scale_termination = -75.0  # penalty for termination (collision, joint limit, etc.) 
    # thresholds
    pos_success_threshold = 0.05        # [m] EE within 5 cm → success
    joint_limit_margin = 0.05

    max_joint_delta = 0.05 # [rad] per control step — tune this, see note below


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

    # collision termination threshold [N] — contacts below this are ignored
    # (avoids terminating on tiny numerical contacts)
    # contact_force_threshold = 10.0  # [N]

    ground_contact_threshold = 5.0    # [N] any arm link touching ground
    self_collision_threshold  = 20.0   # [N] self collision above this force → terminate (should be higher than contact_force_threshold to avoid false positives)

    goal_marker_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/goal_marker"
    )
    ee_marker_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/ee_marker"
    )
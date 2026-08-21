# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from svaya_arm.robots.darm_fixed import DARM_CONFIG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG

from isaaclab.sensors import ContactSensorCfg


@configclass
class ReachConstTargetEnvCfg(DirectRLEnvCfg):
    # ------------------------------------------------------------------
    # env
    # ------------------------------------------------------------------
    decimation = 2
    episode_length_s = 5.0

    action_space = 7
    # left_pos(7) + left_vel(7) + ee_pos(3) + ee_quat(4) + target_pose(7) + last_action(7) = 35
    observation_space = 35
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    robot_cfg: ArticulationCfg = DARM_CONFIG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=15, env_spacing=4.0, replicate_physics=True
    )

    # ------------------------------------------------------------------
    # controllable joints
    # ------------------------------------------------------------------
    left_dof_names = [f"left_arm_joint{i+1}" for i in range(7)]
    right_dof_names = [f"right_arm_joint{i+1}" for i in range(7)]
    dof_names = left_dof_names + right_dof_names 

    ee_link_name = "toolio_left_arm"

    # per-control-step joint delta action, radians
    max_joint_delta = 0.02

    # ------------------------------------------------------------------
    # fixed target: EE pose reached by FK from these left-arm joint angles.
    # If None, target is FK'd from the robot's default_joint_pos instead.
    # ------------------------------------------------------------------
    target_joint_pos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # [rad]

    # ------------------------------------------------------------------
    # reward scales — dense pos, dense rot, sparse success, stability,
    # joint-limit penalty, action smoothness, joint velocity, centering
    # ------------------------------------------------------------------
    rew_scale_alive = 0.5
    rew_scale_termination = -75.0

    rew_scale_pos_error = -5.0
    rew_scale_rot_error = -10.0

    rew_scale_reach_bonus = 500.0
    rew_scale_stability = 30.0

    rew_scale_joint_limit = -2.0
    rew_scale_action_rate = -3.0
    rew_scale_joint_vel = -1.0
    rew_scale_centering = -0.5

    # extra damping penalty that ramps in smoothly as the EE approaches the
    # target (weighted by proximity, not gated on/off) — gives the policy a
    # gradient to slow down on approach instead of relying only on the
    # post-hoc stability bonus once at_target is already true.
    # NOTE: was -60.0, which was strong enough to make the policy avoid the
    # target region entirely (reach bonus collapsed to ~0) — cut way back.
    rew_scale_vel_at_target = -12.0

    # redundant joints used for the centering term (indices into left_dof_names)
    redundant_joint_ids = [4, 6]

    # ------------------------------------------------------------------
    # success / settling thresholds
    # ------------------------------------------------------------------
    pos_success_threshold = 0.02   # [m]   EE within 2 cm of target
    rot_success_threshold = 0.08   # [rad] EE orientation within ~4.6 deg
    vel_ok_threshold = 0.05        # ||lin_vel|| + 0.5*||ang_vel|| — "settled"
    stability_vel_scale = 3.0
    joint_limit_margin = 0.05

    # ------------------------------------------------------------------
    # contact sensors
    # ------------------------------------------------------------------
    ground_contact_sensor_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/l[1-7]_left_arm",
        update_period=0.0,
        history_length=3,
        debug_vis=False,
        filter_prim_paths_expr=["/World/ground"],
    )

    # covers arm links + torso/head/neck so adjacency exclusion can be built
    # against the full body set, not just the arm links
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

    ground_contact_threshold = 5.0   # [N]
    self_collision_threshold = 20.0  # [N]

    # ------------------------------------------------------------------
    # visualization markers
    # ------------------------------------------------------------------
    goal_marker_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/goal_marker"
    )
    ee_marker_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/ee_marker"
    )
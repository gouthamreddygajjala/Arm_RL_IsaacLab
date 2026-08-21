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
class ReachTargetEnvCfg(DirectRLEnvCfg):
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

    # per-control-step joint delta action, radians.
    # Was 0.02 — bumped to 0.05. At action std ~0.05 (normalized), real
    # exploration noise was only ~0.001 rad/step at the old value, which
    # is effectively no exploration at all. This doesn't fix *why* std
    # collapsed (check entropy_coef / init std in your PPO config if it
    # collapses again), but it buys more physical movement per unit of
    # whatever exploration noise the policy does have.
    max_joint_delta = 0.05

    # ------------------------------------------------------------------
    # home pose: every episode STARTS here (all-zero left-arm joints).
    # Target is what's random, sampled fresh every reset — see the
    # curriculum block below.
    # ------------------------------------------------------------------
    home_joint_pos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # [rad]

    # margin (rad) kept away from each joint's soft limit when sampling a
    # random TARGET config — deliberately wider than joint_limit_margin
    # below so sampled targets aren't right at the edge of reachability /
    # right where self-collision starts becoming likely. Also acts as the
    # hard clamp the curriculum offset below expands into.
    target_sample_margin = 0.15

    # ------------------------------------------------------------------
    # target curriculum: sample uniformly from home ± target_offset
    # (clamped to the margin-shrunk joint range), instead of immediately
    # sampling the full range. Asking the policy to solve arbitrary
    # full-range 7-DOF targets from home, from iteration 0, gives it no
    # early path to the sparse reach bonus. Start close, expand once
    # at_target_frac is consistently high (tracked via an EMA in the
    # env). Once target_offset exceeds the per-joint margin-shrunk range
    # it's equivalent to full-range sampling.
    # ------------------------------------------------------------------
    target_offset_init = 0.25              # [rad] starting half-range around home
    target_offset_step = 0.25              # [rad] growth increment when threshold hit
    target_offset_growth_threshold = 0.6   # rolling at_target_frac needed to grow
    target_offset_ema_decay = 0.98         # smoothing factor for the rolling at_target_frac

    # proxy self-collision filter used only while sampling target configs
    # (see _self_collision_proxy in the env) — sphere-approximation min
    # distance [m] between non-adjacent tracked link origins. Tune against
    # your robot's actual link thickness: too small and folded/colliding
    # configs slip through as targets, too large and you'll reject a lot of
    # otherwise-fine, tightly-folded reachable poses.
    target_sample_collision_radius = 0.08

    # max rejection-sampling attempts per env before giving up and using
    # the last (possibly still proxy-colliding) sample anyway
    target_sample_max_tries = 20

    # ------------------------------------------------------------------
    # debug: freeze one joint's commanded target in exactly one env (every
    # step, held at that env's current position) to test whether it
    # passively oscillates around a fixed setpoint — an actuator gain
    # symptom, independent of anything the policy does. The other 14 envs
    # train completely normally; this doesn't touch their actions.
    # Set debug_freeze_env_id = None to disable (default).
    # ------------------------------------------------------------------
    debug_freeze_env_id: int | None = 0
    debug_freeze_joint_idx = 6   # 0-indexed into left_dof_names — 6 = joint7

    # per-step CSV trace of the frozen joint's position/velocity, separate
    # from the sparse LOG_EVERY_N_STEPS print above (which is far too
    # coarse to see an oscillation waveform — only one sample every 200
    # steps). Set a path to enable; None disables it. Only written while
    # debug_freeze_env_id is also set.
    debug_freeze_log_path: str | None = "/home/darms/joint7_freeze_trace.csv"

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
    rew_scale_vel_at_target = -12.0

    # redundant joints used for the centering term (indices into left_dof_names)
    redundant_joint_ids = [4, 6]

    # ------------------------------------------------------------------
    # success / settling thresholds — kept as the final target throughout;
    # the curriculum above eases the task via target *distance*, not by
    # loosening these.
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
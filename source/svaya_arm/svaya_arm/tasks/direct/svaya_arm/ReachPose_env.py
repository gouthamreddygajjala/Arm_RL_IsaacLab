# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import Tuple

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import ContactSensor

from pxr import UsdPhysics, Usd

from .ReachPose_env_cfg import ReachTargetEnvCfg

# how often (in env steps) to print reward-component diagnostics
LOG_EVERY_N_STEPS = 200


class ReachTargetEnv(DirectRLEnv):
    cfg: ReachTargetEnvCfg

    def __init__(
        self, cfg: ReachTargetEnvCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)

        # Print actuator parameters for ENV 0 only
        for actuator_name, actuator in self.robot.actuators.items():
            print(f"[actuator] {actuator_name}:")
            print(f"  joint_names = {actuator.joint_names}")

            stiffness = actuator.stiffness
            damping = actuator.damping

            if hasattr(stiffness, "tolist"):
                stiffness = stiffness[0].tolist() if stiffness.ndim > 1 else stiffness.tolist()

            if hasattr(damping, "tolist"):
                damping = damping[0].tolist() if damping.ndim > 1 else damping.tolist()

            print(f"  stiffness = {stiffness}")
            print(f"  damping   = {damping}")

        # ---- joint index lookups ----
        self._left_dof_idx, _ = self.robot.find_joints(self.cfg.left_dof_names)
        self._right_dof_idx, _ = self.robot.find_joints(self.cfg.right_dof_names)

        ee_bodies, _ = self.robot.find_bodies(self.cfg.ee_link_name)
        self._ee_body_idx = ee_bodies[0]

        self._left_joint_limits = self.robot.data.soft_joint_pos_limits[
            0, self._left_dof_idx, :
        ]  # (7, 2)

        # ---- redundant-joint ids for centering, precomputed once (not per reward call) ----
        self._redundant_joint_ids = torch.tensor(
            self.cfg.redundant_joint_ids, device=self.device
        )

        # ---- home pose (fixed episode START now, not the target) ----
        self._home_joint_pos_left = torch.tensor(
            self.cfg.home_joint_pos, device=self.device
        )

        # ---- target curriculum state: uniform sampling radius (rad) around
        # home, grown over training once at_target_frac is consistently
        # high — see the EMA update in _get_rewards and usage in
        # _sample_target_pose. ----
        self._target_offset = float(self.cfg.target_offset_init)
        self._at_target_ema = 0.0

        # ---- self-collision adjacency exclusion mask ----
        self._sc_body_names = self._self_collision_sensor.body_names
        n = len(self._sc_body_names)
        name_to_idx = {name: i for i, name in enumerate(self._sc_body_names)}

        known_adjacent_pairs = [
            ("l1_left_arm", "l2_left_arm"), ("l2_left_arm", "l3_left_arm"),
            ("l3_left_arm", "l4_left_arm"), ("l4_left_arm", "l5_left_arm"),
            ("l5_left_arm", "l6_left_arm"), ("l6_left_arm", "l7_left_arm"),
            ("l1_left_arm", "torso"),
            ("torso_base", "torso"),
            ("torso", "neck"),
            ("neck", "head"),
        ]

        self._sc_exclude_mask = torch.zeros(n, n, dtype=torch.bool, device=self.device)
        for a, b in known_adjacent_pairs:
            if a in name_to_idx and b in name_to_idx:
                i, j = name_to_idx[a], name_to_idx[b]
                self._sc_exclude_mask[i, j] = True
                self._sc_exclude_mask[j, i] = True
            else:
                print(f"[warn] adjacency pair not found in sensor bodies: {a}, {b}")

        # ---- body indices (into robot.data.body_pos_w) for the same bodies
        # the self-collision sensor tracks, kept in self._sc_body_names order
        # so they line up with self._sc_exclude_mask. Used for a cheap,
        # physics-step-free proxy check when sampling target configs. ----
        sc_link_idx, sc_link_names = self.robot.find_bodies(list(self._sc_body_names))
        name_to_body_idx = {name: idx for name, idx in zip(sc_link_names, sc_link_idx)}
        self._sc_body_idx_ordered = torch.tensor(
            [name_to_body_idx[name] for name in self._sc_body_names], device=self.device
        )

        # ---- per-env target pose: (N, 7) — xyz + wxyz. Re-sampled every
        # reset in _reset_idx (via FK on a random joint config), so no
        # one-time FK probe here like the const-target env had. ----
        self.target_pose = torch.zeros(self.num_envs, 7, device=self.device)
        self.target_pose[:, 3] = 1.0

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        self.actions = torch.zeros(self.num_envs, len(self._left_dof_idx), device=self.device)
        self.prev_actions = torch.zeros(self.num_envs, len(self._left_dof_idx), device=self.device)

        self._step_counter = 0

        if self.cfg.goal_marker_cfg is not None:
            self._goal_marker = VisualizationMarkers(self.cfg.goal_marker_cfg)
            self._ee_marker = VisualizationMarkers(self.cfg.ee_marker_cfg)

        # ---- per-step CSV trace for the freeze test, decoupled from the
        # sparse LOG_EVERY_N_STEPS print (one sample every 200 steps is
        # too coarse to see an oscillation waveform) ----
        self._debug_freeze_log_file = None
        if self.cfg.debug_freeze_env_id is not None and self.cfg.debug_freeze_log_path is not None:
            self._debug_freeze_log_file = open(self.cfg.debug_freeze_log_path, "w")
            self._debug_freeze_log_file.write("step,pos_rad,vel_rad_s\n")

    def close(self):
        if self._debug_freeze_log_file is not None:
            self._debug_freeze_log_file.close()
            self._debug_freeze_log_file = None
        super().close()

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        ground_prim = self.sim.stage.GetPrimAtPath("/World/ground")
        for prim in Usd.PrimRange(ground_prim):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = self.robot

        self._ground_contact_sensor = ContactSensor(self.cfg.ground_contact_sensor_cfg)
        self._self_collision_sensor = ContactSensor(self.cfg.self_collision_sensor_cfg)
        self.scene.sensors["ground_contact"] = self._ground_contact_sensor
        self.scene.sensors["self_collision"] = self._self_collision_sensor

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions = self.actions.clone()
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        # actions are per-step joint deltas in [-1, 1], scaled to a small max
        # angular step per control tick — not absolute positions
        self.actions = torch.clamp(self.actions, -1.0, 1.0)
        delta = self.actions * self.cfg.max_joint_delta

        current_pos = self.joint_pos[:, self._left_dof_idx]
        joint_pos_target = current_pos + delta

        limit_margin = 0.1  # ~5.7 deg
        lower = self._left_joint_limits[:, 0] + limit_margin
        upper = self._left_joint_limits[:, 1] - limit_margin
        joint_pos_target = torch.clamp(joint_pos_target, lower.unsqueeze(0), upper.unsqueeze(0))

        # ---- debug only: freeze one joint's commanded target in ONE env,
        # every step re-holding it at that env's current physical position.
        # Every other env/joint gets the normal policy-commanded target
        # untouched, so this doesn't disturb training. Use this to check
        # whether the joint passively oscillates/rings around a held
        # setpoint (actuator gain issue) independent of anything the
        # policy is doing. Set cfg.debug_freeze_env_id = None to disable. ----
        if self.cfg.debug_freeze_env_id is not None:
            env_id = self.cfg.debug_freeze_env_id
            joint_idx = self.cfg.debug_freeze_joint_idx
            joint_pos_target[env_id, joint_idx] = current_pos[env_id, joint_idx]

        self.robot.set_joint_position_target(joint_pos_target, joint_ids=self._left_dof_idx)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        left_pos = self.joint_pos[:, self._left_dof_idx]
        left_vel = self.joint_vel[:, self._left_dof_idx]

        ee_pos_w = self.robot.data.body_pos_w[:, self._ee_body_idx, :]
        ee_quat_w = self.robot.data.body_quat_w[:, self._ee_body_idx, :]

        obs = torch.cat(
            [left_pos, left_vel, ee_pos_w, ee_quat_w, self.target_pose, self.actions],
            dim=-1,
        )  # 7+7+3+4+7+7 = 35

        return {"policy": obs}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        ee_pos_w = self.robot.data.body_pos_w[:, self._ee_body_idx, :]
        ee_quat_w = self.robot.data.body_quat_w[:, self._ee_body_idx, :]
        ee_lin_vel_w = self.robot.data.body_lin_vel_w[:, self._ee_body_idx, :]
        ee_ang_vel_w = self.robot.data.body_ang_vel_w[:, self._ee_body_idx, :]

        self._update_markers(ee_pos_w, ee_quat_w)

        target_pos = self.target_pose[:, :3]
        target_quat = self.target_pose[:, 3:]

        (total, rew_alive, rew_termination, rew_pos, rew_rot, rew_reach,
         rew_limit, rew_stability, rew_action_rate, rew_joint_vel,
         rew_centering, rew_vel_at_target, at_target) = compute_rewards(
            ee_pos_w, ee_quat_w, ee_lin_vel_w, ee_ang_vel_w,
            target_pos, target_quat,
            self.actions, self.prev_actions,
            self.joint_pos[:, self._left_dof_idx],
            self.joint_vel[:, self._left_dof_idx],
            self._left_joint_limits,
            self._redundant_joint_ids,
            self.reset_terminated,
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_termination,
            self.cfg.rew_scale_pos_error,
            self.cfg.rew_scale_rot_error,
            self.cfg.rew_scale_reach_bonus,
            self.cfg.rew_scale_stability,
            self.cfg.rew_scale_joint_limit,
            self.cfg.rew_scale_action_rate,
            self.cfg.rew_scale_joint_vel,
            self.cfg.rew_scale_centering,
            self.cfg.rew_scale_vel_at_target,
            self.cfg.pos_success_threshold,
            self.cfg.rot_success_threshold,
            self.cfg.vel_ok_threshold,
            self.cfg.stability_vel_scale,
            self.cfg.joint_limit_margin,
        )

        # ---- target curriculum: grow the sampling radius once the policy
        # is consistently reaching, tracked via an EMA so a single lucky
        # batch doesn't trigger growth ----
        at_target_frac = at_target.float().mean().item()
        self._at_target_ema = (
            self.cfg.target_offset_ema_decay * self._at_target_ema
            + (1.0 - self.cfg.target_offset_ema_decay) * at_target_frac
        )
        if self._at_target_ema > self.cfg.target_offset_growth_threshold:
            old_offset = self._target_offset
            self._target_offset += self.cfg.target_offset_step
            self._at_target_ema = 0.0  # avoid growing again next step off the same streak
            print(
                f"[curriculum] at_target_ema exceeded {self.cfg.target_offset_growth_threshold:.2f} "
                f"-> growing target_offset {old_offset:.3f} -> {self._target_offset:.3f} rad"
            )

        self._step_counter += 1

        # ---- freeze-test CSV trace: every step, no gating on LOG_EVERY_N_STEPS ----
        if self._debug_freeze_log_file is not None:
            env_id = self.cfg.debug_freeze_env_id
            joint_idx = self.cfg.debug_freeze_joint_idx
            trace_pos = self.joint_pos[env_id, self._left_dof_idx[joint_idx]].item()
            trace_vel = self.joint_vel[env_id, self._left_dof_idx[joint_idx]].item()
            self._debug_freeze_log_file.write(f"{self._step_counter},{trace_pos:.6f},{trace_vel:.6f}\n")
            if self._step_counter % 500 == 0:
                self._debug_freeze_log_file.flush()  # survive a crash/interrupt without losing the trace

        if self._step_counter % LOG_EVERY_N_STEPS == 0:
            with torch.no_grad():
                pos_err_diag = torch.norm(ee_pos_w - target_pos, dim=-1)
                dot_diag = torch.clamp(torch.abs(torch.sum(ee_quat_w * target_quat, dim=-1)), max=1.0)
                rot_err_diag = 2.0 * torch.acos(dot_diag)
                vel_norm_diag = torch.norm(ee_lin_vel_w, dim=-1) + 0.5 * torch.norm(ee_ang_vel_w, dim=-1)

            print(
                f"[step {self._step_counter}] pos={rew_pos.mean().item():.2f} "
                f"rot={rew_rot.mean().item():.2f} reach={rew_reach.mean().item():.2f} "
                f"stab={rew_stability.mean().item():.2f} limit={rew_limit.mean().item():.2f} "
                f"action_rate={rew_action_rate.mean().item():.2f} jvel={rew_joint_vel.mean().item():.2f} "
                f"vel_at_target={rew_vel_at_target.mean().item():.2f} "
                f"centering={rew_centering.mean().item():.2f} alive={rew_alive.mean().item():.2f} "
                f"term={rew_termination.mean().item():.2f} at_target_frac={at_target_frac:.3f} "
                f"TOTAL={total.mean().item():.2f}"
            )
            print(
                f"[step {self._step_counter}] pos_err[mean/min]={pos_err_diag.mean().item():.4f}/"
                f"{pos_err_diag.min().item():.4f} m  "
                f"rot_err[mean/min]={rot_err_diag.mean().item():.4f}/{rot_err_diag.min().item():.4f} rad  "
                f"vel_norm mean={vel_norm_diag.mean().item():.4f}  "
                f"at_target_ema={self._at_target_ema:.3f}  target_offset={self._target_offset:.3f} rad"
            )

            if torch.any(at_target):
                left_vel_at_target = self.joint_vel[:, self._left_dof_idx][at_target]  # (M, 7)
                per_joint_mean_abs_vel = left_vel_at_target.abs().mean(dim=0)  # (7,)
                joint_str = " ".join(
                    f"j{i+1}={v.item():.4f}" for i, v in enumerate(per_joint_mean_abs_vel)
                )
                print(f"[step {self._step_counter}] per-joint |vel| @ at_target: {joint_str}")

            if self.cfg.debug_freeze_env_id is not None:
                env_id = self.cfg.debug_freeze_env_id
                joint_idx = self.cfg.debug_freeze_joint_idx
                frozen_pos = self.joint_pos[env_id, self._left_dof_idx[joint_idx]].item()
                frozen_vel = self.joint_vel[env_id, self._left_dof_idx[joint_idx]].item()
                print(
                    f"[step {self._step_counter}] [debug env {env_id}] "
                    f"joint{joint_idx+1} pos={frozen_pos:.4f} rad  vel={frozen_vel:.4f} rad/s "
                    f"(target held fixed — nonzero vel here means passive oscillation, not policy)"
                )

        return total

    def _update_markers(self, ee_pos_w: torch.Tensor, ee_quat_w: torch.Tensor) -> None:
        self._goal_marker.visualize(
            translations=self.target_pose[:, :3],
            orientations=self.target_pose[:, 3:],
        )
        self._ee_marker.visualize(translations=ee_pos_w, orientations=ee_quat_w)

    # ------------------------------------------------------------------
    # Dones
    # ------------------------------------------------------------------

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.robot.data.joint_pos

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        left_pos = self.joint_pos[:, self._left_dof_idx]
        lower = self._left_joint_limits[:, 0]
        upper = self._left_joint_limits[:, 1]

        term_tolerance = 0.02  # rad — ignore tiny solver-level overshoot
        joint_violation = torch.any(
            (left_pos < (lower - term_tolerance).unsqueeze(0))
            | (left_pos > (upper + term_tolerance).unsqueeze(0)),
            dim=1,
        )

        self_forces = self._self_collision_sensor.data.force_matrix_w  # (N, B, B, 3)
        self_force_mag = torch.norm(self_forces, dim=-1)
        collision_pairs = (self_force_mag > self.cfg.self_collision_threshold) & ~self._sc_exclude_mask.unsqueeze(0)
        self_collision = torch.any(collision_pairs, dim=(1, 2))

        terminated = joint_violation | self_collision

        if terminated[0]:
            reasons = []
            if joint_violation[0]:
                reasons.append("joint_violation")
            if self_collision[0]:
                reasons.append("self_collision")
            pos_err = torch.norm(
                self.robot.data.body_pos_w[0, self._ee_body_idx] - self.target_pose[0, :3]
            ).item()
            print(f"[ep_len={self.episode_length_buf[0].item()}] TERMINATED: {reasons} | pos_error={pos_err:.3f}")

        return terminated, time_out

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        # ---- root pose/vel must be correct before any FK probe below ----
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # ---- sample this episode's random TARGET pose via a throwaway FK probe ----
        self._sample_target_pose(env_ids)

        # ---- actual episode start: fixed home pose (all-zero left-arm joints) ----
        home_joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        home_joint_pos[:, self._left_dof_idx] = self._home_joint_pos_left.unsqueeze(0)
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        self.robot.write_joint_state_to_sim(home_joint_pos, joint_vel, None, env_ids)
        self.robot.update(dt=0.0)

        self.joint_pos[env_ids] = home_joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0

    def _sample_target_pose(self, env_ids: torch.Tensor) -> None:
        """FK a random in-limits, (proxy) collision-free left-arm config to
        get this episode's target EE pose for `env_ids`, writing the result
        into self.target_pose.

        The random config is drawn uniformly from home ± self._target_offset
        (clamped to the margin-shrunk joint limits), not the full joint
        range. self._target_offset starts small (cfg.target_offset_init)
        and grows over training (see the EMA-gated update in _get_rewards)
        once at_target_frac is consistently high — so early episodes ask
        for nearby, easy-to-reach targets and the task difficulty ramps up
        only once the policy is actually succeeding. Once target_offset
        exceeds the margin-shrunk range on a given joint, sampling on that
        joint is effectively full-range.

        Rejection-samples against a cheap link-origin distance proxy — see
        _self_collision_proxy() — since exact mesh self-collision can only
        be read from the contact sensor after an actual physics step, and
        stepping physics here would advance every other (still-running) env
        in lockstep too. This is a coarse filter, not a guarantee; the real
        self-collision sensor in _get_dones is still the ground truth and
        will terminate early if a proxy-cleared config turns out to overlap.

        Temporarily moves the robot to each candidate config and calls
        robot.update(dt=0.0) — kinematics-only, no physics step — to read
        body_pos_w/body_quat_w. The caller must overwrite the joint state
        with the real start (home) pose right after this returns.
        """
        n = len(env_ids)
        margin = self.cfg.target_sample_margin
        joint_lower = self._left_joint_limits[:, 0] + margin
        joint_upper = self._left_joint_limits[:, 1] - margin

        offset = self._target_offset
        sample_lower = torch.clamp(self._home_joint_pos_left - offset, min=joint_lower)
        sample_upper = torch.clamp(self._home_joint_pos_left + offset, max=joint_upper)

        target_pos = torch.zeros(n, 3, device=self.device)
        target_quat = torch.zeros(n, 4, device=self.device)
        unresolved = torch.ones(n, dtype=torch.bool, device=self.device)

        for attempt in range(self.cfg.target_sample_max_tries):
            n_unresolved = int(unresolved.sum().item())
            if n_unresolved == 0:
                break

            sub_env_ids = env_ids[unresolved]

            rand = torch.rand((n_unresolved, 7), device=self.device)
            random_left_joint_pos = sample_lower.unsqueeze(0) + rand * (sample_upper - sample_lower).unsqueeze(0)

            probe_joint_pos = self.robot.data.default_joint_pos[sub_env_ids].clone()
            probe_joint_pos[:, self._left_dof_idx] = random_left_joint_pos
            zero_vel = torch.zeros_like(probe_joint_pos)

            self.robot.write_joint_state_to_sim(probe_joint_pos, zero_vel, None, sub_env_ids)
            self.robot.update(dt=0.0)

            colliding = self._self_collision_proxy(sub_env_ids)
            clear = ~colliding

            resolved_sub_idx = clear.nonzero(as_tuple=True)[0]
            if len(resolved_sub_idx) > 0:
                unresolved_idx = unresolved.nonzero(as_tuple=True)[0][resolved_sub_idx]
                target_pos[unresolved_idx] = self.robot.data.body_pos_w[
                    sub_env_ids[resolved_sub_idx], self._ee_body_idx, :
                ]
                target_quat[unresolved_idx] = self.robot.data.body_quat_w[
                    sub_env_ids[resolved_sub_idx], self._ee_body_idx, :
                ]
                unresolved[unresolved_idx] = False

        if torch.any(unresolved):
            leftover_idx = unresolved.nonzero(as_tuple=True)[0]
            leftover_env_ids = env_ids[leftover_idx]
            target_pos[leftover_idx] = self.robot.data.body_pos_w[leftover_env_ids, self._ee_body_idx, :]
            target_quat[leftover_idx] = self.robot.data.body_quat_w[leftover_env_ids, self._ee_body_idx, :]
            print(
                f"[warn] target_sample: {len(leftover_idx)} env(s) still proxy-colliding "
                f"after {self.cfg.target_sample_max_tries} tries, using anyway"
            )

        self.target_pose[env_ids, :3] = target_pos
        self.target_pose[env_ids, 3:] = target_quat

    def _self_collision_proxy(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Cheap, physics-step-free self-collision heuristic: flags an env if
        any two non-adjacent tracked links (same set/adjacency as the real
        self-collision sensor) have origins closer than
        cfg.target_sample_collision_radius. Sphere-approximation on link
        origins only — not exact mesh geometry. Tune the radius against your
        robot's actual link thickness; treat this as a filter that reduces
        obviously-folded configs, not a hard collision guarantee.
        """
        body_pos = self.robot.data.body_pos_w[env_ids][:, self._sc_body_idx_ordered, :]  # (n, B, 3)
        dist = torch.cdist(body_pos, body_pos)  # (n, B, B)

        b = dist.shape[-1]
        eye = torch.eye(b, dtype=torch.bool, device=self.device).unsqueeze(0)

        too_close = (dist < self.cfg.target_sample_collision_radius) & ~eye
        too_close = too_close & ~self._sc_exclude_mask.unsqueeze(0)

        return torch.any(too_close, dim=(1, 2))


# ------------------------------------------------------------------
# JIT reward kernel — unchanged from the const-target env, it only
# consumes ee pose vs target_pose and doesn't care how target_pose
# was produced, or how it was sampled (curriculum vs full-range).
# ------------------------------------------------------------------


@torch.jit.script
def compute_rewards(
    ee_pos: torch.Tensor,
    ee_quat: torch.Tensor,
    ee_lin_vel: torch.Tensor,
    ee_ang_vel: torch.Tensor,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    left_joint_pos: torch.Tensor,
    left_joint_vel: torch.Tensor,
    joint_limits: torch.Tensor,
    redundant_joint_ids: torch.Tensor,
    reset_terminated: torch.Tensor,
    rew_scale_alive: float,
    rew_scale_termination: float,
    rew_scale_pos_error: float,
    rew_scale_rot_error: float,
    rew_scale_reach_bonus: float,
    rew_scale_stability: float,
    rew_scale_joint_limit: float,
    rew_scale_action_rate: float,
    rew_scale_joint_vel: float,
    rew_scale_centering: float,
    rew_scale_vel_at_target: float,
    pos_threshold: float,
    rot_success_threshold: float,
    vel_ok_threshold: float,
    stability_vel_scale: float,
    joint_limit_margin: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor]:
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_termination * reset_terminated.float()

    # --- dense positional error ---
    pos_error = torch.norm(ee_pos - target_pos, dim=-1)
    rew_pos = rew_scale_pos_error * pos_error

    # --- dense orientation error ---
    dot = torch.clamp(torch.abs(torch.sum(ee_quat * target_quat, dim=-1)), max=1.0)
    rot_error = 2.0 * torch.acos(dot)
    rew_rot = rew_scale_rot_error * rot_error

    # --- at_target gates the stability bonus; at_target & vel_ok gates the sparse bonus ---
    vel_norm = torch.norm(ee_lin_vel, dim=-1) + 0.5 * torch.norm(ee_ang_vel, dim=-1)
    pos_ok = pos_error < pos_threshold
    rot_ok = rot_error < rot_success_threshold
    vel_ok = vel_norm < vel_ok_threshold
    at_target = pos_ok & rot_ok

    # --- sparse success bonus: requires full pose + settling, discourages touch-and-bounce ---
    rew_reach = rew_scale_reach_bonus * (at_target & vel_ok).float()

    # --- stability bonus: smooth exponential decay by velocity, gated on at_target only ---
    rew_stability = rew_scale_stability * torch.exp(-stability_vel_scale * vel_norm) * at_target.float()

    # --- proximity-weighted damping penalty: ramps in smoothly as pos_error
    # shrinks (not gated on at_target), so the policy gets a gradient to slow
    # down on approach rather than only being rewarded for stillness after
    # it has already arrived ---
    proximity_weight = torch.clamp(1.0 - pos_error / pos_threshold, min=0.0, max=1.0) ** 2
    rew_vel_at_target = rew_scale_vel_at_target * (vel_norm ** 2) * proximity_weight

    # --- joint limit proximity penalty ---
    lower = joint_limits[:, 0]
    upper = joint_limits[:, 1]
    dist_to_lower = left_joint_pos - lower.unsqueeze(0)
    dist_to_upper = upper.unsqueeze(0) - left_joint_pos
    min_dist = torch.minimum(dist_to_lower, dist_to_upper)
    limit_violation = torch.clamp(joint_limit_margin - min_dist, min=0.0)
    rew_limit = rew_scale_joint_limit * torch.sum(limit_violation, dim=-1)

    # --- action smoothness penalty ---
    action_rate_penalty = torch.sum((actions - prev_actions) ** 2, dim=-1)
    rew_action_rate = rew_scale_action_rate * action_rate_penalty

    # --- joint velocity penalty (squared: smooth gradient, discourages jitter) ---
    rew_joint_vel = rew_scale_joint_vel * torch.sum(left_joint_vel ** 2, dim=-1)

    # --- redundant joint centering — faded out near the target so it stops
    # competing with settling on low-authority joints (esp. j7, the lowest
    # stiffness joint and a centering target) right when precision matters most ---
    centers = (joint_limits[redundant_joint_ids, 0] + joint_limits[redundant_joint_ids, 1]) / 2.0
    ranges = joint_limits[redundant_joint_ids, 1] - joint_limits[redundant_joint_ids, 0]
    centering_err = torch.abs(left_joint_pos[:, redundant_joint_ids] - centers.unsqueeze(0)) / ranges.unsqueeze(0)
    rew_centering = rew_scale_centering * torch.sum(centering_err, dim=-1) * (1.0 - proximity_weight)

    total = (rew_alive + rew_termination + rew_pos + rew_rot + rew_reach
             + rew_limit + rew_stability + rew_action_rate + rew_joint_vel
             + rew_centering + rew_vel_at_target)

    return (total, rew_alive, rew_termination, rew_pos, rew_rot, rew_reach,
            rew_limit, rew_stability, rew_action_rate, rew_joint_vel, rew_centering,
            rew_vel_at_target, at_target)
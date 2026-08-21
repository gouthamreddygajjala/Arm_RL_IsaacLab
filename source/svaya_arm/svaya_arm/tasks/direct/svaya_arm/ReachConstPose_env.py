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

from .reach_constPose_env_cfg import ReachConstTargetEnvCfg

# how often (in env steps) to print reward-component diagnostics
LOG_EVERY_N_STEPS = 200


class ReachConstTargetEnv(DirectRLEnv):
    cfg: ReachConstTargetEnvCfg

    def __init__(
        self, cfg: ReachConstTargetEnvCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)

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

        # ---- per-env target pose: (N, 7) — xyz + wxyz ----
        self.target_pose = torch.zeros(self.num_envs, 7, device=self.device)
        self.target_pose[:, 3] = 1.0

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        self.actions = torch.zeros(self.num_envs, len(self._left_dof_idx), device=self.device)
        self.prev_actions = torch.zeros(self.num_envs, len(self._left_dof_idx), device=self.device)

        self._step_counter = 0

        # ---- compute the single constant target EE pose once, via FK ----
        all_env_ids = self.robot._ALL_INDICES

        if self.cfg.target_joint_pos is not None:
            fixed_left_joint_pos = (
                torch.tensor(self.cfg.target_joint_pos, device=self.device)
                .unsqueeze(0)
                .repeat(self.num_envs, 1)
            )
        else:
            fixed_left_joint_pos = self.robot.data.default_joint_pos[
                :, self._left_dof_idx
            ].clone()

        goal_full_joint_pos = self.robot.data.default_joint_pos.clone()
        goal_full_joint_pos[:, self._left_dof_idx] = fixed_left_joint_pos

        self.robot.write_joint_state_to_sim(
            goal_full_joint_pos,
            self.robot.data.default_joint_vel.clone(),
            None,
            all_env_ids,
        )
        self.robot.update(dt=0.0)

        self._fixed_target_pos = self.robot.data.body_pos_w[:, self._ee_body_idx, :].clone()
        self._fixed_target_quat = self.robot.data.body_quat_w[:, self._ee_body_idx, :].clone()

        self.target_pose[:, :3] = self._fixed_target_pos
        self.target_pose[:, 3:] = self._fixed_target_quat

        # restore default joint state after the FK probe
        self.robot.write_joint_state_to_sim(
            self.robot.data.default_joint_pos.clone(),
            self.robot.data.default_joint_vel.clone(),
            None,
            all_env_ids,
        )

        if self.cfg.goal_marker_cfg is not None:
            self._goal_marker = VisualizationMarkers(self.cfg.goal_marker_cfg)
            self._ee_marker = VisualizationMarkers(self.cfg.ee_marker_cfg)

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

        self._step_counter += 1
        if self._step_counter % LOG_EVERY_N_STEPS == 0:
            print(
                f"[step {self._step_counter}] pos={rew_pos.mean().item():.2f} "
                f"rot={rew_rot.mean().item():.2f} reach={rew_reach.mean().item():.2f} "
                f"stab={rew_stability.mean().item():.2f} limit={rew_limit.mean().item():.2f} "
                f"action_rate={rew_action_rate.mean().item():.2f} jvel={rew_joint_vel.mean().item():.2f} "
                f"vel_at_target={rew_vel_at_target.mean().item():.2f} "
                f"centering={rew_centering.mean().item():.2f} alive={rew_alive.mean().item():.2f} "
                f"term={rew_termination.mean().item():.2f} at_target_frac={at_target.float().mean().item():.3f} "
                f"TOTAL={total.mean().item():.2f}"
            )

            # per-joint velocity breakdown, restricted to envs currently at_target,
            # to see which joint(s) are driving the residual settling velocity
            if torch.any(at_target):
                left_vel_at_target = self.joint_vel[:, self._left_dof_idx][at_target]  # (M, 7)
                per_joint_mean_abs_vel = left_vel_at_target.abs().mean(dim=0)  # (7,)
                joint_str = " ".join(
                    f"j{i+1}={v.item():.4f}" for i, v in enumerate(per_joint_mean_abs_vel)
                )
                print(f"[step {self._step_counter}] per-joint |vel| @ at_target: {joint_str}")

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

        n = len(env_ids)

        self.target_pose[env_ids, :3] = self._fixed_target_pos[env_ids]
        self.target_pose[env_ids, 3:] = self._fixed_target_quat[env_ids]

        lower = self._left_joint_limits[:, 0]
        upper = self._left_joint_limits[:, 1]

        # bias start pose toward the center of joint range (Beta(2,2) peaks at 0.5)
        alpha = torch.distributions.Beta(2.0, 2.0).sample((n, 7)).to(self.device)
        start_joint_pos = lower.unsqueeze(0) + alpha * (upper - lower).unsqueeze(0)

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_pos[:, self._left_dof_idx] = start_joint_pos
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0


# ------------------------------------------------------------------
# JIT reward kernel
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
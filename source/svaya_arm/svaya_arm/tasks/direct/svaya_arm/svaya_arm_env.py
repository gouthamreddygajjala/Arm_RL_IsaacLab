# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_error_magnitude, sample_uniform

from isaaclab.markers import VisualizationMarkers

from isaaclab.sensors import ContactSensor



from .svaya_arm_env_cfg import SvayaArmEnvCfg


class SvayaArmEnv(DirectRLEnv):
    cfg: SvayaArmEnvCfg

    def __init__(self, cfg: SvayaArmEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # joint index lookups
        self._left_dof_idx, _ = self.robot.find_joints(self.cfg.left_dof_names)
        self._right_dof_idx, _ = self.robot.find_joints(self.cfg.right_dof_names)

        # end-effector body index
        ee_bodies, _ = self.robot.find_bodies(self.cfg.ee_link_name)
        self._ee_body_idx = ee_bodies[0]

        # joint limits for the left arm: shape (7, 2) → lower, upper
        self._left_joint_limits = self.robot.data.soft_joint_pos_limits[
            0, self._left_dof_idx, :
        ]  # (7, 2)

        # find torso/base body index — ground contact on this means topple (shouldn't happen
        # with fix_root_link, but good to keep for arm links hitting ground)
        # use all body indices except the fixed root
        all_body_names = self.robot.data.body_names
        self._non_root_body_ids, _ = self.robot.find_bodies(".*")  # all bodies


        # per-env target pose: (N, 7) — xyz + wxyz
        self.target_pose = torch.zeros(self.num_envs, 7, device=self.device)
        self.target_pose[:, 3] = 1.0  # unit quaternion w=1

        # cache joint state views
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        # visualization markers (only in non-headless mode)
        if self.cfg.goal_marker_cfg is not None:
            self._goal_marker = VisualizationMarkers(self.cfg.goal_marker_cfg)
            self._ee_marker = VisualizationMarkers(self.cfg.ee_marker_cfg)

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot

        # # register contact sensor
        # self._contact_sensor = ContactSensor(self.cfg.contact_sensor_cfg)
        # self.scene.sensors["contact_sensor"] = self._contact_sensor

            # two separate sensors
        self._ground_contact_sensor = ContactSensor(self.cfg.ground_contact_sensor_cfg)
        self._self_collision_sensor  = ContactSensor(self.cfg.self_collision_sensor_cfg)
        self.scene.sensors["ground_contact"] = self._ground_contact_sensor
        self.scene.sensors["self_collision"]  = self._self_collision_sensor

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    # def _apply_action(self) -> None:
    #     # control only left arm; right arm stays at default (no command written)
    #     self.robot.set_joint_effort_target(
    #         self.actions * self.cfg.action_scale,
    #         joint_ids=self._left_dof_idx,
    #     )

    def _apply_action(self) -> None:
        # actions are joint position targets in [-1, 1], scaled to joint range
        self.robot.set_joint_position_target(
            self.actions,  # directly target joint angles (radians)
            joint_ids=self._left_dof_idx,
        )

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        # refresh joint state
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        left_pos = self.joint_pos[:, self._left_dof_idx]   # (N, 7)
        left_vel = self.joint_vel[:, self._left_dof_idx]   # (N, 7)

        # end-effector pose in world frame
        ee_pos_w = self.robot.data.body_pos_w[:, self._ee_body_idx, :]   # (N, 3)
        ee_quat_w = self.robot.data.body_quat_w[:, self._ee_body_idx, :] # (N, 4) wxyz

        obs = torch.cat(
            [
                left_pos,          # 7
                left_vel,          # 7
                ee_pos_w,          # 3
                ee_quat_w,         # 4  → total EE: 7
                self.target_pose,  # 7  (xyz + wxyz)
            ],
            dim=-1,
        )  # total: 28

        return {"policy": obs}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        ee_pos_w = self.robot.data.body_pos_w[:, self._ee_body_idx, :]
        ee_quat_w = self.robot.data.body_quat_w[:, self._ee_body_idx, :]

        # update markers every reward step
        self._update_markers(ee_pos_w, ee_quat_w)

        target_pos = self.target_pose[:, :3]
        target_quat = self.target_pose[:, 3:]  # wxyz

        return compute_rewards(
            ee_pos_w,
            ee_quat_w,
            target_pos,
            target_quat,
            self.joint_pos[:, self._left_dof_idx],
            self._left_joint_limits,
            self.reset_terminated,
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_pos_error,
            self.cfg.rew_scale_rot_error,
            self.cfg.rew_scale_reach_bonus,
            self.cfg.rew_scale_joint_limit,
            self.cfg.pos_success_threshold,
            self.cfg.joint_limit_margin,
        )


    def _update_markers(self, ee_pos_w: torch.Tensor, ee_quat_w: torch.Tensor) -> None:
        # goal marker: target_pose[:, :3] is position, target_pose[:, 3:] is wxyz quat
        self._goal_marker.visualize(
            translations=self.target_pose[:, :3],
            orientations=self.target_pose[:, 3:],  # wxyz
        )
        # EE marker: actual end-effector pose
        self._ee_marker.visualize(
            translations=ee_pos_w,
            orientations=ee_quat_w,
        )

    # ------------------------------------------------------------------
    # Dones
    # ------------------------------------------------------------------

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.robot.data.joint_pos

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # terminate if any left arm joint violates soft limits
        left_pos = self.joint_pos[:, self._left_dof_idx]  # (N, 7)
        lower = self._left_joint_limits[:, 0]              # (7,)
        upper = self._left_joint_limits[:, 1]              # (7,)
        joint_violation = torch.any(
            (left_pos < lower.unsqueeze(0)) | (left_pos > upper.unsqueeze(0)),
            dim=1,
        )  # (N,)

        # --- ground contact: any arm link touching ground ---
        # net_forces_w: (N, num_bodies, 3) — filtered to ground contacts only
        ground_forces = self._ground_contact_sensor.data.net_forces_w       # (N, 7, 3)
        ground_force_mag = torch.norm(ground_forces, dim=-1)                 # (N, 7)
        ground_collision = torch.any(
            ground_force_mag > self.cfg.ground_contact_threshold, dim=1
        )  # (N,)

        # --- self collision: skip adjacent link pairs ---
        # net_forces_w here includes ALL contacts on each link
        # adjacent links (l1↔l2, l2↔l3, ...) always have joint forces → exclude them
        # strategy: check only non-adjacent links, i.e. links separated by 2+ joints
        # simplest proxy: if link i has high force AND link i+1 also has high force
        # it's likely a joint force — skip. Only flag if a non-adjacent link is involved.
        #
        # Practical approach: use a higher threshold and check force_matrix_w for body pairs
        self_forces = self._self_collision_sensor.data.net_forces_w          # (N, 7, 3)
        self_force_mag = torch.norm(self_forces, dim=-1)                     # (N, 7)

        # mask out adjacent link contributions:
        # if link[i] and link[i+1] BOTH exceed threshold, treat as joint force, not collision
        # real self-collision shows up as isolated high force on a non-adjacent link
        exceed = self_force_mag > self.cfg.self_collision_threshold          # (N, 7) bool

        # adjacent pair mask: (l1,l2), (l2,l3), ..., (l6,l7)
        # if both neighbors exceed threshold → likely joint force, zero them out
        adjacent_both = exceed[:, :-1] & exceed[:, 1:]                      # (N, 6)
        # build a mask of links that are part of an adjacent-both pair
        is_adjacent = torch.zeros_like(exceed)                               # (N, 7)
        is_adjacent[:, :-1] |= adjacent_both
        is_adjacent[:, 1:]  |= adjacent_both

        # only flag as self-collision if exceeds threshold AND not explained by adjacency
        real_self_collision = exceed & ~is_adjacent                          # (N, 7)
        self_collision = torch.any(real_self_collision, dim=1)               # (N,)

        # terminated = joint_violation | ground_collision | self_collision
        terminated = joint_violation 

        return terminated, time_out

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        n = len(env_ids)

        # --- sample random joint angles within soft limits for goal FK ---
        lower = self._left_joint_limits[:, 0]  # (7,)
        upper = self._left_joint_limits[:, 1]  # (7,)

        # # uniform sample in [lower, upper] for each of the 7 joints
        # alpha = torch.rand(n, 7, device=self.device)
        # goal_joint_pos = lower.unsqueeze(0) + alpha * (upper - lower).unsqueeze(0)  # (N, 7)

        # bias toward center of joint range instead of uniform
        # beta distribution with a=b=2 peaks at 0.5 (center of range)
        alpha = torch.distributions.Beta(2.0, 2.0).sample((n, 7)).to(self.device)
        goal_joint_pos = lower.unsqueeze(0) + alpha * (upper - lower).unsqueeze(0)

        # --- write goal joints to sim temporarily to read FK ---
        # build a full joint pos tensor (all joints) starting from default
        goal_full_joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        goal_full_joint_pos[:, self._left_dof_idx] = goal_joint_pos

        # write to sim, step kinematics, read EE pose
        self.robot.write_joint_state_to_sim(
            goal_full_joint_pos,
            self.robot.data.default_joint_vel[env_ids],
            None,
            env_ids,
        )
        self.robot.update(dt=0.0)  # force kinematic update without physics step

        # read EE pose — this is your FK-derived goal
        goal_ee_pos  = self.robot.data.body_pos_w[env_ids, self._ee_body_idx, :].clone()   # (N, 3)
        goal_ee_quat = self.robot.data.body_quat_w[env_ids, self._ee_body_idx, :].clone()  # (N, 4)

        # store as target
        self.target_pose[env_ids, :3] = goal_ee_pos
        self.target_pose[env_ids, 3:] = goal_ee_quat

        # --- now reset robot to default pose (not the goal joints) ---
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel


    # def _reset_idx(self, env_ids: Sequence[int] | None):
    #     if env_ids is None:
    #         env_ids = self.robot._ALL_INDICES
    #     super()._reset_idx(env_ids)

    #     n = len(env_ids)

    #     # reset robot to default pose
    #     joint_pos = self.robot.data.default_joint_pos[env_ids]
    #     joint_vel = self.robot.data.default_joint_vel[env_ids]

    #     default_root_state = self.robot.data.default_root_state[env_ids].clone()
    #     default_root_state[:, :3] += self.scene.env_origins[env_ids]

    #     self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
    #     self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
    #     self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    #     # sample new random target positions for reset envs
    #     target_pos_min = torch.tensor(self.cfg.target_pos_min, device=self.device)
    #     target_pos_max = torch.tensor(self.cfg.target_pos_max, device=self.device)

    #     rand_pos = sample_uniform(
    #         target_pos_min, target_pos_max, (n, 3), device=self.device
    #     )
    #     # offset by env origin so targets are in world frame
    #     rand_pos += self.scene.env_origins[env_ids]

    #     # identity quaternion (wxyz) — orientation goal
    #     # swap for a random quat later if you need full 6-DoF reaching
    #     # rand_quat = torch.zeros(n, 4, device=self.device)
    #     # rand_quat[:, 0] = 1.0  # w=1

    #     rand_quat = random_quaternions(n, self.device) 

    #     self.target_pose[env_ids, :3] = rand_pos
    #     self.target_pose[env_ids, 3:] = rand_quat

    #     self.joint_pos[env_ids] = joint_pos
    #     self.joint_vel[env_ids] = joint_vel


# ------------------------------------------------------------------
# JIT reward kernel
# ------------------------------------------------------------------

@torch.jit.script
def compute_rewards(
    ee_pos: torch.Tensor,
    ee_quat: torch.Tensor,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    left_joint_pos: torch.Tensor,
    joint_limits: torch.Tensor,
    reset_terminated: torch.Tensor,
    rew_scale_alive: float,
    rew_scale_pos_error: float,
    rew_scale_rot_error: float,
    rew_scale_reach_bonus: float,
    rew_scale_joint_limit: float,
    pos_threshold: float,
    joint_limit_margin: float,
) -> torch.Tensor:
    # --- alive reward ---
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())

    # --- positional error ---
    pos_error = torch.norm(ee_pos - target_pos, dim=-1)        # (N,)
    rew_pos = rew_scale_pos_error * pos_error

    # --- orientation error (vectorized, no loop, no external fn) ---
    # dot product between quaternions, clamped to [0, 1] to handle float drift
    # |dot| = cos(half_angle), so angle = 2 * acos(|dot|)
    dot = torch.clamp(torch.abs(torch.sum(ee_quat * target_quat, dim=-1)), max=1.0)  # (N,)
    rot_error = 2.0 * torch.acos(dot)                          # (N,) geodesic angle
    rew_rot = rew_scale_rot_error * rot_error

    # --- sparse success bonus ---
    rew_reach = rew_scale_reach_bonus * (pos_error < pos_threshold).float()

    # --- joint limit proximity penalty ---
    lower = joint_limits[:, 0]                                  # (7,)
    upper = joint_limits[:, 1]                                  # (7,)
    dist_to_lower = left_joint_pos - lower.unsqueeze(0)        # (N, 7)
    dist_to_upper = upper.unsqueeze(0) - left_joint_pos        # (N, 7)
    min_dist = torch.minimum(dist_to_lower, dist_to_upper)     # (N, 7)
    limit_violation = torch.clamp(joint_limit_margin - min_dist, min=0.0)
    rew_limit = rew_scale_joint_limit * torch.sum(limit_violation, dim=-1)

    return rew_alive + rew_pos + rew_rot + rew_reach + rew_limit


@torch.jit.script
def random_quaternions(n: int, device: str) -> torch.Tensor:
    """Sample n uniform random unit quaternions (wxyz)."""
    # Shoemake 1992 method — uniform over SO(3)
    u = torch.rand(n, 3, device=device)
    
    sqrt1_u0 = torch.sqrt(1.0 - u[:, 0])
    sqrt_u0  = torch.sqrt(u[:, 0])
    theta1   = 2.0 * torch.pi * u[:, 1]
    theta2   = 2.0 * torch.pi * u[:, 2]

    w = sqrt1_u0 * torch.sin(theta1)
    x = sqrt1_u0 * torch.cos(theta1)
    y = sqrt_u0  * torch.sin(theta2)
    z = sqrt_u0  * torch.cos(theta2)

    return torch.stack([w, x, y, z], dim=-1)  # (N, 4) wxyz
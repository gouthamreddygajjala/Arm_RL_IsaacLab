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

from pxr import UsdPhysics, Usd

from typing import Tuple



from .reach_constPose_env_cfg import ReachConstTargetEnvCfg


class ReachConstTargetEnv(DirectRLEnv):
    cfg: ReachConstTargetEnvCfg

    def __init__(
        self, cfg: ReachConstTargetEnvCfg, render_mode: str | None = None, **kwargs
    ):
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


        self._sc_body_names = self._self_collision_sensor.body_names
        n = len(self._sc_body_names)
        name_to_idx = {name: i for i, name in enumerate(self._sc_body_names)}

        KNOWN_ADJACENT_PAIRS = [
            ("l1_left_arm", "l2_left_arm"), ("l2_left_arm", "l3_left_arm"),
            ("l3_left_arm", "l4_left_arm"), ("l4_left_arm", "l5_left_arm"),
            ("l5_left_arm", "l6_left_arm"), ("l6_left_arm", "l7_left_arm"),
            ("l1_left_arm", "torso"),
            ("torso_base", "torso"),
            ("torso", "neck"),
            ("neck", "head"),
        ]

        self._sc_exclude_mask = torch.zeros(n, n, dtype=torch.bool, device=self.device)
        for a, b in KNOWN_ADJACENT_PAIRS:
            if a in name_to_idx and b in name_to_idx:
                i, j = name_to_idx[a], name_to_idx[b]
                self._sc_exclude_mask[i, j] = True
                self._sc_exclude_mask[j, i] = True
            else:
                print(f"[warn] adjacency pair not found in sensor bodies: {a}, {b}")


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
        
        self.actions = torch.zeros(self.num_envs, len(self._left_dof_idx), device=self.device)
        self.prev_actions = torch.zeros(self.num_envs, len(self._left_dof_idx), device=self.device)

        # --- compute the single constant target EE pose (once, via FK) ---
        # this is the same goal for every env and every episode; only the
        # starting joint configuration is randomized per episode (see _reset_idx)
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

        # write goal joints to sim temporarily to read FK
        self.robot.write_joint_state_to_sim(
            goal_full_joint_pos,
            self.robot.data.default_joint_vel.clone(),
            None,
            all_env_ids,
        )
        self.robot.update(dt=0.0)  # force kinematic update without physics step

        self._fixed_target_pos = self.robot.data.body_pos_w[
            :, self._ee_body_idx, :
        ].clone()  # (N, 3)
        self._fixed_target_quat = self.robot.data.body_quat_w[
            :, self._ee_body_idx, :
        ].clone()  # (N, 4)

        self.target_pose[:, :3] = self._fixed_target_pos
        self.target_pose[:, 3:] = self._fixed_target_quat

        # restore robot to default joint state after the FK probe
        self.robot.write_joint_state_to_sim(
            self.robot.data.default_joint_pos.clone(),
            self.robot.data.default_joint_vel.clone(),
            None,
            all_env_ids,
        )

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

        # Make ground a kinematic rigid body
        ground_prim = self.sim.stage.GetPrimAtPath("/World/ground")

        for prim in Usd.PrimRange(ground_prim):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                collision_api = UsdPhysics.CollisionAPI(prim)
                collision_api.CreateCollisionEnabledAttr(False)
                print("Disabled collision:", prim.GetPath())

        # print("\n========== GROUND DEBUG ==========")
        # print("Ground prim valid:", ground_prim.IsValid())
        # print("Ground prim path:", ground_prim.GetPath())
        # print("Ground prim type:", ground_prim.GetTypeName())

        # rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(ground_prim)

        # print("RigidBodyAPI valid:", bool(rigid_body_api))
        # print(
        #     "Has RigidBodyAPI:",
        #     ground_prim.HasAPI(UsdPhysics.RigidBodyAPI),
        # )

        # rigid_body_api.CreateKinematicEnabledAttr(True)

        # print(
        #     "Kinematic enabled:",
        #     rigid_body_api.GetKinematicEnabledAttr().Get(),
        # )
        # print(
        #     "Rigid body enabled:",
        #     rigid_body_api.GetRigidBodyEnabledAttr().Get(),
        # )

        # print("Applied API schemas:", ground_prim.GetAppliedSchemas())
        # print("==================================\n")

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = self.robot

        # # register contact sensor
        # self._contact_sensor = ContactSensor(self.cfg.contact_sensor_cfg)
        # self.scene.sensors["contact_sensor"] = self._contact_sensor

        # two separate sensors
        self._ground_contact_sensor = ContactSensor(self.cfg.ground_contact_sensor_cfg)
        self._self_collision_sensor = ContactSensor(self.cfg.self_collision_sensor_cfg)
        # self._arm_body_collision_sensor = ContactSensor(self.cfg.arm_body_collision_sensor_cfg)
        self.scene.sensors["ground_contact"] = self._ground_contact_sensor
        self.scene.sensors["self_collision"] = self._self_collision_sensor
        # self.scene.sensors["arm_body_collision"] = self._arm_body_collision_sensor
        

                
    

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions = self.actions.clone()
        self.actions = actions.clone()

    # def _apply_action(self) -> None:
    #     # control only left arm; right arm stays at default (no command written)
    #     self.robot.set_joint_effort_target(
    #         self.actions * self.cfg.action_scale,
    #         joint_ids=self._left_dof_idx,
    #     )

    # def _apply_action(self) -> None:
    #     # actions are joint position targets in [-1, 1], scaled to joint range
    #     self.robot.set_joint_position_target(
    #         self.actions,  # directly target joint angles (radians)
    #         joint_ids=self._left_dof_idx,
    #     )

    # def _apply_action(self) -> None:
    #     # actions are joint position targets in [-1, 1] — scale per-joint to
    #     # that joint's own soft limit range, not used directly as radians
    #     actions = torch.clamp(self.actions, -1.0, 1.0)

    #     lower = self._left_joint_limits[:, 0]  # (7,)
    #     upper = self._left_joint_limits[:, 1]  # (7,)

    #     # map [-1, 1] -> [lower, upper] per joint
    #     joint_pos_target = lower.unsqueeze(0) + (actions + 1.0) * 0.5 * (
    #         upper - lower
    #     ).unsqueeze(0)

    #     self.robot.set_joint_position_target(
    #         joint_pos_target,
    #         joint_ids=self._left_dof_idx,
    #     )
        
    def _apply_action(self) -> None:
        # actions are per-step joint deltas in [-1, 1], scaled to a small max
        # angular step per control tick — NOT absolute positions
        self.actions = torch.clamp(self.actions, -1.0, 1.0)

        delta = self.actions * self.cfg.max_joint_delta  # (N, 7), radians per control step

        current_pos = self.joint_pos[:, self._left_dof_idx]  # (N, 7) — actual sim state
        joint_pos_target = current_pos + delta

        # still clamp to soft limits so a string of same-sign deltas can't walk
        # the target past the joint's physical range
        limit_margin = 0.1 #5.73 deg
        lower = self._left_joint_limits[:, 0] + limit_margin
        upper = self._left_joint_limits[:, 1] - limit_margin
        joint_pos_target = torch.clamp(joint_pos_target, lower.unsqueeze(0), upper.unsqueeze(0))

        self.robot.set_joint_position_target(
            joint_pos_target,
            joint_ids=self._left_dof_idx,
        )

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        # refresh joint state
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        left_pos = self.joint_pos[:, self._left_dof_idx]  # (N, 7)
        left_vel = self.joint_vel[:, self._left_dof_idx]  # (N, 7)

        # end-effector pose in world frame
        ee_pos_w = self.robot.data.body_pos_w[:, self._ee_body_idx, :]  # (N, 3)
        ee_quat_w = self.robot.data.body_quat_w[:, self._ee_body_idx, :]  # (N, 4) wxyz

        obs = torch.cat(
            [
                left_pos,  # 7
                left_vel,  # 7
                ee_pos_w,  # 3
                ee_quat_w,  # 4  → total EE: 7
                self.target_pose,  # 7  (xyz + wxyz)
                self.actions, #7 Last commanded delta action
            ],
            dim=-1,
        )  # total: 28 # 35

        return {"policy": obs}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        ee_pos_w = self.robot.data.body_pos_w[:, self._ee_body_idx, :]
        ee_quat_w = self.robot.data.body_quat_w[:, self._ee_body_idx, :]

        ee_lin_vel_w = self.robot.data.body_lin_vel_w[:, self._ee_body_idx, :]  # (N, 3)
        ee_ang_vel_w = self.robot.data.body_ang_vel_w[:, self._ee_body_idx, :]  # (N, 3)

        # update markers every reward step
        self._update_markers(ee_pos_w, ee_quat_w)

        target_pos = self.target_pose[:, :3]
        target_quat = self.target_pose[:, 3:]  # wxyz

        (total, rew_alive, rew_termination, rew_pos, rew_rot,
        rew_reach, rew_limit, rew_stability, rew_action_rate, rew_scale_centering) = compute_rewards(
            ee_pos_w, ee_quat_w, ee_lin_vel_w, ee_ang_vel_w,
            target_pos, target_quat,
            self.actions, self.prev_actions,
            self.joint_pos[:, self._left_dof_idx],
            self._left_joint_limits,
            self.reset_terminated,
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_pos_error,
            self.cfg.rew_scale_rot_error,
            self.cfg.rew_scale_reach_bonus,
            self.cfg.rew_scale_joint_limit,
            self.cfg.rew_scale_termination,
            self.cfg.rew_scale_stability,
            self.cfg.rew_scale_action_rate,
            self.cfg.rew_scale_centering,     
            self.cfg.pos_success_threshold,     
            self.cfg.rot_success_threshold,   
            self.cfg.stability_vel_scale,
            self.cfg.joint_limit_margin,
        )

        # --- temporary debug print, remove once diagnosed ---
        print(
            f"pos={rew_pos.mean().item():.2f} rot={rew_rot.mean().item():.2f} "
            f"reach={rew_reach.mean().item():.2f} stab={rew_stability.mean().item():.2f} "
            f"limit={rew_limit.mean().item():.2f} action_rate={rew_action_rate.mean().item():.2f} "
            f"alive={rew_alive.mean().item():.2f} term={rew_termination.mean().item():.2f} "
            f"centering={rew_scale_centering.mean().item():.2f} TOTAL={total.mean().item():.2f}"
        )

        return total

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
        

        # # joint4 angle across all envs
        # joint4_idx = 3  # left_arm_joint4 is index 3 (0-indexed) in left_dof_names
        # joint4_pos = self.joint_pos[:, self._left_dof_idx[joint4_idx]]  # (N,)
        # lo = self._left_joint_limits[joint4_idx, 0].item()
        # hi = self._left_joint_limits[joint4_idx, 1].item()

        # print(f"joint4 limits: [{lo:+.3f}, {hi:+.3f}]")
        # print(f"joint4 angles per env: {joint4_pos.cpu().numpy()}")

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # terminate if any left arm joint violates soft limits
        left_pos = self.joint_pos[:, self._left_dof_idx]  # (N, 7)
        lower = self._left_joint_limits[:, 0]  # (7,)
        upper = self._left_joint_limits[:, 1]  # (7,)
        
        # print("Lower joint limits:", lower)
        # print("Upper joint limits:", upper)
        
        term_tolerance = 0.02  # rad — ignore tiny solver-level overshoot
        joint_violation = torch.any(
            (left_pos < (lower - term_tolerance).unsqueeze(0)) | (left_pos > (upper + term_tolerance).unsqueeze(0)),
            dim=1,
        ) # (N,)

        # --- ground contact: any arm link touching ground ---
        # net_forces_w: (N, num_bodies, 3) — filtered to ground contacts only
        # ground_forces = self._ground_contact_sensor.data.net_forces_w       # (N, 7, 3)
        ground_forces = self._ground_contact_sensor.data.force_matrix_w

        # Print only environment 0
        # print("Ground forces env 0:")
        # print(ground_forces[0])

        ground_force_mag = torch.norm(ground_forces, dim=-1)  # (N, 7)

        # print("Ground force magnitudes env 0:")
        # print(ground_force_mag[0])

        # ground_collision = torch.any(
        #     ground_force_mag > self.cfg.ground_contact_threshold, dim=1
        # )  # (N,)

        # ground_collision = torch.any(
        #     ground_force_mag > self.cfg.ground_contact_threshold, dim=(1, 2)
        # )

        # print("Ground collision env 0:", ground_collision[0].item())

        # # --- self collision: skip adjacent link pairs ---
        # # net_forces_w here includes ALL contacts on each link
        # # adjacent links (l1↔l2, l2↔l3, ...) always have joint forces → exclude them
        # # strategy: check only non-adjacent links, i.e. links separated by 2+ joints
        # # simplest proxy: if link i has high force AND link i+1 also has high force
        # # it's likely a joint force — skip. Only flag if a non-adjacent link is involved.
        # #
        # # Practical approach: use a higher threshold and check force_matrix_w for body pairs
        # self_forces = self._self_collision_sensor.data.net_forces_w          # (N, 7, 3)
        # self_force_mag = torch.norm(self_forces, dim=-1)                     # (N, 7)

        # # mask out adjacent link contributions:
        # # if link[i] and link[i+1] BOTH exceed threshold, treat as joint force, not collision
        # # real self-collision shows up as isolated high force on a non-adjacent link
        # exceed = self_force_mag > self.cfg.self_collision_threshold          # (N, 7) bool

        # # adjacent pair mask: (l1,l2), (l2,l3), ..., (l6,l7)
        # # if both neighbors exceed threshold → likely joint force, zero them out
        # adjacent_both = exceed[:, :-1] & exceed[:, 1:]                      # (N, 6)
        # # build a mask of links that are part of an adjacent-both pair
        # is_adjacent = torch.zeros_like(exceed)                               # (N, 7)
        # is_adjacent[:, :-1] |= adjacent_both
        # is_adjacent[:, 1:]  |= adjacent_both

        # # only flag as self-collision if exceeds threshold AND not explained by adjacency
        # real_self_collision = exceed & ~is_adjacent                          # (N, 7)
        # self_collision = torch.any(real_self_collision, dim=1)               # (N,)

        # ---------------------------------------------------------
        # Self-collision detection
        # ---------------------------------------------------------

        # print(
        #     "Self force matrix shape:",
        #     self._self_collision_sensor.data.force_matrix_w.shape
        # )

        # print("BODY Names:",self._self_collision_sensor.body_names)
        self_forces = self._self_collision_sensor.data.force_matrix_w
        # Expected shape: (N, 7, 7, 3)

        self_force_mag = torch.norm(self_forces, dim=-1)
        # Shape: (N, 7, 7)


        collision_pairs = (self_force_mag > self.cfg.self_collision_threshold) & ~self._sc_exclude_mask.unsqueeze(0)
        self_collision = torch.any(collision_pairs, dim=(1, 2))

        # # Build mask for valid non-adjacent link pairs
        # num_links = self_force_mag.shape[1]

        # link_ids = torch.arange(num_links, device=self.device)

        # i = link_ids[:, None]  # (7, 1)
        # j = link_ids[None, :]  # (1, 7)

        # # Ignore:
        # # i == j          → same link
        # # |i - j| == 1    → adjacent links
        # non_adjacent_mask = torch.abs(i - j) > 1
        # # shape: (7, 7)

        # # Check only non-adjacent pairs
        # collision_pairs = (
        #     self_force_mag > self.cfg.self_collision_threshold
        # ) & non_adjacent_mask.unsqueeze(0)

        # # True if any non-adjacent pair collides
        # self_collision = torch.any(
        #     collision_pairs,
        #     dim=(1, 2),
        # )

        # if self_collision[0]:
        #     pairs = torch.nonzero(collision_pairs[0])

        #     for pair in pairs:
        #         i = pair[0].item()
        #         j = pair[1].item()

        #         # print(
        #         #     f"Self collision: "
        #         #     f"{self._self_collision_sensor.body_names[i]} "
        #         #     f"<-> filtered link index {j}, "
        #         #     f"force = {self_force_mag[0, i, j].item():.2f} N"
        #         # )

 
            
        # print("Source bodies:", self._arm_body_collision_sensor.body_names)
        # print(
        #     "Filter expressions:",
        #     self.cfg.arm_body_collision_sensor_cfg.filter_prim_paths_expr
        # )
        # arm_body_forces = self._arm_body_collision_sensor.data.force_matrix_w  # (N, 7, 4, 3)
        # arm_body_force_mag = torch.norm(arm_body_forces, dim=-1)                # (N, 7, 4)

        # # Only l1 is physically mounted to torso_base — that pair will show a constant
        # # geometric-overlap force even with no real collision, same reason you mask
        # # adjacent arm links. Everything else (l1 vs torso/head/neck, l2-l7 vs anything
        # # here) is a genuine collision if it exceeds threshold.
        # exclude = torch.zeros(7, 4, dtype=torch.bool, device=self.device)
        # exclude[0, 1] = True  # l1_left_arm <-> torso_base (mount point) — verify index order below

        # arm_body_collision_pairs = (arm_body_force_mag > self.cfg.self_collision_threshold) & ~exclude.unsqueeze(0)
        # arm_body_collision = torch.any(arm_body_collision_pairs, dim=(1, 2))    
            
        # --- check specifically for joint4 violations ---
        # joint4_idx = 3
        # joint4_pos = left_pos[:, joint4_idx]  # (N,)
        # joint4_lo = lower[joint4_idx]
        # joint4_hi = upper[joint4_idx]
        # joint4_violation = (joint4_pos < joint4_lo) | (joint4_pos > joint4_hi)  # (N,)

        # if torch.any(joint4_violation):
        #     violating_envs = torch.nonzero(joint4_violation).flatten()
        #     vals = joint4_pos[violating_envs]
        #     print(f"joint4 limits: [{joint4_lo.item():+.3f}, {joint4_hi.item():+.3f}]")
        #     for eid, val in zip(violating_envs.tolist(), vals.tolist()):
        #         print(f"  env {eid}: joint4 VIOLATED at {val:+.4f}")            
            
            
        joint_violation = torch.any(
            (left_pos < lower.unsqueeze(0)) | (left_pos > upper.unsqueeze(0)),
            dim=1,
        )  # (N,)

        if torch.any(joint_violation):
            # find the first violating env instead of assuming env 0
            violating_envs = torch.nonzero(joint_violation).flatten()
            env_id = violating_envs[0].item()

            per_joint_violation = (
                (left_pos[env_id] < lower) | (left_pos[env_id] > upper)
            )
            violating_joint_idx = torch.nonzero(per_joint_violation).flatten().tolist()

            print(f"Joint limit violation detected in env {env_id}, joint(s): {violating_joint_idx}")
            for j in violating_joint_idx:
                val = left_pos[env_id, j].item()
                lo = lower[j].item()
                hi = upper[j].item()
                print(f"  joint{j+1}: angle={val:+.4f}  limits=[{lo:+.3f}, {hi:+.3f}]")        
    
            
            
        # Determine the type of collision
        # if torch.any(joint_violation):
        #     joint4_idx = 3
        #     joint4_pos = self.joint_pos[:, self._left_dof_idx[joint4_idx]]  # (1000,)
        #     lo = self._left_joint_limits[joint4_idx, 0].item()
        #     hi = self._left_joint_limits[joint4_idx, 1].item()

        #     env_ids = torch.arange(0, self.num_envs, 50)
        #     sampled = joint4_pos[env_ids]

        #     print(f"joint4 limits: [{lo:+.3f}, {hi:+.3f}]")
        #     for eid, val in zip(env_ids.tolist(), sampled.cpu().tolist()):
        #         flag = " <-- VIOLATING" if (val < lo or val > hi) else ""
        #         print(f"  env {eid}: joint4 = {val:+.4f}{flag}")
        
        # if torch.any(ground_collision):
        #     print("Ground collision detected.")
            
            
            
            
        if torch.any(self_collision):
            print("Self-collision detected.")

            
        # terminated = joint_violation | ground_collision | self_collision
        terminated = joint_violation | self_collision
        
        if terminated[0]:
            reasons = []
            if joint_violation[0]:
                reasons.append("joint_violation")
            if self_collision[0]:
                reasons.append("self_collision")
            print(
                f"[ep_len={self.episode_length_buf[0].item()}] TERMINATED: {reasons} "
                f"| pos_error={torch.norm(self.robot.data.body_pos_w[0, self._ee_body_idx] - self.target_pose[0, :3]).item():.3f}"
            )

        return terminated, time_out

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        n = len(env_ids)

        # --- target pose is constant: same fixed goal every episode ---
        self.target_pose[env_ids, :3] = self._fixed_target_pos[env_ids]
        self.target_pose[env_ids, 3:] = self._fixed_target_quat[env_ids]

        # --- sample random starting joint angles within soft limits ---
        lower = self._left_joint_limits[:, 0]  # (7,)
        upper = self._left_joint_limits[:, 1]  # (7,)

        # # uniform sample in [lower, upper] for each of the 7 joints
        # alpha = torch.rand(n, 7, device=self.device)
        # start_joint_pos = lower.unsqueeze(0) + alpha * (upper - lower).unsqueeze(0)  # (N, 7)

        # bias toward center of joint range instead of uniform
        # beta distribution with a=b=2 peaks at 0.5 (center of range)
        alpha = torch.distributions.Beta(2.0, 2.0).sample((n, 7)).to(self.device)
        start_joint_pos = lower.unsqueeze(0) + alpha * (upper - lower).unsqueeze(0)

        # build full joint pos tensor (all joints) starting from default, left arm randomized
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
    ee_lin_vel: torch.Tensor,
    ee_ang_vel: torch.Tensor,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    left_joint_pos: torch.Tensor,
    joint_limits: torch.Tensor,
    reset_terminated: torch.Tensor,
    rew_scale_alive: float,
    rew_scale_pos_error: float,
    rew_scale_rot_error: float,
    rew_scale_reach_bonus: float,
    rew_scale_joint_limit: float,
    rew_scale_termination: float,   # <-- new: should be a large negative number, e.g. -50.0
    rew_scale_stability: float,
    rew_scale_action_rate: float,
    rew_scale_centering: float,
    pos_threshold: float,
    rot_success_threshold: float,
    stability_vel_scale: float,
    joint_limit_margin: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # --- alive reward ---
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())

    # --- explicit penalty for crashing (self-collision / joint-limit violation) ---
    # reset_terminated here must be the "real failure" signal, NOT time_out —
    # see _get_dones/_get_rewards wiring below
    rew_termination = rew_scale_termination * reset_terminated.float()

    # --- positional error ---
    pos_error = torch.norm(ee_pos - target_pos, dim=-1)  # (N,)
    rew_pos = rew_scale_pos_error * pos_error

    # --- orientation error ---
    dot = torch.clamp(
        torch.abs(torch.sum(ee_quat * target_quat, dim=-1)), max=1.0
    )  # (N,)
    rot_error = 2.0 * torch.acos(dot)  # (N,) geodesic angle
    rew_rot = rew_scale_rot_error * rot_error

    # --- sparse success bonus ---
    rew_reach = rew_scale_reach_bonus * (pos_error < pos_threshold).float()

    # --- stability bonus: only pays out once EE is at the goal pose, ---
    # --- and decays smoothly as EE velocity grows                     ---
    at_target = (pos_error < pos_threshold) & (rot_error < rot_success_threshold)
    vel_norm = torch.norm(ee_lin_vel, dim=-1) + 0.5 * torch.norm(ee_ang_vel, dim=-1)
    rew_stability = rew_scale_stability * torch.exp(-stability_vel_scale * vel_norm) * at_target.float()

    # --- joint limit proximity penalty ---
    lower = joint_limits[:, 0]  # (7,)
    upper = joint_limits[:, 1]  # (7,)
    dist_to_lower = left_joint_pos - lower.unsqueeze(0)  # (N, 7)
    dist_to_upper = upper.unsqueeze(0) - left_joint_pos  # (N, 7)
    min_dist = torch.minimum(dist_to_lower, dist_to_upper)  # (N, 7)
    limit_violation = torch.clamp(joint_limit_margin - min_dist, min=0.0)
    rew_limit = rew_scale_joint_limit * torch.sum(limit_violation, dim=-1)

    # if torch.equal(actions, prev_actions):
    #     print("actions == previous actions")

    action_rate_penalty = torch.sum((actions - prev_actions) ** 2, dim=-1)
    
    rew_action_rate = rew_scale_action_rate * action_rate_penalty
    
    
    redundant_joint_ids = torch.tensor([4, 6], device=left_joint_pos.device)
    centers = (joint_limits[redundant_joint_ids, 0] + joint_limits[redundant_joint_ids, 1]) / 2.0
    ranges = joint_limits[redundant_joint_ids, 1] - joint_limits[redundant_joint_ids, 0]
    centering_err = torch.abs(left_joint_pos[:, redundant_joint_ids] - centers.unsqueeze(0)) / ranges.unsqueeze(0)
    rew_centering = rew_scale_centering * torch.sum(centering_err, dim=-1)
    
    total = rew_alive + rew_termination + rew_pos + rew_rot + rew_reach + rew_limit + rew_stability + rew_action_rate + rew_centering

    return total, rew_alive, rew_termination, rew_pos, rew_rot, rew_reach, rew_limit, rew_stability, rew_action_rate, rew_centering

@torch.jit.script
def random_quaternions(n: int, device: str) -> torch.Tensor:
    """Sample n uniform random unit quaternions (wxyz)."""
    # Shoemake 1992 method — uniform over SO(3)
    u = torch.rand(n, 3, device=device)

    sqrt1_u0 = torch.sqrt(1.0 - u[:, 0])
    sqrt_u0 = torch.sqrt(u[:, 0])
    theta1 = 2.0 * torch.pi * u[:, 1]
    theta2 = 2.0 * torch.pi * u[:, 2]

    w = sqrt1_u0 * torch.sin(theta1)
    x = sqrt1_u0 * torch.cos(theta1)
    y = sqrt_u0 * torch.sin(theta2)
    z = sqrt_u0 * torch.cos(theta2)

    return torch.stack([w, x, y, z], dim=-1)  # (N, 4) wxyz

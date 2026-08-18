# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations
import os

import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import carb
import isaaclab.sim as sim_utils
import omni.physics.tensors.impl.api as physx
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_conjugate, quat_mul, saturate

if TYPE_CHECKING:
    from .revo3_hand_hora_env_cfg import Revo3HandHoraEnvCfg


class Revo3HandHoraEnv(DirectRLEnv):
    """DirectRLEnv for Revo3 right hand in-hand object rotation.

    Actor observation (141 dims) — 3-frame sliding window, 47 dims/frame:
      [0:21]   joint positions, unscaled to [-1,1] via (2x - hi - lo)/(hi - lo), +-0.02 rad noise
      [21:42]  current joint targets (delta-accumulated, clamped to joint limits)
      [42:47]  object-filtered resultant forces on 5 DIP fingertips, sampled at 20 Hz

    Privileged observation (18 dims): object position delta (3), friction (1),
      mass (1), COM (3), gravity magnitude (1), configured object-axis in world (3),
      object angular velocity (3), and object linear velocity (3).

    Action (21 dims) — delta position control:
      action ∈ [-1,1] → target = prev_target + (1/24)*action → clamp(joint_limits)
      Torque control: torque = p_gain*(target - pos) - d_gain*vel
      p_gain/d_gain from cfg, randomized per reset: ×[0.8, 1.2] per-DOF

    Reward (total ×0.01 for PPO): normalized directed rotation progress and a
      stable-rotation bonus; reverse-rotation, object-axis tilt, off-axis angular
      velocity and XY/Z drift penalties; explicit drop and hand self-collision
      penalties; normalized torque and per-joint work regularization.  The
      policy is not tied to the cached reset posture after an episode starts.

    Termination:
      height:    obj_z outside [init_z - 2cm, init_z + 2cm]
      timeout:   episode_length >= max_episode_length (400 steps @20Hz)
      gravity curriculum: evaluate height-reset rate over 200 policy steps,
      advance/rollback by 0.10 m/s², and cap exactly at 9.81 m/s²

    Key design decisions:
      - the sampled grasp-cache row defines reset state only, not a reward reference
      - PD gains per-joint-type from cfg.pgain_dict/dgain_dict, not read from URDF/USD
      - torque/work penalty uses self.torques (our explicit PD command), not PhysX applied_torque
      - tactile Stage2 keeps real five-channel contacts in actor obs and proprio_hist
    """
    cfg: Revo3HandHoraEnvCfg

    def __init__(self, cfg: Revo3HandHoraEnvCfg, render_mode: str | None = None, **kwargs):
        self.reset_height_lower = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.reset_height_upper = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)

        super().__init__(cfg, render_mode, **kwargs)

        self.num_hand_dofs = self.hand.num_joints

        # Canonical init joint pose from assets.py — used for cache-less reset.
        self.init_joint_pos = torch.zeros((1, self.num_hand_dofs), device=self.device)
        _cfg_pos = self.cfg.robot_cfg.init_state.joint_pos
        if _cfg_pos:
            for _name, _val in _cfg_pos.items():
                if _name in self.hand.joint_names:
                    self.init_joint_pos[0, self.hand.joint_names.index(_name)] = float(_val)
        # Per-environment sampled reset pose.  It is not used as a posture
        # reward target, so the learned finger gait may leave this pose freely.
        self.grasp_joint_pos = self.init_joint_pos.expand(self.num_envs, -1).clone()

        self._axes_visualizer = None
        if getattr(self.cfg, 'debug_show_axes', True):
            try:
                from isaaclab.markers import VisualizationMarkers
                from isaaclab.markers.config import FRAME_MARKER_CFG
                # create frame marker configuration for the object's local axes
                axes_marker_cfg = FRAME_MARKER_CFG.replace(
                    prim_path="/Visuals/ObjectAxes"
                )
                # adjust the axes size based on config (default 0.06 m)
                axes_length = getattr(self.cfg, 'vis_object_axes_length', 0.06)
                axes_marker_cfg.markers["frame"].scale = (axes_length, axes_length, axes_length)
                # create the visualization marker
                self._axes_visualizer = VisualizationMarkers(axes_marker_cfg)
            except Exception as e:
                self._axes_visualizer = None

        # buffers for position targets
        self.prev_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)

        # buffers for object
        self.object_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.object_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.object_pos_prev = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.object_rot_prev = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.object_default_pose = torch.zeros((self.num_envs, 7), dtype=torch.float, device=self.device)
        self.rb_forces = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

        # buffers for data
        # The actor needs 3 frames and ProprioAdapt needs prop_hist_len frames.
        # A ring buffer avoids cloning/cat-ing a legacy 80-frame tensor every step.
        self._obs_history_len = max(3, int(self.cfg.prop_hist_len))
        self._obs_history_index = -1
        self.obs_buf_lag_history = torch.zeros(
            (self.num_envs, self._obs_history_len, self.cfg.observation_space // 3),
            device=self.device,
            dtype=torch.float,
        )
        self._obs_history_offsets = torch.arange(self._obs_history_len, device=self.device)
        self.at_reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.proprio_hist_buf = self.obs_buf_lag_history[:, -self.cfg.prop_hist_len:]
        self.priv_info_buf = torch.zeros((self.num_envs, self.cfg.priv_info_dim), device=self.device, dtype=torch.float)

        # list of actuated joints
        self.actuated_dof_indices = list()
        for joint_name in cfg.actuated_joint_names:
            self.actuated_dof_indices.append(self.hand.joint_names.index(joint_name))
        self.actuated_dof_indices.sort()

        # finger bodies
        self.finger_bodies = list()
        for body_name in self.cfg.fingertip_body_names:
            self.finger_bodies.append(self.hand.body_names.index(body_name))
        self.num_fingertips = len(self.finger_bodies)

        # joint limits
        joint_pos_limits = self.hand.root_physx_view.get_dof_limits().to(self.device)
        self.hand_dof_lower_limits = joint_pos_limits[..., 0] * self.cfg.dof_limits_scale
        self.hand_dof_upper_limits = joint_pos_limits[..., 1] * self.cfg.dof_limits_scale

        # Hardcoded PD gains — not reading from URDF/USD baked-in defaults
        # Per-joint-type base gains from identified hardware dynamics
        ndof = self.num_hand_dofs
        _p_base = torch.zeros(ndof, device=self.device)
        _d_base = torch.zeros(ndof, device=self.device)
        _name_to_dof = {name: i for i, name in enumerate(self.hand.joint_names)}
        for joint_name in self.cfg.actuated_joint_names:
            dof_idx = _name_to_dof[joint_name]
            # joint group lookup: CMP/CMR > thumb_flexion > DIP > MPR > MCP > PIP
            if "CMP" in joint_name:
                group = "thumb_CMP"
            elif "CMR" in joint_name:
                group = "thumb_CMR"
            elif "thumb" in joint_name and ("MCP" in joint_name or "PIP" in joint_name):
                group = "thumb_flexion"
            elif "DIP" in joint_name:
                group = "DIP"
            elif "MPR" in joint_name:
                group = "MPR"
            elif "MCP" in joint_name:
                group = "MCP"
            else:
                group = "PIP"
            _p_base[dof_idx] = self.cfg.pgain_dict[group]
            _d_base[dof_idx] = self.cfg.dgain_dict[group]
        self._p_gain_base = _p_base
        self._d_gain_base = _d_base
        self.p_gain = _p_base.unsqueeze(0).expand(self.num_envs, -1).contiguous()
        self.d_gain = _d_base.unsqueeze(0).expand(self.num_envs, -1).contiguous()

        # grasp_cache
        self.scale_ids = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.int32)
        cache_path = f"{self.cfg.grasp_cache_path}.npy"
        if os.path.exists(cache_path):
            self.saved_grasping_states = torch.from_numpy(np.load(cache_path)).float().to(self.device)
            self.bucket_grasp = self.saved_grasping_states.shape[0]
            self.bucket_env = self.num_envs
        else:
            print(f"[WARN] Grasp cache not found: {cache_path}, falling back to default pose.")
            self.saved_grasping_states = None

        self.object_rotation_axis_local = torch.tensor(
            self.cfg.object_rotation_axis_local, dtype=torch.float32, device=self.device
        ).repeat(self.num_envs, 1)
        self.target_rotation_axis_world = torch.tensor(
            self.cfg.target_rotation_axis_world, dtype=torch.float32, device=self.device
        ).repeat(self.num_envs, 1)

        # contact buffers
        self._contact_body_ids = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        self._contact_body_ids_disable = torch.tensor(self.cfg.disable_tactile_ids, dtype=torch.long)
        self.last_contacts = torch.zeros((self.num_envs, len(self._contact_body_ids)), dtype=torch.float, device=self.device)
        self.elastomer_ids = [self.hand.body_names.index(body_name) for body_name in self.cfg.elastomer_body_names]

        # randomize
        if self.cfg.randomize_friction:
            rand_friction = torch.empty(self.num_envs).uniform_(self.cfg.randomize_friction_scale_lower, self.cfg.randomize_friction_scale_upper)
            rand_friction = rand_friction.reshape(self.num_envs, 1)
            rand_friction_object = rand_friction.clone() * self.cfg.object_base_friction
            self.set_friction(self.object, rand_friction_object, self.num_envs)
            n_hand_mats = self.hand.root_physx_view.get_material_properties().shape[1]
            rand_friction_hand = rand_friction.clone().repeat(1, n_hand_mats) * self.cfg.metal_base_friction
            self.set_friction(self.hand, rand_friction_hand, self.num_envs)
            self.priv_info_buf[:, 3] = rand_friction.squeeze()
        if self.cfg.randomize_com:
            rand_com = torch.empty([self.num_envs, 3]).uniform_(self.cfg.randomize_com_lower, self.cfg.randomize_com_upper)
            self.set_com(self.object, rand_com, self.num_envs)
            self.priv_info_buf[:, 5:8] = self.object.root_physx_view.get_coms().reshape(self.num_envs, -1)[:, :3]
        if self.cfg.randomize_mass:
            rand_mass = torch.empty(self.num_envs).uniform_(self.cfg.randomize_mass_lower, self.cfg.randomize_mass_upper)
            self.set_mass(self.object, rand_mass, self.num_envs)
            self.priv_info_buf[:, 4] = self.object.root_physx_view.get_masses().reshape(self.num_envs)

        # physics_sim_view
        self.physics_sim_view: physx.SimulationView = sim_utils.SimulationContext.instance().physics_sim_view
        gravity = self.physics_sim_view.get_gravity()
        self._gravity_magnitude = float(
            (gravity[0] ** 2 + gravity[1] ** 2 + gravity[2] ** 2) ** 0.5
        )
        self._gravity_reset_sum = torch.zeros((), dtype=torch.float32, device=self.device)
        self._gravity_window_steps = 0
        self._gravity_window_reset_rate = 1.0

    def _setup_scene(self):
        # add hand, in-hand object, and goal object
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # Clone heterogeneous environments, then explicitly filter cross-env
        # collisions while keeping collisions with the shared ground plane.
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        # add articulation to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["hand"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        # contact sensors
        self._contact_sensor = []
        for id in range(len(self.cfg.contact_sensor)):
            self._contact_sensor.append(ContactSensor(self.cfg.contact_sensor[id]))
            self.scene.sensors[f"contact_sensor_{id}"] = self._contact_sensor[id]
        # Separate filtered sensors ensure that normal fingertip-object contact
        # is never mistaken for hand self-collision.
        self._self_collision_sensors = []
        for sensor_id, sensor_cfg in enumerate(self.cfg.self_collision_sensor):
            sensor = ContactSensor(sensor_cfg)
            self._self_collision_sensors.append(sensor)
            self.scene.sensors[f"self_collision_sensor_{sensor_id}"] = sensor
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Delta position control: action ∈ [-1,1] → target += (1/24)*action → clamp to joint limits.
        Also updates object_rot_prev/pos_prev for angular velocity computation in reward."""
        actions = saturate(actions, torch.tensor(-self.cfg.clip_actions), torch.tensor(self.cfg.clip_actions))
        self.actions = actions.clone()
        targets = self.prev_targets + self.cfg.action_scale * self.actions
        self.cur_targets[:, self.actuated_dof_indices] = saturate(
            targets,
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )
        self.object_pos_prev[:] = self.object_pos
        self.object_rot_prev[:] = self.object_rot

        if self.cfg.force_scale > 0.0:
            self.rb_forces *= torch.pow(torch.tensor(self.cfg.force_decay, dtype=torch.float32), self.physics_dt / self.cfg.force_decay_interval)
            # apply new forces
            obj_mass = self.object.root_physx_view.get_masses().reshape(self.num_envs).to(self.device)
            prob = self.cfg.random_force_prob_scalar
            force_indices = (torch.less(torch.rand(self.num_envs, device=self.device), prob)).nonzero().to(self.device)
            self.rb_forces[force_indices, :] = torch.randn(self.rb_forces[force_indices, :].shape, device=self.device) * obj_mass[force_indices, None] * self.cfg.force_scale
            self.object.permanent_wrench_composer.set_forces_and_torques(
                forces=self.rb_forces.reshape(self.num_envs, 1, 3),
                torques=torch.zeros(self.num_envs, 1, 3, device=self.device),
            )

    def _apply_action(self) -> None:
        """Torque control: torques = p_gain*(target - pos) - d_gain*vel, sent via set_joint_effort_target.
        p_gain/d_gain are per-joint-type from cfg.pgain_dict/dgain_dict, NOT read from URDF/USD stiffness/damping."""
        self._refresh_lab()
        if self.cfg.torque_control:
            self.torques = self.p_gain * (self.cur_targets - self.hand_dof_pos) - self.d_gain * self.hand_dof_vel
            self.hand.set_joint_effort_target(self.torques[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices)
        else:
            self.hand.set_joint_position_target(self.cur_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices)
        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

    def _get_observations(self) -> dict:
        self._refresh_lab()
        obs = self.compute_observations()
        return {
            "obs":          obs,
            "priv_info":    self.priv_info_buf.clone(),
            # compute_observations creates a fresh chronological tensor, so a
            # second ~90 MB clone is unnecessary at 16k environments.
            "proprio_hist": self.proprio_hist_buf,
        }

    def _get_rewards(self) -> torch.Tensor:
        """Reward the configured rotation while preserving axis alignment."""
        # PhysX reports angular velocity directly in the world frame.  The
        # task registry supplies the desired world-frame rotation axis.
        object_angvel = self.object_angvel
        target_axis_angvel = (
            object_angvel * self.target_rotation_axis_world
        ).sum(-1)
        # Signed progress: positive rotation is rewarded and reverse rotation
        # receives an equal-magnitude penalty.  Saturation at the target speed
        # avoids rewarding unsafe angular velocity beyond the task objective.
        rotate_reward = torch.clamp(
            target_axis_angvel / self.cfg.target_angvel,
            min=-1.0,
            max=1.0,
        )

        # Rotate the task-specific local object axis into world coordinates and
        # compare it with the independently configured target world axis.
        object_axis_world = rotate_axis_by_quat(
            self.object_rotation_axis_local, self.object_rot
        )
        if self.cfg.enforce_object_axis_alignment:
            axis_cos = (object_axis_world * self.target_rotation_axis_world).sum(-1)
            if self.cfg.object_axis_bidirectional:
                axis_cos = torch.abs(axis_cos)
                axis_cos = torch.clamp(axis_cos, 0.0, 1.0)
            else:
                axis_cos = torch.clamp(axis_cos, -1.0, 1.0)
            object_axis_tilt = torch.acos(axis_cos)
            object_axis_tilt_penalty = (
                object_axis_tilt / self.cfg.object_axis_tilt_tolerance
            ) ** 2
        else:
            object_axis_tilt = torch.zeros_like(object_angvel[:, 0])
            object_axis_tilt_penalty = torch.zeros_like(object_angvel[:, 0])

        target_angvel_vector = (
            target_axis_angvel.unsqueeze(-1)
            * self.target_rotation_axis_world
        )
        off_axis_angvel_penalty = ((object_angvel - target_angvel_vector) ** 2).sum(-1)

        object_pos_delta = self.object_pos - self.object_default_pose[:, :3]
        xy_drift = torch.norm(object_pos_delta[:, :2], dim=-1)
        z_drift = torch.abs(object_pos_delta[:, 2])
        xy_drift_excess = torch.relu(xy_drift - self.cfg.xy_drift_deadzone)
        xy_drift_penalty = smooth_l1_normalized(
            xy_drift_excess, self.cfg.xy_drift_tolerance
        )
        z_drift_penalty = smooth_l1_normalized(z_drift, self.cfg.z_drift_tolerance)

        if self._self_collision_sensors:
            self_collision_forces = torch.stack(
                [
                    torch.linalg.vector_norm(
                        sensor.data.force_matrix_w[:, 0, :, :], dim=-1
                    ).amax(dim=-1)
                    for sensor in self._self_collision_sensors
                ],
                dim=-1,
            )
            self_collision_force_max = self_collision_forces.amax(dim=-1)
        else:
            self_collision_force_max = torch.zeros_like(xy_drift)
        self_collision_force_excess = torch.relu(
            self_collision_force_max - self.cfg.self_collision_force_threshold
        )
        self_collision_penalty = smooth_l1_normalized(
            self_collision_force_excess,
            self.cfg.self_collision_force_tolerance,
        )
        normalized_torque = (
            self.torques[:, self.actuated_dof_indices]
            / self.cfg.torque_normalization
        )
        torque_penalty = normalized_torque.square().mean(-1)
        joint_power = normalized_torque * self.hand_dof_vel[:, self.actuated_dof_indices]
        work_penalty = joint_power.abs().mean(-1)
        # Applied on the terminal transition, before DirectRLEnv resets it.
        drop_penalty = (
            (self.object_pos[:, 2] < self.reset_height_lower)
            | (self.object_pos[:, 2] > self.reset_height_upper)
        ).float()
        alive_reward = 1.0 - drop_penalty
        stable_rotation_bonus = (
            (target_axis_angvel >= self.cfg.stable_rotation_min_angvel)
            & (object_axis_tilt <= self.cfg.object_axis_tilt_tolerance)
            & (xy_drift <= self.cfg.xy_drift_deadzone)
            & (z_drift <= self.cfg.z_drift_tolerance)
            & (drop_penalty == 0.0)
        ).float()

        total_reward = compute_rewards(
            rotate_reward, self.cfg.rotate_reward_scale,
            stable_rotation_bonus, self.cfg.stable_rotation_bonus_scale,
            alive_reward, self.cfg.alive_reward_scale,
            object_axis_tilt_penalty, self.cfg.object_axis_tilt_penalty_scale,
            off_axis_angvel_penalty, self.cfg.off_axis_angvel_penalty_scale,
            xy_drift_penalty, self.cfg.xy_drift_penalty_scale,
            z_drift_penalty, self.cfg.z_drift_penalty_scale,
            drop_penalty, self.cfg.drop_penalty_scale,
            self_collision_penalty, self.cfg.self_collision_penalty_scale,
            torque_penalty, self.cfg.torque_penalty_scale,
            work_penalty, self.cfg.work_penalty_scale,
        )

        self.extras["rew/rotate"] = (rotate_reward * self.cfg.rotate_reward_scale).mean()
        self.extras["rew/stable_rotation"] = (
            stable_rotation_bonus * self.cfg.stable_rotation_bonus_scale
        ).mean()
        self.extras["rew/alive"] = (alive_reward * self.cfg.alive_reward_scale).mean()
        self.extras["rew/object_axis_tilt"] = (
            object_axis_tilt_penalty * self.cfg.object_axis_tilt_penalty_scale
        ).mean()
        self.extras["rew/off_axis_angvel"] = (off_axis_angvel_penalty * self.cfg.off_axis_angvel_penalty_scale).mean()
        self.extras["rew/xy_drift"] = (xy_drift_penalty * self.cfg.xy_drift_penalty_scale).mean()
        self.extras["rew/z_drift"] = (z_drift_penalty * self.cfg.z_drift_penalty_scale).mean()
        self.extras["rew/drop"] = (drop_penalty * self.cfg.drop_penalty_scale).mean()
        self.extras["rew/self_collision"] = (
            self_collision_penalty * self.cfg.self_collision_penalty_scale
        ).mean()
        self.extras["rew/torque"] = (torque_penalty * self.cfg.torque_penalty_scale).mean()
        self.extras["rew/work"] = (work_penalty * self.cfg.work_penalty_scale).mean()
        self.extras['object_axis_tilt_deg'] = torch.rad2deg(object_axis_tilt).mean()
        self.extras['axis_aligned_rate'] = (
            object_axis_tilt <= self.cfg.object_axis_tilt_tolerance
        ).float().mean()
        self.extras['xy_drift_mm'] = (xy_drift * 1000.0).mean()
        self.extras['z_drift_mm'] = (z_drift * 1000.0).mean()
        self.extras['self_collision_rate'] = (
            self_collision_force_max > self.cfg.self_collision_force_threshold
        ).float().mean()
        self.extras['self_collision_force_max_n'] = self_collision_force_max.max()
        self.extras['self_collision_force_mean_n'] = self_collision_force_max.mean()
        self.extras['angvelX'] = object_angvel[:, 0].mean()
        self.extras['angvelY'] = object_angvel[:, 1].mean()
        self.extras['angvelZ'] = object_angvel[:, 2].mean()
        self.extras['target_angvel'] = target_axis_angvel.mean()
        self.extras['rotation_progress'] = rotate_reward.mean()
        self.extras['stable_rotation_rate'] = stable_rotation_bonus.mean()
        self.extras['reverse_rotation_rate'] = (target_axis_angvel < 0.0).float().mean()
        self.extras['total_reward'] = total_reward.mean()
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Terminate out-of-height grasps and update the windowed gravity curriculum."""
        self._refresh_lab()
        height_reset_upper = self.object_pos[:, 2] > self.reset_height_upper
        height_reset_lower = self.object_pos[:, 2] < self.reset_height_lower
        height_reset = height_reset_upper | height_reset_lower
        time_out = self.episode_length_buf >= self.max_episode_length
        self.extras['height_reset_upper'] = height_reset_upper.float().mean()
        self.extras['height_reset_lower'] = height_reset_lower.float().mean()
        self.extras['time_out'] = time_out.float().mean()
        instant_reset_rate = height_reset.float().mean()

        if self.cfg.gravity_curriculum:
            if self.common_step_counter <= self.cfg.gravity_curriculum_warmup_steps:
                self._gravity_reset_sum.zero_()
                self._gravity_window_steps = 0
            else:
                self._gravity_reset_sum += instant_reset_rate.detach()
                self._gravity_window_steps += 1

            if self._gravity_window_steps >= self.cfg.gravity_curriculum_window:
                self._gravity_window_reset_rate = float(
                    (self._gravity_reset_sum / self._gravity_window_steps).item()
                )
                old_gravity = self._gravity_magnitude
                if self._gravity_window_reset_rate <= self.cfg.gravity_curriculum_advance_reset_rate:
                    new_gravity = min(
                        self.cfg.gravity_curriculum_target,
                        old_gravity + self.cfg.gravity_curriculum_step,
                    )
                elif self._gravity_window_reset_rate >= self.cfg.gravity_curriculum_rollback_reset_rate:
                    new_gravity = max(
                        abs(float(self.cfg.sim.gravity[2])),
                        old_gravity - self.cfg.gravity_curriculum_step,
                    )
                else:
                    new_gravity = old_gravity

                if abs(new_gravity - old_gravity) > 1.0e-6:
                    self.set_gravity_magnitude(new_gravity)
                    direction = "advance" if new_gravity > old_gravity else "rollback"
                    print(
                        f"[GRAVITY] {direction}: {old_gravity:.2f} -> {new_gravity:.2f} m/s^2 "
                        f"(height-reset rate={self._gravity_window_reset_rate:.5f}, "
                        f"window={self._gravity_window_steps} steps)",
                        flush=True,
                    )
                self._gravity_reset_sum.zero_()
                self._gravity_window_steps = 0

        at_full_gravity = (
            self._gravity_magnitude >= self.cfg.gravity_curriculum_target - 0.02
        )
        self.extras['gravity_z'] = -self._gravity_magnitude
        self.extras['gravity_magnitude'] = self._gravity_magnitude
        self.extras['gravity_reset_rate_instant'] = instant_reset_rate
        self.extras['gravity_reset_rate_window'] = self._gravity_window_reset_rate
        self.extras['gravity_full_stable'] = float(
            at_full_gravity
            and self._gravity_window_reset_rate <= self.cfg.gravity_curriculum_advance_reset_rate
        )
        return height_reset, time_out

    def set_gravity_magnitude(self, magnitude: float) -> None:
        """Set downward world-Z gravity and keep curriculum/checkpoint state synchronized."""
        self._gravity_magnitude = float(magnitude)
        self.physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, -self._gravity_magnitude))

    def _rand_pd_scales(self, lower, upper, num_envs, n_dofs):
        rand_scale_s = torch.distributions.Uniform(lower, 1).sample((num_envs, n_dofs)).to(self.device)
        rand_scale_l = torch.distributions.Uniform(1, upper).sample((num_envs, n_dofs)).to(self.device)
        mask_choice = torch.rand((num_envs, n_dofs), device=self.device) > 0.5
        rand_scale = torch.where(mask_choice, rand_scale_s, rand_scale_l)
        return rand_scale

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset hand to grasp pose (from cache or init_joint_pos), object to default state.
        PD gains randomized per-DOF each reset: p_gain × [0.5,2.0], d_gain × [0.5,2.0].
        Height bounds computed dynamically: obj_z ± 2cm window."""
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        # resets articulation and rigid body attributes
        super()._reset_idx(env_ids)

        # pd randomize — multiply per-DOF base gains by random scale
        if self.cfg.randomize_pd_gains:
            assert self.cfg.randomize_p_gain_scale_lower <= 1, "pd scale lower bound must be <= 1, upper bound must be >= 1"
            assert self.cfg.randomize_p_gain_scale_upper >= 1, "pd scale lower bound must be <= 1, upper bound must be >= 1"
            assert self.cfg.randomize_d_gain_scale_lower <= 1, "pd scale lower bound must be <= 1, upper bound must be >= 1"
            assert self.cfg.randomize_d_gain_scale_upper >= 1, "pd scale lower bound must be <= 1, upper bound must be >= 1"
            rand_scale = self._rand_pd_scales(self.cfg.randomize_p_gain_scale_lower, self.cfg.randomize_p_gain_scale_upper, len(env_ids), self.num_hand_dofs)
            self.p_gain[env_ids] = self._p_gain_base.unsqueeze(0) * rand_scale
            rand_scale = self._rand_pd_scales(self.cfg.randomize_d_gain_scale_lower, self.cfg.randomize_d_gain_scale_upper, len(env_ids), self.num_hand_dofs)
            self.d_gain[env_ids] = self._d_gain_base.unsqueeze(0) * rand_scale

        # pose cache
        ndof_cache = self.num_hand_dofs
        if self.saved_grasping_states is not None:
            if self.cfg.grasp_cache_sequential:
                sampled_idx = env_ids.to(device=self.device) % self.saved_grasping_states.shape[0]
            else:
                sampled_idx = torch.randint(
                    0, self.saved_grasping_states.shape[0], (len(env_ids),), device=self.device
                )
            sampled_pose = self.saved_grasping_states[sampled_idx].clone()
            # Grasp-cache quaternions are stored as xyzw, while Isaac Lab's
            # simulation APIs expect wxyz.
            sampled_object_quat = torch.cat(
                [sampled_pose[:, ndof_cache + 6:ndof_cache + 7], sampled_pose[:, ndof_cache + 3:ndof_cache + 6]],
                dim=-1,
            )
        else:
            sampled_pose = torch.cat([
                self.init_joint_pos.expand(len(env_ids), -1),
                self.object.data.default_root_state[env_ids, :3],
                self.object.data.default_root_state[env_ids, 3:7],
            ], dim=-1)
            sampled_object_quat = sampled_pose[:, ndof_cache + 3:ndof_cache + 7]

        # reset object
        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        if self.cfg.reset_random_quat:
            rotate_center = self.hand.data.default_root_state.clone()[env_ids, :3]
            q_rand = get_random_rotation(env_ids, self.device)
            _, object_default_pos = apply_random_rotation_with_center(object_default_state[:, 3:7], object_default_state[:, 0:3], rotate_center, q_rand)
            self.object_default_pose[env_ids, :3] = object_default_pos.clone()
            object_default_state[:, 3:7], object_default_state[:, 0:3] = apply_random_rotation_with_center(sampled_object_quat, sampled_pose[:, ndof_cache:ndof_cache+3], rotate_center, q_rand)
            object_default_state[:, 0:3] += self.scene.env_origins[env_ids]
        else:
            self.object_default_pose[env_ids, :3] = object_default_state[:, :3].clone()
            object_default_state[:, 0:3] = sampled_pose[:, ndof_cache:ndof_cache+3] + self.scene.env_origins[env_ids]
            object_default_state[:, 3:7] = sampled_object_quat
        object_default_state[:, 7:] = torch.zeros_like(self.object.data.default_root_state[env_ids, 7:])
        self.object.write_root_pose_to_sim(object_default_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(object_default_state[:, 7:], env_ids)
        self.object_default_pose[env_ids, 3:7] = object_default_state[:, 3:7]
        self.rb_forces[env_ids, :] = 0.0

        self.reset_height_lower[env_ids] = object_default_state[:, 2] - (self.cfg.reset_height_upper - self.cfg.reset_height_lower) / 2
        self.reset_height_upper[env_ids] = object_default_state[:, 2] + (self.cfg.reset_height_upper - self.cfg.reset_height_lower) / 2

        # reset hand
        hand_default_state = self.hand.data.default_root_state.clone()[env_ids]
        if self.cfg.reset_random_quat:
            hand_default_state[:, 3:7], hand_default_state[:, 0:3] = apply_random_rotation_with_center(hand_default_state[:, 3:7], hand_default_state[:, :3], rotate_center, q_rand)
        hand_default_state[:, 0:3] += self.scene.env_origins[env_ids]
        self.hand.write_root_state_to_sim(hand_default_state, env_ids)
        dof_pos = sampled_pose[:, :ndof_cache]
        self.grasp_joint_pos[env_ids] = dof_pos
        dof_vel = torch.zeros_like(self.hand.data.default_joint_vel[env_ids])
        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos
        self.hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)
        self._refresh_lab()
        self.object_pos_prev[env_ids] = self.object_pos[env_ids]
        self.object_rot_prev[env_ids] = self.object_rot[env_ids]

        # reset data buffers
        self.last_contacts[env_ids] = 0
        self.at_reset_buf[env_ids] = 1

    def _refresh_lab(self):
        # data for hand
        self.fingertip_pos = self.hand.data.body_pos_w[:, self.finger_bodies]
        self.fingertip_rot = self.hand.data.body_quat_w[:, self.finger_bodies]
        self.fingertip_pos -= self.scene.env_origins.repeat((1, self.num_fingertips)).reshape(self.num_envs, self.num_fingertips, 3)
        self.fingertip_velocities = self.hand.data.body_vel_w[:, self.finger_bodies]

        self.hand_dof_pos = self.hand.data.joint_pos
        self.hand_dof_vel = self.hand.data.joint_vel

        # data for object
        self.object_pos = self.object.data.root_pos_w - self.scene.env_origins
        self.object_rot = self.object.data.root_quat_w
        self.object_velocities = self.object.data.root_vel_w
        self.object_linvel = self.object.data.root_lin_vel_w
        self.object_angvel = self.object.data.root_ang_vel_w

        # visualize object-local coordinate axes using VisualizationMarkers
        if getattr(self.cfg, 'debug_show_axes', True) and self._axes_visualizer is not None and self.num_envs > 0:
            try:
                # world poses are already with env origins; add back origins for vis API if needed
                object_pos_w = self.object.data.root_pos_w
                object_quat_w = self.object.data.root_quat_w
                self._axes_visualizer.visualize(
                    translations=object_pos_w, orientations=object_quat_w
                )
            except Exception:
                pass

    def compute_observations(self):
        # Object-filtered resultant force for each fingertip, sampled exactly
        # once per policy step (20 Hz / 0.05 s).  force_matrix_w excludes
        # self-collision and contacts with anything except the target object.
        object_contact_forces = torch.stack(
            [sensor.data.force_matrix_w[:, 0, 0, :] for sensor in self._contact_sensor],
            dim=1,
        )
        contact_forces = torch.nan_to_num(torch.norm(object_contact_forces, dim=-1))
        contact_forces[:, self._contact_body_ids_disable] = 0.0
        if self.cfg.binary_contact:
            binary_contacts = torch.where(contact_forces > self.cfg.contact_threshold, 1.0, 0.0)
            latency_samples = torch.rand_like(self.last_contacts)
            latency = torch.where(latency_samples < self.cfg.contact_latency, 1.0, 0.0)
            self.last_contacts = self.last_contacts * latency + binary_contacts * (1 - latency)
            mask = torch.rand_like(self.last_contacts)
            mask = torch.where(mask < self.cfg.contact_sensor_noise, 0.0, 1.0)
            sensed_contacts = torch.where(self.last_contacts > 0.1, mask * self.last_contacts, self.last_contacts)
        else:
            latency_samples = torch.rand_like(self.last_contacts)
            latency = torch.where(latency_samples < self.cfg.contact_latency, 1.0, 0.0)
            self.last_contacts = self.last_contacts * latency + contact_forces * (1 - latency)
            sensed_contacts = self.last_contacts.clone()

        # contact_pos computation retained for future reference (always zeroed: enable_contact_pos=False)
        # not_contact_mask = sensed_contacts < 1.0e-6
        # not_contact_mask[:, self._contact_body_ids_disable] = True
        # contact_mask = ~not_contact_mask
        # contact_pos = torch.cat([self._contact_sensor[id].data.contact_pos_w[:, 0, 0, :].unsqueeze(1) for id in self._contact_body_ids], dim=1)
        # contact_pos = torch.nan_to_num(contact_pos, nan=0.0)
        # contact_pos[contact_mask, :] = transform_between_frames(contact_pos[contact_mask, :] - tactile_frame_pos[contact_mask, :], world_quat[contact_mask, :], tactile_frame_quat[contact_mask, :])
        # contact_pos[not_contact_mask, :] = 0.0
        # contact_pos = contact_pos.reshape(self.num_envs, -1)
        # if not self.cfg.enable_contact_pos:
        #     contact_pos[:] = 0.0

        if not self.cfg.enable_tactile:
            sensed_contacts[:] = 0.0
        self.extras['tactile/force_mean_n'] = sensed_contacts.mean()
        self.extras['tactile/force_max_n'] = sensed_contacts.max()
        self.extras['tactile/contact_rate'] = (sensed_contacts > self.cfg.contact_threshold).float().mean()

        # Build the current frame and append it to a chronological ring buffer.
        joint_noise_matrix = (torch.rand(self.hand_dof_pos.shape, device=self.device) * 2.0 - 1.0) * self.cfg.joint_noise_scale
        cur_obs_buf = unscale(
            joint_noise_matrix + self.hand_dof_pos, 
            self.hand_dof_lower_limits, 
            self.hand_dof_upper_limits
        ).unsqueeze(1)
        cur_tar_buf = self.cur_targets[:, None]
        cur_frame = torch.cat([cur_obs_buf, cur_tar_buf, sensed_contacts.unsqueeze(1)], dim=-1).squeeze(1)
        self._obs_history_index = (self._obs_history_index + 1) % self._obs_history_len
        self.obs_buf_lag_history[:, self._obs_history_index] = cur_frame

        # refill the initialized buffers
        at_reset_env_ids = self.at_reset_buf.nonzero(as_tuple=False).squeeze(-1)
        ndof = self.num_hand_dofs
        self.obs_buf_lag_history[at_reset_env_ids, :, 0:ndof] = unscale(
            self.hand_dof_pos[at_reset_env_ids],
            self.hand_dof_lower_limits[at_reset_env_ids],
            self.hand_dof_upper_limits[at_reset_env_ids],
        ).clone().unsqueeze(1)
        self.obs_buf_lag_history[at_reset_env_ids, :, ndof:ndof*2] = self.hand_dof_pos[at_reset_env_ids].unsqueeze(1)
        self.obs_buf_lag_history[at_reset_env_ids, :, ndof*2:ndof*2+5] = sensed_contacts[at_reset_env_ids].unsqueeze(1)
        self.at_reset_buf[at_reset_env_ids] = 0
        history_indices = (self._obs_history_offsets + self._obs_history_index + 1) % self._obs_history_len
        chronological_history = self.obs_buf_lag_history.index_select(1, history_indices)
        obs_buf = chronological_history[:, -3:].reshape(self.num_envs, -1)

        # Optional ablation/no-tactile mode. Tactile Stage2 keeps this enabled.
        if not self.cfg.enable_contact_in_obs:
            obs_buf = obs_buf.clone()
            obs_single = ndof * 2 + 5
            for f in range(3):
                obs_buf[:, f * obs_single + ndof * 2:f * obs_single + ndof * 2 + 5] = 0.0

        self.proprio_hist_buf = chronological_history[:, -self.cfg.prop_hist_len:]
        self.priv_info_buf[:, 0:3] = self.object_pos - self.object_default_pose[:, :3]
        object_axis_world = rotate_axis_by_quat(
            self.object_rotation_axis_local, self.object_rot
        )
        self.priv_info_buf[:, 8] = self._gravity_magnitude
        self.priv_info_buf[:, 9:12] = object_axis_world
        self.priv_info_buf[:, 12:15] = self.object_angvel
        self.priv_info_buf[:, 15:18] = self.object_linvel

        return obs_buf
    
    def set_friction(self, asset, value, num_envs):
        materials = asset.root_physx_view.get_material_properties()
        materials[..., 0] = value  # Static friction.
        materials[..., 1] = value  # Dynamic friction.
        env_ids = torch.arange(num_envs, device="cpu")
        asset.root_physx_view.set_material_properties(materials, env_ids)

    def set_com(self, asset, value, num_envs):
        coms = asset.root_physx_view.get_coms().clone()
        coms[:, :3] += value
        env_ids = torch.arange(num_envs, device="cpu")
        asset.root_physx_view.set_coms(coms, env_ids)

    def set_mass(self, asset, value, num_envs):
        env_ids = torch.arange(num_envs, device="cpu")
        asset.root_physx_view.set_masses(value, env_ids)


@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)

@torch.jit.script
def compute_rewards(
    rotate_reward: torch.Tensor, rotate_reward_scale: float,
    stable_rotation_bonus: torch.Tensor, stable_rotation_bonus_scale: float,
    alive_reward: torch.Tensor, alive_reward_scale: float,
    object_axis_tilt_penalty: torch.Tensor, object_axis_tilt_penalty_scale: float,
    off_axis_angvel_penalty: torch.Tensor, off_axis_angvel_penalty_scale: float,
    xy_drift_penalty: torch.Tensor, xy_drift_penalty_scale: float,
    z_drift_penalty: torch.Tensor, z_drift_penalty_scale: float,
    drop_penalty: torch.Tensor, drop_penalty_scale: float,
    self_collision_penalty: torch.Tensor, self_collision_penalty_scale: float,
    torque_penalty: torch.Tensor, torque_penalty_scale: float,
    work_penalty: torch.Tensor, work_penalty_scale: float,
):
    reward = rotate_reward * rotate_reward_scale
    reward += stable_rotation_bonus * stable_rotation_bonus_scale
    reward += alive_reward * alive_reward_scale
    reward += object_axis_tilt_penalty * object_axis_tilt_penalty_scale
    reward += off_axis_angvel_penalty * off_axis_angvel_penalty_scale
    reward += xy_drift_penalty * xy_drift_penalty_scale
    reward += z_drift_penalty * z_drift_penalty_scale
    reward += drop_penalty * drop_penalty_scale
    reward += self_collision_penalty * self_collision_penalty_scale
    reward += torque_penalty * torque_penalty_scale
    reward += work_penalty * work_penalty_scale
    return reward

@torch.jit.script
def smooth_l1_normalized(x: torch.Tensor, tolerance: float) -> torch.Tensor:
    """Dimensionless Huber penalty: quadratic near zero and linear outside tolerance."""
    normalized = torch.abs(x) / tolerance
    return torch.where(normalized < 1.0, 0.5 * normalized ** 2, normalized - 0.5)

@torch.jit.script
def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    B = q.shape[0]
    R = torch.zeros((B, 3, 3), device=q.device, dtype=q.dtype)

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)

    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)

    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R

@torch.jit.script
def get_random_rotation(env_ids: torch.Tensor, device: str) -> torch.Tensor:
    N = env_ids.shape[0]

    u1 = torch.rand(N, device=device)
    u2 = torch.rand(N, device=device) * 2.0 * torch.pi
    u3 = torch.rand(N, device=device) * 2.0 * torch.pi
    q1 = torch.sqrt(1.0 - u1) * torch.sin(u2)
    q2 = torch.sqrt(1.0 - u1) * torch.cos(u2)
    q3 = torch.sqrt(u1) * torch.sin(u3)
    q4 = torch.sqrt(u1) * torch.cos(u3)
    q_rand = torch.stack([q4, q1, q2, q3], dim=-1)

    return q_rand

@torch.jit.script
def apply_random_rotation_with_center(
    qs_init: torch.Tensor, pos_init: torch.Tensor, center: torch.Tensor, q_rand: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    qs_new = quat_mul(q_rand, qs_init)

    R = quat_to_rotmat(q_rand)
    offset = pos_init - center
    new_offset = torch.bmm(R, offset.unsqueeze(-1)).squeeze(-1)
    pos_new = new_offset + center

    return qs_new, pos_new

@torch.jit.script
def rotate_axis_by_quat(axis: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    axis_q = torch.cat([torch.zeros(axis.shape[:-1] + (1,), device=axis.device), axis], dim=-1)
    quat_conj = quat_conjugate(quat)
    rotated_q = quat_mul(quat_mul(quat, axis_q), quat_conj)
    return rotated_q[..., 1:]

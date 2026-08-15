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

import isaaclab.sim as sim_utils
import omni.physics.tensors.impl.api as physx
from pxr import Usd, UsdGeom
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, quat_conjugate, quat_mul, saturate

if TYPE_CHECKING:
    from .revo3_hand_hora_env_cfg import Revo3HandHoraEnvCfg


class Revo3HandHoraEnv(DirectRLEnv):
    """DirectRLEnv for Revo3 right hand in-hand object rotation.

    Actor observation (141 dims) — 3-frame sliding window, 47 dims/frame:
      [0:21]   joint positions, unscaled to [-1,1] via (2x - hi - lo)/(hi - lo), +-0.02 rad noise
      [21:42]  current joint targets (delta-accumulated, clamped to joint limits)
      [42:47]  object-filtered resultant-force magnitudes on 5 DIP fingertips,
               scaled by 0.1 and sensor-randomized, sampled at 20 Hz

    Privileged observation (21 dims): object position delta (3), friction (1),
      mass (1), COM (3), world gravity direction (3), normalized radius (1),
      cylinder world axis (3), object angular velocity (3), and object linear
      velocity (3).

    Action (21 dims) — delta position control:
      action ∈ [-1,1] → target = prev_target + (1/24)*action → clamp(joint_limits)
      Torque control: torque = p_gain*(target - pos) - d_gain*vel
      p_gain/d_gain use the per-joint-type cfg values and are randomized per
      reset: ×[0.5, 2.0] per-DOF

    Reward (total ×0.01 for PPO): target world-Z rotation; cylinder tilt and
      off-axis angular-velocity penalties; independent smooth radial/axial drift
      penalties; explicit drop penalty; sampled-cache posture, torque and work
      regularization.

    Termination:
      drift:     object exceeds the radial/axial workspace around reset pose
      timeout:   episode_length >= max_episode_length (400 steps @20Hz)
      gravity:   fixed 9.81 m/s² magnitude, uniformly random sphere direction
                 per environment and episode

    Key design decisions:
      - the sampled grasp-cache row is the per-environment posture reference
      - PD gains per-joint-type from cfg.pgain_dict/dgain_dict, not read from URDF/USD
      - torque/work penalty uses self.torques (our explicit PD command), not PhysX applied_torque
      - tactile Stage2 keeps all five force magnitudes in actor obs and proprio_hist
    """
    cfg: Revo3HandHoraEnvCfg

    def __init__(self, cfg: Revo3HandHoraEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.num_hand_dofs = self.hand.num_joints

        # Canonical init joint pose from assets.py — used for cache-less reset.
        self.init_joint_pos = torch.zeros((1, self.num_hand_dofs), device=self.device)
        _cfg_pos = self.cfg.robot_cfg.init_state.joint_pos
        if _cfg_pos:
            for _name, _val in _cfg_pos.items():
                if _name in self.hand.joint_names:
                    self.init_joint_pos[0, self.hand.joint_names.index(_name)] = float(_val)
        # Per-environment posture reference.  This is updated to the actual
        # sampled grasp-cache joint pose on every reset.
        self.grasp_joint_pos = self.init_joint_pos.expand(self.num_envs, -1).clone()

        self._axes_visualizer = None
        if getattr(self.cfg, 'debug_show_axes', True):
            try:
                from isaaclab.markers import VisualizationMarkers
                from isaaclab.markers.config import FRAME_MARKER_CFG
                # create frame marker configuration for cylinder
                axes_marker_cfg = FRAME_MARKER_CFG.replace(
                    prim_path="/Visuals/CylinderAxes"
                )
                # adjust the axes size based on config (default 0.06 m)
                axes_length = getattr(self.cfg, 'vis_cylinder_axes_length', 0.06)
                axes_marker_cfg.markers["frame"].scale = (axes_length, axes_length, axes_length)
                # create the visualization marker
                self._axes_visualizer = VisualizationMarkers(axes_marker_cfg)
            except Exception:
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
        self.object_reset_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.rb_forces = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.gravity_direction_w = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.gravity_direction_w[:, 2] = -1.0
        self._gravity_magnitude = float(self.cfg.gravity_magnitude)
        self._gravity_reset_sum = torch.zeros((), dtype=torch.float32, device=self.device)
        self._gravity_window_steps = 0
        self._gravity_window_reset_rate = 1.0

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

        # Radius geometry is read back from USD rather than inferred from env
        # indices, so its cache and privileged value cannot drift out of sync
        # with the actual collision shape.
        self.cylinder_radius_mm = self._read_cylinder_radii_mm()
        nominal = float(self.cfg.cylinder_radius_nominal_mm)
        half_range = float(self.cfg.cylinder_radius_normalization_half_range_mm)
        self.normalized_cylinder_radius = (self.cylinder_radius_mm.float() - nominal) / half_range
        self.radius_local_index = torch.empty(self.num_envs, dtype=torch.long, device=self.device)
        for radius_mm in torch.unique(self.cylinder_radius_mm).tolist():
            radius_env_ids = (self.cylinder_radius_mm == int(radius_mm)).nonzero(as_tuple=False).squeeze(-1)
            self.radius_local_index[radius_env_ids] = torch.arange(
                len(radius_env_ids), dtype=torch.long, device=self.device
            )
        self.saved_grasping_states_by_radius = self._load_grasp_caches()

        self.rot_axis = torch.tensor(self.cfg.rot_axis, dtype=torch.float32).repeat(self.num_envs, 1).to(self.device)

        # contact buffers
        self._contact_body_ids = torch.arange(self.num_fingertips, dtype=torch.long, device=self.device)
        self._contact_body_ids_disable = torch.tensor(
            self.cfg.disable_tactile_ids, dtype=torch.long, device=self.device
        )
        self.last_contacts = torch.zeros(
            (self.num_envs, len(self._contact_body_ids)), dtype=torch.float, device=self.device
        )
        self.joint_zero_offset = torch.zeros(
            (self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device
        )
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
        # Mass is static for the lifetime of an environment instance.  Cache
        # it on the simulation device instead of synchronizing a PhysX property
        # query on every one of the 12 physics substeps.
        self.object_mass = self.object.root_physx_view.get_masses().reshape(
            self.num_envs, 1
        ).to(self.device)
        self.priv_info_buf[:, 4] = self.object_mass.squeeze(-1)

        # Physics scene gravity is intentionally zero.  Per-environment gravity
        # is applied at each object's COM in _apply_action.
        self.physics_sim_view: physx.SimulationView = sim_utils.SimulationContext.instance().physics_sim_view
        gravity = self.physics_sim_view.get_gravity()
        if max(abs(float(gravity[i])) for i in range(3)) > 1.0e-6:
            raise RuntimeError(f"Scene gravity must be zero for per-environment gravity, got {gravity}")

    def _setup_scene(self):
        # add hand, in-hand object, and goal object
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # replicate_physics=False pre-creates independent env Xforms before
        # this hook.  Re-cloning env_0 here would overwrite heterogeneous
        # cylinder geometries created by MultiAssetSpawnerCfg.
        if self.cfg.scene.replicate_physics:
            self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions()
        # add articulation to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["hand"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        # contact sensors
        self._contact_sensor = []
        for id in range(len(self.cfg.contact_sensor)):
            self._contact_sensor.append(ContactSensor(self.cfg.contact_sensor[id]))
            self.scene.sensors[f"contact_sensor_{id}"] = self._contact_sensor[id]
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _read_cylinder_radii_mm(self) -> torch.Tensor:
        """Read actual USD radii and verify PhysX/environment tensor ordering."""
        if self.object.num_instances != self.num_envs:
            raise RuntimeError(
                f"RigidObject has {self.object.num_instances} instances for {self.num_envs} environments"
            )
        view_paths = list(self.object.root_physx_view.prim_paths)
        if len(view_paths) != self.num_envs:
            raise RuntimeError(
                f"RigidObject tensor view exposes {len(view_paths)} paths for {self.num_envs} environments"
            )
        view_env_ids: list[int] = []
        for path in view_paths:
            try:
                view_env_ids.append(int(path.split("/env_")[1].split("/")[0]))
            except (IndexError, ValueError) as exc:
                raise RuntimeError(f"Cannot parse environment id from PhysX path: {path}") from exc
        if view_env_ids != list(range(self.num_envs)):
            raise RuntimeError(
                "RigidObject tensor-view order does not match env_0..env_N; "
                "radius/cache assignment would be unsafe"
            )

        object_prims = sim_utils.find_matching_prims(self.cfg.object_cfg.prim_path)
        if len(object_prims) != self.num_envs:
            raise RuntimeError(
                f"USD stage exposes {len(object_prims)} object prims for {self.num_envs} environments"
            )
        radii_by_env: dict[int, int] = {}
        non_cylinder_envs: list[int] = []
        for object_prim in object_prims:
            path = object_prim.GetPath().pathString
            try:
                env_id = int(path.split("/env_")[1].split("/")[0])
            except (IndexError, ValueError) as exc:
                raise RuntimeError(f"Cannot parse environment id from object path: {path}") from exc
            cylinder_prims = [
                prim for prim in Usd.PrimRange(object_prim)
                if prim.IsA(UsdGeom.Cylinder)
            ]
            if len(cylinder_prims) == 0:
                non_cylinder_envs.append(env_id)
                continue
            if len(cylinder_prims) != 1:
                raise RuntimeError(
                    f"Expected one cylinder geometry below {path}, found {len(cylinder_prims)}"
                )
            radius_m = float(UsdGeom.Cylinder(cylinder_prims[0]).GetRadiusAttr().Get())
            radii_by_env[env_id] = int(round(radius_m * 1000.0))

        if non_cylinder_envs:
            if radii_by_env or self.cfg.randomize_cylinder_radius:
                raise RuntimeError(
                    "Object batch mixes cylinder and non-cylinder geometry, or radius randomization "
                    "was enabled for a non-cylinder object"
                )
            if sorted(non_cylinder_envs) != list(range(self.num_envs)):
                raise RuntimeError("Non-cylinder geometry paths do not cover every environment exactly once")
            # The shared environment also supports the legacy sphere task.  It
            # has no meaningful radius privilege, so encode the nominal value
            # (normalized to zero) and use its single legacy cache.
            return torch.full(
                (self.num_envs,), int(self.cfg.cylinder_radius_nominal_mm),
                dtype=torch.int64, device=self.device,
            )

        if sorted(radii_by_env) != list(range(self.num_envs)):
            raise RuntimeError("Cylinder geometry paths do not cover every environment exactly once")
        radii = torch.tensor(
            [radii_by_env[env_id] for env_id in range(self.num_envs)],
            dtype=torch.int64,
            device=self.device,
        )
        allowed = torch.tensor(self.cfg.cylinder_radius_bins_mm, device=self.device)
        if not torch.isin(radii, allowed).all():
            raise RuntimeError(f"Unexpected cylinder radii in USD: {torch.unique(radii).tolist()}")
        counts = {int(radius): int((radii == radius).sum()) for radius in allowed.tolist()}
        if self.cfg.randomize_cylinder_radius and self.num_envs == 16384:
            expected = {radius: (6554 if radius == 30 else 983) for radius in allowed.tolist()}
            if counts != expected:
                raise RuntimeError(
                    f"Unexpected 16384-environment radius distribution: {counts}; expected {expected}"
                )
        print(f"[INFO] Cylinder radius counts (mm): {counts}", flush=True)
        return radii

    def _cache_path_for_radius(self, radius_mm: int) -> str:
        return f"{self.cfg.grasp_cache_path}_r{int(radius_mm)}mm.npy"

    def _load_grasp_caches(self) -> dict[int, torch.Tensor]:
        caches: dict[int, torch.Tensor] = {}
        if self.cfg.randomize_cylinder_radius:
            cache_paths = {
                radius_mm: self._cache_path_for_radius(radius_mm)
                for radius_mm in sorted(set(int(value) for value in self.cylinder_radius_mm.tolist()))
            }
        else:
            cache_paths = {
                int(self.cfg.cylinder_radius_nominal_mm): f"{self.cfg.grasp_cache_path}.npy"
            }
        missing: list[str] = []
        for radius_mm, path in cache_paths.items():
            if not os.path.exists(path):
                missing.append(path)
                continue
            states_np = np.load(path)
            expected_width = self.num_hand_dofs + 7
            if states_np.ndim != 2 or states_np.shape[1] != expected_width or states_np.shape[0] == 0:
                raise ValueError(
                    f"Invalid grasp cache {path}: expected non-empty [N,{expected_width}], got {states_np.shape}"
                )
            if not np.isfinite(states_np).all():
                raise ValueError(f"Grasp cache contains NaN/Inf: {path}")
            quat_norm = np.linalg.norm(states_np[:, -4:], axis=1)
            if not np.allclose(quat_norm, 1.0, rtol=1.0e-3, atol=1.0e-3):
                raise ValueError(f"Grasp cache contains non-unit xyzw quaternions: {path}")
            caches[radius_mm] = torch.from_numpy(states_np).float().to(self.device)
        if missing and self.cfg.strict_grasp_caches:
            mode = "Multi-radius training" if self.cfg.randomize_cylinder_radius else "This environment"
            raise FileNotFoundError(
                f"{mode} requires the following grasp cache(s). Missing:\n  "
                + "\n  ".join(missing)
                + "\nGenerate them with gen_grasp.py --cylinder_radius_mm <25..35>."
            )
        for path in missing:
            print(f"[WARN] Grasp cache not found: {path}; using configured default pose.", flush=True)
        return caches

    def _sample_grasp_states(self, env_ids: torch.Tensor) -> torch.Tensor:
        default_state = self.object.data.default_root_state[env_ids]
        default_quat_wxyz = default_state[:, 3:7]
        sampled = torch.cat(
            (
                self.init_joint_pos.expand(len(env_ids), -1),
                default_state[:, :3],
                default_quat_wxyz[:, 1:4],
                default_quat_wxyz[:, 0:1],
            ),
            dim=-1,
        ).clone()
        env_radii = self.cylinder_radius_mm[env_ids]
        for radius_mm, cache in self.saved_grasping_states_by_radius.items():
            if self.cfg.randomize_cylinder_radius:
                local_ids = (env_radii == radius_mm).nonzero(as_tuple=False).squeeze(-1)
            else:
                local_ids = torch.arange(len(env_ids), device=self.device)
            if local_ids.numel() == 0:
                continue
            if self.cfg.grasp_cache_sequential:
                cache_ids = self.radius_local_index[env_ids[local_ids]] % cache.shape[0]
            else:
                cache_ids = torch.randint(0, cache.shape[0], (len(local_ids),), device=self.device)
            sampled[local_ids] = cache[cache_ids]
        return sampled

    def _sample_gravity_directions(self, env_ids: torch.Tensor) -> None:
        if self.cfg.randomize_gravity_direction:
            directions = torch.randn((len(env_ids), 3), device=self.device)
            directions /= torch.linalg.vector_norm(directions, dim=-1, keepdim=True).clamp_min(1.0e-8)
        else:
            directions = torch.zeros((len(env_ids), 3), device=self.device)
            directions[:, 2] = -1.0
        self.gravity_direction_w[env_ids] = directions

    def _object_drift(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        delta = self.object_pos - self.object_reset_pos
        task_axis = self.rot_axis
        signed_axial = (delta * task_axis).sum(-1)
        axial = torch.abs(signed_axial)
        radial = torch.linalg.vector_norm(delta - signed_axial.unsqueeze(-1) * task_axis, dim=-1)
        dropped = (radial > self.cfg.drop_radial_distance) | (axial > self.cfg.drop_axial_distance)
        return radial, axial, dropped

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
            self.rb_forces *= self.cfg.force_decay ** (
                self.step_dt / self.cfg.force_decay_interval
            )
            # apply new forces
            prob = self.cfg.random_force_prob_scalar
            force_indices = (
                torch.rand(self.num_envs, device=self.device) < prob
            ).nonzero(as_tuple=False).squeeze(-1)
            if force_indices.numel() > 0:
                self.rb_forces[force_indices] = (
                    torch.randn((len(force_indices), 3), device=self.device)
                    * self.object_mass[force_indices]
                    * self.cfg.force_scale
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

        # Instantaneous composer is reset by RigidObject.write_data_to_sim(),
        # so this is refreshed on every physics substep.  Convert the desired
        # world force to the current link frame and apply it at the COM offset
        # in that same frame.  Keeping force and position in one frame avoids
        # the mixed-frame torque bug in Isaac Lab 0.54.x's global-position
        # WrenchComposer path.
        gravity_force_w = (
            self.object_mass * self._gravity_magnitude * self.gravity_direction_w
        )
        link_quat_w = self.object.data.body_link_quat_w[:, 0]
        gravity_force_b = quat_apply_inverse(link_quat_w, gravity_force_w).unsqueeze(1)
        com_position_b = self.object.data.body_com_pos_b
        composer = self.object.instantaneous_wrench_composer
        composer.set_forces_and_torques(
            forces=gravity_force_b,
            positions=com_position_b,
            is_global=False,
        )
        if self.cfg.force_scale > 0.0:
            random_force_b = quat_apply_inverse(link_quat_w, self.rb_forces).unsqueeze(1)
            composer.add_forces_and_torques(
                forces=random_force_b,
                positions=com_position_b,
                is_global=False,
            )

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
        """Reward world-Z rotation while preserving a stable upright grasp."""
        # PhysX reports angular velocity directly in the world frame.  The
        # target component remains world Z, as required by the task.
        object_angvel = self.object_angvel
        rotate_reward = saturate((object_angvel * self.rot_axis).sum(-1), torch.tensor(self.cfg.angvel_clip_min), torch.tensor(self.cfg.angvel_clip_max))

        # CylinderCfg's long axis is local Z.  Its sign is irrelevant for an
        # axis, hence abs(dot) maps the tilt to [0, pi/2].
        cylinder_axis_world = rotate_axis_by_quat(self.rot_axis, self.object_rot)
        upright_cos = torch.clamp(
            torch.abs((cylinder_axis_world * self.rot_axis).sum(-1)), 0.0, 1.0
        )
        cylinder_tilt = torch.acos(upright_cos)
        cylinder_tilt_penalty = (cylinder_tilt / self.cfg.cylinder_tilt_tolerance) ** 2

        target_angvel = (object_angvel * self.rot_axis).sum(-1, keepdim=True) * self.rot_axis
        off_axis_angvel_penalty = ((object_angvel - target_angvel) ** 2).sum(-1)

        radial_drift, axial_drift, dropped = self._object_drift()
        xy_drift_penalty = smooth_l1_normalized(radial_drift, self.cfg.xy_drift_tolerance)
        z_drift_penalty = smooth_l1_normalized(axial_drift, self.cfg.z_drift_tolerance)

        # The penalty reference must match the specific cache row sampled for
        # each environment, not one canonical pose shared by the whole batch.
        pos_diff_penalty = ((self.hand_dof_pos[:, self.actuated_dof_indices] - self.grasp_joint_pos[:, self.actuated_dof_indices]) ** 2).sum(-1)
        torque_penalty = (self.torques[:, self.actuated_dof_indices] ** 2).sum(-1)
        work_penalty = ((self.torques[:, self.actuated_dof_indices] * self.hand_dof_vel[:, self.actuated_dof_indices]).sum(-1)) ** 2
        # Applied on the terminal transition, before DirectRLEnv resets it.
        drop_penalty = dropped.float()

        total_reward = compute_rewards(
            rotate_reward, self.cfg.rotate_reward_scale,
            cylinder_tilt_penalty, self.cfg.cylinder_tilt_penalty_scale,
            off_axis_angvel_penalty, self.cfg.off_axis_angvel_penalty_scale,
            xy_drift_penalty, self.cfg.xy_drift_penalty_scale,
            z_drift_penalty, self.cfg.z_drift_penalty_scale,
            drop_penalty, self.cfg.drop_penalty_scale,
            pos_diff_penalty, self.cfg.pos_diff_penalty_scale,
            torque_penalty, self.cfg.torque_penalty_scale,
            work_penalty, self.cfg.work_penalty_scale,
        )

        self.extras["rew/rotate"] = (rotate_reward * self.cfg.rotate_reward_scale).mean()
        self.extras["rew/cylinder_tilt"] = (cylinder_tilt_penalty * self.cfg.cylinder_tilt_penalty_scale).mean()
        self.extras["rew/off_axis_angvel"] = (off_axis_angvel_penalty * self.cfg.off_axis_angvel_penalty_scale).mean()
        self.extras["rew/xy_drift"] = (xy_drift_penalty * self.cfg.xy_drift_penalty_scale).mean()
        self.extras["rew/z_drift"] = (z_drift_penalty * self.cfg.z_drift_penalty_scale).mean()
        self.extras["rew/drop"] = (drop_penalty * self.cfg.drop_penalty_scale).mean()
        self.extras["rew/posture"] = (pos_diff_penalty * self.cfg.pos_diff_penalty_scale).mean()
        self.extras["rew/torque"] = (torque_penalty * self.cfg.torque_penalty_scale).mean()
        self.extras["rew/work"] = (work_penalty * self.cfg.work_penalty_scale).mean()
        self.extras['cylinder_tilt_deg'] = torch.rad2deg(cylinder_tilt).mean()
        self.extras['radial_drift_mm'] = (radial_drift * 1000.0).mean()
        self.extras['axial_drift_mm'] = (axial_drift * 1000.0).mean()
        self.extras['angvelX'] = object_angvel[:, 0].mean()
        self.extras['angvelY'] = object_angvel[:, 1].mean()
        self.extras['angvelZ'] = object_angvel[:, 2].mean()
        self.extras['total_reward'] = total_reward.mean()
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Terminate escaped objects and update the full-gravity drop-rate window."""
        self._refresh_lab()
        _, _, drop_reset = self._object_drift()
        # A transition that both reaches the horizon and drops is a true task
        # termination, not a truncation eligible for value bootstrapping.
        time_out = (self.episode_length_buf >= self.max_episode_length) & ~drop_reset
        self.extras['drop_reset'] = drop_reset.float().mean()
        self.extras['time_out'] = time_out.float().mean()
        instant_reset_rate = drop_reset.float().mean()
        self._gravity_reset_sum += instant_reset_rate.detach()
        self._gravity_window_steps += 1
        if self._gravity_window_steps >= self.cfg.drop_reset_rate_window:
            self._gravity_window_reset_rate = float(
                (self._gravity_reset_sum / self._gravity_window_steps).item()
            )
            self._gravity_reset_sum.zero_()
            self._gravity_window_steps = 0

        self.extras['gravity_magnitude'] = self._gravity_magnitude
        self.extras['gravity_dir_x_mean'] = self.gravity_direction_w[:, 0].mean()
        self.extras['gravity_dir_y_mean'] = self.gravity_direction_w[:, 1].mean()
        self.extras['gravity_dir_z_mean'] = self.gravity_direction_w[:, 2].mean()
        self.extras['gravity_reset_rate_instant'] = instant_reset_rate
        self.extras['drop_reset_rate_window'] = self._gravity_window_reset_rate
        # Compatibility alias for older log readers/checkpoints.
        self.extras['gravity_reset_rate_window'] = self._gravity_window_reset_rate
        self.extras['gravity_full_stable'] = float(
            self._gravity_window_reset_rate <= self.cfg.drop_stable_reset_rate
        )
        return drop_reset, time_out

    def set_gravity_magnitude(self, magnitude: float) -> None:
        """Set the fixed magnitude used by per-environment equivalent gravity forces."""
        magnitude = float(magnitude)
        if magnitude < 0.0:
            raise ValueError("gravity magnitude must be non-negative")
        self._gravity_magnitude = magnitude

    def _rand_pd_scales(self, lower, upper, num_envs, n_dofs):
        rand_scale_s = torch.distributions.Uniform(lower, 1).sample((num_envs, n_dofs)).to(self.device)
        rand_scale_l = torch.distributions.Uniform(1, upper).sample((num_envs, n_dofs)).to(self.device)
        mask_choice = torch.rand((num_envs, n_dofs), device=self.device) > 0.5
        rand_scale = torch.where(mask_choice, rand_scale_s, rand_scale_l)
        return rand_scale

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset hand to grasp pose (from cache or init_joint_pos), object to default state.
        PD gains randomized per-DOF each reset: p_gain × [0.5,2.0], d_gain × [0.5,2.0].
        Drop bounds are radial/axial around the sampled object reset pose."""
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        # resets articulation and rigid body attributes
        super()._reset_idx(env_ids)

        # Encoder zero drift is constant within an episode and affects only
        # the joint positions observed by the policy.
        if self.cfg.randomize_joint_zero:
            self.joint_zero_offset[env_ids] = torch.empty(
                (len(env_ids), self.num_hand_dofs), device=self.device
            ).uniform_(
                self.cfg.joint_zero_offset_lower, self.cfg.joint_zero_offset_upper
            )
        else:
            self.joint_zero_offset[env_ids] = 0.0

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

        # Sample only from the cache generated for each environment's actual
        # cylinder radius.
        ndof_cache = self.num_hand_dofs
        sampled_pose = self._sample_grasp_states(env_ids)
        # Grasp-cache quaternions are stored as xyzw, while Isaac Lab's
        # simulation APIs expect wxyz.
        sampled_object_quat = torch.cat(
            [sampled_pose[:, ndof_cache + 6:ndof_cache + 7], sampled_pose[:, ndof_cache + 3:ndof_cache + 6]],
            dim=-1,
        )

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
        self.object_reset_pos[env_ids] = object_default_state[:, :3] - self.scene.env_origins[env_ids]
        self.object_default_pose[env_ids, :3] = self.object_reset_pos[env_ids]
        self._sample_gravity_directions(env_ids)

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

        # visualize coordinate axes for cylinder using VisualizationMarkers
        if getattr(self.cfg, 'debug_show_axes', True) and self._axes_visualizer is not None and self.num_envs > 0:
            try:
                # world poses are already with env origins; add back origins for vis API if needed
                cyl_pos_w = self.object.data.root_pos_w
                cyl_quat_w = self.object.data.root_quat_w
                self._axes_visualizer.visualize(translations=cyl_pos_w, orientations=cyl_quat_w)
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
        contact_force_magnitudes = torch.nan_to_num(torch.norm(object_contact_forces, dim=-1))
        contact_force_magnitudes[:, self._contact_body_ids_disable] = 0.0
        if self.cfg.binary_contact:
            contacts = (contact_force_magnitudes > self.cfg.contact_threshold).float()
            latency = torch.rand_like(self.last_contacts) < self.cfg.contact_latency
            self.last_contacts = torch.where(latency, self.last_contacts, contacts)
            dropout = torch.rand_like(self.last_contacts) >= self.cfg.contact_sensor_noise
            sensed_contacts = self.last_contacts * dropout
        else:
            latency = torch.rand_like(self.last_contacts) < self.cfg.contact_latency
            self.last_contacts = torch.where(latency, self.last_contacts, contact_force_magnitudes)
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
            normalized_contacts = torch.zeros_like(sensed_contacts)
        else:
            sensed_contacts[:, self._contact_body_ids_disable] = 0.0
            normalized_contacts = sensed_contacts * self.cfg.contact_force_scale
            if self.cfg.randomize_contact_force and self.cfg.contact_force_noise_std > 0.0:
                normalized_contacts += (
                    torch.randn_like(normalized_contacts) * self.cfg.contact_force_noise_std
                )
            normalized_contacts[:, self._contact_body_ids_disable] = 0.0

        self.extras['tactile/force_mean_n'] = sensed_contacts.mean()
        self.extras['tactile/force_max_n'] = sensed_contacts.max()
        self.extras['tactile/contact_rate'] = (contact_force_magnitudes > self.cfg.contact_threshold).float().mean()

        # Build the current frame and append it to a chronological ring buffer.
        joint_noise_matrix = (torch.rand(self.hand_dof_pos.shape, device=self.device) * 2.0 - 1.0) * self.cfg.joint_noise_scale
        sensed_joint_pos = self.hand_dof_pos + self.joint_zero_offset
        cur_obs_buf = unscale(
            joint_noise_matrix + sensed_joint_pos,
            self.hand_dof_lower_limits, 
            self.hand_dof_upper_limits
        ).unsqueeze(1)
        cur_tar_buf = self.cur_targets[:, None]
        cur_frame = torch.cat([cur_obs_buf, cur_tar_buf, normalized_contacts.unsqueeze(1)], dim=-1).squeeze(1)
        self._obs_history_index = (self._obs_history_index + 1) % self._obs_history_len
        self.obs_buf_lag_history[:, self._obs_history_index] = cur_frame

        # refill the initialized buffers
        at_reset_env_ids = self.at_reset_buf.nonzero(as_tuple=False).squeeze(-1)
        ndof = self.num_hand_dofs
        # Fill reset history with the same current noisy encoder sample used by
        # cur_frame, instead of silently dropping per-step joint noise on the
        # first policy observation of every episode.
        self.obs_buf_lag_history[at_reset_env_ids, :, 0:ndof] = cur_obs_buf[at_reset_env_ids]
        self.obs_buf_lag_history[at_reset_env_ids, :, ndof:ndof*2] = self.cur_targets[at_reset_env_ids].unsqueeze(1)
        self.obs_buf_lag_history[at_reset_env_ids, :, ndof*2:] = normalized_contacts[at_reset_env_ids].unsqueeze(1)
        self.at_reset_buf[at_reset_env_ids] = 0
        history_indices = (self._obs_history_offsets + self._obs_history_index + 1) % self._obs_history_len
        chronological_history = self.obs_buf_lag_history.index_select(1, history_indices)
        obs_buf = chronological_history[:, -3:].reshape(self.num_envs, -1)

        # Optional ablation/no-tactile mode. Tactile Stage2 keeps this enabled.
        if not self.cfg.enable_contact_in_obs:
            obs_buf = obs_buf.clone()
            contact_dim = self.num_fingertips
            obs_single = ndof * 2 + contact_dim
            for f in range(3):
                obs_buf[:, f * obs_single + ndof * 2:f * obs_single + ndof * 2 + contact_dim] = 0.0

        self.proprio_hist_buf = chronological_history[:, -self.cfg.prop_hist_len:]
        self.priv_info_buf[:, 0:3] = self.object_pos - self.object_reset_pos
        cylinder_axis_world = rotate_axis_by_quat(self.rot_axis, self.object_rot)
        self.priv_info_buf[:, 8:11] = self.gravity_direction_w
        self.priv_info_buf[:, 11] = self.normalized_cylinder_radius
        self.priv_info_buf[:, 12:15] = cylinder_axis_world
        self.priv_info_buf[:, 15:18] = self.object_angvel
        self.priv_info_buf[:, 18:21] = self.object_linvel

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
        masses = asset.root_physx_view.get_masses()
        new_mass = value.reshape(num_envs).to(device=masses.device, dtype=masses.dtype)
        masses[:, 0] = new_mass
        asset.root_physx_view.set_masses(masses, env_ids)

        # Scale every environment's own default inertia.  This preserves the
        # radius-dependent cylinder inertia instead of reusing a 30 mm tensor.
        default_mass = asset.data.default_mass.reshape(num_envs).to(
            device=asset.data.default_inertia.device,
            dtype=asset.data.default_inertia.dtype,
        )
        mass_scale = new_mass.to(default_mass.device) / default_mass.clamp_min(1.0e-8)
        inertias = asset.root_physx_view.get_inertias()
        inertias[:] = asset.data.default_inertia.reshape(num_envs, 9) * mass_scale.unsqueeze(-1)
        asset.root_physx_view.set_inertias(inertias, env_ids)


@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)

@torch.jit.script
def compute_rewards(
    rotate_reward: torch.Tensor, rotate_reward_scale: float,
    cylinder_tilt_penalty: torch.Tensor, cylinder_tilt_penalty_scale: float,
    off_axis_angvel_penalty: torch.Tensor, off_axis_angvel_penalty_scale: float,
    xy_drift_penalty: torch.Tensor, xy_drift_penalty_scale: float,
    z_drift_penalty: torch.Tensor, z_drift_penalty_scale: float,
    drop_penalty: torch.Tensor, drop_penalty_scale: float,
    pos_diff_penalty: torch.Tensor, pos_diff_penalty_scale: float,
    torque_penalty: torch.Tensor, torque_penalty_scale: float,
    work_penalty: torch.Tensor, work_penalty_scale: float,
):
    reward = rotate_reward * rotate_reward_scale
    reward += cylinder_tilt_penalty * cylinder_tilt_penalty_scale
    reward += off_axis_angvel_penalty * off_axis_angvel_penalty_scale
    reward += xy_drift_penalty * xy_drift_penalty_scale
    reward += z_drift_penalty * z_drift_penalty_scale
    reward += drop_penalty * drop_penalty_scale
    reward += pos_diff_penalty * pos_diff_penalty_scale
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

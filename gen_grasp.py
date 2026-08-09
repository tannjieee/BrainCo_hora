"""Generate grasp cache for Isaac Lab Revo3 Stage 1 training.

Grasp detection (reset_buf approach in _get_rewards):
  cond1: all 5 fingertips within 0.1m of object center
  cond2: >=4 fingertips each contact in >=70% of a 20-step rolling window
  cond3: after settling, >=2 live contacts and cylinder-axis tilt <=10 deg
  cond4: after settling, XY drift <=5mm and Z drift <=15mm

Gravity testing defaults to fixed -Z gravity. Six-axis cycling is opt-in after the
  fixed-gravity baseline produces stable grasps.

Joint exploration: ±noise_scale (default 0.15) rad around init_joint_pos, clamped to
  joint limits. Noise scale is NOT zeroed for any task — cylinder benefits from
  exploration too.

Cache format: [joint_pos(21), obj_local_xyz(3), obj_quat_xyzw(4)] = 28 floats per grasp.
Saved to cache/<grasp_cache_path>.npy.

Gotcha — timeout check: uses >= max_episode_length (NOT -1), must match env's _get_dones.
"""

import argparse
import copy
import os
import sys
import time
from collections.abc import Sequence

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="ball", choices=["ball", "cylinder"], help="Task variant")
parser.add_argument("--num_envs", type=int, default=8192)
parser.add_argument("--target_count", type=int, default=8192)
parser.add_argument("--cache_file", type=str, default="", help="Override output cache filename under cache/.")
parser.add_argument("--usd", type=str, default="", help="Override hand USD path.")
parser.add_argument("--noise_scale", type=float, default=0.15, help="±noise added to init_joint_pos")
parser.add_argument("--progress_interval", type=float, default=10.0, help="Seconds between progress updates.")
parser.add_argument("--settle_steps", type=int, default=20, help="Steps allowed for contact establishment.")
parser.add_argument("--contact_force_threshold", type=float, default=0.05, help="Per-fingertip object contact threshold in N.")
parser.add_argument(
    "--min_contact_fingertips",
    type=int,
    default=4,
    help="Distinct fingertips that must establish contact during settling.",
)
parser.add_argument(
    "--contact_window_steps",
    type=int,
    default=20,
    help="Rolling contact window in policy steps; 20 steps equals 1s at 20Hz.",
)
parser.add_argument(
    "--min_contact_ratio",
    type=float,
    default=0.70,
    help="Minimum above-threshold fraction in the rolling window for an established fingertip.",
)
parser.add_argument(
    "--min_live_contact_fingertips",
    type=int,
    default=2,
    help="Simultaneous fingertip contacts required after settling.",
)
parser.add_argument(
    "--max_axis_tilt_deg",
    type=float,
    default=10.0,
    help="Maximum cylinder long-axis deviation from world Z after settling.",
)
parser.add_argument(
    "--max_horizontal_drift_m",
    type=float,
    default=0.005,
    help="Maximum XY displacement from the candidate start position after settling.",
)
parser.add_argument(
    "--max_height_drift_m",
    type=float,
    default=0.015,
    help="Maximum object Z drift after settling; stricter than the training reset window.",
)
parser.add_argument(
    "--gravity_mode",
    choices=["fixed", "six_axis"],
    default="fixed",
    help="Use fixed -Z gravity or cycle through all six axis directions.",
)
parser.add_argument(
    "--gravity_interval",
    type=int,
    default=15,
    help="Global steps per direction in six_axis mode; 15 visits all six axes within a 100-step episode.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.num_envs <= 0:
    parser.error("--num_envs must be greater than 0")
if args.target_count <= 0:
    parser.error("--target_count must be greater than 0")
if args.progress_interval <= 0:
    parser.error("--progress_interval must be greater than 0")
if args.settle_steps < 0:
    parser.error("--settle_steps must be greater than or equal to 0")
if args.contact_force_threshold < 0:
    parser.error("--contact_force_threshold must be greater than or equal to 0")
if not 1 <= args.min_contact_fingertips <= 5:
    parser.error("--min_contact_fingertips must be between 1 and 5")
if args.contact_window_steps <= 0:
    parser.error("--contact_window_steps must be greater than 0")
if args.contact_window_steps > args.settle_steps:
    parser.error("--contact_window_steps must not exceed --settle_steps")
if not 0.0 < args.min_contact_ratio <= 1.0:
    parser.error("--min_contact_ratio must be in (0, 1]")
if not 0 <= args.min_live_contact_fingertips <= 5:
    parser.error("--min_live_contact_fingertips must be between 0 and 5")
if not 0.0 <= args.max_axis_tilt_deg <= 90.0:
    parser.error("--max_axis_tilt_deg must be between 0 and 90")
if args.max_horizontal_drift_m <= 0.0:
    parser.error("--max_horizontal_drift_m must be greater than 0")
if args.max_height_drift_m <= 0.0:
    parser.error("--max_height_drift_m must be greater than 0")
if args.gravity_interval <= 0:
    parser.error("--gravity_interval must be greater than 0")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import carb
import numpy as np
import torch

from isaaclab.utils.math import quat_conjugate, quat_mul, saturate

from hora.tasks.isaaclab import Revo3HandHoraEnv, Revo3HandHoraEnvCfg
from hora.tasks.isaaclab.assets import (
    BALL_OBJECT_CFG, CYLINDER_OBJECT_CFG,
    REVO3_HAND_BALL_CFG, REVO3_HAND_CYLINDER_CFG,
)


class GraspGenEnv(Revo3HandHoraEnv):
    """Revo3HandHoraEnv subclass for grasp cache collection — reset_buf approach."""

    FINGERTIP_NEAR_THRESHOLD = 0.10
    def __init__(
        self,
        cfg,
        render_mode=None,
        noise_scale: float = 0.15,
        target_count: int | None = None,
        progress_interval: float = 10.0,
        settle_steps: int = 20,
        contact_force_threshold: float = 0.05,
        min_contact_fingertips: int = 4,
        contact_window_steps: int = 20,
        min_contact_ratio: float = 0.70,
        min_live_contact_fingertips: int = 2,
        max_axis_tilt_deg: float = 10.0,
        max_horizontal_drift_m: float = 0.005,
        max_height_drift_m: float = 0.015,
        check_axis_tilt: bool = True,
        gravity_mode: str = "fixed",
        gravity_interval: int = 15,
        **kwargs,
    ):
        self._noise_scale = noise_scale
        if target_count is None:
            raise ValueError("target_count must be provided explicitly from CLI.")
        self._target_count = int(target_count)
        self._collected: list[np.ndarray] = []
        self._collected_count = 0
        self._progress_interval = float(progress_interval)
        self._progress_started = False
        self._progress_start_time = 0.0
        self._last_progress_time = 0.0
        self._last_progress_total = 0
        self._attempt_count = 0
        self._settle_steps = int(settle_steps)
        self._contact_force_threshold = float(contact_force_threshold)
        self._min_contact_fingertips = int(min_contact_fingertips)
        self._contact_window_steps = int(contact_window_steps)
        self._min_contact_ratio = float(min_contact_ratio)
        self._min_live_contact_fingertips = int(min_live_contact_fingertips)
        self._max_axis_tilt = float(max_axis_tilt_deg) * torch.pi / 180.0
        self._max_horizontal_drift = float(max_horizontal_drift_m)
        self._max_height_drift = float(max_height_drift_m)
        self._check_axis_tilt = bool(check_axis_tilt)
        self._gravity_mode = gravity_mode
        self._gravity_interval = int(gravity_interval)
        self._contact_history = torch.zeros(
            (cfg.scene.num_envs, self._contact_window_steps, 5),
            dtype=torch.bool,
            device=cfg.sim.device,
        )
        self._contact_window_counts = torch.zeros(
            cfg.scene.num_envs, dtype=torch.int16, device=cfg.sim.device
        )
        self._contact_history_index = 0
        self._gravity_id = 0
        self._gravity_directions = [
            carb.Float3(0.0, 0.0, -9.81),
            carb.Float3(0.0, 0.0, 9.81),
            carb.Float3(9.81, 0.0, 0.0),
            carb.Float3(-9.81, 0.0, 0.0),
            carb.Float3(0.0, 9.81, 0.0),
            carb.Float3(0.0, -9.81, 0.0),
        ]
        self._reset_condition_stats()
        super().__init__(cfg, render_mode, **kwargs)
        # The state that must be cached is the candidate at the start of its
        # validation episode.  Saving the terminal state can reintroduce a
        # settled pose with incompatible zero velocity/contact penetration.
        self._candidate_joint_pos = torch.zeros(
            (self.num_envs, self.num_hand_dofs), device=self.device
        )
        self._candidate_object_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._candidate_object_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.physics_sim_view.set_gravity(self._gravity_directions[0])
        if self._gravity_mode == "six_axis":
            self._gravity_id = 1

    def _reset_condition_stats(self):
        self._condition_samples = 0
        self._cond_fingertip_passes = 0
        self._cond_contact_passes = 0
        self._contact_established_passes = 0
        self._cond_rotation_passes = 0
        self._cond_all_passes = 0
        self._cond_height_passes = 0
        self._cond_horizontal_passes = 0
        self._contact_count_sum = 0.0
        self._contact_force_sum = 0.0
        self._contact_force_samples = 0
        self._active_contact_force_sum = 0.0
        self._active_contact_force_samples = 0
        self._contact_force_max = 0.0
        self._axis_tilt_sum = 0.0
        self._axis_tilt_max = 0.0
        self._finger_contact_passes = [0.0] * 5
        self._finger_force_sums = [0.0] * 5
        self._finger_force_max = [0.0] * 5
        self._finger_distance_sums = [0.0] * 5
        self._finger_contact_ratio_sums = [0.0] * 5

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = max(0, int(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def start_progress_tracking(self):
        """Start tracking after the initial environment reset."""
        now = time.perf_counter()
        self._progress_started = True
        self._progress_start_time = now
        self._last_progress_time = now
        self._last_progress_total = min(self._collected_count, self._target_count)
        self._attempt_count = 0
        self._reset_condition_stats()

    def _print_progress(self, force: bool = False):
        if not self._progress_started:
            return

        now = time.perf_counter()
        since_last = now - self._last_progress_time
        if not force and since_last < self._progress_interval:
            return

        completed = min(self._collected_count, self._target_count)
        elapsed = max(now - self._progress_start_time, 1.0e-6)
        new_grasps = completed - self._last_progress_total
        overall_rate = completed / elapsed
        recent_rate = new_grasps / max(since_last, 1.0e-6)
        success_rate = 100.0 * completed / max(self._attempt_count, 1)
        percentage = 100.0 * completed / self._target_count
        eta = (
            self._format_duration((self._target_count - completed) / overall_rate)
            if overall_rate > 0.0
            else "--:--:--"
        )

        print(
            f"[PROGRESS] {completed:>6}/{self._target_count} ({percentage:6.2f}%)"
            f" | attempts={self._attempt_count}"
            f" | success={success_rate:6.2f}%"
            f" | rate={overall_rate:7.2f}/s (recent={recent_rate:7.2f}/s)"
            f" | elapsed={self._format_duration(elapsed)} | ETA={eta}",
            flush=True,
        )

        if self._condition_samples > 0:
            samples = self._condition_samples
            force_mean = self._contact_force_sum / max(self._contact_force_samples, 1)
            active_force_mean = self._active_contact_force_sum / max(self._active_contact_force_samples, 1)
            print(
                f"[CONDITIONS] fingertip={100.0 * self._cond_fingertip_passes / samples:6.2f}%"
                f" | contact_frame={100.0 * self._cond_contact_passes / samples:6.2f}%"
                f" | established={100.0 * self._contact_established_passes / samples:6.2f}%"
                f" | axis/rotation={100.0 * self._cond_rotation_passes / samples:6.2f}%"
                f" | horizontal={100.0 * self._cond_horizontal_passes / samples:6.2f}%"
                f" | height={100.0 * self._cond_height_passes / samples:6.2f}%"
                f" | all_frame={100.0 * self._cond_all_passes / samples:6.2f}%"
                f" | contacts={self._contact_count_sum / samples:4.2f}/5"
                f" | force mean={force_mean:6.3f}N"
                f" active_mean={active_force_mean:6.3f}N max={self._contact_force_max:6.3f}N"
                f" | tilt mean={self._axis_tilt_sum / samples:5.2f}deg"
                f" max={self._axis_tilt_max:5.2f}deg",
                flush=True,
            )
            finger_names = ("thumb", "index", "middle", "ring", "little")
            finger_stats = []
            for index, name in enumerate(finger_names):
                finger_stats.append(
                    f"{name}:contact={100.0 * self._finger_contact_passes[index] / samples:5.1f}%"
                    f",force={self._finger_force_sums[index] / samples:5.3f}N"
                    f",max={self._finger_force_max[index]:5.2f}N"
                    f",window={100.0 * self._finger_contact_ratio_sums[index] / samples:5.1f}%"
                    f",dist={100.0 * self._finger_distance_sums[index] / samples:4.1f}cm"
                )
            print(f"[FINGERS] {' | '.join(finger_stats)}", flush=True)

        self._last_progress_time = now
        self._last_progress_total = completed
        self._reset_condition_stats()

    def _update_condition_stats(
        self,
        cond_fingertip: torch.Tensor,
        cond_contact: torch.Tensor,
        cond_rotation: torch.Tensor,
        contact_forces: torch.Tensor,
        contact_mask: torch.Tensor,
        contact_established: torch.Tensor,
        fingertip_distances: torch.Tensor,
        axis_tilt: torch.Tensor,
        cond_height: torch.Tensor,
        cond_horizontal: torch.Tensor,
        contact_ratio: torch.Tensor,
    ):
        force_magnitudes = torch.norm(contact_forces, dim=-1)
        all_conditions = cond_fingertip & cond_contact & cond_rotation & cond_height & cond_horizontal
        active_force_sum = (force_magnitudes * contact_mask).sum()
        scalar_stats = torch.stack(
            (
                cond_fingertip.float().sum(),
                cond_contact.float().sum(),
                contact_established.float().sum(),
                cond_rotation.float().sum(),
                all_conditions.float().sum(),
                contact_mask.float().sum(),
                force_magnitudes.sum(),
                active_force_sum,
                force_magnitudes.max(),
                torch.rad2deg(axis_tilt).sum(),
                torch.rad2deg(axis_tilt).max(),
                cond_height.float().sum(),
                cond_horizontal.float().sum(),
            )
        )
        per_finger_stats = torch.cat(
            (
                contact_mask.float().sum(dim=0),
                force_magnitudes.sum(dim=0),
                force_magnitudes.max(dim=0).values,
                fingertip_distances.sum(dim=0),
                contact_ratio.sum(dim=0),
            )
        )
        step_stats = torch.cat((scalar_stats, per_finger_stats)).tolist()

        self._condition_samples += cond_fingertip.numel()
        self._cond_fingertip_passes += int(step_stats[0])
        self._cond_contact_passes += int(step_stats[1])
        self._contact_established_passes += int(step_stats[2])
        self._cond_rotation_passes += int(step_stats[3])
        self._cond_all_passes += int(step_stats[4])
        self._contact_count_sum += step_stats[5]
        self._contact_force_sum += step_stats[6]
        self._contact_force_samples += force_magnitudes.numel()
        self._active_contact_force_sum += step_stats[7]
        self._active_contact_force_samples += int(step_stats[5])
        self._contact_force_max = max(self._contact_force_max, step_stats[8])
        self._axis_tilt_sum += step_stats[9]
        self._axis_tilt_max = max(self._axis_tilt_max, step_stats[10])
        self._cond_height_passes += int(step_stats[11])
        self._cond_horizontal_passes += int(step_stats[12])
        for finger_index in range(5):
            self._finger_contact_passes[finger_index] += step_stats[13 + finger_index]
            self._finger_force_sums[finger_index] += step_stats[18 + finger_index]
            self._finger_force_max[finger_index] = max(
                self._finger_force_max[finger_index], step_stats[23 + finger_index]
            )
            self._finger_distance_sums[finger_index] += step_stats[28 + finger_index]
            self._finger_contact_ratio_sums[finger_index] += step_stats[33 + finger_index]

    def _get_rewards(self) -> torch.Tensor:
        self._refresh_lab()
        # cond1: all 5 fingertips within 0.1m of object
        fingertip_distances = torch.norm(self.fingertip_pos - self.object_pos.unsqueeze(1), dim=-1)
        cond1 = (fingertip_distances < self.FINGERTIP_NEAR_THRESHOLD).all(-1)
        # cond2: use object-filtered forces, excluding self and unrelated contacts
        object_contact_forces = torch.stack(
            [sensor.data.force_matrix_w[:, 0, 0, :] for sensor in self._contact_sensor],
            dim=1,
        )
        contact_mask = torch.norm(object_contact_forces, dim=-1) > self._contact_force_threshold
        cond2 = contact_mask.sum(-1) >= self._min_contact_fingertips
        self._contact_history[:, self._contact_history_index, :] = contact_mask
        self._contact_history_index = (self._contact_history_index + 1) % self._contact_window_steps
        self._contact_window_counts.add_(1).clamp_(max=self._contact_window_steps)
        contact_ratio = self._contact_history.float().sum(dim=1) / self._contact_window_counts.clamp_min(1).unsqueeze(-1)
        window_ready = self._contact_window_counts >= self._contact_window_steps
        established_fingers = contact_ratio >= self._min_contact_ratio
        contact_established = window_ready & (
            established_fingers.sum(-1) >= self._min_contact_fingertips
        )
        live_contact = contact_mask.sum(-1) >= self._min_live_contact_fingertips

        # Cylinder local Z in world coordinates.  Only the long-axis alignment
        # matters; yaw about that axis is intentionally not rejected.
        _, quat_x, quat_y, _ = self.object_rot.unbind(-1)
        axis_alignment = torch.abs(1.0 - 2.0 * (quat_x.square() + quat_y.square()))
        axis_tilt = torch.acos(torch.clamp(axis_alignment, 0.0, 1.0))
        if self._check_axis_tilt:
            cond3 = axis_tilt <= self._max_axis_tilt
        else:
            dq = quat_mul(self.object_rot, quat_conjugate(self.object.data.default_root_state[:, 3:7]))
            angle = 2.0 * torch.acos(torch.clamp(torch.abs(dq[:, 0]), 0.0, 1.0))
            cond3 = angle < (45.0 / 180.0 * torch.pi)
        position_delta = self.object_pos - self._candidate_object_pos
        horizontal_drift = torch.norm(position_delta[:, :2], dim=-1)
        cond_horizontal = horizontal_drift <= self._max_horizontal_drift
        cond_height = torch.abs(position_delta[:, 2]) <= self._max_height_drift

        self._update_condition_stats(
            cond1,
            cond2,
            cond3,
            object_contact_forces,
            contact_mask,
            contact_established,
            fingertip_distances,
            axis_tilt,
            cond_height,
            cond_horizontal,
            contact_ratio,
        )

        # During settling, do not reject a candidate for missing contact or
        # tilt. Afterwards require multi-frame contact evidence, a modest live
        # contact floor, and tight cylinder-axis alignment.
        settling = self.episode_length_buf <= self._settle_steps
        stable = contact_established & live_contact & cond3 & cond_height & cond_horizontal
        cond = cond1 & (stable | settling)
        self.reset_buf[~cond] = 1

        # Six-axis testing is deliberately opt-in. Baseline cache generation
        # uses fixed -Z gravity until stable candidates have been confirmed.
        if self._gravity_mode == "six_axis" and self.common_step_counter % self._gravity_interval == 0:
            self.physics_sim_view.set_gravity(self._gravity_directions[self._gravity_id])
            self._gravity_id = (self._gravity_id + 1) % len(self._gravity_directions)

        self._print_progress()
        return torch.zeros(self.num_envs, device=self.device)

    def _reset_idx(self, env_ids: Sequence[int]):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES

        self._refresh_lab()
        if self._progress_started:
            self._attempt_count += len(env_ids)

        # collect states that survived full episode (successful grasps)
        success = self.episode_length_buf >= self.max_episode_length
        n_success = success.sum().item()
        if n_success > 0:
            joint_pos = self._candidate_joint_pos[success].cpu().numpy()
            obj_local = self._candidate_object_pos[success].cpu().numpy()
            obj_quat_wxyz = self._candidate_object_quat[success].cpu().numpy()
            obj_quat_xyzw = np.concatenate([obj_quat_wxyz[:, 1:], obj_quat_wxyz[:, :1]], axis=-1)
            entry = np.concatenate([joint_pos, obj_local, obj_quat_xyzw], axis=-1)
            self._collected.append(entry)

        self._collected_count += n_success
        total = self._collected_count
        if total >= self._target_count:
            self._print_progress(force=True)
            self._save_and_exit()

        # A time-based heartbeat is also emitted when no grasp succeeds, so a
        # slow search can be distinguished from a stalled process.
        self._print_progress()

        # full scene reset
        self.scene.reset(env_ids)

        # reset episode length buffer
        self.episode_length_buf[env_ids] = 0
        self._contact_history[env_ids] = False
        self._contact_window_counts[env_ids] = 0

        # random joint exploration: ±0.15 rad around default
        ndof = self.num_hand_dofs
        rand_floats = 2.0 * torch.rand((len(env_ids), ndof), device=self.device) - 1.0
        dof_pos = self.init_joint_pos.expand(len(env_ids), -1) + self._noise_scale * rand_floats
        dof_pos = saturate(dof_pos, self.hand_dof_lower_limits[env_ids], self.hand_dof_upper_limits[env_ids])
        dof_vel = torch.zeros_like(self.hand.data.default_joint_vel[env_ids])

        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos
        self.hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

        # reset object to default state
        obj_default = self.object.data.default_root_state.clone()[env_ids]
        obj_default[:, 0:3] += self.scene.env_origins[env_ids]
        obj_default[:, 7:] = 0.0
        self.object.write_root_pose_to_sim(obj_default[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(obj_default[:, 7:], env_ids)
        self.rb_forces[env_ids, :] = 0.0

        self.reset_height_lower[env_ids] = self.cfg.reset_height_lower
        self.reset_height_upper[env_ids] = self.cfg.reset_height_upper

        self._refresh_lab()
        # Record the exact initial state whose complete episode will be judged.
        # These buffers are allocated after the constructor's implicit reset,
        # so the hasattr guard only skips that one bootstrap reset.
        if hasattr(self, "_candidate_joint_pos"):
            self._candidate_joint_pos[env_ids] = dof_pos
            self._candidate_object_pos[env_ids] = self.object_pos[env_ids]
            self._candidate_object_quat[env_ids] = self.object_rot[env_ids]
        self.object_pos_prev[env_ids] = self.object_pos[env_ids]
        self.object_rot_prev[env_ids] = self.object_rot[env_ids]
        self.last_contacts[env_ids] = 0
        self.proprio_hist_buf[env_ids] = 0
        self.at_reset_buf[env_ids] = 1

    def _save_and_exit(self):
        all_states = np.concatenate(self._collected, axis=0)[: self._target_count]
        os.makedirs("cache", exist_ok=True)
        path = f"{self.cfg.grasp_cache_path}.npy"
        np.save(path, all_states.astype(np.float32))
        print(f"\n[INFO] Saved {len(all_states)} grasps -> {path}")
        print(f"       shape={all_states.shape}  dtype={all_states.dtype}")
        self.close()
        simulation_app.close()
        sys.exit(0)


env_cfg = Revo3HandHoraEnvCfg()

# Select correct robot_cfg and object_cfg based on task
_TASK_ROBOT_CFG = {"ball": REVO3_HAND_BALL_CFG, "cylinder": REVO3_HAND_CYLINDER_CFG}
_TASK_OBJECT_CFG = {"ball": BALL_OBJECT_CFG, "cylinder": CYLINDER_OBJECT_CFG}
env_cfg.robot_cfg = _TASK_ROBOT_CFG.get(args.task, REVO3_HAND_CYLINDER_CFG)
env_cfg.object_cfg = _TASK_OBJECT_CFG.get(args.task, CYLINDER_OBJECT_CFG)

# Set output cache filename based on task
_TASK_CACHE_FILE = {"ball": "cache/revo3_right_grasp_ball", "cylinder": "cache/revo3_right_grasp_cylinder"}
env_cfg.grasp_cache_path = _TASK_CACHE_FILE.get(args.task, _TASK_CACHE_FILE["cylinder"])
if args.cache_file:
    env_cfg.grasp_cache_path = f"cache/{args.cache_file.replace('.npy', '')}"

cache_path = f"{env_cfg.grasp_cache_path}.npy"

if args.usd:
    usd_path = os.path.abspath(args.usd)
    if not os.path.exists(usd_path):
        raise FileNotFoundError(f"--usd path not found: {usd_path}")
    env_cfg.robot_cfg = copy.deepcopy(env_cfg.robot_cfg)
    if env_cfg.robot_cfg.spawn is None or not hasattr(env_cfg.robot_cfg.spawn, "usd_path"):
        raise RuntimeError("env_cfg.robot_cfg.spawn has no usd_path to override.")
    env_cfg.robot_cfg.spawn.usd_path = usd_path

env_cfg.gravity_curriculum = False
env_cfg.scene.num_envs = args.num_envs
env_cfg.episode_length_s = 5.0
env_cfg.randomize_pd_gains = False
env_cfg.randomize_com = False
env_cfg.randomize_friction = False
env_cfg.randomize_mass = False  # use cfg default 0.10 kg
env_cfg.force_scale = 0.0
env_cfg.random_force_prob_scalar = 0.0
env_cfg.sim.device = "cuda:0"

env = GraspGenEnv(
    env_cfg,
    render_mode=None,
    noise_scale=args.noise_scale,
    target_count=args.target_count,
    progress_interval=args.progress_interval,
    settle_steps=args.settle_steps,
    contact_force_threshold=args.contact_force_threshold,
    min_contact_fingertips=args.min_contact_fingertips,
    contact_window_steps=args.contact_window_steps,
    min_contact_ratio=args.min_contact_ratio,
    min_live_contact_fingertips=args.min_live_contact_fingertips,
    max_axis_tilt_deg=args.max_axis_tilt_deg,
    max_horizontal_drift_m=args.max_horizontal_drift_m,
    max_height_drift_m=args.max_height_drift_m,
    check_axis_tilt=args.task == "cylinder",
    gravity_mode=args.gravity_mode,
    gravity_interval=args.gravity_interval,
)
env.reset()
env.start_progress_tracking()

print("\n[INFO] Grasp cache generation started.")
print(f"  task        : {args.task}")
print(f"  num_envs    : {args.num_envs}")
print(f"  noise_scale : ±{args.noise_scale} rad")
print(f"  episode_len : {env_cfg.episode_length_s}s  ({env.max_episode_length} steps)")
print(f"  target      : {args.target_count} grasps")
print(f"  progress    : every {args.progress_interval:g}s")
print(f"  settle      : {args.settle_steps} steps")
print(
    f"  contact     : >= {args.min_contact_fingertips} fingertips present in"
    f" >= {100.0 * args.min_contact_ratio:g}% of a {args.contact_window_steps}-step"
    f" ({args.contact_window_steps / 20.0:g}s at 20Hz) rolling window"
)
print(f"  force gate  : > {args.contact_force_threshold:g}N per fingertip per frame")
print(f"  live contact: >= {args.min_live_contact_fingertips} fingertips after settle")
if args.task == "cylinder":
    print(f"  axis tilt   : <= {args.max_axis_tilt_deg:g}deg after settle")
print(f"  XY drift    : <= {1000.0 * args.max_horizontal_drift_m:g}mm after settle")
print(f"  height drift: <= {1000.0 * args.max_height_drift_m:g}mm after settle")
print(f"  gravity     : {args.gravity_mode} (9.81m/s²)")
print("  disturbance : random force off, friction randomization off")
print(f"  output      : {cache_path}\n")
if args.usd:
    print(f"[INFO] hand usd   : {os.path.abspath(args.usd)}")

zero_actions = torch.zeros((args.num_envs, env_cfg.action_space), device=env.device)

while simulation_app.is_running():
    with torch.inference_mode():
        env.step(zero_actions)

"""Generate grasp cache for Isaac Lab Revo3 Stage 1 training.

Grasp detection (reset_buf approach in _get_rewards):
  cond1: all 5 fingertips within 0.1m of object center
  cond2: >=4 fingertips each contact in >=70% of a 20-step rolling window
  cond3: after settling, >=2 live contacts and cylinder-axis tilt <=10 deg
  cond4: after settling, XY drift <=5mm and Z drift <=15mm

Gravity testing defaults to a sphere-uniform direction sampled independently for
  every environment at reset.  The direction remains fixed for the complete
  candidate episode; scene gravity stays zero and the base environment applies
  the equivalent world-frame force on every physics substep.

Joint exploration: ±noise_scale (default 0.15) rad around init_joint_pos, clamped to
  joint limits. Noise scale is NOT zeroed for any task — cylinder benefits from
  exploration too.

Cache format: [joint_pos(21), obj_local_xyz(3), obj_quat_xyzw(4)] = 28 floats per grasp.
Saved to cache/<grasp_cache_path>.npy. Incomplete runs atomically checkpoint to
  matching .partial.npy + .partial.json files. Resume is allowed only when the
  quality settings, cylinder radius, and hand USD fingerprint match exactly.

Gotcha — timeout check: uses >= max_episode_length (NOT -1), must match env's _get_dones.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import tempfile
import time
import traceback
from collections.abc import Sequence

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="ball", choices=["ball", "cylinder"], help="Task variant")
parser.add_argument("--num_envs", type=int, default=8192)
parser.add_argument("--target_count", type=int, default=8192)
parser.add_argument(
    "--episode_length_s",
    type=float,
    default=5.0,
    help="Seconds that a candidate must remain valid before it is cached.",
)
parser.add_argument(
    "--checkpoint_interval",
    type=float,
    default=60.0,
    help="Seconds between atomic .partial.npy checkpoints; existing partial files resume automatically.",
)
parser.add_argument(
    "--cache_file",
    type=str,
    default="",
    help=(
        "Override cache name under cache/. For cylinders this is a shared prefix; "
        "_rXXmm is appended unless already present."
    ),
)
parser.add_argument(
    "--cylinder_radius_mm",
    type=int,
    choices=range(25, 36),
    default=30,
    help="Single cylinder radius to collect, in millimetres (25 through 35).",
)
parser.add_argument(
    "--force_overwrite",
    action="store_true",
    help="Start clean: allow replacing the final cache and discard any partial checkpoint.",
)
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
    choices=["sphere", "fixed"],
    default="sphere",
    help="Sample one sphere-uniform direction per environment/episode, or keep fixed -Z.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

default_cache_stem = "revo3_right_grasp_ball"
if args.task == "cylinder":
    radius_suffix = f"_r{args.cylinder_radius_mm}mm"
    default_cache_stem = f"revo3_right_grasp_cylinder{radius_suffix}"
    custom_cache_stem = args.cache_file.removesuffix(".npy")
    if custom_cache_stem and not custom_cache_stem.endswith(radius_suffix):
        custom_cache_stem += radius_suffix
else:
    custom_cache_stem = args.cache_file.removesuffix(".npy")
cache_stem = custom_cache_stem if custom_cache_stem else default_cache_stem
cache_relative_path = os.path.normpath(f"{cache_stem}.npy")
if os.path.isabs(cache_relative_path) or cache_relative_path == ".." or cache_relative_path.startswith(f"..{os.sep}"):
    parser.error("--cache_file must resolve below the repository cache/ directory")
cache_path = os.path.join("cache", cache_relative_path)
cache_path_without_suffix = os.path.splitext(cache_path)[0]
partial_cache_path = f"{cache_path_without_suffix}.partial.npy"
partial_metadata_path = f"{cache_path_without_suffix}.partial.json"

if args.num_envs <= 0:
    parser.error("--num_envs must be greater than 0")
if args.target_count <= 0:
    parser.error("--target_count must be greater than 0")
if args.episode_length_s <= 0.0:
    parser.error("--episode_length_s must be greater than 0")
if args.checkpoint_interval <= 0.0:
    parser.error("--checkpoint_interval must be greater than 0")
if args.progress_interval <= 0:
    parser.error("--progress_interval must be greater than 0")
if args.settle_steps < 0:
    parser.error("--settle_steps must be greater than or equal to 0")
collection_max_episode_steps = math.ceil(args.episode_length_s / (12.0 / 240.0))
if args.settle_steps >= collection_max_episode_steps:
    parser.error(
        f"--settle_steps must be less than the {collection_max_episode_steps}-step collection episode"
    )
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
if os.path.exists(cache_path) and not args.force_overwrite:
    parser.error(
        f"output cache already exists: {cache_path}; pass --force_overwrite to replace it"
    )
partial_exists = os.path.exists(partial_cache_path)
metadata_exists = os.path.exists(partial_metadata_path)
if args.force_overwrite:
    for stale_path in (partial_cache_path, partial_metadata_path):
        if os.path.exists(stale_path):
            try:
                os.unlink(stale_path)
            except OSError as error:
                parser.error(f"failed to discard stale partial checkpoint {stale_path}: {error}")
            print(f"[FORCE] Discarded stale partial checkpoint: {stale_path}", flush=True)
elif partial_exists != metadata_exists:
    parser.error(
        "incomplete partial checkpoint: both "
        f"{partial_cache_path} and {partial_metadata_path} must exist; "
        "use --force_overwrite to discard it and start clean"
    )

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch

from isaaclab.utils.math import quat_conjugate, quat_mul, saturate

from hora.tasks.isaaclab import Revo3HandHoraEnv, Revo3HandHoraEnvCfg
from hora.tasks.isaaclab.assets import (
    BALL_OBJECT_CFG,
    REVO3_HAND_BALL_CFG, REVO3_HAND_CYLINDER_CFG,
    make_cylinder_object_cfg,
)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        checkpoint_interval: float = 60.0,
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
        force_overwrite: bool = False,
        collection_signature: dict | None = None,
        **kwargs,
    ):
        self._noise_scale = noise_scale
        if target_count is None:
            raise ValueError("target_count must be provided explicitly from CLI.")
        self._target_count = int(target_count)
        self._collected: list[np.ndarray] = []
        self._collected_count = 0
        self._progress_interval = float(progress_interval)
        self._checkpoint_interval = float(checkpoint_interval)
        self._progress_started = False
        self._progress_start_time = 0.0
        self._last_progress_time = 0.0
        self._last_progress_total = 0
        self._session_start_count = 0
        self._attempt_count = 0
        self._last_checkpoint_time = time.perf_counter()
        self._last_checkpoint_count = 0
        self._cache_path = f"{cfg.grasp_cache_path}.npy"
        self._partial_cache_path = f"{cfg.grasp_cache_path}.partial.npy"
        self._partial_metadata_path = f"{cfg.grasp_cache_path}.partial.json"
        self._collection_complete = False
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
        self._force_overwrite = bool(force_overwrite)
        if collection_signature is None:
            raise ValueError("collection_signature is required for safe partial-cache resume.")
        self._collection_signature = copy.deepcopy(collection_signature)
        self._contact_history = torch.zeros(
            (cfg.scene.num_envs, self._contact_window_steps, 5),
            dtype=torch.bool,
            device=cfg.sim.device,
        )
        self._contact_window_counts = torch.zeros(
            cfg.scene.num_envs, dtype=torch.int16, device=cfg.sim.device
        )
        self._contact_history_index = 0
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
        self._candidate_is_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        if self._force_overwrite:
            self._discard_partial_checkpoint()
        else:
            self._resume_partial_cache()

    def _read_cylinder_radii_mm(self) -> torch.Tensor:
        """Keep ball collection compatible with the cylinder-oriented base environment."""
        if self._check_axis_tilt:
            return super()._read_cylinder_radii_mm()
        return torch.full(
            (self.num_envs,),
            int(self.cfg.cylinder_radius_nominal_mm),
            dtype=torch.int64,
            device=self.device,
        )

    def _load_grasp_caches(self) -> dict[int, torch.Tensor]:
        """A collector creates caches and must never require an existing cache."""
        return {}

    @property
    def collection_complete(self) -> bool:
        return self._collection_complete

    def _discard_partial_checkpoint(self):
        """Remove both partial components so forced runs can never mix old rows."""
        for stale_path in (self._partial_cache_path, self._partial_metadata_path):
            if os.path.exists(stale_path):
                os.unlink(stale_path)
                print(f"[FORCE] Discarded stale partial checkpoint: {stale_path}", flush=True)

    def _resume_partial_cache(self):
        """Resume only when the checkpoint and its strict provenance sidecar agree."""
        partial_exists = os.path.exists(self._partial_cache_path)
        metadata_exists = os.path.exists(self._partial_metadata_path)
        if not partial_exists and not metadata_exists:
            return
        if partial_exists != metadata_exists:
            raise RuntimeError(
                "Partial checkpoint is incomplete; both "
                f"{self._partial_cache_path} and {self._partial_metadata_path} are required. "
                "Use --force_overwrite to discard it and start clean."
            )

        with open(self._partial_metadata_path, encoding="utf-8") as file_obj:
            metadata = json.load(file_obj)
        expected_metadata_keys = {
            "schema_version",
            "state_count",
            "target_count_at_checkpoint",
            "cache_sha256",
            "collection_signature",
        }
        if not isinstance(metadata, dict) or set(metadata) != expected_metadata_keys:
            actual_keys = sorted(metadata) if isinstance(metadata, dict) else type(metadata).__name__
            raise ValueError(
                f"Invalid partial metadata keys in {self._partial_metadata_path}: "
                f"expected {sorted(expected_metadata_keys)}, found {actual_keys}."
            )
        if metadata["schema_version"] != 1:
            raise ValueError(
                f"Unsupported partial metadata schema {metadata['schema_version']!r} in "
                f"{self._partial_metadata_path}; expected 1."
            )
        if (
            not isinstance(metadata["cache_sha256"], str)
            or len(metadata["cache_sha256"]) != 64
            or _sha256_file(self._partial_cache_path) != metadata["cache_sha256"]
        ):
            raise ValueError(
                f"Partial cache checksum does not match {self._partial_metadata_path}; "
                "use --force_overwrite to discard the inconsistent checkpoint."
            )
        if metadata["collection_signature"] != self._collection_signature:
            expected = json.dumps(self._collection_signature, indent=2, sort_keys=True)
            actual = json.dumps(metadata["collection_signature"], indent=2, sort_keys=True)
            raise ValueError(
                "Partial checkpoint was produced with different collection settings and "
                "cannot be mixed with this run. Use --force_overwrite to start clean.\n"
                f"Expected signature:\n{expected}\nCheckpoint signature:\n{actual}"
            )
        if type(metadata["state_count"]) is not int or metadata["state_count"] < 0:
            raise ValueError(
                f"Invalid state_count in partial metadata: {metadata['state_count']!r}."
            )
        if (
            type(metadata["target_count_at_checkpoint"]) is not int
            or metadata["target_count_at_checkpoint"] <= 0
        ):
            raise ValueError(
                "Invalid target_count_at_checkpoint in partial metadata: "
                f"{metadata['target_count_at_checkpoint']!r}."
            )

        resumed = np.load(self._partial_cache_path, allow_pickle=False)
        expected_width = self.num_hand_dofs + 3 + 4
        if resumed.ndim != 2 or resumed.shape[1] != expected_width:
            raise ValueError(
                f"Invalid partial cache shape {resumed.shape} in {self._partial_cache_path}; "
                f"expected (N, {expected_width})."
            )
        if (
            not np.issubdtype(resumed.dtype, np.number)
            or np.iscomplexobj(resumed)
            or not np.isfinite(resumed).all()
        ):
            raise ValueError(
                f"Partial cache {self._partial_cache_path} must contain finite real values."
            )
        if len(resumed) != metadata["state_count"]:
            raise ValueError(
                f"Partial checkpoint count mismatch: array has {len(resumed)} rows but "
                f"sidecar records {metadata['state_count']}. The checkpoint may have been "
                "interrupted mid-update; use --force_overwrite to start clean."
            )
        quaternion_norms = np.linalg.norm(resumed[:, -4:].astype(np.float64), axis=1)
        if not np.all(np.abs(quaternion_norms - 1.0) <= 1.0e-3):
            worst_index = int(np.argmax(np.abs(quaternion_norms - 1.0)))
            raise ValueError(
                f"Partial cache contains a non-unit quaternion at row {worst_index}: "
                f"norm={quaternion_norms[worst_index]:.8f}."
            )

        original_count = len(resumed)
        resumed = np.ascontiguousarray(resumed[: self._target_count], dtype=np.float32)
        if len(resumed) > 0:
            self._collected = [resumed]
        self._collected_count = len(resumed)
        self._last_checkpoint_count = self._collected_count
        self._last_checkpoint_time = time.perf_counter()
        suffix = (
            f" (using the first {len(resumed)} for target {self._target_count})"
            if original_count != len(resumed)
            else ""
        )
        print(
            f"[RESUME] Loaded {original_count} grasps from {self._partial_cache_path}{suffix}",
            flush=True,
        )
        previous_target = metadata["target_count_at_checkpoint"]
        if previous_target != self._target_count:
            print(
                f"[RESUME] Target count changed from {previous_target} to {self._target_count}; "
                "collection quality settings are unchanged.",
                flush=True,
            )

    def _all_collected_states(self) -> np.ndarray:
        expected_width = self.num_hand_dofs + 3 + 4
        if not self._collected:
            return np.empty((0, expected_width), dtype=np.float32)
        return np.concatenate(self._collected, axis=0)[: self._target_count].astype(
            np.float32, copy=False
        )

    @staticmethod
    def _atomic_save(path: str, states: np.ndarray):
        """Write a NumPy cache in the destination directory, then atomically replace it."""
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        fd, temporary_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory
        )
        try:
            with os.fdopen(fd, "wb") as file_obj:
                np.save(file_obj, states.astype(np.float32, copy=False))
                file_obj.flush()
                os.fsync(file_obj.fileno())
            os.replace(temporary_path, path)
            temporary_path = ""
        finally:
            if temporary_path and os.path.exists(temporary_path):
                os.unlink(temporary_path)

    @staticmethod
    def _atomic_save_json(path: str, payload: dict):
        """Atomically write a strict provenance sidecar next to a partial cache."""
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        fd, temporary_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file_obj:
                json.dump(payload, file_obj, allow_nan=False, indent=2, sort_keys=True)
                file_obj.write("\n")
                file_obj.flush()
                os.fsync(file_obj.fileno())
            os.replace(temporary_path, path)
            temporary_path = ""
        finally:
            if temporary_path and os.path.exists(temporary_path):
                os.unlink(temporary_path)

    def save_partial(self, force: bool = False):
        """Checkpoint newly collected states without exposing a half-written file."""
        if self._collection_complete or self._collected_count <= 0:
            return

        now = time.perf_counter()
        if not force and now - self._last_checkpoint_time < self._checkpoint_interval:
            return
        if (
            self._collected_count == self._last_checkpoint_count
            and os.path.exists(self._partial_cache_path)
            and os.path.exists(self._partial_metadata_path)
        ):
            self._last_checkpoint_time = now
            return

        states = self._all_collected_states()
        self._atomic_save(self._partial_cache_path, states)
        cache_sha256 = _sha256_file(self._partial_cache_path)
        self._atomic_save_json(
            self._partial_metadata_path,
            {
                "schema_version": 1,
                "state_count": len(states),
                "target_count_at_checkpoint": self._target_count,
                "cache_sha256": cache_sha256,
                "collection_signature": self._collection_signature,
            },
        )
        self._last_checkpoint_count = len(states)
        self._last_checkpoint_time = now
        print(
            f"[CHECKPOINT] Saved {len(states)} grasps -> {self._partial_cache_path}",
            flush=True,
        )

    def _reset_condition_stats(self):
        self._condition_samples = 0
        self._cond_fingertip_passes = 0
        self._cond_contact_passes = 0
        self._contact_established_passes = 0
        self._live_contact_passes = 0
        self._cond_rotation_passes = 0
        self._post_settle_samples = 0
        self._stable_passes = 0
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
        self._session_start_count = self._last_progress_total
        self._attempt_count = 0
        self._reset_condition_stats()
        if self._collected_count >= self._target_count:
            self._save_and_exit()

    def _print_progress(self, force: bool = False):
        if not self._progress_started:
            return

        now = time.perf_counter()
        since_last = now - self._last_progress_time
        if not force and since_last < self._progress_interval:
            return

        completed = min(self._collected_count, self._target_count)
        session_completed = completed - self._session_start_count
        elapsed = max(now - self._progress_start_time, 1.0e-6)
        new_grasps = completed - self._last_progress_total
        overall_rate = session_completed / elapsed
        recent_rate = new_grasps / max(since_last, 1.0e-6)
        success_rate = 100.0 * session_completed / max(self._attempt_count, 1)
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
            post_settle_samples = max(self._post_settle_samples, 1)
            force_mean = self._contact_force_sum / max(self._contact_force_samples, 1)
            active_force_mean = self._active_contact_force_sum / max(self._active_contact_force_samples, 1)
            print(
                f"[CONDITIONS] fingertip={100.0 * self._cond_fingertip_passes / samples:6.2f}%"
                f" | frame_contact>={self._min_contact_fingertips}="
                f"{100.0 * self._cond_contact_passes / samples:6.2f}%"
                f" | established={100.0 * self._contact_established_passes / samples:6.2f}%"
                f" | live={100.0 * self._live_contact_passes / samples:6.2f}%"
                f" | axis/rotation={100.0 * self._cond_rotation_passes / samples:6.2f}%"
                f" | horizontal={100.0 * self._cond_horizontal_passes / samples:6.2f}%"
                f" | height={100.0 * self._cond_height_passes / samples:6.2f}%"
                f" | stable_post={100.0 * self._stable_passes / post_settle_samples:6.2f}%"
                f" | valid_post={100.0 * self._cond_all_passes / post_settle_samples:6.2f}%"
                f" (post_n={self._post_settle_samples})"
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
        live_contact: torch.Tensor,
        fingertip_distances: torch.Tensor,
        axis_tilt: torch.Tensor,
        cond_height: torch.Tensor,
        cond_horizontal: torch.Tensor,
        contact_ratio: torch.Tensor,
        stable: torch.Tensor,
        valid: torch.Tensor,
        settling: torch.Tensor,
    ):
        force_magnitudes = torch.norm(contact_forces, dim=-1)
        post_settle = ~settling
        active_force_sum = (force_magnitudes * contact_mask).sum()
        scalar_stats = torch.stack(
            (
                cond_fingertip.float().sum(),
                cond_contact.float().sum(),
                contact_established.float().sum(),
                live_contact.float().sum(),
                cond_rotation.float().sum(),
                cond_height.float().sum(),
                cond_horizontal.float().sum(),
                post_settle.float().sum(),
                (stable & post_settle).float().sum(),
                (valid & post_settle).float().sum(),
                contact_mask.float().sum(),
                force_magnitudes.sum(),
                active_force_sum,
                force_magnitudes.max(),
                torch.rad2deg(axis_tilt).sum(),
                torch.rad2deg(axis_tilt).max(),
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
        self._live_contact_passes += int(step_stats[3])
        self._cond_rotation_passes += int(step_stats[4])
        self._cond_height_passes += int(step_stats[5])
        self._cond_horizontal_passes += int(step_stats[6])
        self._post_settle_samples += int(step_stats[7])
        self._stable_passes += int(step_stats[8])
        self._cond_all_passes += int(step_stats[9])
        self._contact_count_sum += step_stats[10]
        self._contact_force_sum += step_stats[11]
        self._contact_force_samples += force_magnitudes.numel()
        self._active_contact_force_sum += step_stats[12]
        self._active_contact_force_samples += int(step_stats[10])
        self._contact_force_max = max(self._contact_force_max, step_stats[13])
        self._axis_tilt_sum += step_stats[14]
        self._axis_tilt_max = max(self._axis_tilt_max, step_stats[15])
        for finger_index in range(5):
            self._finger_contact_passes[finger_index] += step_stats[16 + finger_index]
            self._finger_force_sums[finger_index] += step_stats[21 + finger_index]
            self._finger_force_max[finger_index] = max(
                self._finger_force_max[finger_index], step_stats[26 + finger_index]
            )
            self._finger_distance_sums[finger_index] += step_stats[31 + finger_index]
            self._finger_contact_ratio_sums[finger_index] += step_stats[36 + finger_index]

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

        # During settling, do not reject a candidate for missing contact or
        # tilt. Afterwards require multi-frame contact evidence, a modest live
        # contact floor, and tight cylinder-axis alignment.
        settling = self.episode_length_buf <= self._settle_steps
        stable = contact_established & live_contact & cond3 & cond_height & cond_horizontal
        cond = cond1 & (stable | settling)

        self._update_condition_stats(
            cond1,
            cond2,
            cond3,
            object_contact_forces,
            contact_mask,
            contact_established,
            live_contact,
            fingertip_distances,
            axis_tilt,
            cond_height,
            cond_horizontal,
            contact_ratio,
            stable,
            cond,
            settling,
        )

        self._candidate_is_valid[:] = cond
        self.reset_buf[~cond] = 1

        self._print_progress()
        return torch.zeros(self.num_envs, device=self.device)

    def _reset_idx(self, env_ids: Sequence[int]):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES

        self._refresh_lab()
        if self._progress_started:
            self._attempt_count += len(env_ids)

        # collect states that survived full episode (successful grasps)
        successful_env_ids = env_ids[
            (self.episode_length_buf[env_ids] >= self.max_episode_length)
            & self._candidate_is_valid[env_ids]
        ]
        n_success = len(successful_env_ids)
        if n_success > 0:
            joint_pos = self._candidate_joint_pos[successful_env_ids].cpu().numpy()
            obj_local = self._candidate_object_pos[successful_env_ids].cpu().numpy()
            obj_quat_wxyz = self._candidate_object_quat[successful_env_ids].cpu().numpy()
            obj_quat_xyzw = np.concatenate([obj_quat_wxyz[:, 1:], obj_quat_wxyz[:, :1]], axis=-1)
            entry = np.concatenate([joint_pos, obj_local, obj_quat_xyzw], axis=-1)
            self._collected.append(entry)

        self._collected_count += n_success
        total = self._collected_count
        if total >= self._target_count:
            self._print_progress(force=True)
            self._save_and_exit()
        self.save_partial()

        # A time-based heartbeat is also emitted when no grasp succeeds, so a
        # slow search can be distinguished from a stalled process.
        self._print_progress()

        # full scene reset
        self.scene.reset(env_ids)
        # Sample exactly once for this candidate.  The base environment applies
        # the resulting world-frame gravity force on every physics substep and
        # does not change the direction again until this environment resets.
        self._sample_gravity_directions(env_ids)

        # reset episode length buffer
        self.episode_length_buf[env_ids] = 0
        self._contact_history[env_ids] = False
        self._contact_window_counts[env_ids] = 0
        self._candidate_is_valid[env_ids] = False

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

        self._refresh_lab()
        self.object_reset_pos[env_ids] = self.object_pos[env_ids]
        self.object_default_pose[env_ids, :3] = self.object_pos[env_ids]
        # Record the exact initial state whose complete episode will be judged.
        # The guard also keeps this override robust if a future DirectRLEnv
        # constructor performs a bootstrap reset.
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
        all_states = self._all_collected_states()
        if os.path.exists(self._cache_path) and not self._force_overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing grasp cache: {self._cache_path}. "
                "Pass --force_overwrite to replace it."
            )
        self._atomic_save(self._cache_path, all_states)
        self._collection_complete = True
        for stale_path in (self._partial_cache_path, self._partial_metadata_path):
            if os.path.exists(stale_path):
                try:
                    os.unlink(stale_path)
                except OSError as error:
                    print(
                        "[WARN] Final cache is complete, but stale checkpoint removal "
                        f"failed for {stale_path}: {error}",
                        flush=True,
                    )
        print(f"\n[INFO] Saved {len(all_states)} grasps -> {self._cache_path}")
        print(f"       shape={all_states.shape}  dtype={all_states.dtype}")
        self.close()
        simulation_app.close()
        sys.exit(0)


env_cfg = Revo3HandHoraEnvCfg()

# Select one hand/object pair.  Cache generation always uses a single cylinder
# radius; the multi-radius object distribution is reserved for training.
_TASK_ROBOT_CFG = {"ball": REVO3_HAND_BALL_CFG, "cylinder": REVO3_HAND_CYLINDER_CFG}
env_cfg.robot_cfg = _TASK_ROBOT_CFG.get(args.task, REVO3_HAND_CYLINDER_CFG)
env_cfg.object_cfg = (
    BALL_OBJECT_CFG
    if args.task == "ball"
    else make_cylinder_object_cfg(
        radius_mm=args.cylinder_radius_mm,
        use_radius_distribution=False,
    )
)
env_cfg.grasp_cache_path = os.path.splitext(cache_path)[0]

if args.usd:
    usd_path = os.path.abspath(args.usd)
    if not os.path.exists(usd_path):
        raise FileNotFoundError(f"--usd path not found: {usd_path}")
    env_cfg.robot_cfg = copy.deepcopy(env_cfg.robot_cfg)
    if env_cfg.robot_cfg.spawn is None or not hasattr(env_cfg.robot_cfg.spawn, "usd_path"):
        raise RuntimeError("env_cfg.robot_cfg.spawn has no usd_path to override.")
    env_cfg.robot_cfg.spawn.usd_path = usd_path

env_cfg.scene.num_envs = args.num_envs
env_cfg.episode_length_s = args.episode_length_s
env_cfg.randomize_cylinder_radius = False
env_cfg.strict_grasp_caches = False
env_cfg.randomize_pd_gains = False
env_cfg.randomize_com = False
env_cfg.randomize_friction = False
env_cfg.randomize_mass = False  # use cfg default 0.10 kg
env_cfg.force_scale = 0.0
env_cfg.random_force_prob_scalar = 0.0
env_cfg.randomize_gravity_direction = args.gravity_mode == "sphere"
env_cfg.gravity_magnitude = 9.81
env_cfg.sim.gravity = (0.0, 0.0, 0.0)
if getattr(args, "device", None):
    env_cfg.sim.device = args.device

if env_cfg.robot_cfg.spawn is None or not hasattr(env_cfg.robot_cfg.spawn, "usd_path"):
    raise RuntimeError("env_cfg.robot_cfg.spawn has no usd_path for checkpoint provenance.")
effective_usd_path = os.path.realpath(os.path.abspath(env_cfg.robot_cfg.spawn.usd_path))
if not os.path.isfile(effective_usd_path):
    raise FileNotFoundError(f"effective hand USD path not found: {effective_usd_path}")
collection_signature = {
    "collector_protocol_version": 1,
    "cache_format": "joint21_object_xyz3_quat_xyzw4",
    "task": args.task,
    "cylinder_radius_mm": args.cylinder_radius_mm if args.task == "cylinder" else None,
    "hand_usd": {
        "path": effective_usd_path,
        "sha256": _sha256_file(effective_usd_path),
    },
    "noise_scale_rad": args.noise_scale,
    "episode_length_s": args.episode_length_s,
    "episode_steps": collection_max_episode_steps,
    "settle_steps": args.settle_steps,
    "fingertip_near_threshold_m": GraspGenEnv.FINGERTIP_NEAR_THRESHOLD,
    "contact_force_threshold_n": args.contact_force_threshold,
    "min_contact_fingertips": args.min_contact_fingertips,
    "contact_window_steps": args.contact_window_steps,
    "min_contact_ratio": args.min_contact_ratio,
    "min_live_contact_fingertips": args.min_live_contact_fingertips,
    "max_axis_tilt_deg": args.max_axis_tilt_deg,
    "max_horizontal_drift_m": args.max_horizontal_drift_m,
    "max_height_drift_m": args.max_height_drift_m,
    "gravity_mode": args.gravity_mode,
    "gravity_magnitude_m_s2": env_cfg.gravity_magnitude,
    "randomize_mass": env_cfg.randomize_mass,
    "randomize_friction": env_cfg.randomize_friction,
    "random_force_scale": env_cfg.force_scale,
}

env = GraspGenEnv(
    env_cfg,
    render_mode=None,
    noise_scale=args.noise_scale,
    target_count=args.target_count,
    progress_interval=args.progress_interval,
    checkpoint_interval=args.checkpoint_interval,
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
    force_overwrite=args.force_overwrite,
    collection_signature=collection_signature,
)
env.reset()
env.start_progress_tracking()

print("\n[INFO] Grasp cache generation started.")
print(f"  task        : {args.task}")
if args.task == "cylinder":
    print(f"  radius      : {args.cylinder_radius_mm}mm")
print(f"  num_envs    : {args.num_envs}")
print(f"  noise_scale : ±{args.noise_scale} rad")
print(f"  episode_len : {env_cfg.episode_length_s}s  ({env.max_episode_length} steps)")
print(f"  target      : {args.target_count} grasps")
print(f"  progress    : every {args.progress_interval:g}s")
print(f"  checkpoint  : every {args.checkpoint_interval:g}s -> {env._partial_cache_path}")
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
print(f"  gravity     : {args.gravity_mode} per env/episode (9.81m/s², scene gravity=0)")
print("  disturbance : random force off, friction randomization off")
print(f"  output      : {cache_path}\n")
if args.usd:
    print(f"[INFO] hand usd   : {os.path.abspath(args.usd)}")

zero_actions = torch.zeros((args.num_envs, env_cfg.action_space), device=env.device)

try:
    while simulation_app.is_running():
        with torch.inference_mode():
            env.step(zero_actions)
except KeyboardInterrupt:
    print("\n[INFO] Collection interrupted; saving the latest partial cache.", flush=True)
    raise
except BaseException:
    # SimulationApp.close may terminate the interpreter on some Kit builds;
    # emit the original failure before cleanup so it cannot be hidden.
    traceback.print_exc()
    raise
finally:
    if not env.collection_complete:
        try:
            env._print_progress(force=True)
            env.save_partial(force=True)
        finally:
            env.close()
            simulation_app.close()

if not env.collection_complete:
    print(
        f"[ERROR] Simulation stopped before reaching target: "
        f"{env._collected_count}/{args.target_count} grasps. Partial progress was saved.",
        file=sys.stderr,
        flush=True,
    )
    sys.exit(2)

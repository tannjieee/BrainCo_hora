"""Pure-Python registry shared by CLI parsing, grasp collection, and training."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OBJECT_ASSET_ROOT = PROJECT_ROOT / "assets/usd/objects"
OBJECT_MANIFEST_PATH = OBJECT_ASSET_ROOT / "manifest.json"


@dataclass(frozen=True)
class ObjectTaskSpec:
    name: str
    display_name: str
    kind: str
    cache_stem: str
    hand_pose: str
    hand_joint_pos_rad: tuple[tuple[str, float], ...]
    object_init_pos_m: tuple[float, float, float]
    object_init_quat_wxyz: tuple[float, float, float, float]
    rotation_axis_local: tuple[float, float, float]
    target_axis_world: tuple[float, float, float]
    axis_bidirectional: bool
    enforce_axis_alignment: bool
    axis_tilt_tolerance_deg: float
    fingertip_near_threshold_m: float
    scale: float = 1.0
    source_size_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scaled_size_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    usd_path: str | None = None

    def metadata(self) -> dict[str, object]:
        return {
            "task": self.name,
            "display_name": self.display_name,
            "kind": self.kind,
            "usd_path": self.usd_path,
            "scale": self.scale,
            "source_size_m": self.source_size_m,
            "scaled_size_m": self.scaled_size_m,
            "hand_pose": self.hand_pose,
            "hand_joint_pos_rad": dict(self.hand_joint_pos_rad),
            "object_init_pos_m": self.object_init_pos_m,
            "object_init_quat_wxyz": self.object_init_quat_wxyz,
            "rotation_axis_local": self.rotation_axis_local,
            "target_axis_world": self.target_axis_world,
            "axis_bidirectional": self.axis_bidirectional,
            "enforce_axis_alignment": self.enforce_axis_alignment,
            "axis_tilt_tolerance_deg": self.axis_tilt_tolerance_deg,
            "grasp_cache_path": self.cache_stem,
        }


def _vector(values, length: int, field: str) -> tuple[float, ...]:
    if len(values) != length:
        raise ValueError(f"{field} must contain {length} values, got {values!r}")
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{field} must contain only finite values, got {values!r}")
    return result


def _unit_vector(values, field: str) -> tuple[float, float, float]:
    vector = _vector(values, 3, field)
    norm = math.sqrt(sum(value * value for value in vector))
    if norm < 1e-8:
        raise ValueError(f"{field} must be non-zero")
    return tuple(value / norm for value in vector)


def _unit_quaternion(values, field: str) -> tuple[float, float, float, float]:
    quaternion = _vector(values, 4, field)
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm < 1e-8:
        raise ValueError(f"{field} must be non-zero")
    return tuple(value / norm for value in quaternion)


def _load_scanned_objects() -> dict[str, ObjectTaskSpec]:
    with OBJECT_MANIFEST_PATH.open("r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    specs: dict[str, ObjectTaskSpec] = {}
    for name, item in manifest.items():
        usd_path = (OBJECT_ASSET_ROOT / item["usd"]).resolve()
        grasp_seed = item["grasp_seed"]
        rotation = item["rotation"]
        hand_pose = str(grasp_seed["hand_pose_profile"])
        if hand_pose not in {"ball", "cylinder", "custom"}:
            raise ValueError(
                f"{name}.grasp_seed.hand_pose_profile must be 'ball', 'cylinder', or 'custom'"
            )
        axis_tilt_tolerance_deg = float(rotation["tilt_tolerance_deg"])
        if not 0.0 < axis_tilt_tolerance_deg <= 180.0:
            raise ValueError(
                f"{name}.rotation.tilt_tolerance_deg must be in (0, 180]"
            )
        hand_joint_pos_rad = tuple(
            (str(joint), float(value))
            for joint, value in grasp_seed["hand_joint_pos_rad"].items()
        )
        if not all(math.isfinite(value) for _, value in hand_joint_pos_rad):
            raise ValueError(f"{name}.grasp_seed.hand_joint_pos_rad contains non-finite values")
        if hand_pose == "custom" and len(hand_joint_pos_rad) != 21:
            raise ValueError(
                f"{name}.grasp_seed.hand_joint_pos_rad must contain all 21 joints "
                "when hand_pose_profile is 'custom'"
            )
        scale = float(item["scale"])
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"{name}.scale must be a finite value greater than zero")
        source_size_m = _vector(item["source_size_m"], 3, f"{name}.source_size_m")
        specs[name] = ObjectTaskSpec(
            name=name,
            display_name=item["display_name"],
            kind="usd",
            usd_path=str(usd_path),
            scale=scale,
            source_size_m=source_size_m,
            scaled_size_m=tuple(value * scale for value in source_size_m),
            cache_stem=item["cache_stem"],
            hand_pose=hand_pose,
            hand_joint_pos_rad=hand_joint_pos_rad,
            object_init_pos_m=_vector(
                grasp_seed["object_pos_m"], 3, f"{name}.grasp_seed.object_pos_m"
            ),
            object_init_quat_wxyz=_unit_quaternion(
                grasp_seed["object_quat_wxyz"],
                f"{name}.grasp_seed.object_quat_wxyz",
            ),
            rotation_axis_local=_unit_vector(
                rotation["local_axis"], f"{name}.rotation.local_axis"
            ),
            target_axis_world=_unit_vector(
                rotation["target_axis_world"], f"{name}.rotation.target_axis_world"
            ),
            axis_bidirectional=bool(rotation["axis_bidirectional"]),
            enforce_axis_alignment=bool(rotation["enforce_axis_alignment"]),
            axis_tilt_tolerance_deg=axis_tilt_tolerance_deg,
            fingertip_near_threshold_m=float(item["fingertip_near_threshold_m"]),
        )
    return specs


OBJECT_TASK_SPECS: dict[str, ObjectTaskSpec] = {
    "ball": ObjectTaskSpec(
        name="ball",
        display_name="30 mm radius sphere",
        kind="sphere",
        cache_stem="cache/revo3_right_grasp_ball",
        hand_pose="ball",
        hand_joint_pos_rad=(),
        object_init_pos_m=(0.0, -0.08, 1.65),
        object_init_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        rotation_axis_local=(0.0, 0.0, 1.0),
        target_axis_world=(0.0, 0.0, 1.0),
        axis_bidirectional=True,
        enforce_axis_alignment=False,
        axis_tilt_tolerance_deg=10.0,
        fingertip_near_threshold_m=0.10,
        scaled_size_m=(0.06, 0.06, 0.06),
        source_size_m=(0.06, 0.06, 0.06),
    ),
    "cylinder": ObjectTaskSpec(
        name="cylinder",
        display_name="30 mm radius x 70 mm cylinder",
        kind="cylinder",
        cache_stem="cache/revo3_right_grasp_cylinder",
        hand_pose="cylinder",
        hand_joint_pos_rad=(),
        object_init_pos_m=(0.0, -0.08, 1.635),
        object_init_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        rotation_axis_local=(0.0, 0.0, 1.0),
        target_axis_world=(0.0, 0.0, 1.0),
        axis_bidirectional=True,
        enforce_axis_alignment=True,
        axis_tilt_tolerance_deg=10.0,
        fingertip_near_threshold_m=0.10,
        scaled_size_m=(0.06, 0.06, 0.07),
        source_size_m=(0.06, 0.06, 0.07),
    ),
    **_load_scanned_objects(),
}
OBJECT_TASK_NAMES = tuple(OBJECT_TASK_SPECS)
SCANNED_OBJECT_TASK_NAMES = tuple(
    name for name, spec in OBJECT_TASK_SPECS.items() if spec.kind == "usd"
)


def get_object_task_spec(name: str) -> ObjectTaskSpec:
    try:
        return OBJECT_TASK_SPECS[name]
    except KeyError as error:
        raise ValueError(
            f"Unknown object task {name!r}; choose one of: {', '.join(OBJECT_TASK_NAMES)}"
        ) from error

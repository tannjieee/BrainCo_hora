from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


DEFAULT_PRESSURE_TIP_SLICES = ((1, 4), (10, 13), (18, 21), (26, 29), (34, 37))
DEFAULT_MATRIX_TIP_MODULE_IDS = (1, 3, 5, 7, 9)


@dataclass(frozen=True)
class FingertipForceAdapter:
    """Convert Revo3 tactile SDK values to [thumb,index,middle,ring,little] N."""

    pressure_tip_slices: tuple[tuple[int, int], ...]
    matrix_tip_module_ids: tuple[int, ...]
    pressure_scale_to_n: float
    matrix_scale_to_n: float
    gain: np.ndarray
    bias_n: np.ndarray
    clip_min_n: float
    clip_max_n: float

    @classmethod
    def from_profile(cls, cfg: dict | None = None) -> "FingertipForceAdapter":
        cfg = cfg or {}
        pressure_cfg = dict(cfg.get("pressure") or {})
        matrix_cfg = dict(cfg.get("matrix") or {})
        calibration_cfg = dict(cfg.get("calibration") or {})

        pressure_slices = tuple(
            (int(value[0]), int(value[1]))
            for value in pressure_cfg.get("summary_tip_slices", DEFAULT_PRESSURE_TIP_SLICES)
        )
        matrix_ids = tuple(
            int(value)
            for value in matrix_cfg.get("tip_module_ids", DEFAULT_MATRIX_TIP_MODULE_IDS)
        )
        if len(pressure_slices) != 5 or any(end <= start for start, end in pressure_slices):
            raise ValueError("Pressure summary must define five non-empty fingertip slices.")
        if len(matrix_ids) != 5 or len(set(matrix_ids)) != 5:
            raise ValueError("Matrix tactile config must define five unique fingertip modules.")

        gain = _five_vector(calibration_cfg.get("gain", [1.0] * 5), "tactile gain")
        bias = _five_vector(calibration_cfg.get("bias_n", [0.0] * 5), "tactile bias_n")
        clip = calibration_cfg.get("clip_n", [0.0, 100.0])
        if not isinstance(clip, list) or len(clip) != 2:
            raise ValueError("tactile calibration.clip_n must be [min, max].")
        clip_values = np.asarray(clip, dtype=np.float32)
        if not np.isfinite(clip_values).all() or clip_values[1] <= clip_values[0]:
            raise ValueError("tactile calibration.clip_n must be finite with max > min.")
        pressure_scale = float(pressure_cfg.get("unit_scale_to_n", 0.001))
        matrix_scale = float(matrix_cfg.get("unit_scale_to_n", 0.0001))
        if (
            not np.isfinite((pressure_scale, matrix_scale)).all()
            or pressure_scale <= 0.0
            or matrix_scale <= 0.0
        ):
            raise ValueError("Tactile unit scales must be finite and positive.")
        if np.any(gain <= 0.0):
            raise ValueError("Tactile calibration gains must be positive.")
        return cls(
            pressure_tip_slices=pressure_slices,
            matrix_tip_module_ids=matrix_ids,
            pressure_scale_to_n=pressure_scale,
            matrix_scale_to_n=matrix_scale,
            gain=gain,
            bias_n=bias,
            clip_min_n=float(clip_values[0]),
            clip_max_n=float(clip_values[1]),
        )

    def from_pressure_summary(self, summary: Sequence[float]) -> np.ndarray:
        values = np.asarray(summary, dtype=np.float32).reshape(-1)
        required = max(end for _, end in self.pressure_tip_slices)
        if values.size < required or not np.isfinite(values).all():
            raise ValueError(f"Pressure summary must contain at least {required} finite values.")
        forces = np.asarray(
            [np.maximum(values[start:end], 0.0).sum() for start, end in self.pressure_tip_slices],
            dtype=np.float32,
        )
        return self._calibrate(forces * self.pressure_scale_to_n)

    def from_matrix_modules(
        self,
        modules: Mapping[int, Sequence[float]] | Sequence[Sequence[float]],
    ) -> np.ndarray:
        def module_values(module_id: int) -> Sequence[float]:
            return modules[module_id]

        totals: list[float] = []
        for module_id in self.matrix_tip_module_ids:
            values = np.asarray(module_values(module_id), dtype=np.float32).reshape(-1)
            if values.size == 0 or not np.isfinite(values).all():
                raise ValueError(f"Matrix tactile module {module_id} returned invalid data.")
            totals.append(float(np.maximum(values, 0.0).sum()))
        return self._calibrate(np.asarray(totals, dtype=np.float32) * self.matrix_scale_to_n)

    def from_force_vector(self, forces_n: Sequence[float]) -> np.ndarray:
        """Apply the shared five-finger calibration to values already in newtons."""
        values = _five_vector(forces_n, "fingertip forces")
        if np.any(values < 0.0):
            raise ValueError("Fingertip force magnitudes must be non-negative.")
        return self._calibrate(values)

    def _calibrate(self, forces_n: np.ndarray) -> np.ndarray:
        calibrated = forces_n * self.gain + self.bias_n
        return np.clip(calibrated, self.clip_min_n, self.clip_max_n).astype(np.float32)


def _five_vector(value: Sequence[float], name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.shape != (5,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain five finite values.")
    return vector

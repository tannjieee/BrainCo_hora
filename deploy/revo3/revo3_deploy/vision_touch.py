from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np


class VisionTouchCollector:
    """Background five-sensor VisionTouch Force6D collector.

    The pyvitaisdk calls are synchronous and each sensor owns a separate USB
    stream, so collection is performed concurrently in a bounded thread pool.
    Public values are fingertip force norms in newtons, in the configured
    thumb/index/middle/ring/little order.
    """

    EXPECTED_ORDER = ("thumb", "index", "middle", "ring", "little")

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = dict(config or {})
        self.model_dir = Path(str(self.config.get("model_dir", ""))).expanduser()
        if not self.model_dir.is_absolute():
            self.model_dir = (Path.cwd() / self.model_dir).resolve()
        sensor_order = tuple(str(value).strip() for value in self.config.get("sensor_order", ()))
        if len(sensor_order) != 5 or any(not value for value in sensor_order):
            raise ValueError(
                "vision_touch.sensor_order must contain five non-empty sensor serial numbers."
            )
        if len(set(sensor_order)) != 5:
            raise ValueError("vision_touch.sensor_order must contain unique serial numbers.")
        self.sensor_order = sensor_order
        self.calibrate_on_open = bool(self.config.get("calibrate_on_open", True))
        self.mapping_verified = bool(self.config.get("mapping_verified", False))
        self.sample_period_s = _positive(self.config, "sample_period_s", 0.02)
        self.sample_timeout_s = _positive(self.config, "sample_timeout_s", 0.045)
        self.max_sample_age_s = _positive(self.config, "max_sample_age_s", 0.045)
        workers = int(self.config.get("workers", 5))
        if workers != 5:
            raise ValueError("vision_touch.workers must be exactly 5 for the five sensors.")
        self._workers = workers
        self._sensors: list[tuple[str, Any]] = []
        self._executor: ThreadPoolExecutor | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._first_sample = threading.Event()
        self._lock = threading.Lock()
        self._latest_forces = np.zeros(5, dtype=np.float32)
        self._latest_timestamp = 0.0
        self._last_error: BaseException | None = None
        self._closed = False

    def open(self) -> None:
        if self._thread is not None:
            raise RuntimeError("VisionTouchCollector is already open.")
        self._stop.clear()
        self._first_sample.clear()
        with self._lock:
            self._latest_forces = np.zeros(5, dtype=np.float32)
            self._latest_timestamp = 0.0
            self._last_error = None
        try:
            from pyvitaisdk import VTSDataType, VTSDeviceFinder, VTSensor
        except ImportError as exc:
            raise RuntimeError(
                "pyvitaisdk is required for Revo3UltraVisionTouch. Install the BrainCo "
                "pyvitaisdk4bc wheel in the revo3 environment."
            ) from exc

        finder = VTSDeviceFinder()
        available = tuple(str(value) for value in finder.get_sns())
        missing = [serial for serial in self.sensor_order if serial not in available]
        if missing:
            raise RuntimeError(
                f"VisionTouch sensors are missing: {missing}; available sensors: {available}."
            )
        if not self.model_dir.is_dir():
            raise RuntimeError(f"VisionTouch model directory does not exist: {self.model_dir}")

        created: list[tuple[str, Any]] = []
        try:
            for serial in self.sensor_order:
                model_path = self.model_dir / serial / f"{serial}.onnx.enc"
                if not model_path.is_file():
                    raise RuntimeError(f"Missing VisionTouch force model: {model_path}")
                device_config = finder.get_device_by_sn(serial)
                sensor = VTSensor(config=device_config, force_model_path=str(model_path))
                if self.calibrate_on_open:
                    sensor.calibrate()
                sensor_type = str(getattr(getattr(sensor, "sensor_type", None), "value", ""))
                if serial == self.sensor_order[0] and sensor_type not in {"GFBCT", "GF515T"}:
                    raise RuntimeError(
                        f"Expected a thumb VisionTouch sensor first, got {sensor_type!r} "
                        f"for {serial}."
                    )
                if serial != self.sensor_order[0] and sensor_type in {"GFBCT", "GF515T"}:
                    raise RuntimeError(
                        f"Thumb VisionTouch sensor {serial} is not first in sensor_order."
                    )
                if serial != self.sensor_order[0] and sensor_type not in {"GFBCI", "GF515I"}:
                    raise RuntimeError(
                        f"Expected a non-thumb VisionTouch sensor, got {sensor_type!r} "
                        f"for {serial}."
                    )
                created.append((serial, sensor))
            self._sensors = created
            self._executor = ThreadPoolExecutor(
                max_workers=self._workers,
                thread_name_prefix="revo3-vision-touch",
            )
            self._closed = False
            self._thread = threading.Thread(
                target=self._collect_loop,
                args=(VTSDataType.FORCE6D_VECTOR,),
                name="revo3-vision-touch-collector",
                daemon=True,
            )
            self._thread.start()
            if not self._first_sample.wait(timeout=self.sample_timeout_s * 2.0):
                detail = f": {self._last_error!r}" if self._last_error else ""
                raise RuntimeError(f"VisionTouch produced no fresh force sample{detail}.")
        except BaseException:
            if self._sensors:
                self.close()
            else:
                for _, sensor in created:
                    try:
                        sensor.release()
                    except Exception:
                        pass
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(1.0, self.sample_timeout_s * 4.0))
        self._thread = None
        executor = self._executor
        self._executor = None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        sensors = self._sensors
        self._sensors = []
        for _, sensor in sensors:
            try:
                sensor.release()
            except Exception:
                pass

    def read_latest(self, enforce_freshness: bool = True) -> tuple[np.ndarray, float]:
        with self._lock:
            timestamp = self._latest_timestamp
            forces = self._latest_forces.copy()
            error = self._last_error
        if timestamp <= 0.0:
            detail = f": {error!r}" if error else ""
            raise RuntimeError(f"VisionTouch has no force sample{detail}.")
        age = max(0.0, time.monotonic() - timestamp)
        if enforce_freshness and age > self.max_sample_age_s:
            detail = f"; last error={error!r}" if error else ""
            raise RuntimeError(
                f"VisionTouch force sample is stale ({age * 1000.0:.1f} ms > "
                f"{self.max_sample_age_s * 1000.0:.1f} ms){detail}."
            )
        if not np.isfinite(forces).all() or np.any(forces < 0.0):
            raise RuntimeError("VisionTouch force sample is non-finite or negative.")
        return forces, age

    @property
    def sensor_serials(self) -> tuple[str, ...]:
        return self.sensor_order

    @property
    def last_error(self) -> BaseException | None:
        with self._lock:
            return self._last_error

    def _collect_loop(self, data_type: Any) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            executor = self._executor
            if executor is None:
                return
            futures = [
                executor.submit(self._read_sensor_force, sensor, data_type)
                for _, sensor in self._sensors
            ]
            try:
                values = [future.result(timeout=self.sample_timeout_s) for future in futures]
                forces = np.asarray(
                    [float(np.linalg.norm(value[:3])) for value in values], dtype=np.float32
                )
                if forces.shape != (5,) or not np.isfinite(forces).all():
                    raise RuntimeError("VisionTouch force output has an invalid shape or value.")
                with self._lock:
                    self._latest_forces = forces
                    self._latest_timestamp = time.monotonic()
                    self._last_error = None
                self._first_sample.set()
            except BaseException as exc:
                with self._lock:
                    self._last_error = exc
                for future in futures:
                    future.cancel()
            remaining = self.sample_period_s - (time.monotonic() - started)
            if remaining > 0.0:
                self._stop.wait(remaining)

    @staticmethod
    def _read_sensor_force(sensor: Any, data_type: Any) -> np.ndarray:
        data = sensor.collect_sensor_data(data_type)
        value = np.asarray(data[data_type], dtype=np.float32)
        if value.ndim == 1:
            if value.size < 3:
                raise RuntimeError("VisionTouch Force6D output has fewer than three values.")
            return value
        if value.shape[-1] < 3:
            raise RuntimeError("VisionTouch Force6D output has fewer than three channels.")
        return value[..., :6].reshape(-1, value.shape[-1]).mean(axis=0)


def _positive(config: dict[str, Any], key: str, default: float) -> float:
    value = float(config.get(key, default))
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"vision_touch.{key} must be finite and positive.")
    return value

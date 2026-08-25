"""Validated acquisition and projection contracts for dual-energy AM CT.

This module intentionally contains no reconstruction code. It defines the
serializable physical inputs and provenance boundary shared by simulation,
measured-data ingestion, calibration, tests, and the application UI.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AMSpectrum:
    """Discrete normalized X-ray spectrum."""

    energies_kev: tuple[float, ...]
    weights: tuple[float, ...]
    label: str

    def __post_init__(self) -> None:
        energies = tuple(float(value) for value in self.energies_kev)
        weights = tuple(float(value) for value in self.weights)
        if len(energies) < 2 or len(energies) != len(weights):
            raise ValueError("spectrum energies and weights must have the same length >= 2")
        if not np.isfinite(energies).all() or not np.isfinite(weights).all():
            raise ValueError("spectrum energies and weights must be finite")
        if any(right <= left for left, right in zip(energies, energies[1:])):
            raise ValueError("spectrum energies must be strictly increasing")
        if energies[0] <= 0:
            raise ValueError("spectrum energies must be positive")
        if any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError("spectrum weights must have positive total and be nonnegative")
        total = float(sum(weights))
        object.__setattr__(self, "energies_kev", energies)
        object.__setattr__(self, "weights", tuple(weight / total for weight in weights))
        object.__setattr__(self, "label", str(self.label).strip() or "unnamed")

    @classmethod
    def triangular(
        cls, minimum_kev: float, maximum_kev: float, *, bins: int = 200, label: str
    ) -> "AMSpectrum":
        if bins < 2 or maximum_kev <= minimum_kev:
            raise ValueError("triangular spectrum requires bins >= 2 and max > min")
        energies = np.linspace(minimum_kev, maximum_kev, bins)
        weights = np.maximum(0.0, maximum_kev - energies)
        return cls(tuple(energies), tuple(weights), label)

    def to_dict(self) -> dict[str, Any]:
        return {
            "energies_kev": list(self.energies_kev),
            "weights": list(self.weights),
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AMSpectrum":
        return cls(
            tuple(payload["energies_kev"]),
            tuple(payload["weights"]),
            payload.get("label", "unnamed"),
        )


@dataclass(frozen=True)
class AMAcquisitionConfig:
    """Serializable acquisition settings required to interpret AM data."""

    low_spectrum: AMSpectrum
    high_spectrum: AMSpectrum
    num_angles: int
    pixel_size_cm: float = 0.01
    angle_start_deg: float = 0.0
    angle_stop_deg: float = 180.0
    photon_count: int | None = None
    random_seed: int | None = None
    calibration_id: str | None = None
    source_kind: str = "simulated"
    electron_density_unit: str = "relative_to_water"
    lut_a1_max: float = 1500.0
    lut_a2_max: float = 30.0
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.num_angles, (int, np.integer)) or self.num_angles < 2:
            raise ValueError("num_angles must be an integer >= 2")
        if not np.isfinite(self.pixel_size_cm) or self.pixel_size_cm <= 0:
            raise ValueError("pixel_size_cm must be positive")
        if self.angle_stop_deg <= self.angle_start_deg:
            raise ValueError("angle_stop_deg must be greater than angle_start_deg")
        if self.source_kind not in {"simulated", "measured"}:
            raise ValueError("source_kind must be 'simulated' or 'measured'")
        if self.electron_density_unit != "relative_to_water":
            raise ValueError("electron_density_unit must be 'relative_to_water'")
        if self.photon_count is not None:
            if not isinstance(self.photon_count, (int, np.integer)) or self.photon_count <= 0:
                raise ValueError("photon_count must be a positive integer")
            if self.source_kind == "simulated" and self.random_seed is None:
                raise ValueError("random_seed is required for simulated photon noise")
        elif self.random_seed is not None:
            raise ValueError("random_seed requires photon_count")
        if self.source_kind == "measured" and self.random_seed is not None:
            raise ValueError("measured acquisition cannot declare a random_seed")
        if self.lut_a1_max <= 0 or self.lut_a2_max <= 0:
            raise ValueError("LUT coefficient bounds must be positive")
        if self.schema_version != 1:
            raise ValueError("unsupported acquisition schema_version")

    @classmethod
    def industrial_default(
        cls,
        *,
        num_angles: int,
        photon_count: int | None = None,
        random_seed: int | None = None,
        calibration_id: str | None = None,
    ) -> "AMAcquisitionConfig":
        return cls(
            low_spectrum=AMSpectrum.triangular(
                60.0, 160.0, label="industrial-low-160kVp"
            ),
            high_spectrum=AMSpectrum.triangular(
                60.0, 320.0, label="industrial-high-320kVp"
            ),
            num_angles=num_angles,
            photon_count=photon_count,
            random_seed=random_seed,
            calibration_id=calibration_id,
        )

    @property
    def theta_deg(self) -> np.ndarray:
        return np.linspace(
            self.angle_start_deg,
            self.angle_stop_deg,
            self.num_angles,
            endpoint=False,
            dtype=np.float64,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "low_spectrum": self.low_spectrum.to_dict(),
            "high_spectrum": self.high_spectrum.to_dict(),
            "num_angles": int(self.num_angles),
            "pixel_size_cm": float(self.pixel_size_cm),
            "angle_start_deg": float(self.angle_start_deg),
            "angle_stop_deg": float(self.angle_stop_deg),
            "photon_count": self.photon_count,
            "random_seed": self.random_seed,
            "calibration_id": self.calibration_id,
            "source_kind": self.source_kind,
            "electron_density_unit": self.electron_density_unit,
            "lut_a1_max": float(self.lut_a1_max),
            "lut_a2_max": float(self.lut_a2_max),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AMAcquisitionConfig":
        data = dict(payload)
        data["low_spectrum"] = AMSpectrum.from_dict(data["low_spectrum"])
        data["high_spectrum"] = AMSpectrum.from_dict(data["high_spectrum"])
        return cls(**data)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "AMAcquisitionConfig":
        return cls.from_dict(json.loads(payload))


@dataclass(frozen=True)
class PairedProjectionData:
    """Validated low/high negative-log projections and acquisition metadata.

    Arrays may represent one slice ``(detector, angle)`` or a volume
    ``(slice, detector, angle)``. The last axis always corresponds to theta.
    """

    low_log: np.ndarray
    high_log: np.ndarray
    theta_deg: np.ndarray
    config: AMAcquisitionConfig
    a1_max_bound: float | None = None
    a2_max_bound: float | None = None

    def __post_init__(self) -> None:
        low = np.asarray(self.low_log, dtype=np.float32)
        high = np.asarray(self.high_log, dtype=np.float32)
        theta = np.asarray(self.theta_deg, dtype=np.float64)
        if low.shape != high.shape:
            raise ValueError("low/high projections must have matching shapes")
        if low.ndim not in (2, 3):
            raise ValueError("projections must have shape (detector, angle) or (slice, detector, angle)")
        if theta.ndim != 1 or low.shape[-1] != theta.size:
            raise ValueError("projection angle axis must match theta_deg")
        if theta.size != self.config.num_angles:
            raise ValueError("projection angle axis must match config.num_angles")
        if not np.isfinite(low).all() or not np.isfinite(high).all() or not np.isfinite(theta).all():
            raise ValueError("projection data and angles must be finite")
        if np.any(low < 0) or np.any(high < 0):
            raise ValueError("negative-log projection data must be nonnegative")
        if np.any(np.diff(theta) <= 0):
            raise ValueError("projection angles must be strictly increasing")
        a1_bound = self.config.lut_a1_max if self.a1_max_bound is None else float(self.a1_max_bound)
        a2_bound = self.config.lut_a2_max if self.a2_max_bound is None else float(self.a2_max_bound)
        if a1_bound <= 0 or a2_bound <= 0:
            raise ValueError("projection LUT bounds must be positive")
        object.__setattr__(self, "low_log", low)
        object.__setattr__(self, "high_log", high)
        object.__setattr__(self, "theta_deg", theta)
        object.__setattr__(self, "a1_max_bound", a1_bound)
        object.__setattr__(self, "a2_max_bound", a2_bound)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.low_log.shape

    @property
    def source_kind(self) -> str:
        return self.config.source_kind

    @property
    def is_measured(self) -> bool:
        return self.config.source_kind == "measured"

    @property
    def low_log_zhw(self) -> np.ndarray:
        return self.low_log[None, ...] if self.low_log.ndim == 2 else self.low_log

    @property
    def high_log_zhw(self) -> np.ndarray:
        return self.high_log[None, ...] if self.high_log.ndim == 2 else self.high_log


def load_paired_projection_files(
    low_path: str | Path,
    high_path: str | Path,
    theta_path: str | Path,
    config: AMAcquisitionConfig,
) -> PairedProjectionData:
    """Load measured ``.npy`` negative-log projections with strict metadata.

    File loading is deliberately narrow: callers must convert proprietary raw
    detector formats upstream and explicitly provide a measured acquisition
    configuration rather than relying on filename or scanner heuristics.
    """
    if config.source_kind != "measured":
        raise ValueError("projection file import requires source_kind='measured'")
    paths = tuple(Path(path) for path in (low_path, high_path, theta_path))
    if any(path.suffix.lower() != ".npy" for path in paths):
        raise ValueError("paired projection import currently requires .npy files")
    try:
        low, high, theta = (
            np.load(path, allow_pickle=False) for path in paths
        )
    except (OSError, ValueError) as error:
        raise ValueError(f"could not load paired projection data: {error}") from error
    return PairedProjectionData(low, high, theta, config)

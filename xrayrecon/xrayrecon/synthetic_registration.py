"""Deterministic synthetic datasets for registration demos and regression tests.

The generators live in the package rather than under ``tests`` so the desktop
application never imports test code. Each case carries its known physical-space
transform and identifiability expectations, making it useful as an executable
registration contract as well as a demo source.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .registration import RigidScaleTransform, resample_labels, resample_volume


@dataclass(frozen=True)
class SyntheticRegistrationSpec:
    name: str
    transform: RigidScaleTransform = field(default_factory=RigidScaleTransform.identity)
    moving_shape: tuple[int, int, int] = (40, 44, 12)
    fixed_shape: tuple[int, int, int] | None = None
    moving_spacing: tuple[float, float, float] = (0.15, 0.15, 0.10)
    fixed_spacing: tuple[float, float, float] = (0.15, 0.15, 0.10)
    noise_std: float = 0.0
    intensity_gamma: float = 1.0
    crop_fraction: float = 0.0
    geometry: str = "asymmetric"
    identifiable: bool = True
    minimum_score_gain: float = 0.03
    seed: int = 1729


@dataclass(frozen=True)
class SyntheticRegistrationCase:
    spec: SyntheticRegistrationSpec
    moving_hwz: np.ndarray
    fixed_hwz: np.ndarray
    moving_labels_hwz: np.ndarray
    fixed_labels_hwz: np.ndarray

    @property
    def known_transform(self) -> RigidScaleTransform:
        return self.spec.transform

    @property
    def moving_spacing(self) -> tuple[float, float, float]:
        return self.spec.moving_spacing

    @property
    def fixed_spacing(self) -> tuple[float, float, float]:
        return self.spec.fixed_spacing


def _asymmetric_phantom(shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    axes = np.meshgrid(
        *[np.linspace(-1.0, 1.0, size, dtype=np.float32) for size in shape],
        indexing="ij",
    )
    x, y, z = axes
    volume = np.zeros(shape, dtype=np.float32)
    blobs = (
        (0.95, (-0.42, -0.30, -0.25), (0.18, 0.13, 0.22)),
        (0.72, (0.36, -0.12, 0.28), (0.12, 0.22, 0.16)),
        (0.55, (-0.08, 0.46, 0.05), (0.25, 0.10, 0.28)),
    )
    for amplitude, center, widths in blobs:
        exponent = sum(
            ((axis - axis_center) / width) ** 2
            for axis, axis_center, width in zip(axes, center, widths)
        )
        volume += amplitude * np.exp(-0.5 * exponent)

    # Asymmetric trace and via features break rotational/translation symmetry.
    trace = (x > -0.75) & (x < 0.15) & (y > 0.05) & (y < 0.16) & (abs(z) < 0.45)
    via = ((x - 0.57) ** 2 + (y - 0.42) ** 2 < 0.065**2) & (z > -0.65)
    corner = (x < -0.55) & (y > 0.45) & (z < -0.05)
    volume[trace] += 0.70
    volume[via] += 0.90
    volume[corner] += 0.45
    volume = np.clip(volume, 0.0, 1.0)

    labels = np.zeros(shape, dtype=np.int16)
    labels[volume > 0.12] = 1
    labels[volume > 0.42] = 2
    labels[volume > 0.72] = 3
    return volume, labels


def _symmetric_phantom(shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    axes = np.meshgrid(
        *[np.linspace(-1.0, 1.0, size, dtype=np.float32) for size in shape],
        indexing="ij",
    )
    radius_xy = np.sqrt(axes[0] ** 2 + axes[1] ** 2)
    volume = np.exp(-((radius_xy - 0.48) ** 2) / 0.012) * np.exp(
        -(axes[2] ** 2) / 0.35
    )
    volume = volume.astype(np.float32)
    labels = (volume > 0.25).astype(np.int16)
    return volume, labels


def make_synthetic_registration_case(
    spec: SyntheticRegistrationSpec,
) -> SyntheticRegistrationCase:
    """Build one reproducible moving/fixed pair from ``spec``."""
    if spec.geometry == "asymmetric":
        moving, moving_labels = _asymmetric_phantom(spec.moving_shape)
    elif spec.geometry == "symmetric":
        moving, moving_labels = _symmetric_phantom(spec.moving_shape)
    elif spec.geometry == "constant":
        moving = np.ones(spec.moving_shape, dtype=np.float32)
        moving_labels = np.ones(spec.moving_shape, dtype=np.int16)
    else:
        raise ValueError(f"unknown synthetic geometry: {spec.geometry}")

    fixed_shape = spec.fixed_shape or spec.moving_shape
    fixed = resample_volume(
        moving,
        spec.moving_spacing,
        fixed_shape,
        spec.fixed_spacing,
        spec.transform,
        order=1,
    ).astype(np.float32)
    fixed_labels = resample_labels(
        moving_labels,
        spec.moving_spacing,
        fixed_shape,
        spec.fixed_spacing,
        spec.transform,
    )

    if spec.intensity_gamma != 1.0:
        fixed = np.power(np.clip(fixed, 0.0, 1.0), spec.intensity_gamma).astype(
            np.float32
        )
    if spec.noise_std > 0.0:
        rng = np.random.default_rng(spec.seed)
        fixed = np.clip(
            fixed + rng.normal(0.0, spec.noise_std, size=fixed.shape), 0.0, 1.0
        ).astype(np.float32)
    if spec.crop_fraction > 0.0:
        crop_slices = int(round(fixed.shape[0] * spec.crop_fraction))
        if crop_slices:
            fixed[:crop_slices] = 0.0
            fixed_labels[:crop_slices] = 0

    return SyntheticRegistrationCase(
        spec=spec,
        moving_hwz=moving,
        fixed_hwz=fixed,
        moving_labels_hwz=moving_labels,
        fixed_labels_hwz=fixed_labels,
    )


def default_synthetic_registration_specs() -> tuple[SyntheticRegistrationSpec, ...]:
    """Return the canonical deterministic regression matrix."""
    transform = RigidScaleTransform
    return (
        SyntheticRegistrationSpec("identity", minimum_score_gain=0.0),
        SyntheticRegistrationSpec("translate_x", transform(tx=0.45)),
        SyntheticRegistrationSpec("translate_y", transform(ty=-0.45)),
        SyntheticRegistrationSpec("translate_z", transform(tz=0.20)),
        SyntheticRegistrationSpec("rotate_z", transform(rot_z_deg=6.0)),
        SyntheticRegistrationSpec("uniform_scale", transform(scale=1.06)),
        SyntheticRegistrationSpec(
            "combined", transform(tx=0.30, ty=-0.30, tz=0.10, rot_z_deg=5.0, scale=1.04)
        ),
        SyntheticRegistrationSpec(
            "different_spacing",
            transform(tx=0.30, ty=-0.15, rot_z_deg=4.0, scale=1.03),
            fixed_shape=(36, 48, 15),
            fixed_spacing=(0.17, 0.13, 0.08),
        ),
        SyntheticRegistrationSpec(
            "noise_and_intensity_remap",
            transform(tx=0.30, ty=-0.30, rot_z_deg=4.0),
            noise_std=0.025,
            intensity_gamma=1.6,
            minimum_score_gain=0.015,
        ),
        SyntheticRegistrationSpec(
            "partial_crop",
            transform(tx=0.30, ty=-0.30, rot_z_deg=4.0),
            crop_fraction=0.12,
            minimum_score_gain=0.015,
        ),
        SyntheticRegistrationSpec(
            "constant_low_information",
            geometry="constant",
            identifiable=False,
            minimum_score_gain=0.0,
        ),
        SyntheticRegistrationSpec(
            "symmetric_ambiguous",
            transform=transform(rot_z_deg=8.0),
            geometry="symmetric",
            identifiable=False,
            minimum_score_gain=0.0,
        ),
    )


def make_default_synthetic_registration_cases() -> tuple[SyntheticRegistrationCase, ...]:
    return tuple(
        make_synthetic_registration_case(spec)
        for spec in default_synthetic_registration_specs()
    )

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import affine_transform


@dataclass
class RigidScaleTransform:
    """5-DOF transform mapping MOVING-volume mm coordinates to FIXED-volume
    mm coordinates: translation + a single in-plane rotation + one uniform
    scale. Deliberately not full 6-DOF/independent-axis-scale -- PCBs are
    thin/flat and mounted roughly parallel to the CT stage, so the dominant
    real-world misalignment is in-plane shift/rotation plus an unknown
    relative voxel size between the two volumes (absorbed into `scale`).

    Axis convention: every volume/spacing in this module uses the
    (axis0, axis1, axis2) array-index order already standard throughout this
    codebase (`LayerModel.dataCube`, `_gerber_intensity_hwz`, i.e. (H, W, Z)).
    `tx`/`ty`/`tz` are mm translations along axis0/axis1/axis2 respectively;
    `rot_z_deg` is an in-plane rotation within the axis0-axis1 plane (i.e.
    about the depth axis, axis2 -- named "z" for the rotation axis by CT
    convention, not to be confused with the `tz` translation component).
    """

    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.0
    rot_z_deg: float = 0.0
    scale: float = 1.0

    @classmethod
    def identity(cls) -> "RigidScaleTransform":
        return cls()


@dataclass(frozen=True)
class RegistrationProposal:
    """Result of a bounded initial-registration search.

    ``initial_score`` is the identity-transform baseline and ``score`` is the
    score for ``transform``. Scores are comparable for the same volume pair,
    but are not calibrated probabilities of a correct registration.
    """

    transform: RigidScaleTransform
    initial_score: float
    score: float
    evaluations: int

    @property
    def improvement(self) -> float:
        return self.score - self.initial_score

    @property
    def confidence(self) -> str:
        """Conservative, user-facing interpretation of the score gain."""
        if self.score >= 0.75 and self.improvement >= 0.10:
            return "high"
        if self.score >= 0.45 and self.improvement >= 0.03:
            return "moderate"
        return "low"


@dataclass(frozen=True)
class RegistrationInputSignature:
    """Identity and physical metadata for one moving/fixed volume pair."""

    moving_id: int
    moving_shape: tuple[int, int, int]
    moving_spacing: tuple[float, float, float]
    fixed_id: int
    fixed_shape: tuple[int, int, int]
    fixed_spacing: tuple[float, float, float]

    @staticmethod
    def _spacing(values) -> tuple[float, float, float]:
        spacing = tuple(float(value) for value in values)
        if len(spacing) != 3 or any(not np.isfinite(value) or value <= 0 for value in spacing):
            raise ValueError("registration spacing must contain three positive finite values")
        return spacing

    @classmethod
    def capture(
        cls,
        moving: np.ndarray,
        moving_spacing,
        fixed: np.ndarray,
        fixed_spacing,
    ) -> "RegistrationInputSignature":
        moving_array = np.asarray(moving)
        fixed_array = np.asarray(fixed)
        if moving_array.ndim != 3 or fixed_array.ndim != 3:
            raise ValueError("registration inputs must both be 3-D volumes")
        return cls(
            moving_id=id(moving),
            moving_shape=tuple(moving_array.shape),
            moving_spacing=cls._spacing(moving_spacing),
            fixed_id=id(fixed),
            fixed_shape=tuple(fixed_array.shape),
            fixed_spacing=cls._spacing(fixed_spacing),
        )


@dataclass
class RegistrationSession:
    """Non-UI registration lifecycle and stale-input protection.

    A session distinguishes the currently edited transform from the last
    explicitly accepted transform. Configuring an identical volume pair keeps
    the edit in progress; any new array identity or physical metadata resets it.
    """

    inputs: RegistrationInputSignature | None = None
    current_transform: RigidScaleTransform = field(
        default_factory=RigidScaleTransform.identity
    )
    current_score: float | None = None
    accepted_transform: RigidScaleTransform | None = None
    accepted_score: float | None = None
    proposal: RegistrationProposal | None = None
    dirty: bool = False
    invalid_reason: str | None = None

    def configure(
        self,
        moving: np.ndarray,
        moving_spacing,
        fixed: np.ndarray,
        fixed_spacing,
    ) -> bool:
        """Configure inputs, returning True when prior state was reset."""
        signature = RegistrationInputSignature.capture(
            moving, moving_spacing, fixed, fixed_spacing
        )
        if signature == self.inputs:
            return False
        self.inputs = signature
        self.current_transform = RigidScaleTransform.identity()
        self.current_score = None
        self.accepted_transform = None
        self.accepted_score = None
        self.proposal = None
        self.dirty = False
        self.invalid_reason = None
        return True

    def update_current(
        self, transform: RigidScaleTransform, score: float | None = None
    ) -> None:
        if self.inputs is None:
            raise RuntimeError("registration inputs have not been configured")
        self.current_transform = transform
        self.current_score = None if score is None else float(score)
        self.dirty = self.current_transform != self.accepted_transform

    def record_proposal(self, proposal: RegistrationProposal) -> None:
        self.proposal = proposal
        self.update_current(proposal.transform, proposal.score)

    def accept(self) -> RigidScaleTransform:
        if self.inputs is None:
            raise RuntimeError("registration inputs have not been configured")
        self.accepted_transform = self.current_transform
        self.accepted_score = self.current_score
        self.dirty = False
        return self.accepted_transform

    def invalidate(self, reason: str) -> None:
        self.inputs = None
        self.current_transform = RigidScaleTransform.identity()
        self.current_score = None
        self.accepted_transform = None
        self.accepted_score = None
        self.proposal = None
        self.dirty = False
        self.invalid_reason = reason

    def is_compatible(
        self,
        moving: np.ndarray,
        moving_spacing,
        fixed: np.ndarray,
        fixed_spacing,
    ) -> bool:
        if self.accepted_transform is None or self.inputs is None:
            return False
        return self.inputs == RegistrationInputSignature.capture(
            moving, moving_spacing, fixed, fixed_spacing
        )


def invert_transform(transform: RigidScaleTransform) -> RigidScaleTransform:
    """Returns the transform mapping FIXED mm -> MOVING mm, i.e. the inverse
    of `transform` (which maps moving -> fixed). Useful for tests (recovering
    a known applied misalignment) and for swapping which volume is treated
    as "moving" vs "fixed"."""
    inv_scale = 1.0 / transform.scale
    theta = -transform.rot_z_deg
    c, s = np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))
    inv_linear = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    ) * inv_scale
    translation = np.array([transform.tx, transform.ty, transform.tz], dtype=np.float64)
    new_translation = -inv_linear @ translation
    return RigidScaleTransform(
        tx=float(new_translation[0]),
        ty=float(new_translation[1]),
        tz=float(new_translation[2]),
        rot_z_deg=theta,
        scale=inv_scale,
    )


def _forward_linear_matrix(transform: RigidScaleTransform) -> np.ndarray:
    """3x3 matrix s.t. fixed_mm = matrix @ moving_mm + translation (rotation
    within the axis0-axis1 plane, scaled uniformly, axis2 untouched by
    rotation)."""
    theta = np.deg2rad(transform.rot_z_deg)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return rot * transform.scale


def transform_to_matrix(
    transform: RigidScaleTransform,
    moving_spacing,
    fixed_spacing,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (matrix, offset) for `scipy.ndimage.affine_transform`, which
    maps an OUTPUT (fixed-grid) voxel index to the INPUT (moving-grid) voxel
    index to sample -- i.e. the algebraic inverse of `transform`, since that
    is the direction `affine_transform` requires (it pulls each output voxel
    from a location in the input).
    """
    moving_spacing = np.asarray(moving_spacing, dtype=np.float64)
    fixed_spacing = np.asarray(fixed_spacing, dtype=np.float64)
    translation = np.array([transform.tx, transform.ty, transform.tz], dtype=np.float64)

    forward_inv = np.linalg.inv(_forward_linear_matrix(transform))
    inv_moving_spacing = np.diag(1.0 / moving_spacing)

    matrix = inv_moving_spacing @ forward_inv @ np.diag(fixed_spacing)
    offset = -(inv_moving_spacing @ forward_inv) @ translation
    return matrix, offset


def resample_volume(
    moving: np.ndarray,
    moving_spacing,
    fixed_shape,
    fixed_spacing,
    transform: RigidScaleTransform,
    order: int = 1,
    cval: float = 0.0,
) -> np.ndarray:
    """Resample `moving` (intensity/continuous data) onto a grid of
    `fixed_shape` at `fixed_spacing`, given `transform` mapping moving mm ->
    fixed mm. `order=1` (trilinear) is intended for live-preview/intensity
    use; pass `order=0` only if discrete output is also acceptable here --
    for label volumes prefer `resample_labels`, which additionally preserves
    the input's integer dtype.

    Note: `scipy.ndimage.affine_transform`'s default 'constant' boundary mode
    zeroes an entire interpolated sample outright the instant its source
    coordinate goes even slightly out of range (no partial blending at the
    edge) -- so content sitting at/near a volume's boundary can vanish
    completely under even a sub-voxel shift toward that edge. This is
    expected scipy behavior, not a bug in this function.
    """
    matrix, offset = transform_to_matrix(transform, moving_spacing, fixed_spacing)
    return affine_transform(
        moving,
        matrix=matrix,
        offset=offset,
        output_shape=tuple(fixed_shape),
        order=order,
        cval=cval,
    )


def resample_labels(
    label_vol: np.ndarray,
    moving_spacing,
    fixed_shape,
    fixed_spacing,
    transform: RigidScaleTransform,
    background_label: int = 0,
) -> np.ndarray:
    """Nearest-neighbor resample of a discrete material-ID volume, guaranteed
    to only ever emit label values already present in `label_vol` (plus
    `background_label` for out-of-bounds regions) -- used to transfer the
    Gerber ground-truth per-voxel materials onto the real-CT grid once a
    transform has been established.
    """
    matrix, offset = transform_to_matrix(transform, moving_spacing, fixed_spacing)
    return affine_transform(
        label_vol,
        matrix=matrix,
        offset=offset,
        output_shape=tuple(fixed_shape),
        output=label_vol.dtype,
        order=0,
        mode="constant",
        cval=background_label,
    )


def _robust_normalize(volume: np.ndarray) -> np.ndarray:
    """Return a finite float32 volume normalized to [0, 1]."""
    values = np.asarray(volume, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"registration volumes must be 3-D, got shape {values.shape}")
    values = np.nan_to_num(values, copy=True)
    lo, hi = np.percentile(values, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(values)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def _feature_volume(volume: np.ndarray) -> np.ndarray:
    """Create a modality-tolerant feature image for registration scoring.

    Most Gerber/CT intensity differences are monotonic while their material
    boundaries coincide. Combining normalized intensity with gradient
    magnitude therefore retains broad board structure while emphasizing the
    shared edges used for alignment.
    """
    normalized = _robust_normalize(volume)
    gradients = [
        np.gradient(normalized, axis=axis)
        for axis, size in enumerate(normalized.shape)
        if size > 1
    ]
    if not gradients:
        return normalized
    magnitude = np.sqrt(sum(component * component for component in gradients))
    magnitude = _robust_normalize(magnitude)
    return (0.25 * normalized + 0.75 * magnitude).astype(np.float32)


def _normalized_feature_correlation(
    fixed_feature: np.ndarray, moving_feature: np.ndarray
) -> float:
    """Positive normalized correlation over the union of informative voxels."""
    if fixed_feature.shape != moving_feature.shape:
        raise ValueError(
            "registration-quality inputs must have the same shape after resampling; "
            f"got {fixed_feature.shape} and {moving_feature.shape}"
        )
    mask = (fixed_feature > 0.03) | (moving_feature > 0.03)
    if np.count_nonzero(mask) < 16:
        return 0.0
    fixed_values = fixed_feature[mask].astype(np.float64)
    moving_values = moving_feature[mask].astype(np.float64)
    fixed_values -= fixed_values.mean()
    moving_values -= moving_values.mean()
    denominator = np.linalg.norm(fixed_values) * np.linalg.norm(moving_values)
    if denominator <= np.finfo(np.float64).eps:
        return 0.0
    return float(np.clip(np.dot(fixed_values, moving_values) / denominator, 0.0, 1.0))


def registration_quality(fixed: np.ndarray, resampled_moving: np.ndarray) -> float:
    """Score the alignment of same-grid fixed and moving volumes in [0, 1].

    The score is a positive normalized correlation of robust intensity/edge
    features. It is deterministic and useful for comparing transforms for a
    single volume pair. It is deliberately *not* a calibrated probability or
    proof that a registration is correct; users should review the overlay.
    """
    return _normalized_feature_correlation(
        _feature_volume(fixed), _feature_volume(resampled_moving)
    )


def _downsample_for_search(volume: np.ndarray, spacing, max_dim: int):
    factors = tuple(max(1, int(np.ceil(size / max_dim))) for size in volume.shape)
    if factors == (1, 1, 1):
        return volume, tuple(float(value) for value in spacing)
    return (
        volume[tuple(slice(None, None, factor) for factor in factors)],
        tuple(float(value) * factor for value, factor in zip(spacing, factors)),
    )


def _weighted_geometry(feature: np.ndarray, spacing) -> tuple[np.ndarray, float]:
    """Return feature centroid in mm and RMS radius about that centroid."""
    weights = np.asarray(feature, dtype=np.float64)
    total = weights.sum()
    if total <= np.finfo(np.float64).eps:
        center_idx = (np.asarray(feature.shape, dtype=np.float64) - 1.0) / 2.0
        return center_idx * np.asarray(spacing), 0.0
    coordinates = np.indices(feature.shape, dtype=np.float64)
    centroid_idx = np.array(
        [(coordinates[axis] * weights).sum() / total for axis in range(3)]
    )
    spacing_array = np.asarray(spacing, dtype=np.float64)
    centroid_mm = centroid_idx * spacing_array
    squared_radius = sum(
        ((coordinates[axis] - centroid_idx[axis]) * spacing_array[axis]) ** 2
        for axis in range(3)
    )
    rms_radius = float(np.sqrt((squared_radius * weights).sum() / total))
    return centroid_mm, rms_radius


def propose_initial_transform(
    moving: np.ndarray,
    moving_spacing,
    fixed: np.ndarray,
    fixed_spacing,
    *,
    max_dim: int = 48,
    rotation_limit_deg: float = 15.0,
    translation_limit_fraction: float = 0.25,
    scale_bounds: tuple[float, float] = (0.8, 1.2),
) -> RegistrationProposal:
    """Propose a deterministic 5-DOF starting transform.

    The search is intentionally bounded for interactive use. It begins with
    identity and feature-centroid seeds, then performs a three-resolution
    coordinate search over translation, in-plane rotation, and uniform scale.
    The result is a reviewable starting point for the existing manual tool,
    not an automatically accepted registration.
    """
    if max_dim < 8:
        raise ValueError("max_dim must be at least 8")
    if not 0.0 < translation_limit_fraction <= 1.0:
        raise ValueError("translation_limit_fraction must be in (0, 1]")
    scale_low, scale_high = (float(scale_bounds[0]), float(scale_bounds[1]))
    if not 0.0 < scale_low <= 1.0 <= scale_high:
        raise ValueError("scale_bounds must be positive and contain 1.0")

    moving_small, moving_small_spacing = _downsample_for_search(
        np.asarray(moving), moving_spacing, max_dim
    )
    fixed_small, fixed_small_spacing = _downsample_for_search(
        np.asarray(fixed), fixed_spacing, max_dim
    )
    moving_feature = _feature_volume(moving_small)
    fixed_feature = _feature_volume(fixed_small)
    fixed_extent = np.asarray(fixed_small.shape, dtype=np.float64) * np.asarray(
        fixed_small_spacing
    )
    translation_limits = fixed_extent * translation_limit_fraction
    evaluations = 0

    def score(transform: RigidScaleTransform) -> float:
        nonlocal evaluations
        evaluations += 1
        resampled = resample_volume(
            moving_feature,
            moving_small_spacing,
            fixed_feature.shape,
            fixed_small_spacing,
            transform,
            order=1,
        )
        return _normalized_feature_correlation(fixed_feature, resampled)

    identity = RigidScaleTransform.identity()
    initial_score = score(identity)
    moving_center, moving_radius = _weighted_geometry(moving_feature, moving_small_spacing)
    fixed_center, fixed_radius = _weighted_geometry(fixed_feature, fixed_small_spacing)
    radius_scale = 1.0
    if moving_radius > np.finfo(float).eps:
        radius_scale = float(np.clip(fixed_radius / moving_radius, scale_low, scale_high))

    def centroid_seed(scale: float) -> RigidScaleTransform:
        translation = np.clip(
            fixed_center - scale * moving_center, -translation_limits, translation_limits
        )
        return RigidScaleTransform(
            tx=float(translation[0]),
            ty=float(translation[1]),
            tz=float(translation[2]),
            scale=scale,
        )

    seeds = [identity, centroid_seed(1.0), centroid_seed(radius_scale)]
    best_transform = identity
    best_score = initial_score
    for seed in seeds[1:]:
        seed_score = score(seed)
        if seed_score > best_score:
            best_transform, best_score = seed, seed_score

    parameter_bounds = {
        "tx": (-translation_limits[0], translation_limits[0]),
        "ty": (-translation_limits[1], translation_limits[1]),
        "tz": (-translation_limits[2], translation_limits[2]),
        "rot_z_deg": (-abs(rotation_limit_deg), abs(rotation_limit_deg)),
        "scale": (scale_low, scale_high),
    }
    base_steps = {
        "tx": max(fixed_small_spacing[0], fixed_extent[0] * 0.05),
        "ty": max(fixed_small_spacing[1], fixed_extent[1] * 0.05),
        "tz": max(fixed_small_spacing[2], fixed_extent[2] * 0.05),
        "rot_z_deg": min(5.0, abs(rotation_limit_deg)),
        "scale": min(0.05, (scale_high - scale_low) / 4.0),
    }
    parameter_order = ("scale", "rot_z_deg", "tx", "ty", "tz")

    for resolution in (1.0, 0.5, 0.25):
        for _ in range(3):
            improved = False
            for name in parameter_order:
                current_value = getattr(best_transform, name)
                candidates = []
                for direction in (-1.0, 1.0):
                    candidate_value = float(
                        np.clip(
                            current_value + direction * base_steps[name] * resolution,
                            *parameter_bounds[name],
                        )
                    )
                    if np.isclose(candidate_value, current_value):
                        continue
                    values = {
                        "tx": best_transform.tx,
                        "ty": best_transform.ty,
                        "tz": best_transform.tz,
                        "rot_z_deg": best_transform.rot_z_deg,
                        "scale": best_transform.scale,
                    }
                    values[name] = candidate_value
                    candidate = RigidScaleTransform(**values)
                    candidates.append((score(candidate), candidate))
                if candidates:
                    candidate_score, candidate = max(candidates, key=lambda item: item[0])
                    if candidate_score > best_score + 1e-6:
                        best_transform, best_score = candidate, candidate_score
                        improved = True
            if not improved:
                break

    return RegistrationProposal(
        transform=best_transform,
        initial_score=initial_score,
        score=best_score,
        evaluations=evaluations,
    )

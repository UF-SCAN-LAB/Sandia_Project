from __future__ import annotations

from dataclasses import dataclass

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

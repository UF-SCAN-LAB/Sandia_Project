from __future__ import annotations

import numpy as np

from xrayrecon.multimaterial import MultiMaterialConfig


def normalize_volume(vol: np.ndarray) -> np.ndarray:
    """Min-max normalize *vol* to [0.0, 1.0] float32."""
    vol = vol.astype(np.float32)
    vmin, vmax = float(vol.min()), float(vol.max())
    if vmax > vmin:
        return (vol - vmin) / (vmax - vmin)
    return np.zeros_like(vol)


def voxel_to_material_id(
    volume: np.ndarray,
    config: MultiMaterialConfig,
    channel_idx: int = 0,
) -> np.ndarray:
    """Assign each voxel a material ID (1-based) by threshold on *channel_idx*.

    volume: (Z, H, W) or (C, Z, H, W) float32, values in [0, 1].
    Returns uint8 array of shape (Z, H, W); 0 = unassigned, 1..N = material index.
    Materials are evaluated in order; first match wins so ranges should be non-overlapping.
    """
    if volume.ndim == 4:
        vol_3d = volume[channel_idx]
    else:
        vol_3d = volume

    labels = np.zeros(vol_3d.shape, dtype=np.uint8)
    for idx, mat in enumerate(config.materials, start=1):
        lo, hi = mat.threshold
        unassigned = labels == 0
        in_range = (vol_3d >= lo) & (vol_3d <= hi)
        labels[unassigned & in_range] = idx

    return labels


def label_map_to_rgba(
    label_map: np.ndarray,
    config: MultiMaterialConfig,
) -> np.ndarray:
    """Convert an integer label map to a (H, W, Z, 4) uint8 RGBA volume.

    label_map: uint8 (Z, H, W); 0 = unassigned (rendered as transparent).
    Output format matches PCBViewer3D.setRGBADataCube expectations.
    """
    Z, H, W = label_map.shape
    rgba = np.zeros((H, W, Z, 4), dtype=np.uint8)

    for idx, mat in enumerate(config.materials, start=1):
        mask_zhw = label_map == idx            # (Z, H, W) bool
        mask_hwz = np.moveaxis(mask_zhw, 0, -1)  # (H, W, Z) bool
        r, g, b, a = mat.color
        rgba[mask_hwz, 0] = r
        rgba[mask_hwz, 1] = g
        rgba[mask_hwz, 2] = b
        rgba[mask_hwz, 3] = a

    return rgba

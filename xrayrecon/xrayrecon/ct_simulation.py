from __future__ import annotations

import numpy as np
from skimage.transform import iradon, radon


def simulate_ct_volume(vol: np.ndarray, num_angles: int | None = None) -> np.ndarray:
    """Simulate an FBP CT reconstruction of a Voxelizer-produced volume.

    vol: (Z, H, W, 2) = [z_eff, electron_density], as returned by
    Voxelizer.voxelize_stackup(). For each Z slice, builds a 2D attenuation
    proxy (z_eff * density, the same convention demo_gerber_xray.py uses),
    projects it via radon() at `num_angles` evenly-spaced angles over 180
    degrees, and reconstructs via iradon() filtered back-projection. This
    reproduces real reconstruction artifacts (streaking, blur) without an
    explicit noise model.

    circle=False on both calls is required: radon()/iradon() otherwise
    assume the image is zero outside the circle inscribed in the square
    slice, silently corrupting content near the corners (e.g. PCB traces
    that reach the layout edges). circle=False makes skimage pad to the
    diagonal internally and return output already cropped back to (H, W).

    Returns a (Z, H, W) float32 volume, ready to feed into the same
    real-data code path as loadCubeFromFile/loadEDData.
    """
    z_depth, h, w, _ = vol.shape
    if num_angles is None:
        num_angles = max(h, w)
    theta = np.linspace(0.0, 180.0, num_angles, endpoint=False)

    out = np.zeros((z_depth, h, w), dtype=np.float32)
    for z in range(z_depth):
        atten = (vol[z, :, :, 0] * vol[z, :, :, 1]).astype(np.float32)
        sinogram = radon(atten, theta=theta, circle=False)
        out[z] = iradon(sinogram, theta=theta, filter_name="ramp", circle=False).astype(np.float32)
    return out

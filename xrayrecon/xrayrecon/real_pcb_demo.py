"""Loader for real X-ray CT data of a real PCB, for the "real Gerber + real
CT" demo. No Qt dependency -- standalone-testable, same shape as demo_pcb.py.

A full-resolution sub-scan here is ~1375 slices at 2940x2940 uint16 (~22 GB)
-- too large to load into a GLVolumeItem-backed viewer directly. This reads
every `slice_stride`-th slice and resizes each to (target_xy, target_xy) as
it's read, never materializing the full-resolution stack, so the result
stays genuinely real data at a responsive resolution.
"""

from __future__ import annotations

import glob
import os

import cv2
import numpy as np

try:
    import tifffile as _tifffile
except ImportError:
    _tifffile = None

DEFAULT_DATA_DIR = (
    r"E:\Data\20250701_PCB2\20250701_PCB2_stitched_montage_recon_fdk"
)


def load_real_pcb_ct(
    data_dir: str = DEFAULT_DATA_DIR,
    slice_stride: int = 4,
    target_xy: int = 512,
) -> np.ndarray:
    """Load a downsampled real CT volume from a directory of TIFF slices.

    Returns a (Z, H, W) float32 array, normalized to [0, 1].
    """
    if _tifffile is None:
        raise ImportError("tifffile is required to load real CT TIFF slices")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Real CT data directory not found: {data_dir}")

    files = sorted(glob.glob(os.path.join(data_dir, "*.tif*")))
    if not files:
        raise ValueError(f"No TIFF slices found in {data_dir}")
    files = files[::slice_stride]

    slices = []
    for f in files:
        img = _tifffile.imread(f).astype(np.float32)
        img = cv2.resize(img, (target_xy, target_xy), interpolation=cv2.INTER_AREA)
        slices.append(img)

    vol = np.stack(slices, axis=0)  # (Z, H, W)
    lo, hi = float(vol.min()), float(vol.max())
    if hi > lo:
        vol = (vol - lo) / (hi - lo)
    else:
        vol = np.zeros_like(vol)
    return vol


if __name__ == "__main__":
    v = load_real_pcb_ct()
    print(f"Real PCB CT volume: {v.shape} {v.dtype} min={v.min():.3f} max={v.max():.3f}")

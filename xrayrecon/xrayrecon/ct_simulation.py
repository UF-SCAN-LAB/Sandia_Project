from __future__ import annotations

import numpy as np
from skimage.transform import iradon, radon


def _match_reconstruction_shape(
    reconstruction: np.ndarray, target_shape: tuple[int, int]
) -> np.ndarray:
    """Center-crop/pad ``iradon`` output to the source slice shape."""
    target_h, target_w = (int(target_shape[0]), int(target_shape[1]))
    source_h, source_w = reconstruction.shape
    output = np.zeros((target_h, target_w), dtype=reconstruction.dtype)

    copy_h = min(source_h, target_h)
    copy_w = min(source_w, target_w)
    source_y = max(0, (source_h - copy_h) // 2)
    source_x = max(0, (source_w - copy_w) // 2)
    target_y = max(0, (target_h - copy_h) // 2)
    target_x = max(0, (target_w - copy_w) // 2)
    output[target_y : target_y + copy_h, target_x : target_x + copy_w] = (
        reconstruction[
            source_y : source_y + copy_h,
            source_x : source_x + copy_w,
        ]
    )
    return output


def simulate_ct_volume(
    vol: np.ndarray,
    num_angles: int | None = None,
    *,
    projection_backend: str = "cpu",
    projection_batch_size: int = 8,
) -> np.ndarray:
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
    diagonal internally; iradon may return that reconstruction as a square,
    so the result is center-cropped/padded back to (H, W).

    ``projection_backend="cuda"`` uses the optional CUDA Radon projector for
    the forward projection. Filtered backprojection remains on CPU. Explicit
    CUDA selection raises if PyTorch/CUDA is unavailable and never falls back.

    Returns a (Z, H, W) float32 volume, ready to feed into the same real-data
    code path as loadCubeFromFile/loadEDData.
    """
    z_depth, h, w, _ = vol.shape
    if num_angles is None:
        num_angles = max(h, w)
    theta = np.linspace(0.0, 180.0, num_angles, endpoint=False)

    if projection_backend == "cpu":
        projector = None
    elif projection_backend == "cuda":
        from .cuda_projection import CudaRadonProjector

        projector = CudaRadonProjector(
            (h, w), theta, angle_batch_size=projection_batch_size
        )
    else:
        raise ValueError("projection_backend must be 'cpu' or 'cuda'")

    out = np.zeros((z_depth, h, w), dtype=np.float32)
    for z in range(z_depth):
        atten = (vol[z, :, :, 0] * vol[z, :, :, 1]).astype(np.float32)
        if projector is None:
            sinogram = radon(atten, theta=theta, circle=False)
        else:
            sinogram = projector.project(atten[None])[0]
        reconstruction = iradon(
            sinogram, theta=theta, filter_name="ramp", circle=False
        )
        out[z] = _match_reconstruction_shape(reconstruction, (h, w)).astype(np.float32)
    return out

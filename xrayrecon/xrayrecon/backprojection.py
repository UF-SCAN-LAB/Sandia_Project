"""Select the CPU reference or explicit CUDA filtered backprojection."""
from __future__ import annotations

import numpy as np
from skimage.transform import iradon


def make_backprojector(detector_size, theta, *, backend="cpu"):
    if backend == "cpu":
        def reconstruct(sinograms, *, filter_name="ramp"):
            return np.stack([
                iradon(sinogram, theta=theta, filter_name=filter_name, circle=False)
                for sinogram in sinograms
            ])
        return reconstruct
    if backend == "cuda":
        from .cuda_backprojection import CudaBackprojector

        return CudaBackprojector(detector_size, theta).reconstruct
    raise ValueError("backprojection_backend must be 'cpu' or 'cuda'")

"""Optional CUDA Radon projection matching scikit-image's ``circle=False``.

PyTorch is imported only when a projector is constructed. Explicit CUDA requests
raise on missing dependencies/devices or execution failures; there is no CPU
fallback and no system CUDA compiler is required.
"""
from __future__ import annotations

from collections.abc import MutableMapping

import numpy as np


class CudaRadonProjector:
    """Reuse one projection geometry across float32 and float64 image batches.

    ``project`` accepts finite NumPy-compatible ``(C, H, W)`` floating arrays
    and returns independent, synchronized NumPy ``(C, D, A)`` arrays in the
    same dtype. Here ``D = ceil(sqrt(2) * max(H, W))`` and ``A`` is the number
    of angles in degrees. Only scikit-image's zero-padded, ``circle=False``
    geometry is supported. Angle order and repetitions are preserved.

    The two small cached geometries contain axes and rotation coefficients,
    not full sampling grids. Sampling and rotated-image temporaries are bounded
    by ``angle_batch_size * C * D**2``; input storage scales with ``C * D**2``.
    Output storage lives on the host. No grids for all angles are retained.
    """

    def __init__(
        self,
        image_shape: tuple[int, int],
        theta_deg: np.ndarray,
        *,
        angle_batch_size: int = 8,
    ) -> None:
        try:
            shape = tuple(image_shape)
        except TypeError as exc:
            raise ValueError("image_shape must contain two positive integers") from exc
        if len(shape) != 2 or any(
            isinstance(size, (bool, np.bool_))
            or not isinstance(size, (int, np.integer))
            or size <= 0
            for size in shape
        ):
            raise ValueError("image_shape must contain two positive integers")
        if (
            isinstance(angle_batch_size, (bool, np.bool_))
            or not isinstance(angle_batch_size, (int, np.integer))
            or angle_batch_size <= 0
        ):
            raise ValueError("angle_batch_size must be a positive integer")
        try:
            theta = np.asarray(theta_deg)
            valid_angles = (
                theta.ndim == 1
                and theta.size > 0
                and theta.dtype.kind in "fiu"
                and np.isfinite(theta).all()
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("theta_deg must be a nonempty finite real 1-D array") from exc
        if not valid_angles:
            raise ValueError("theta_deg must be a nonempty finite real 1-D array")
        self.image_shape = tuple(int(size) for size in shape)
        self.theta_deg = theta.copy()
        self.theta_deg.flags.writeable = False
        self.angle_batch_size = int(angle_batch_size)
        self.detector_size = int(np.ceil(np.sqrt(2) * max(self.image_shape)))
        center = self.detector_size // 2
        self._pad_before = tuple(center - size // 2 for size in self.image_shape)
        # Use exactly Radon's NumPy expressions before warp's input-dtype cast.
        self._matrices = np.asarray([
            [
                [np.cos(a), np.sin(a), -center * (np.cos(a) + np.sin(a) - 1)],
                [-np.sin(a), np.cos(a), -center * (np.cos(a) - np.sin(a) - 1)],
            ]
            for a in np.deg2rad(self.theta_deg)
        ])
        self._geometry: dict[str, tuple] = {}
        try:
            import torch
        except (ImportError, OSError, RuntimeError) as exc:
            raise RuntimeError(
                "CUDA projection backend requires an installed CUDA-enabled "
                "PyTorch; the optional dependency could not be imported."
            ) from exc
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("PyTorch reports no available CUDA device")
            self.device = torch.device("cuda", torch.cuda.current_device())
            self._torch = torch
            torch.cuda.synchronize(self.device)
        except (RuntimeError, AssertionError, OSError) as exc:
            raise RuntimeError(f"CUDA projection backend is unavailable: {exc}") from exc

    def _get_geometry(self, dtype: np.dtype) -> tuple:
        key = dtype.str
        if key not in self._geometry:
            torch = self._torch
            device_dtype = torch.float32 if dtype == np.float32 else torch.float64
            axis = torch.arange(
                self.detector_size, dtype=device_dtype, device=self.device
            )
            coefficients = torch.from_numpy(self._matrices.astype(dtype)).to(self.device)
            self._geometry[key] = (axis, coefficients)
        return self._geometry[key]

    def _sample_pixels(self, source, xin, yin):
        """Bilinear zero-exterior interpolation without grid normalization.

        Keeping pixel coordinates avoids the float32 normalized-coordinate
        round trip in ``grid_sample``, which moves sparse features by measurable
        fractions of a pixel on larger images. Each neighbor is masked before
        blending, so points just outside the image retain partial edge weights.
        """
        torch = self._torch
        side = self.detector_size
        x0 = torch.floor(xin).to(torch.int64)
        y0 = torch.floor(yin).to(torch.int64)
        dx = (xin - x0).unsqueeze(0)
        dy = (yin - y0).unsqueeze(0)

        def neighbor(xx, yy):
            valid = (xx >= 0) & (xx < side) & (yy >= 0) & (yy < side)
            values = source[0, :, yy.clamp(0, side - 1), xx.clamp(0, side - 1)]
            return values * valid.unsqueeze(0)

        top = neighbor(x0, y0) * (1 - dx) + neighbor(x0 + 1, y0) * dx
        bottom = neighbor(x0, y0 + 1) * (1 - dx) + neighbor(x0 + 1, y0 + 1) * dx
        return (top * (1 - dy) + bottom * dy).permute(1, 0, 2, 3)

    def project(
        self,
        images: np.ndarray,
        *,
        timings: MutableMapping[str, float] | None = None,
    ) -> np.ndarray:
        """Project all channels together using bounded angle batches.

        Optional ``timings['compute_ms']`` sums CUDA event intervals around
        coordinate generation, bilinear resampling and row reduction. These
        intervals exclude explicit host transfers and geometry initialization,
        but include dispatch gaps. Measure host-to-host elapsed time separately;
        instrumentation is absent unless requested.
        """
        images = np.asarray(images)
        if images.ndim != 3 or images.shape[0] < 1 or images.shape[1:] != self.image_shape:
            raise ValueError(f"images must have shape (C, {self.image_shape[0]}, {self.image_shape[1]}) with C > 0")
        if images.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError("CUDA projection images must have dtype float32 or float64")
        if not np.isfinite(images).all():
            raise ValueError("CUDA projection images must contain only finite values")
        if timings is not None:
            timings["compute_ms"] = 0.0
        torch = self._torch
        side = self.detector_size
        channels = images.shape[0]
        angles = self.theta_deg.size
        output = np.empty((channels, side, angles), dtype=images.dtype)
        # Both noncontiguous and read-only arrays are safe; no caller data is
        # exposed to a writable Torch view or changed by the device operation.
        host_images = np.array(images, copy=True, order="C")
        try:
            with torch.cuda.device(self.device), torch.no_grad():
                axis, coefficients = self._get_geometry(images.dtype)
                source = torch.zeros(
                    (1, channels, side, side), dtype=axis.dtype, device=self.device
                )
                top, left = self._pad_before
                height, width = self.image_shape
                source[0, :, top:top + height, left:left + width].copy_(
                    torch.from_numpy(host_images)
                )
                x = axis.reshape(1, 1, side)
                y = axis.reshape(1, side, 1)
                if timings is not None:
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                for offset in range(0, angles, self.angle_batch_size):
                    stop = min(offset + self.angle_batch_size, angles)
                    matrix = coefficients[offset:stop]
                    if timings is not None:
                        start.record()
                    xin = (
                        matrix[:, 0, 0, None, None] * x
                        + matrix[:, 0, 1, None, None] * y
                        + matrix[:, 0, 2, None, None]
                    )
                    yin = (
                        matrix[:, 1, 0, None, None] * x
                        + matrix[:, 1, 1, None, None] * y
                        + matrix[:, 1, 2, None, None]
                    )
                    rotated = self._sample_pixels(source, xin, yin)
                    projected = rotated.sum(dim=2).permute(1, 2, 0)
                    if timings is not None:
                        end.record()
                    output[:, :, offset:stop] = projected.cpu().numpy()
                    if timings is not None:
                        timings["compute_ms"] += start.elapsed_time(end)
                    # Release batch temporaries before allocating the next batch.
                    del xin, yin, rotated, projected
                torch.cuda.synchronize(self.device)
        except (RuntimeError, AssertionError, OSError) as exc:
            raise RuntimeError(
                f"CUDA projection execution failed: {exc}. "
                "Select projection_backend='cpu' explicitly to use the CPU projector."
            ) from exc
        return output

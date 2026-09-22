"""Optional float64 CUDA spectral evaluation using installed PyTorch kernels.

Importing this module does not import PyTorch or initialize CUDA. Construction
is an explicit CUDA request: unavailable dependencies/devices raise an error,
and execution never falls back to CPU. No system CUDA compiler is needed.
"""
from __future__ import annotations

from collections.abc import MutableMapping, Sequence

import numpy as np

from .am_acquisition import AMSpectrum


class CudaSpectrumEvaluator:
    """Cache spectrum tables and evaluate one or more spectra in bounded batches.

    Inputs are converted to flattened NumPy float64 arrays, matching the CPU
    spectral API. Equal input sizes are required. Each result is an independent
    flat NumPy float64 array; all device-to-host copies finish before returning.

    Two workspaces of at most ``batch_size * max_energy_bins`` float64 values
    are allocated per evaluation and reused across spectra and batches. Ray
    coefficients are uploaded once per batch for all spectra. Spectrum tables
    remain on the device for the lifetime of this object.
    """

    def __init__(
        self, spectra: Sequence[AMSpectrum], *, batch_size: int = 4096
    ) -> None:
        if (
            isinstance(batch_size, (bool, np.bool_))
            or not isinstance(batch_size, (int, np.integer))
            or batch_size <= 0
        ):
            raise ValueError("CUDA batch_size must be a positive integer")
        spectra = tuple(spectra)
        if not spectra:
            raise ValueError("CUDA spectral evaluation requires at least one spectrum")
        self.batch_size = int(batch_size)

        # Import only after explicit selection and basic argument validation.
        try:
            import torch
        except (ImportError, OSError, RuntimeError) as exc:
            raise RuntimeError(
                "CUDA spectral backend requires an installed CUDA-enabled PyTorch; "
                "the optional dependency could not be imported."
            ) from exc

        from .alvarez_macovski import f1, f2

        try:
            if not torch.cuda.is_available():
                raise RuntimeError("PyTorch reports no available CUDA device")
            self.device = torch.device("cuda", torch.cuda.current_device())
            self._torch = torch
            self._tables = tuple(
                tuple(
                    torch.as_tensor(values, dtype=torch.float64, device=self.device)
                    for values in (
                        f1(np.asarray(spectrum.energies_kev, dtype=np.float64)),
                        f2(np.asarray(spectrum.energies_kev, dtype=np.float64)),
                        np.asarray(spectrum.weights, dtype=np.float64),
                    )
                )
                for spectrum in spectra
            )
            # Surface initialization/upload errors here, including driver errors.
            torch.cuda.synchronize(self.device)
        except (RuntimeError, AssertionError, OSError) as exc:
            raise RuntimeError(
                f"CUDA spectral backend is unavailable: {exc}. "
                "Select backend='cpu' explicitly to use the CPU implementation."
            ) from exc

    def evaluate(
        self,
        a1: np.ndarray,
        a2: np.ndarray,
        *,
        timings: MutableMapping[str, float] | None = None,
    ) -> tuple[np.ndarray, ...]:
        """Evaluate all spectra and optionally report CUDA compute milliseconds.

        ``timings['compute_ms']`` sums CUDA event intervals around only device
        arithmetic, excluding allocation and input/output transfers. Host-to-host
        elapsed time must be measured separately. Event instrumentation is absent
        unless timings is provided, and may increase overall call latency.
        """
        a1 = np.asarray(a1, dtype=np.float64).ravel()
        a2 = np.asarray(a2, dtype=np.float64).ravel()
        if a1.size != a2.size:
            raise ValueError("CUDA spectral inputs must have equal flattened sizes")
        if timings is not None:
            timings["compute_ms"] = 0.0
        outputs = tuple(np.empty(a1.size, dtype=np.float64) for _ in self._tables)
        if not a1.size:
            return outputs

        # torch.from_numpy warns about read-only memory even though these host
        # tensors are only copied from; make a writable copy in that rare case.
        if not a1.flags.writeable:
            a1 = a1.copy()
        if not a2.flags.writeable:
            a2 = a2.copy()
        torch = self._torch
        rows = min(self.batch_size, a1.size)
        bins = max(table[0].numel() for table in self._tables)
        try:
            with torch.cuda.device(self.device), torch.no_grad():
                device_a1 = torch.empty(rows, dtype=torch.float64, device=self.device)
                device_a2 = torch.empty_like(device_a1)
                transmitted = torch.empty_like(device_a1)
                exponent = torch.empty(
                    (rows, bins), dtype=torch.float64, device=self.device
                )
                scratch = torch.empty_like(exponent)
                if timings is not None:
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)

                for offset in range(0, a1.size, self.batch_size):
                    stop = min(offset + self.batch_size, a1.size)
                    count = stop - offset
                    batch_a1 = device_a1[:count]
                    batch_a2 = device_a2[:count]
                    batch_a1.copy_(torch.from_numpy(a1[offset:stop]))
                    batch_a2.copy_(torch.from_numpy(a2[offset:stop]))
                    result = transmitted[:count]
                    for output, (basis1, basis2, weights) in zip(outputs, self._tables):
                        exp_batch = exponent[:count, :basis1.numel()]
                        scratch_batch = scratch[:count, :basis1.numel()]
                        if timings is not None:
                            start.record()
                        torch.mul(batch_a1[:, None], basis1, out=exp_batch)
                        exp_batch.neg_()
                        torch.mul(batch_a2[:, None], basis2, out=scratch_batch)
                        exp_batch.sub_(scratch_batch)
                        exp_batch.exp_()
                        exp_batch.mul_(weights)
                        torch.sum(exp_batch, dim=1, out=result)
                        result.clamp_(min=1e-12)
                        result.log_().neg_()
                        if timings is not None:
                            end.record()
                        # Blocking CPU copy synchronizes this batch's arithmetic
                        # before the workspace or event pair can be reused.
                        output[offset:stop] = result.cpu().numpy()
                        if timings is not None:
                            timings["compute_ms"] += start.elapsed_time(end)
        except (RuntimeError, AssertionError, OSError) as exc:
            raise RuntimeError(
                f"CUDA spectral evaluation failed: {exc}. "
                "No CPU fallback was performed; try a smaller batch_size if "
                "device memory is exhausted."
            ) from exc
        return outputs

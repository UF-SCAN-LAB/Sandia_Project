"""Reference-based fidelity of numerical images, before display normalization."""
from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


def reconstruction_fidelity(reference, reconstruction, *, data_range=None) -> dict:
    """SNR = 10 log10(sum(reference**2) / sum((reconstruction-reference)**2)).

    SSI is scikit-image's SSIM with uniform windows, sample covariance, K1=.01,
    K2=.03, and a reference-only data range. For volumes, average 2-D SSI over
    Z slices; SNR uses all voxels. Include background and boundaries without
    clipping, masking, fitting, or independently normalizing either image.
    Constant references or sides smaller than 3 have unavailable SSI; zero
    reference energy has unavailable SNR. Exact nonzero recovery has +inf SNR.
    """
    reference, reconstruction = np.asarray(reference), np.asarray(reconstruction)
    if reference.shape != reconstruction.shape or reference.ndim not in (2, 3):
        raise ValueError("reference and reconstruction must share a 2-D or (Z, H, W) shape")
    if reference.size == 0:
        raise ValueError("reference and reconstruction must be nonempty")
    if not np.isfinite(reference).all() or not np.isfinite(reconstruction).all():
        raise ValueError("reference and reconstruction must be finite")
    if data_range is None:
        data_range = float(reference.max()) - float(reference.min())
    if not np.isfinite(data_range) or data_range < 0:
        raise ValueError("data_range must be finite and nonnegative")
    side = min(reference.shape[-2:])
    window = min(7, side if side % 2 else side - 1)
    status = "constant_reference" if data_range == 0 else "image_too_small" if window < 3 else "available"
    refs = reference[None] if reference.ndim == 2 else reference
    recs = reconstruction[None] if reconstruction.ndim == 2 else reconstruction
    signal_energy = error_energy = 0.0
    similarities = []
    for truth, recovered in zip(refs, recs):
        truth, recovered = truth.astype(np.float64), recovered.astype(np.float64)
        signal_energy += float(np.sum(truth ** 2))
        error_energy += float(np.sum((recovered - truth) ** 2))
        if status == "available":
            similarities.append(float(structural_similarity(
                truth, recovered, data_range=float(data_range), win_size=window,
            )))
    snr = None
    if signal_energy > 0:
        snr = float("inf") if error_energy == 0 else float(10 * (np.log10(signal_energy) - np.log10(error_energy)))
    return {
        "snr_db": snr,
        "ssi": float(np.mean(similarities)) if similarities else None,
        "ssi_status": status,
        "ssi_data_range": float(data_range),
        "ssi_window_size": window if status == "available" else None,
        "scope": "all_voxels_including_background_and_boundaries",
    }


def am_reconstruction_metrics(result) -> dict:
    """Compare each available AM reconstruction mode to matching ground truth."""
    metrics = {}
    for mode, suffix in (("raw", "recovered"), ("regularized", "regularized"), ("calibrated", "calibrated")):
        fields = {}
        for field in ("zeff", "rhoe"):
            reference, recovered = result.get(f"{field}_ground_truth"), result.get(f"{field}_{suffix}")
            if reference is not None and recovered is not None:
                fields[field] = reconstruction_fidelity(reference, recovered)
        if fields:
            metrics[mode] = fields
    return metrics

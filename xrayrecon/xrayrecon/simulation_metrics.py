"""Terminal run records for comparing host-to-host simulation performance."""
from __future__ import annotations

import json
import math
import os
import platform
import sys
from decimal import Decimal, ROUND_DOWN
from time import perf_counter


def simulation_record(
    started: float,
    *,
    model: str,
    shape_zhw,
    num_angles: int,
    projection_backend: str,
    sample_file=None,
    voxel_spacing_mm=None,
    spectrum_backend: str | None = None,
    backprojection_backend: str = "cpu",
    action_name: str | None = None,
    **settings,
) -> dict:
    """Record completed computation, including initialization and transfers.

    CUDA projectors/evaluators finish device-to-host copies before returning,
    so this measures completed work. CPU runs never import optional torch.
    Spacing is metadata in (X, Y, Z) order; unknown values remain null.
    """
    elapsed_s = perf_counter() - started
    uses_cuda = "cuda" in (projection_backend, spectrum_backend, backprojection_backend)
    gpu = None
    if uses_cuda:
        torch = sys.modules["torch"]  # Already initialized by the selected backend.
        gpu = torch.cuda.get_device_name(torch.cuda.current_device())
    return {
        "model": model,
        "action_name": action_name or f"{model.upper()} simulation",
        "status": "complete",
        "elapsed_s": elapsed_s,
        "timing_scope": "simulation_including_setup_and_transfers",
        "sample_file": None if sample_file is None else str(sample_file),
        "shape_zhw": [int(value) for value in shape_zhw],
        "voxel_spacing_mm_xyz": (
            None if voxel_spacing_mm is None else list(voxel_spacing_mm)
        ),
        "num_angles": int(num_angles),
        "processor": "cuda" if uses_cuda else "cpu",
        "cpu": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER") or platform.machine(),
        "gpu": gpu,
        "projection_backend": projection_backend,
        "spectrum_backend": spectrum_backend,
        "backprojection_backend": backprojection_backend,
        **settings,
    }


def print_simulation_record(record: dict) -> None:
    """Print a single parseable JSON record per completed run."""
    def json_safe(value):
        if isinstance(value, float) and not math.isfinite(value):
            return "inf" if value > 0 else "-inf" if value < 0 else None
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [json_safe(item) for item in value]
        return value
    print("[simulation] " + json.dumps(json_safe(record), allow_nan=False), flush=True)


def print_simulation_timing(record: dict) -> None:
    """Print only the action, CPU/GPU, and times truncated to two decimals."""
    def seconds(value):
        return f"{Decimal(str(value)).quantize(Decimal('0.01'), rounding=ROUND_DOWN):.2f} s"

    metrics_s = record.get("metrics_elapsed_s", 0.0)
    processor = "GPU" if record["processor"] == "cuda" else "CPU"
    parts = [
        f"[{record['action_name']}] {processor}",
        f"Total: {seconds(record['elapsed_s'] + metrics_s)}",
    ]
    for key, label in (("projection_s", "Projection"), ("acquisition_s", "Acquisition"),
                       ("reconstruction_s", "Reconstruction")):
        if key in record:
            step = f"{label}: {seconds(record[key])}"
            if key == "reconstruction_s":
                details = [f"{name}: {seconds(record[field])}"
                           for field, name in (("basis_backprojection_s", "Basis BP"),
                                               ("quality_backprojection_s", "Residual BP"))
                           if field in record]
                if details:
                    step += " (" + "; ".join(details) + ")"
            parts.append(step)
    # AM backprojection is already shown inside its reconstruction step.
    if "backprojection_s" in record and "reconstruction_s" not in record:
        parts.append(f"Backprojection: {seconds(record['backprojection_s'])}")
    if "metrics_elapsed_s" in record:
        parts.append(f"Metrics: {seconds(metrics_s)}")
    print(" | ".join(parts), flush=True)

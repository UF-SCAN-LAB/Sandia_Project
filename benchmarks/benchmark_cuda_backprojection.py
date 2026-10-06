"""Compare CPU FBP, custom CUDA FBP, and CT/AM before/after GPU workflows.

python -B benchmarks/benchmark_cuda_backprojection.py --quick --repeats 3
JSON report goes to stdout; simulation records go to stderr. CPU, previous
partial GPU, and current GPU workflows use identical inputs and settings.
Host timings include setup/transfers and metrics, with first calls separated
from alternating warmed repeats. FBP event timing excludes host transfers.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
from skimage.transform import iradon, radon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "xrayrecon"))
from xrayrecon.cuda_backprojection import CudaBackprojector
from xrayrecon.ct_simulation import simulate_ct_volume
from xrayrecon.alvarez_macovski import simulate_am_volume
from xrayrecon.materials import MATERIAL_NAMES, material_labels_to_zeff_rhoe
from benchmark_cuda_projection import summary, errors, workflow_gate


def measure(calls, repeats, torch):
    first, samples, results = {}, {name: [] for name in calls}, {}
    for repeat in range(repeats + 1):
        order = list(calls) if repeat % 2 == 0 else list(reversed(calls))
        for name in order:
            torch.cuda.synchronize()
            started = perf_counter()
            with redirect_stdout(sys.stderr):
                results[name] = calls[name]()
            torch.cuda.synchronize()
            duration = perf_counter() - started
            if repeat == 0:
                first[name] = duration
            else:
                samples[name].append(duration)
    return results, {"first_host_seconds": first,
                     "warm_host_seconds": {name: summary(values) for name, values in samples.items()}}


def fbp_case(shape, angles, dtype, args, torch):
    image = np.random.default_rng(281).random(shape).astype(dtype)
    theta = np.linspace(0, 180, angles, endpoint=False)
    sinogram = radon(image, theta=theta, circle=False)
    def gpu():
        return CudaBackprojector(sinogram.shape[0], theta).reconstruct(sinogram[None])[0]
    results, report = measure({"cpu": lambda: iradon(sinogram, theta=theta, circle=False),
                               "cuda": gpu}, args.repeats, torch)
    projector = CudaBackprojector(sinogram.shape[0], theta)
    compute = []
    for _ in range(args.repeats):
        timings = {}
        projector.reconstruct(sinogram[None], timings=timings)
        compute.append(timings["compute_ms"])
    error = errors(results["cuda"], results["cpu"])
    report.update(shape=shape, angles=angles, dtype=np.dtype(dtype).name,
                  cuda_fft_and_kernel_ms=summary(compute), numerical_errors=error,
                  numerical_gate_pass=error["normalized_max"] < (2e-5 if dtype == np.float32 else 1e-11),
                  median_speedup=report["warm_host_seconds"]["cpu"]["median"] / report["warm_host_seconds"]["cuda"]["median"])
    return report


def workflow_case(model, args, torch):
    labels = np.full((2, 96, 128), MATERIAL_NAMES.index("AIR"), dtype=np.int16)
    labels[:, 12:84, 16:112] = MATERIAL_NAMES.index("FR4")
    labels[:, 24:48, 32:60] = MATERIAL_NAMES.index("COPPER")
    labels[:, 54:76, 78:104] = MATERIAL_NAMES.index("TIN")
    volume = material_labels_to_zeff_rhoe(labels)
    simulator = simulate_ct_volume if model == "ct" else simulate_am_volume
    def run(projection, backprojection, spectrum):
        kwargs = dict(num_angles=64, projection_backend=projection,
                      backprojection_backend=backprojection, sample_file="synthetic_FR4_Cu_Tin",
                      voxel_spacing_mm=(0.1, 0.1, 0.05))
        if model == "am":
            kwargs["backend"] = spectrum
        return simulator(volume, **kwargs)
    calls = {"cpu": lambda: run("cpu", "cpu", "cpu"),
             "partial_gpu": lambda: run("cuda", "cpu", "cuda"),
             "gpu": lambda: run("cuda", "cuda", "cuda")}
    results, report = measure(calls, args.repeats, torch)
    report.update(model=model, shape_zhw=labels.shape, angles=64,
                  sample_file="synthetic_FR4_Cu_Tin", voxel_spacing_mm_xyz=(0.1, 0.1, 0.05),
                  gpu_backprojection_speedup=report["warm_host_seconds"]["partial_gpu"]["median"] / report["warm_host_seconds"]["gpu"]["median"],
                  total_cpu_to_gpu_speedup=report["warm_host_seconds"]["cpu"]["median"] / report["warm_host_seconds"]["gpu"]["median"])
    if model == "am":
        report["scientific_gate"] = workflow_gate(results["gpu"], results["cpu"], labels)
        report["last_run_metrics"] = {key: result.run_metrics for key, result in results.items()}
    else:
        report["numerical_errors"] = errors(results["gpu"], results["cpu"])
        report["numerical_gate_pass"] = report["numerical_errors"]["normalized_max"] <= 2e-5
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--fbp-only", action="store_true")
    parser.add_argument("--output", type=Path, help="Write the JSON report as UTF-8")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    import torch
    if not torch.cuda.is_available():
        parser.error("a working CUDA-enabled PyTorch installation is required")
    torch.cuda.init()
    report = {"environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
                              "cpu": platform.processor(), "repeats": args.repeats,
                              "kernel": "NVRTC one-thread-per-pixel, CUDA FFT ramp filter"},
              "backprojection": [], "workflow": []}
    cases = [((96, 128), 64)]
    if not args.quick:
        cases += [((256, 384), 180), ((512, 512), 180)]
    for shape, angles in cases:
        for dtype in (np.float32, np.float64):
            report["backprojection"].append(fbp_case(shape, angles, dtype, args, torch))
    if not args.fbp_only:
        for model in ("ct", "am"):
            report["workflow"].append(workflow_case(model, args, torch))
    payload = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    if any(not row["numerical_gate_pass"] for row in report["backprojection"]) or any(
        not row.get("numerical_gate_pass", row.get("scientific_gate", {}).get("pass_gate", False))
        for row in report["workflow"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

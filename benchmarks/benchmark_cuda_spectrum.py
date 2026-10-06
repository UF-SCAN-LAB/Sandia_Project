"""Measure optional CUDA against the unchanged NumPy spectral implementation.

Run from any directory, for example:
    python benchmarks/benchmark_cuda_spectrum.py --repeats 5

JSON goes to stdout; progress goes to stderr. Run without other GPU tests or
benchmarks. No packages are installed and no system CUDA compiler is needed.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import gc
import json
import platform
from pathlib import Path
import statistics
import sys
from time import perf_counter

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "xrayrecon"))

from xrayrecon.alvarez_macovski import forward_spectrum, simulate_am_volume
from xrayrecon.am_acquisition import AMAcquisitionConfig
from xrayrecon.cuda_backend import CudaSpectrumEvaluator
from xrayrecon.materials import material_labels_to_zeff_rhoe
from xrayrecon.synthetic_registration import make_pcb_registration_showcase


# Initial full-showcase maximum differences: confidence 1.19e-7, residual
# 7.45e-8; material properties and labels matched exactly. The absolute gate
# allows roughly four times the observed maximum, plus a relative allowance
# for float32 output maps. Label equivalence is always required exactly.
RECOVERY_RTOL = 1e-6
RECOVERY_ATOL = 5e-7


def summary(samples):
    values = [float(value) for value in samples]
    return {
        "samples": values,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "population_stddev": statistics.pstdev(values),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
    }


def errors(actual, expected):
    delta = np.abs(np.asarray(actual, dtype=np.float64) - expected)
    return {
        "max_abs": float(np.max(delta, initial=0)),
        "rms": float(np.sqrt(np.mean(delta ** 2))) if delta.size else 0.0,
        "max_relative_above_1e-8": float(np.max(
            delta / np.maximum(np.abs(expected), 1e-8), initial=0
        )),
    }


def recovery_errors(actual, expected):
    report = {}
    for name in expected:
        report[name] = errors(actual[name], expected[name])
        report[name]["allclose_with_recovery_tolerance"] = bool(np.allclose(
            actual[name], expected[name], rtol=RECOVERY_RTOL, atol=RECOVERY_ATOL
        ))
    labels = "material_labels_recovered"
    report[labels]["disagreement_count"] = int(np.count_nonzero(actual[labels] != expected[labels]))
    return report


def timed(call, torch=None):
    if torch is not None:
        torch.cuda.synchronize()
    started = perf_counter()
    with redirect_stdout(sys.stderr):
        result = call()
    if torch is not None:
        torch.cuda.synchronize()
    return result, perf_counter() - started


def memory_begin(torch):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    return torch.cuda.memory_allocated()


def memory_end(torch, baseline):
    return {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_extra_allocated_bytes": int(torch.cuda.max_memory_allocated() - baseline),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def spectrum_case(count, spectra, batch_size, repeats, torch):
    print(f"Spectra: {count} rays x {len(spectra[0].energies_kev)} bins, {repeats} repeats", file=sys.stderr, flush=True)
    rng = np.random.default_rng(20260921 + count)
    a1 = np.exp(rng.uniform(np.log(1e-4), np.log(1500), count))
    a2 = rng.uniform(0, 30, count)

    def cpu_pair():
        # Two calls retain the pre-CUDA NumPy formula and allocation behavior.
        return tuple(forward_spectrum(a1, a2, spectrum) for spectrum in spectra)

    def gpu_pair():
        # Construction/table allocation and upload are intentionally timed.
        return CudaSpectrumEvaluator(spectra, batch_size=batch_size).evaluate(a1, a2)

    cpu_output, cpu_first = timed(cpu_pair)
    gpu_output, gpu_first = timed(gpu_pair, torch)
    cpu_times, gpu_times, peaks = [], [], []
    for repeat in range(repeats):
        # Alternate execution order to reduce persistent order bias.
        order = ("cpu", "cuda") if repeat % 2 == 0 else ("cuda", "cpu")
        for backend in order:
            if backend == "cpu":
                cpu_output, elapsed = timed(cpu_pair)
                cpu_times.append(elapsed)
            else:
                baseline = memory_begin(torch)
                gpu_output, elapsed = timed(gpu_pair, torch)
                gpu_times.append(elapsed)
                peaks.append(memory_end(torch, baseline))
    spectral_errors = [errors(actual, expected) for actual, expected in zip(gpu_output, cpu_output)]
    for actual, expected in zip(gpu_output, cpu_output):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    # Profile compute separately so CUDA event instrumentation is excluded
    # from the primary host-to-host wall-clock measurements above.
    evaluator = CudaSpectrumEvaluator(spectra, batch_size=batch_size)
    compute_ms, reused_host_seconds = [], []
    evaluator.evaluate(a1, a2)
    for _ in range(repeats):
        details = {}
        _, elapsed = timed(lambda: evaluator.evaluate(a1, a2, timings=details), torch)
        compute_ms.append(details["compute_ms"])
        reused_host_seconds.append(elapsed)
    del evaluator
    return {
        "rays": count,
        "energy_bins": [len(spectrum.energies_kev) for spectrum in spectra],
        "batch_size": batch_size,
        "first_cpu_seconds": cpu_first,
        "first_cuda_seconds_after_context_initialization": gpu_first,
        "cpu_host_to_host_seconds": summary(cpu_times),
        "cuda_host_to_host_seconds": summary(gpu_times),
        "cpu_to_cuda_median_speedup": statistics.median(cpu_times) / statistics.median(gpu_times),
        "cuda_compute_only_ms": summary(compute_ms),
        "cuda_cached_tables_instrumented_host_to_host_seconds": summary(reused_host_seconds),
        "gpu_memory": {name: max(row[name] for row in peaks) for name in peaks[0]},
        "spectral_errors": spectral_errors,
    }


def workflow_case(batch_size, repeats, torch):
    showcase = make_pcb_registration_showcase()
    labels = np.moveaxis(showcase.moving_labels_hwz, -1, 0)
    volume = material_labels_to_zeff_rhoe(labels)
    print(f"Complete AM: {volume.shape}, 64 angles, {repeats} repeats per backend", file=sys.stderr, flush=True)

    def run(backend):
        return simulate_am_volume(volume, num_angles=64, backend=backend, batch_size=batch_size)

    cpu, cpu_first = timed(lambda: run("cpu"))
    gpu, gpu_first = timed(lambda: run("cuda"), torch)
    cpu_times, gpu_times, peaks = [], [], []
    for repeat in range(repeats):
        order = ("cpu", "cuda") if repeat % 2 == 0 else ("cuda", "cpu")
        for backend in order:
            print(f"  AM repeat {repeat + 1}/{repeats}: {backend}", file=sys.stderr, flush=True)
            if backend == "cpu":
                cpu, elapsed = timed(lambda: run("cpu"))
                cpu_times.append(elapsed)
            else:
                baseline = memory_begin(torch)
                gpu, elapsed = timed(lambda: run("cuda"), torch)
                gpu_times.append(elapsed)
                peaks.append(memory_end(torch, baseline))
    return {
        "volume_shape": list(volume.shape),
        "num_angles": 64,
        "photon_count": None,
        "first_cpu_seconds": cpu_first,
        "first_cuda_seconds_after_context_initialization": gpu_first,
        "cpu_seconds": summary(cpu_times),
        "cuda_seconds": summary(gpu_times),
        "cpu_to_cuda_median_speedup": statistics.median(cpu_times) / statistics.median(gpu_times),
        "gpu_memory": {name: max(row[name] for row in peaks) for name in peaks[0]},
        "recovery_errors": recovery_errors(gpu, cpu),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--workflow-repeats", type=int, default=3)
    parser.add_argument("--rays", nargs="+", type=int, default=[128, 16384, 262144])
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--skip-workflow", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or args.workflow_repeats < 1 or min(args.rays) < 1 or args.batch_size < 1:
        parser.error("repeats, rays, and batch size must all be positive")
    config = AMAcquisitionConfig.industrial_default(num_angles=64)
    spectra = (config.low_spectrum, config.high_spectrum)

    # No torch imports or CUDA discovery precede this stopwatch. This includes
    # lazy dependency import, context setup, tables, input/output transfers,
    # allocation and actual synchronized execution for a 128-ray pair.
    print("Cold CUDA initialization and 128-ray paired evaluation", file=sys.stderr, flush=True)
    initial_a1 = np.linspace(0, 1000, 128)
    initial_a2 = np.linspace(0, 30, 128)
    started = perf_counter()
    evaluator = CudaSpectrumEvaluator(spectra, batch_size=args.batch_size)
    cold_output = evaluator.evaluate(initial_a1, initial_a2)
    cold_seconds = perf_counter() - started
    del evaluator
    import torch
    torch.cuda.synchronize()
    cold_errors = [errors(actual, forward_spectrum(initial_a1, initial_a2, spectrum))
                   for actual, spectrum in zip(cold_output, spectra)]
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    report = {
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "device_memory_bytes": int(properties.total_memory),
            "torch_cpu_threads": torch.get_num_threads(),
        },
        "method": {
            "spectral_wall_time": "Both spectra, fresh evaluator/tables per call; includes allocations, host/device copies and synchronization; NumPy inputs prepared before timing.",
            "compute_time": "Separate CUDA-event samples on a reusable evaluator measure arithmetic regions, including device idle gaps between host launches; not a sum of kernel profiler durations. Excludes explicit allocations and host/device transfers; event instrumentation is excluded from primary wall times.",
            "memory": "PyTorch allocator peak allocated/reserved bytes, including persistent tables; excludes CUDA context and driver allocations. Warm allocator caches remain enabled.",
            "order": "One unmeasured warm-up per backend/size, then alternating CPU/CUDA order; first calls reported separately.",
            "workflow": "Complete noiseless AM forward projection, spectral evaluation, LUT construction/decomposition, FBP, regularization and quality maps, with one evaluator reused inside each CUDA invocation.",
            "recovery_tolerance": {"rtol": RECOVERY_RTOL, "atol": RECOVERY_ATOL, "labels_must_match_exactly": True},
        },
        "cold_cuda_128_ray_pair_seconds": cold_seconds,
        "cold_spectral_errors": cold_errors,
        "spectra": [],
    }
    for count in args.rays:
        gc.collect()
        torch.cuda.empty_cache()
        report["spectra"].append(spectrum_case(count, spectra, args.batch_size, args.repeats, torch))
    if not args.skip_workflow:
        gc.collect()
        torch.cuda.empty_cache()
        report["complete_am"] = workflow_case(args.batch_size, args.workflow_repeats, torch)
        recovery = report["complete_am"]["recovery_errors"]
        report["complete_am"]["correctness_passed"] = (
            all(item["allclose_with_recovery_tolerance"] for item in recovery.values())
            and recovery["material_labels_recovered"]["disagreement_count"] == 0
        )
    print(json.dumps(report, indent=2))
    # Preserve the numerical/performance evidence even if the recovery gate
    # fails. A failing process status prevents a benchmark from implying parity.
    if not args.skip_workflow and not report["complete_am"]["correctness_passed"]:
        print("Complete AM correctness gate failed; inspect recovery_errors in JSON.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

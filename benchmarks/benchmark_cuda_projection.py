"""Host-to-host optional Radon and complete-AM benchmark; JSON stdout.

Run alone: python -B benchmarks/benchmark_cuda_projection.py --repeats 5
No packages are installed. CPU backprojection and CUDA spectra are fixed in
both workflow branches. --projector-only is usable before AM integration.
CUDA event intervals measure coordinate arithmetic, sampling, row reductions,
temporary allocation and dispatch gaps; they exclude explicit host transfers
and geometry setup, and are not a sum of isolated kernel execution times.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter

import numpy as np
from skimage.transform import radon
from scipy.ndimage import binary_erosion

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'xrayrecon'))
from xrayrecon.cuda_projection import CudaRadonProjector
from xrayrecon.alvarez_macovski import simulate_am_volume
from xrayrecon.materials import MATERIAL_NAMES, MATERIALS_DB, material_labels_to_zeff_rhoe
from xrayrecon.synthetic_registration import make_pcb_registration_showcase


def summary(values):
    return dict(samples=values, median=statistics.median(values), min=min(values), max=max(values),
                population_stddev=statistics.pstdev(values), p25=float(np.percentile(values, 25)), p75=float(np.percentile(values, 75)))


def errors(actual, reference):
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    delta = np.abs(a-b)
    scale = max(float(np.max(np.abs(b), initial=0)), 1e-12)
    near = np.abs(b) <= scale * 1e-8
    return dict(max_abs=float(np.max(delta, initial=0)), rmse=float(np.sqrt(np.mean(delta**2))) if delta.size else 0.0,
                normalized_max=float(np.max(delta, initial=0)/scale), reference_scale=scale,
                near_zero_count=int(near.sum()), near_zero_max_abs=float(np.max(delta[near], initial=0)))


def timed(call, torch):
    torch.cuda.synchronize()
    start = perf_counter()
    value = call()
    torch.cuda.synchronize()
    return value, perf_counter()-start


def paired_measure(cpu_call, gpu_call, repeats, torch):
    cpu, first_cpu = timed(cpu_call, torch)
    gpu, first_gpu = timed(gpu_call, torch)
    times = {'cpu': [], 'cuda': []}
    peaks = []
    for index in range(repeats):
        for name in (('cpu', 'cuda') if index % 2 == 0 else ('cuda', 'cpu')):
            if name == 'cuda':
                torch.cuda.reset_peak_memory_stats()
                before = torch.cuda.memory_allocated()
                gpu, duration = timed(gpu_call, torch)
                peaks.append(dict(peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                                  peak_extra_allocated_bytes=int(torch.cuda.max_memory_allocated()-before),
                                  peak_reserved_bytes=int(torch.cuda.max_memory_reserved())))
            else:
                cpu, duration = timed(cpu_call, torch)
            times[name].append(duration)
    return cpu, gpu, dict(first_cpu_seconds=first_cpu, first_cuda_seconds_after_context_initialization=first_gpu,
                          cpu_host_to_host_seconds=summary(times['cpu']), cuda_host_to_host_seconds=summary(times['cuda']),
                          median_speedup=statistics.median(times['cpu'])/statistics.median(times['cuda']),
                          gpu_memory={key: max(row[key] for row in peaks) for key in peaks[0]})


def projector_case(shape, angles, dtype, args, torch):
    print(f'Paired projection {shape}, {angles} angles, {np.dtype(dtype).name}', file=sys.stderr, flush=True)
    rng = np.random.default_rng(9182)
    images = rng.random((2, *shape)).astype(dtype)
    images[:, :shape[0]//8, :] = 0
    theta = np.linspace(0, 180, angles, endpoint=False)
    def cpu_call():
        return np.stack([radon(image, theta=theta, circle=False) for image in images])
    def gpu_call():
        return CudaRadonProjector(shape, theta, angle_batch_size=args.batch_size).project(images)
    cpu, gpu, report = paired_measure(cpu_call, gpu_call, args.repeats, torch)
    channel_errors = [errors(a, b) for a, b in zip(gpu, cpu)]
    limit = 2e-5 if dtype == np.float32 else 1e-11
    projector = CudaRadonProjector(shape, theta, angle_batch_size=args.batch_size)
    projector.project(images)
    compute, reused = [], []
    for _ in range(args.repeats):
        details = {}
        _, duration = timed(lambda: projector.project(images, timings=details), torch)
        compute.append(details['compute_ms'])
        reused.append(duration)
    report.update(shape=shape, channels=2, angles=angles, dtype=np.dtype(dtype).name, batch_size=args.batch_size,
                  numerical_errors=channel_errors, projection_gate_pass=all(row['normalized_max'] <= limit for row in channel_errors),
                  cuda_compute_only_ms=summary(compute), cuda_reused_geometry_host_seconds=summary(reused))
    return report


def recovery_errors(actual, reference, labels):
    foreground = labels != MATERIAL_NAMES.index('AIR')
    report = {}
    for key in reference:
        report[key] = dict(all=errors(actual[key], reference[key]), foreground=errors(actual[key][foreground], reference[key][foreground]),
                           background=errors(actual[key][~foreground], reference[key][~foreground]))
    recovered = 'material_labels_recovered'
    report[recovered]['disagreement_count'] = int(np.count_nonzero(actual[recovered] != reference[recovered]))
    report[recovered]['cpu_foreground_accuracy'] = float(np.mean(reference[recovered][foreground] == labels[foreground]))
    report[recovered]['cuda_foreground_accuracy'] = float(np.mean(actual[recovered][foreground] == labels[foreground]))
    return report


def workflow_gate(actual, reference, labels):
    """Apply the predeclared physical drift budget to trusted interiors.

    Thin structures without sixteen eroded voxels are reported as unevaluable,
    not silently counted as passes. Physical accuracy of the general AM model
    is covered separately by unchanged existing material/showcase contracts.
    """
    combined = np.zeros(labels.shape, dtype=bool)
    materials = []
    passed = all(np.isfinite(value).all() for value in actual.values())
    for material_id in np.unique(labels):
        if material_id == MATERIAL_NAMES.index('AIR'):
            continue
        interior = np.stack([binary_erosion(plane == material_id, iterations=2) for plane in labels])
        count = int(interior.sum())
        name = MATERIAL_NAMES[material_id]
        if count < 16:
            materials.append(dict(material=name, interior_voxels=count, skipped='fewer than 16 trusted interior voxels'))
            continue
        combined |= interior
        row = dict(material=name, interior_voxels=count, fields={})
        for key, truth in [('zeff_recovered', MATERIALS_DB[name].z_eff), ('rhoe_recovered', MATERIALS_DB[name].electron_density)]:
            cpu_median = float(np.median(reference[key][interior]))
            cuda_median = float(np.median(actual[key][interior]))
            drift = abs(cuda_median-cpu_median)/truth
            row['fields'][key] = dict(cpu_median=cpu_median, cuda_median=cuda_median, drift_over_truth=drift,
                                      gate_pass=drift <= 0.005)
            passed = passed and drift <= 0.005
        materials.append(row)
    if not combined.any():
        return dict(pass_gate=False, materials=materials, reason='no evaluable material interiors')
    key = 'material_labels_recovered'
    cpu_accuracy = float(np.mean(reference[key][combined] == labels[combined]))
    cuda_accuracy = float(np.mean(actual[key][combined] == labels[combined]))
    passed = passed and cuda_accuracy >= cpu_accuracy-0.005
    return dict(pass_gate=bool(passed), materials=materials, interior_voxels=int(combined.sum()),
                cpu_interior_accuracy=cpu_accuracy, cuda_interior_accuracy=cuda_accuracy,
                classification_deterioration=cpu_accuracy-cuda_accuracy,
                material_median_drift_budget=0.005, classification_deterioration_budget=0.005)


def workflow_case(larger, args, torch):
    if larger:
        labels = np.full((4, 192, 256), MATERIAL_NAMES.index('AIR'), dtype=np.int16)
        labels[:, 20:172, 23:231] = MATERIAL_NAMES.index('FR4')
        labels[:, 45:92, 52:104] = MATERIAL_NAMES.index('COPPER')
        labels[1:, 105:155, 143:217] = MATERIAL_NAMES.index('TIN')
        angles = 128
    else:
        labels = np.moveaxis(make_pcb_registration_showcase().moving_labels_hwz, -1, 0)
        angles = 64
    volume = material_labels_to_zeff_rhoe(labels)
    print(f'AM {labels.shape}, {angles} angles, CUDA spectra fixed', file=sys.stderr, flush=True)
    def run(backend):
        return simulate_am_volume(volume, num_angles=angles, backend='cuda', projection_backend=backend, projection_batch_size=args.batch_size)
    cpu, gpu, report = paired_measure(lambda: run('cpu'), lambda: run('cuda'), args.workflow_repeats, torch)
    report.update(shape=labels.shape, angles=angles, spectral_backend='cuda', recovery_errors=recovery_errors(gpu, cpu, labels),
                  scientific_gate=workflow_gate(gpu, cpu, labels),
                  finite_outputs=all(np.isfinite(value).all() for value in gpu.values()))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--workflow-repeats', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--projector-only', action='store_true')
    parser.add_argument('--workflow-only', action='store_true')
    parser.add_argument('--quick', action='store_true', help='Only representative projector/workflow size')
    args = parser.parse_args()
    if min(args.repeats, args.workflow_repeats, args.batch_size) < 1:
        parser.error('repeat and batch counts must be positive')
    start = perf_counter()
    import torch
    imported = perf_counter()-start
    start = perf_counter()
    cold = CudaRadonProjector((5, 7), [0, 37]).project(np.ones((2, 5, 7), dtype=np.float64))
    torch.cuda.synchronize()
    cold_seconds = perf_counter()-start
    assert np.isfinite(cold).all()
    report = dict(environment=dict(executable=sys.executable, python=platform.python_version(), torch=torch.__version__,
                                  cuda_runtime=torch.version.cuda, gpu=torch.cuda.get_device_name(), torch_import_seconds=imported,
                                  cold_context_and_first_projection_seconds=cold_seconds), projector=[], workflow=[])
    if not args.workflow_only:
        cases = [((96, 128), 64)]
        if not args.quick:
            cases += [((256, 384), 180), ((512, 512), 180)]
        for shape, angles in cases:
            for dtype in (np.float32, np.float64):
                report['projector'].append(projector_case(shape, angles, dtype, args, torch))
    if not args.projector_only:
        report['workflow'].append(workflow_case(False, args, torch))
        if not args.quick:
            report['workflow'].append(workflow_case(True, args, torch))
    print(json.dumps(report, indent=2))
    if any(not case['projection_gate_pass'] for case in report['projector']) or any(not case['scientific_gate']['pass_gate'] for case in report['workflow']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()

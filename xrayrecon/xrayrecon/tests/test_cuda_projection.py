"""Optional projector equivalence and physical recovery acceptance.

The projection envelope is a diagnostic prerequisite, not sufficient physical
acceptance: a 0.5% material-median drift budget consumes at most one tenth of
the established 5% noiseless recovery allowance. Existing physical contracts
must also pass unchanged. Near-zero rays use absolute, never relative, error.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from skimage.transform import radon

from xrayrecon import alvarez_macovski as am
from xrayrecon.cuda_projection import CudaRadonProjector
from xrayrecon.materials import MATERIAL_NAMES, MATERIALS_DB, material_labels_to_zeff_rhoe


def error_report(actual, reference):
    actual, reference = np.asarray(actual, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    delta = np.abs(actual - reference)
    scale = max(float(np.max(np.abs(reference), initial=0)), 1e-12)
    near = np.abs(reference) <= scale * 1e-8
    return dict(max_abs=float(np.max(delta, initial=0)), normalized_max=float(np.max(delta, initial=0) / scale),
                rmse=float(np.sqrt(np.mean(delta ** 2))) if delta.size else 0.0,
                near_zero_count=int(near.sum()), near_zero_max_abs=float(np.max(delta[near], initial=0)),
                reference_scale=scale)


def assert_projection(actual, reference):
    assert actual.shape == reference.shape
    assert actual.dtype == reference.dtype
    assert np.isfinite(actual).all()
    limit = 2e-5 if reference.dtype == np.float32 else 1e-11
    for channel in range(reference.shape[0]):
        report = error_report(actual[channel], reference[channel])
        assert report['normalized_max'] <= limit, report
        # Absolute error on zero rays is bounded in physical projection units.
        assert report['near_zero_max_abs'] <= limit * report['reference_scale'], report


@pytest.fixture
def cuda_torch():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA execution unavailable')
    assert torch.tensor([3.0], device='cuda', dtype=torch.float64).square().cpu().item() == 9
    return torch


def phantoms(shape, dtype):
    h, w = shape
    images = np.zeros((4, h, w), dtype=dtype)
    images[0, h // 3, 3 * w // 4] = 2.3
    images[0, 2 * h // 3, w // 4] += 0.7
    images[1, 1:max(2, h - 1), w // 3] = 0.8
    images[1, max(0, h - 2), w // 3:] = 1.9
    images[2, h // 3, :] = 3
    images[2, :, 2 * w // 3] += 0.2
    images[3, 0, 0], images[3, -1, -1] = 4, 1
    images[3, 0, -1], images[3, -1, 0] = 2, 0.4
    return images


@pytest.mark.cuda
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_isolated_geometry_and_boundary_matrix(cuda_torch, dtype):
    theta = np.array([0, 90, 180, 270, -37, 17.3, 89.9, 360, 17.3, -90, 143.2])
    reports = []
    for shape in [(5, 7), (6, 8), (7, 5), (8, 6), (5, 8), (8, 5), (1, 1), (1, 7), (6, 1), (31, 48), (192, 256), (511, 512), (512, 512)]:
        images = phantoms(shape, dtype)
        expected = np.stack([radon(i, theta, circle=False) for i in images])
        projector = CudaRadonProjector(shape, theta, angle_batch_size=3)
        assert str(projector.device).startswith('cuda')
        actual = projector.project(images)
        assert_projection(actual, expected)
        # Independent mass checks catch gross boundary loss and wrapping.
        np.testing.assert_allclose(actual.sum(axis=1), expected.sum(axis=1), rtol=2e-5 if dtype == np.float32 else 1e-11, atol=2e-5 if dtype == np.float32 else 1e-11)
        reports.append(dict(shape=shape, **error_report(actual, expected),
                            channels=[error_report(a, b) for a, b in zip(actual, expected)]))
    print(json.dumps(dict(dtype=np.dtype(dtype).name, geometry_errors=reports)))


@pytest.mark.cuda
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_isolated_cardinal_orientation_no_shift(cuda_torch, dtype):
    image = np.zeros((1, 5, 7), dtype=dtype)
    image[0, 1, 5] = 1
    actual = CudaRadonProjector((5, 7), [0, 90, 180, 270]).project(image)
    assert actual.argmax(axis=1).tolist() == [[7, 6, 3, 4]]
    np.testing.assert_allclose(actual.sum(axis=1), 1, atol=2e-6 if dtype == np.float32 else 1e-12)


@pytest.mark.cuda
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_isolated_batches_noncontiguous_readonly_and_zero(cuda_torch, dtype):
    source = phantoms((13, 34), dtype)
    images = source[:, :, ::-2]
    images.setflags(write=False)
    original = source.copy()
    theta = np.linspace(-180, 181, 17)
    expected = np.stack([radon(i, theta, circle=False) for i in images])
    first = None
    for batch in (1, 2, 8, 16, 17, 18):
        projector = CudaRadonProjector(images.shape[1:], theta, angle_batch_size=batch)
        actual = projector.project(images)
        assert_projection(actual, expected)
        if first is None:
            first = actual
        else:
            np.testing.assert_array_equal(actual, first)
        zeros = projector.project(np.zeros_like(images))
        np.testing.assert_array_equal(zeros, 0)
        near = projector.project(images * dtype(1e-14))
        assert_projection(near, np.stack([radon(i * dtype(1e-14), theta, circle=False) for i in images]))
    np.testing.assert_array_equal(source, original)


@pytest.mark.cuda
def test_isolated_observes_device_execution_and_timings(cuda_torch, monkeypatch):
    # Instrument a real GPU reduction used by either grid_sample or direct
    # interpolation. A CPU-returning fake backend cannot satisfy this check.
    devices = []
    original = cuda_torch.Tensor.sum
    def observed(tensor, *args, **kwargs):
        devices.append(tensor.device.type)
        return original(tensor, *args, **kwargs)
    monkeypatch.setattr(cuda_torch.Tensor, 'sum', observed)
    details = {}
    result = CudaRadonProjector((5, 7), [0, 31]).project(phantoms((5, 7), np.float64), timings=details)
    assert isinstance(result, np.ndarray)
    assert 'cuda' in devices
    assert details['compute_ms'] > 0


@pytest.mark.parametrize('shape,angles,batch', [((0, 3), [0], 1), ((3,), [0], 1), ((2.5, 3), [0], 1), ((3, 3), [], 1), ((3, 3), [[0]], 1), ((3, 3), [np.nan], 1), ((3, 3), [np.inf], 1), ((3, 3), [0], 0), ((3, 3), [0], 1.5)])
def test_isolated_invalid_geometry(shape, angles, batch):
    with pytest.raises(ValueError):
        CudaRadonProjector(shape, angles, angle_batch_size=batch)


@pytest.mark.cuda
def test_isolated_invalid_images(cuda_torch):
    projector = CudaRadonProjector((3, 4), [0])
    for value in [np.zeros((3, 4)), np.zeros((0, 3, 4)), np.zeros((1, 4, 3)), np.full((1, 3, 4), np.nan), np.full((1, 3, 4), np.inf)]:
        with pytest.raises(ValueError):
            projector.project(value)
    for dtype in (np.int32, np.float16, np.complex128):
        with pytest.raises(TypeError):
            projector.project(np.ones((1, 3, 4), dtype=dtype))


def test_isolated_lazy_import_and_missing_torch():
    code = '''
import sys, importlib.abc
sys.path.insert(0, sys.argv[1])
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ImportError("blocked optional torch")
sys.meta_path.insert(0, NoTorch())
from xrayrecon.cuda_projection import CudaRadonProjector
assert "torch" not in sys.modules
try:
    CudaRadonProjector((3, 4), [0])
except RuntimeError as exc:
    assert "cuda" in str(exc).lower() or "torch" in str(exc).lower()
else:
    raise AssertionError("explicit CUDA silently fell back")
'''
    subprocess.run([sys.executable, '-B', '-c', code, str(Path(__file__).resolve().parents[2])], check=True)


def test_isolated_unavailable_device_fails_explicitly(monkeypatch):
    torch = pytest.importorskip('torch')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='CUDA projection backend is unavailable'):
        CudaRadonProjector((3, 4), [0])


@pytest.mark.cuda
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_isolated_signed_input_does_not_clip(cuda_torch, dtype):
    images = phantoms((11, 17), dtype) - dtype(0.7)
    theta = np.array([0, -17, 90, 132.6])
    expected = np.stack([radon(image, theta, circle=False) for image in images])
    actual = CudaRadonProjector((11, 17), theta).project(images)
    assert_projection(actual, expected)
    assert actual.min() < 0


def _patch_projector(monkeypatch):
    cache = {}
    def project(image, theta, circle=False):
        assert circle is False
        key = (image.shape, tuple(theta))
        if key not in cache:
            cache[key] = CudaRadonProjector(image.shape, theta, angle_batch_size=8)
        return cache[key].project(np.asarray(image)[None])[0]
    monkeypatch.setattr(am, 'radon', project)
    return cache


@pytest.mark.cuda
@pytest.mark.parametrize('material', ['FR4', 'ALUMINA', 'ALUMINUM', 'COPPER', 'TIN', 'TUNGSTEN', 'GOLD'])
def test_isolated_downstream_material_budget(cuda_torch, monkeypatch, material):
    labels = np.full((1, 40, 40), MATERIAL_NAMES.index('AIR'), dtype=np.int16)
    labels[:, 8:32, 8:32] = MATERIAL_NAMES.index(material)
    volume = material_labels_to_zeff_rhoe(labels)
    cpu = am.simulate_am_volume(volume, num_angles=64, backend='cuda')
    _patch_projector(monkeypatch)
    gpu = am.simulate_am_volume(volume, num_angles=64, backend='cuda')
    report = dict(material=material)
    for key, truth in [('zeff_recovered', MATERIALS_DB[material].z_eff), ('rhoe_recovered', MATERIALS_DB[material].electron_density)]:
        region = (0, slice(12, 28), slice(12, 28))
        a, b = float(np.median(gpu[key][region])), float(np.median(cpu[key][region]))
        assert abs(a - truth) / truth <= 0.05
        assert abs(b - truth) / truth <= 0.05
        assert abs(a - b) / truth <= 0.005
        report[key] = dict(cpu_median=b, cuda_median=a, drift_over_truth=abs(a-b)/truth)
    print(json.dumps(report))


@pytest.mark.cuda
def test_isolated_existing_material_and_quality_contracts(cuda_torch, monkeypatch):
    path = Path(__file__).with_name('test_alvarez_macovski_showcase.py')
    spec = importlib.util.spec_from_file_location('_projection_existing_contracts', path)
    contracts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contracts)
    _patch_projector(monkeypatch)
    monkeypatch.setattr(contracts, 'simulate_am_volume', lambda *args, **kwargs: am.simulate_am_volume(*args, **kwargs, backend='cuda'))
    for material in ('TIN', 'TUNGSTEN', 'GOLD'):
        contracts.test_noisy_high_z_materials_meet_approved_error_thresholds(material)
    contracts.test_showcase_slice_meets_approved_am_recovery_contract()
    contracts.test_noisy_showcase_meets_density_classification_and_confidence_contract()
    contracts.test_reduced_angle_showcase_retains_interpretable_material_accuracy()
    contracts.test_noisy_am_recovery_is_seeded_and_reports_auditable_quality_maps()


@pytest.mark.cuda
@pytest.mark.parametrize('noisy', [False, True])
def test_isolated_showcase_map_comparison(cuda_torch, monkeypatch, noisy):
    from scipy.ndimage import binary_erosion
    from xrayrecon.synthetic_registration import make_pcb_registration_showcase
    labels = make_pcb_registration_showcase().moving_labels_hwz[::2, 16:112:2, 19]
    volume = material_labels_to_zeff_rhoe(labels[None])
    kwargs = dict(num_angles=96, backend='cuda')
    if noisy:
        kwargs.update(photon_count=2000000, random_seed=71)
    cpu = am.simulate_am_volume(volume, **kwargs)
    _patch_projector(monkeypatch)
    gpu = am.simulate_am_volume(volume, **kwargs)
    foreground = labels != MATERIAL_NAMES.index('AIR')
    interior = np.zeros(labels.shape, dtype=bool)
    for material in np.unique(labels):
        if material != MATERIAL_NAMES.index('AIR'):
            interior |= binary_erosion(labels == material, iterations=2)
    report = dict(noisy=noisy, fields={}, comparison=(
        'CPU Poisson sampling retained; slightly different means may change random sampling branches, so cross-backend voxel equality is not expected'
        if noisy else 'Noiseless projector numerical propagation with spectral CUDA fixed'))
    for key in cpu:
        a, b = gpu[key][0], cpu[key][0]
        assert a.shape == b.shape
        assert np.isfinite(a).all()
        report['fields'][key] = dict(all=error_report(a, b), foreground=error_report(a[foreground], b[foreground]),
                                    background=error_report(a[~foreground], b[~foreground]))
    predicted = 'material_labels_recovered'
    cpu_accuracy = float(np.mean(cpu[predicted][0][interior] == labels[interior]))
    gpu_accuracy = float(np.mean(gpu[predicted][0][interior] == labels[interior]))
    report.update(cpu_interior_accuracy=cpu_accuracy, cuda_interior_accuracy=gpu_accuracy,
                  label_disagreements=int(np.count_nonzero(cpu[predicted] != gpu[predicted])))
    print(json.dumps(report))
    # Classification is discrete and particularly sensitive near boundaries;
    # permit at most 0.5 percentage points of deterioration in trusted interiors.
    assert gpu_accuracy >= cpu_accuracy - 0.005


@pytest.mark.cuda
@pytest.mark.parametrize('projection_backend', ['cpu', 'cuda'])
def test_integration_seeded_noise_and_separable_api(cuda_torch, projection_backend):
    labels = np.full((1, 24, 32), MATERIAL_NAMES.index('AIR'), dtype=np.int16)
    labels[:, 4:20, 5:27] = MATERIAL_NAMES.index('FR4')
    labels[:, 9:16, 12:21] = MATERIAL_NAMES.index('COPPER')
    volume = material_labels_to_zeff_rhoe(labels)
    kwargs = dict(num_angles=32, backend='cuda', projection_backend=projection_backend, projection_batch_size=3, photon_count=200000, random_seed=71)
    first = am.simulate_am_volume(volume, **kwargs)
    second = am.simulate_am_volume(volume, **kwargs)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    from xrayrecon.am_acquisition import AMAcquisitionConfig
    config = AMAcquisitionConfig.industrial_default(num_angles=32, photon_count=200000, random_seed=71)
    projections = am.simulate_am_projections(volume, config=config, backend='cuda', projection_backend=projection_backend, projection_batch_size=3)
    separate = am.reconstruct_am_projections(projections, output_shape=labels.shape[1:], backend='cuda', projection_backend=projection_backend, projection_batch_size=3,
        zeff_clip=float(volume[..., 0].max()) * 1.5, rhoe_clip=float(volume[..., 1].max()) * 1.5)
    for key in separate:
        np.testing.assert_array_equal(first[key], separate[key])


def test_integration_cpu_without_optional_import():
    code = '''
import sys, importlib.abc, numpy as np
sys.path.insert(0, sys.argv[1])
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise AssertionError("CPU path imported torch")
sys.meta_path.insert(0, NoTorch())
from xrayrecon.alvarez_macovski import simulate_am_volume
v = np.zeros((1, 8, 10, 2), dtype=np.float32)
v[:, 2:6, 2:8] = [7, 1]
simulate_am_volume(v, num_angles=8)
assert "torch" not in sys.modules
try:
    simulate_am_volume(v, num_angles=8, projection_backend="bad")
except ValueError:
    pass
else:
    raise AssertionError("invalid backend accepted")
'''
    subprocess.run([sys.executable, '-B', '-c', code, str(Path(__file__).resolve().parents[2])], check=True)


@pytest.mark.cuda
def test_integration_reuses_geometry_and_independent_spectral_cpu(cuda_torch, monkeypatch):
    calls, projected_dtypes = [], []
    original_init, original_project = CudaRadonProjector.__init__, CudaRadonProjector.project
    def observed_init(self, *args, **kwargs):
        calls.append((args, kwargs))
        original_init(self, *args, **kwargs)
    def observed_project(self, images, **kwargs):
        projected_dtypes.append(images.dtype)
        return original_project(self, images, **kwargs)
    monkeypatch.setattr(CudaRadonProjector, '__init__', observed_init)
    monkeypatch.setattr(CudaRadonProjector, 'project', observed_project)
    # Spectral CUDA evaluator must never be requested in this independent mode.
    from xrayrecon.cuda_backend import CudaSpectrumEvaluator
    def forbidden(*args, **kwargs):
        raise AssertionError('spectral CUDA selected despite backend=cpu')
    monkeypatch.setattr(CudaSpectrumEvaluator, '__init__', forbidden)
    volume = np.zeros((2, 12, 16, 2), dtype=np.float32)
    volume[:, 3:9, 4:12] = (7, 1)
    result = am.simulate_am_volume(volume, num_angles=12, backend='cpu', projection_backend='cuda', projection_batch_size=3)
    assert len(calls) == 1
    assert projected_dtypes.count(np.dtype(np.float32)) == 2
    assert projected_dtypes.count(np.dtype(np.float64)) == 2
    assert np.isfinite(result['confidence']).all()


@pytest.mark.cuda
def test_integration_explicit_cuda_execution_error_propagates(cuda_torch, monkeypatch):
    def unavailable(*args, **kwargs):
        raise RuntimeError('CUDA projection deliberately unavailable')
    monkeypatch.setattr(CudaRadonProjector, 'project', unavailable)
    volume = np.zeros((1, 8, 10, 2), dtype=np.float32)
    volume[:, 2:6, 2:8] = (7, 1)
    with pytest.raises(RuntimeError, match='deliberately unavailable'):
        am.simulate_am_volume(volume, num_angles=8, projection_backend='cuda')


@pytest.mark.cuda
def test_integration_existing_physical_contracts(cuda_torch, monkeypatch):
    # Re-run the unchanged physical assertions through the actual integrated
    # two-channel path, not the stage-one one-channel radon substitution.
    spec = importlib.util.spec_from_file_location('_projection_integrated_contracts', Path(__file__).with_name('test_alvarez_macovski_showcase.py'))
    contracts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contracts)
    monkeypatch.setattr(contracts, 'simulate_am_volume', lambda *args, **kwargs: am.simulate_am_volume(*args, **kwargs, backend='cuda', projection_backend='cuda'))
    for material in ('FR4', 'ALUMINA', 'ALUMINUM', 'COPPER', 'TIN', 'TUNGSTEN', 'GOLD'):
        contracts.test_noiseless_material_medians_recover_within_five_percent(material)
    for material in ('TIN', 'TUNGSTEN', 'GOLD'):
        contracts.test_noisy_high_z_materials_meet_approved_error_thresholds(material)
    contracts.test_showcase_slice_meets_approved_am_recovery_contract()
    contracts.test_noisy_showcase_meets_density_classification_and_confidence_contract()
    contracts.test_reduced_angle_showcase_retains_interpretable_material_accuracy()

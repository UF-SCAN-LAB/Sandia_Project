"""Optional CUDA spectral contract; CPU-only checks never require torch."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest

from xrayrecon import alvarez_macovski as am
from xrayrecon.am_acquisition import AMAcquisitionConfig, AMSpectrum
from xrayrecon.materials import MATERIAL_NAMES, material_labels_to_zeff_rhoe


# Float64 reduction order differs between NumPy and CUDA. On the initial RTX
# 4060/PyTorch 2.3.1 run the largest edge-case error was 3.55e-15; 1e-12 leaves
# room for different CUDA reduction trees without accepting float32 arithmetic.
# The downstream gate is separate: LUT triangulation can amplify much smaller
# projection errors. Small-volume recovery and the full 24x96x128 showcase's
# material properties/labels were bitwise equal. Full-showcase residual and
# confidence maxima were 7.45e-8 and 1.19e-7: atol=5e-7 permits about four times
# that observed maximum, plus a 1e-6 relative allowance across float32 maps.
# These are CPU/GPU equivalence tolerances, not material-accuracy guarantees.
SPECTRAL_RTOL = 1e-12
SPECTRAL_ATOL = 1e-12
RECOVERY_RTOL = 1e-6
RECOVERY_ATOL = 5e-7


def _spectra():
    config = AMAcquisitionConfig.industrial_default(num_angles=16)
    return config.low_spectrum, config.high_spectrum


def _volume():
    labels = np.full((2, 24, 32), MATERIAL_NAMES.index("AIR"), dtype=np.int16)
    labels[:, 3:21, 4:28] = MATERIAL_NAMES.index("FR4")
    labels[:, 8:16, 10:20] = MATERIAL_NAMES.index("COPPER")
    labels[1, 12:19, 21:27] = MATERIAL_NAMES.index("TIN")
    return material_labels_to_zeff_rhoe(labels)


def _report_errors(label, actual, expected):
    delta = np.abs(np.asarray(actual, dtype=np.float64) - expected)
    report = {
        "case": label,
        "max_abs": float(np.max(delta, initial=0)),
        "rms": float(np.sqrt(np.mean(delta ** 2))) if delta.size else 0.0,
    }
    print(json.dumps(report, sort_keys=True))


@pytest.fixture
def cuda_torch():
    torch = pytest.importorskip("torch", reason="optional torch is not installed")
    if not torch.cuda.is_available():
        pytest.skip("torch.cuda.is_available() is false")
    # A real synchronized operation, rather than relying on driver discovery.
    assert torch.tensor([2.0], device="cuda", dtype=torch.float64).square().cpu().item() == 4
    return torch


@pytest.mark.unit
def test_default_cpu_spectrum_preserves_original_formula():
    a1 = np.linspace(0, 1000, 34).reshape(2, 17)[:, ::2]
    a2 = np.linspace(0, 25, 34).reshape(2, 17)[:, ::2]
    for spectrum in _spectra():
        energies = np.asarray(spectrum.energies_kev, dtype=float)
        exponent = -np.outer(a1.ravel(), am.f1(energies)) - np.outer(a2.ravel(), am.f2(energies))
        expected = -np.log(np.clip(np.sum(np.asarray(spectrum.weights) * np.exp(exponent), axis=1), 1e-12, None))
        np.testing.assert_array_equal(am.forward_spectrum(a1, a2, spectrum), expected)
    pair = am.forward_spectra(a1, a2, _spectra())
    assert isinstance(pair, tuple)
    for actual, spectrum in zip(pair, _spectra()):
        np.testing.assert_array_equal(actual, am.forward_spectrum(a1, a2, spectrum))


@pytest.mark.unit
def test_cpu_pipeline_and_backend_module_import_without_torch():
    # A fresh interpreter proves the CPU path does not merely reuse an already
    # imported torch module. Block every optional torch import deliberately.
    code = textwrap.dedent("""
        import importlib.abc
        import sys
        class NoTorch(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == 'torch' or fullname.startswith('torch.'):
                    raise AssertionError('CPU execution attempted to import torch')
        sys.meta_path.insert(0, NoTorch())
        sys.path.insert(0, sys.argv[1])
        import numpy as np
        import xrayrecon.cuda_backend
        from xrayrecon.alvarez_macovski import forward_spectra, simulate_am_volume
        from xrayrecon.am_acquisition import AMAcquisitionConfig
        from xrayrecon.materials import MATERIAL_NAMES, material_labels_to_zeff_rhoe
        cfg = AMAcquisitionConfig.industrial_default(num_angles=12)
        pair = forward_spectra([0., 1.], [0., 2.], (cfg.low_spectrum, cfg.high_spectrum))
        assert all(x.dtype == np.float64 for x in pair)
        labels = np.full((1, 12, 16), MATERIAL_NAMES.index('FR4'), dtype=np.int16)
        result = simulate_am_volume(material_labels_to_zeff_rhoe(labels), config=cfg)
        assert np.isfinite(result['zeff_recovered']).all()
        assert not any(name == 'torch' or name.startswith('torch.') for name in sys.modules)
    """)
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, str(Path(__file__).resolve().parents[2])],
        capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.unit
def test_explicit_cuda_fails_when_torch_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(RuntimeError, match="(?i)(torch|cuda)"):
        am.forward_spectrum([0.], [0.], _spectra()[0], backend="cuda")


@pytest.mark.unit
def test_explicit_cuda_fails_when_device_is_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    with pytest.raises(RuntimeError, match="(?i)cuda"):
        am.forward_spectrum([0.], [0.], _spectra()[0], backend="cuda")


@pytest.mark.unit
def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="(?i)backend"):
        am.forward_spectrum([0.], [0.], _spectra()[0], backend="automatic")


@pytest.mark.cuda
@pytest.mark.parametrize("count", [0, 1, 3, 4, 5, 9])
def test_cuda_pair_batch_boundaries_and_noncontiguous_inputs(cuda_torch, count):
    from xrayrecon.cuda_backend import CudaSpectrumEvaluator

    storage1 = np.linspace(0, 1000, max(2 * count, 1), dtype=np.float64)
    storage2 = np.linspace(0, 30, max(2 * count, 1), dtype=np.float64)
    a1, a2 = storage1[:2 * count:2], storage2[:2 * count:2]
    before1, before2 = storage1.copy(), storage2.copy()
    evaluator = CudaSpectrumEvaluator(_spectra(), batch_size=4)
    assert cuda_torch.device(evaluator.device).type == "cuda"
    timings = {}
    actual = evaluator.evaluate(a1, a2, timings=timings)
    assert isinstance(actual, tuple) and len(actual) == 2
    assert timings["compute_ms"] >= 0
    for spectrum, output in zip(_spectra(), actual):
        expected = am.forward_spectrum(a1, a2, spectrum)
        assert output.shape == (count,) and output.dtype == np.float64
        _report_errors(f"batch-{count}-{spectrum.label}", output, expected)
        np.testing.assert_allclose(output, expected, rtol=SPECTRAL_RTOL, atol=SPECTRAL_ATOL)
    np.testing.assert_array_equal(storage1, before1)
    np.testing.assert_array_equal(storage2, before2)


@pytest.mark.cuda
def test_cuda_custom_spectra_zero_weights_and_transmission_floor(cuda_torch):
    spectra = (
        AMSpectrum((15., 27., 51., 123., 401.), (0., 1., 0., 3., 2.), "custom-low"),
        AMSpectrum((20., 45., 95.), (0., 0., 1.), "single-weight-high"),
    )
    a1 = np.array([[0., 1e-12, 0.5], [30., 1e4, 1e8]], dtype=np.float64).T
    a2 = np.array([[0., 1e-12, 0.1], [3., 1e4, 1e8]], dtype=np.float64).T
    outputs = am.forward_spectra(a1, a2, spectra, backend="cuda", batch_size=2)
    for spectrum, actual in zip(spectra, outputs):
        expected = am.forward_spectrum(a1, a2, spectrum)
        _report_errors(spectrum.label, actual, expected)
        np.testing.assert_allclose(actual, expected, rtol=SPECTRAL_RTOL, atol=SPECTRAL_ATOL)
        assert actual[-1] == pytest.approx(-np.log(1e-12), abs=SPECTRAL_ATOL)
        assert abs(actual[0]) <= SPECTRAL_ATOL


@pytest.mark.cuda
@pytest.mark.parametrize("seed", [17, 71, 2026])
def test_dense_cuda_spectra_over_broad_float32_input_range(cuda_torch, seed):
    rng = np.random.default_rng(seed)
    count = 8193  # Two full default batches and one trailing ray.
    a1 = np.exp(rng.uniform(np.log(1e-7), np.log(1e5), count)).astype(np.float32)
    a2 = np.exp(rng.uniform(np.log(1e-7), np.log(1e3), count)).astype(np.float32)
    a1[::100] = 0
    a2[::100] = 0
    before1, before2 = a1.copy(), a2.copy()
    weights = rng.uniform(0, 1, 37)
    weights[::7] = 0
    custom = AMSpectrum(tuple(np.geomspace(20, 400, 37)), tuple(weights), "custom-37-bin")
    spectra = (*_spectra(), custom)
    actual = am.forward_spectra(a1, a2, spectra, backend="cuda", batch_size=4096)
    for spectrum, output in zip(spectra, actual):
        expected = am.forward_spectrum(a1, a2, spectrum)
        assert output.dtype == np.float64 and output.shape == (count,)
        _report_errors(f"dense-{seed}-{spectrum.label}", output, expected)
        np.testing.assert_allclose(output, expected, rtol=SPECTRAL_RTOL, atol=SPECTRAL_ATOL)
    np.testing.assert_array_equal(a1, before1)
    np.testing.assert_array_equal(a2, before2)


@pytest.mark.cuda
def test_cuda_copies_actual_float64_device_results(cuda_torch, monkeypatch):
    original_cpu = cuda_torch.Tensor.cpu
    observed = []

    def record_cpu(tensor, *args, **kwargs):
        if tensor.is_cuda:
            observed.append((tensor.dtype, tensor.numel()))
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(cuda_torch.Tensor, "cpu", record_cpu)
    output = am.forward_spectrum(np.arange(9), np.arange(9), _spectra()[0], backend="cuda", batch_size=4)
    assert output.shape == (9,)
    assert observed, "CUDA selection returned without observing any device-to-host result copy"
    assert all(dtype == cuda_torch.float64 for dtype, _ in observed)


@pytest.mark.cuda
@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_cuda_rejects_invalid_batch_size(cuda_torch, batch_size):
    from xrayrecon.cuda_backend import CudaSpectrumEvaluator
    with pytest.raises(ValueError, match="batch"):
        CudaSpectrumEvaluator(_spectra(), batch_size=batch_size)


@pytest.mark.cuda
def test_cuda_rejects_mismatched_rays_and_empty_spectra(cuda_torch):
    from xrayrecon.cuda_backend import CudaSpectrumEvaluator
    evaluator = CudaSpectrumEvaluator(_spectra())
    with pytest.raises(ValueError):
        evaluator.evaluate([1., 2.], [1.])
    with pytest.raises(ValueError):
        CudaSpectrumEvaluator(())


@pytest.mark.cuda
@pytest.mark.parametrize("custom", [False, True])
def test_complete_am_recovery_and_projection_parity(cuda_torch, custom):
    config = AMAcquisitionConfig.industrial_default(num_angles=24)
    if custom:
        config = AMAcquisitionConfig(
            AMSpectrum((60., 73., 95., 130., 160.), (1., 3., 0., 2., 1.), "custom-low"),
            AMSpectrum((60., 90., 140., 210., 320.), (1., 2., 0., 3., 1.), "custom-high"),
            num_angles=24,
        )
    volume = _volume()
    before = volume.copy()
    cpu_projections = am.simulate_am_projections(volume, config)
    gpu_projections = am.simulate_am_projections(volume, config, backend="cuda", batch_size=127)
    for name in ("low_log", "high_log"):
        actual, expected = getattr(gpu_projections, name), getattr(cpu_projections, name)
        _report_errors(f"projections-{custom}-{name}", actual, expected)
        np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=2e-7)
    expected = am.simulate_am_volume(volume, config=config)
    actual = am.simulate_am_volume(volume, config=config, backend="cuda", batch_size=127)
    assert actual.keys() == expected.keys()
    assert actual.acquisition_config == expected.acquisition_config
    assert actual.has_ground_truth == expected.has_ground_truth
    for name in expected:
        assert actual[name].dtype == expected[name].dtype
        _report_errors(f"recovery-{custom}-{name}", actual[name], expected[name])
        if name == "material_labels_recovered":
            np.testing.assert_array_equal(actual[name], expected[name])
        else:
            np.testing.assert_allclose(actual[name], expected[name], rtol=RECOVERY_RTOL, atol=RECOVERY_ATOL)
    np.testing.assert_array_equal(volume, before)


@pytest.mark.cuda
def test_public_cuda_reconstruction_accepts_cpu_generated_projections(cuda_torch):
    config = AMAcquisitionConfig.industrial_default(num_angles=16)
    volume = _volume()
    projections = am.simulate_am_projections(volume, config)
    expected = am.reconstruct_am_projections(projections, output_shape=volume.shape[1:3])
    actual = am.reconstruct_am_projections(
        projections, output_shape=volume.shape[1:3], backend="cuda", batch_size=257
    )
    for name in expected:
        _report_errors(f"public-reconstruction-{name}", actual[name], expected[name])
        if name == "material_labels_recovered":
            np.testing.assert_array_equal(actual[name], expected[name])
        else:
            np.testing.assert_allclose(actual[name], expected[name], rtol=RECOVERY_RTOL, atol=RECOVERY_ATOL)


@pytest.mark.parametrize("backend", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
def test_seeded_noise_remains_on_cpu_and_repeats_within_backend(request, monkeypatch, backend):
    if backend == "cuda":
        request.getfixturevalue("cuda_torch")
    original = am._poisson_log_measurement
    calls = []

    def observe_noise(line_integral, photon_count, rng):
        assert isinstance(line_integral, np.ndarray)
        assert isinstance(rng, np.random.Generator)
        calls.append(line_integral.shape)
        return original(line_integral, photon_count, rng)

    monkeypatch.setattr(am, "_poisson_log_measurement", observe_noise)
    options = dict(num_angles=16, photon_count=200_000, random_seed=71, backend=backend)
    first = am.simulate_am_volume(_volume(), **options)
    second = am.simulate_am_volume(_volume(), **options)
    assert len(calls) == 8  # two spectra x two slices x two complete runs
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from skimage.transform import iradon, radon

from xrayrecon.cuda_backprojection import CudaBackprojector
from xrayrecon.backprojection import make_backprojector


@pytest.fixture
def cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA execution unavailable")
    return torch


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("shape", [(5, 8), (16, 16), (17, 24), (32, 21)])
@pytest.mark.parametrize("filter_name", ["ramp", None])
def test_cuda_backprojection_matches_iradon(cuda, dtype, shape, filter_name):
    images = np.random.default_rng(71).normal(size=(2, *shape)).astype(dtype)
    images[:, 0, -1] = 8  # Include a strong feature outside the inscribed circle.
    theta = np.array([0, 3.7, 37, 90, 120, 177, 37], dtype=np.float64)
    sinograms = np.stack([radon(image, theta=theta, circle=False) for image in images])
    expected = np.stack([iradon(s, theta=theta, filter_name=filter_name, circle=False) for s in sinograms])
    projector = CudaBackprojector(sinograms.shape[1], theta)
    timings = {}
    actual = projector.reconstruct(sinograms, filter_name=filter_name, timings=timings)
    assert actual.dtype == dtype
    assert actual.shape == expected.shape
    assert timings["compute_ms"] > 0
    tolerance = 3e-6 if dtype == np.float32 else 2e-13
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.cuda
def test_readonly_noncontiguous_input_and_nondefault_stream(cuda):
    storage = np.random.default_rng(10).random((2, 19, 22)).astype(np.float64)
    sinograms = storage[:, :, ::2]
    before = storage.copy()
    sinograms.flags.writeable = False
    theta = np.linspace(-20, 190, 11)
    with cuda.cuda.stream(cuda.cuda.Stream()):
        projector = CudaBackprojector(19, theta, output_size=23)
        actual = projector.reconstruct(sinograms)
    expected = np.stack([iradon(s, theta=theta, output_size=23, circle=False) for s in sinograms])
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)
    np.testing.assert_array_equal(storage, before)
    assert actual.flags.writeable


@pytest.mark.parametrize("detectors,theta,side", [(0, [0], None), (True, [0], None), (8, [], None), (8, [np.nan], None), (8, [[0]], None), (8, [0], 0)])
def test_invalid_geometry_rejected_before_cuda_import(detectors, theta, side):
    with pytest.raises(ValueError):
        CudaBackprojector(detectors, theta, output_size=side)


@pytest.mark.cuda
def test_invalid_sinograms_and_filter_rejected(cuda):
    projector = CudaBackprojector(8, [0, 90])
    with pytest.raises(ValueError, match="shape"):
        projector.reconstruct(np.ones((8, 2)))
    with pytest.raises(TypeError, match="dtype"):
        projector.reconstruct(np.ones((1, 8, 2), dtype=np.int32))
    with pytest.raises(ValueError, match="finite"):
        projector.reconstruct(np.full((1, 8, 2), np.nan))
    with pytest.raises(ValueError, match="filter"):
        projector.reconstruct(np.ones((1, 8, 2)), filter_name="hann")


def test_cpu_backprojection_does_not_import_torch():
    code = """
import sys
import numpy as np
from xrayrecon.backprojection import make_backprojector
assert 'torch' not in sys.modules
result = make_backprojector(8, [0, 90])(np.ones((1, 8, 2)))
assert result.shape == (1, 5, 5)
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, "-B", "-c", code], cwd=Path(__file__).resolve().parents[2], check=True)


def test_unknown_backend_rejected():
    with pytest.raises(ValueError, match="backprojection_backend"):
        make_backprojector(8, [0, 90], backend="automatic")

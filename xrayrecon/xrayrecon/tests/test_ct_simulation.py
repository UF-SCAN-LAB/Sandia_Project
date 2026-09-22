import numpy as np
import pytest

from xrayrecon.ct_simulation import simulate_ct_volume


def _phantom_volume(shape: tuple[int, int, int]) -> np.ndarray:
    z_depth, h, w = shape
    vol = np.zeros((z_depth, h, w, 2), dtype=np.float32)
    vol[..., 0] = 4.0
    vol[..., 1] = 1.0
    vol[:, h // 3 : 2 * h // 3, w // 4 : 3 * w // 4, 0] = 29.0
    vol[:, h // 3 : 2 * h // 3, w // 4 : 3 * w // 4, 1] = 7.0
    return vol


def test_simulate_ct_volume_preserves_square_slice_shape():
    vol = _phantom_volume((2, 16, 16))

    result = simulate_ct_volume(vol, num_angles=16)

    assert result.shape == (2, 16, 16)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()


def test_simulate_ct_volume_preserves_rectangular_slice_shape():
    vol = _phantom_volume((2, 14, 20))

    result = simulate_ct_volume(vol, num_angles=20)

    assert result.shape == (2, 14, 20)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    assert float(np.ptp(result)) > 0.0


def test_simulate_ct_volume_rejects_unknown_projection_backend():
    with pytest.raises(ValueError, match="projection_backend"):
        simulate_ct_volume(
            _phantom_volume((1, 8, 10)),
            num_angles=8,
            projection_backend="automatic",
        )


@pytest.mark.cuda
def test_simulate_ct_volume_cuda_projection_matches_cpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA execution unavailable")
    volume = _phantom_volume((2, 14, 20))

    cpu = simulate_ct_volume(volume, num_angles=23)
    gpu = simulate_ct_volume(
        volume,
        num_angles=23,
        projection_backend="cuda",
        projection_batch_size=3,
    )

    assert gpu.shape == cpu.shape
    assert gpu.dtype == np.float32
    assert np.isfinite(gpu).all()
    np.testing.assert_allclose(gpu, cpu, rtol=2e-5, atol=2e-4)

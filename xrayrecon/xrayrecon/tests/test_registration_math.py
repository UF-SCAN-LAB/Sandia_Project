import numpy as np

from xrayrecon.registration import (
    RigidScaleTransform,
    invert_transform,
    resample_labels,
    resample_volume,
    transform_to_matrix,
)
from xrayrecon.tests.fixtures import make_synthetic_pair


def _manual_forward_linear(t: RigidScaleTransform) -> np.ndarray:
    """Independent re-derivation of the forward rotate+scale matrix, used to
    cross-check transform_to_matrix without relying on the module's own
    (potentially buggy) internals."""
    theta = np.deg2rad(t.rot_z_deg)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return rot * t.scale


def test_transform_to_matrix_maps_fixed_index_to_correct_moving_index():
    t = RigidScaleTransform(tx=2.0, ty=-1.0, tz=0.5, rot_z_deg=30.0, scale=1.5)
    moving_spacing = np.array([0.2, 0.2, 0.1])
    fixed_spacing = np.array([0.3, 0.3, 0.1])

    moving_idx = np.array([10.0, 4.0, 2.0])
    moving_mm = moving_idx * moving_spacing
    fixed_mm = _manual_forward_linear(t) @ moving_mm + np.array([t.tx, t.ty, t.tz])
    fixed_idx = fixed_mm / fixed_spacing

    matrix, offset = transform_to_matrix(t, moving_spacing, fixed_spacing)
    recovered_moving_idx = matrix @ fixed_idx + offset

    np.testing.assert_allclose(recovered_moving_idx, moving_idx, atol=1e-8)


def test_identity_transform_is_noop():
    rng = np.random.default_rng(0)
    moving = rng.random((10, 12, 6)).astype(np.float32)
    spacing = (1.0, 1.0, 1.0)

    result = resample_volume(moving, spacing, moving.shape, spacing, RigidScaleTransform.identity())

    np.testing.assert_allclose(result, moving, atol=1e-5)


def test_pure_translation_shifts_content_by_expected_voxels():
    moving = np.zeros((20, 20, 4), dtype=np.float32)
    moving[5:8, 5:8, :] = 1.0
    spacing = (1.0, 1.0, 1.0)
    t = RigidScaleTransform(tx=3.0)

    result = resample_volume(moving, spacing, moving.shape, spacing, t, order=0)

    expected = np.zeros_like(moving)
    expected[8:11, 5:8, :] = 1.0
    np.testing.assert_array_equal(result, expected)


def test_uniform_scale_matches_expected_extent():
    moving = np.zeros((40, 40, 4), dtype=np.float32)
    moving[10:14, 10:14, :] = 1.0  # a 4-voxel-wide block centered at 12
    spacing = (1.0, 1.0, 1.0)
    t = RigidScaleTransform(scale=2.0)

    result = resample_volume(moving, spacing, moving.shape, spacing, t, order=0)

    # Scaling by 2 about the origin should double the block's extent and position.
    assert result[20:28, 20:28, :].sum() > 0
    assert result[:15, :15, :].sum() == 0


def test_forward_inverse_roundtrip_recovers_original():
    # Smooth, spatially-correlated field (a few overlapping Gaussian blobs) --
    # i.i.d. random noise has no spatial structure, so any interpolation
    # blur (inevitable across two resampling passes) destroys correlation
    # almost immediately and isn't representative of real volumes.
    axis0, axis1, axis2 = np.meshgrid(
        np.arange(24), np.arange(24), np.arange(8), indexing="ij"
    )
    moving = np.zeros((24, 24, 8), dtype=np.float32)
    for cx, cy, cz in [(8, 8, 3), (16, 14, 5), (10, 18, 2)]:
        moving += np.exp(
            -(((axis0 - cx) ** 2 + (axis1 - cy) ** 2) / 18.0 + ((axis2 - cz) ** 2) / 6.0)
        )
    spacing = (1.0, 1.0, 1.0)
    t = RigidScaleTransform(tx=3.2, ty=-1.7, tz=0.5, rot_z_deg=7.0, scale=1.05)

    misaligned = resample_volume(moving, spacing, moving.shape, spacing, t, order=1)
    recovered = resample_volume(
        misaligned, spacing, moving.shape, spacing, invert_transform(t), order=1
    )

    # Two rounds of trilinear interpolation blur things somewhat, plus edge
    # effects near the volume boundary -- compare interior region with a
    # normalized-correlation-style tolerance rather than exact/near-exact match.
    interior = np.s_[4:-4, 4:-4, 2:-2]
    corr = np.corrcoef(recovered[interior].ravel(), moving[interior].ravel())[0, 1]
    assert corr > 0.9


def test_resample_labels_never_invents_new_labels():
    label_vol = np.zeros((10, 10, 4), dtype=np.int32)
    label_vol[2:5, 2:5, :] = 3
    spacing = (1.0, 1.0, 1.0)
    t = RigidScaleTransform(tx=1.3, ty=-2.7, rot_z_deg=17.0, scale=1.2)

    out = resample_labels(label_vol, spacing, (12, 12, 4), spacing, t, background_label=0)

    assert out.dtype == label_vol.dtype
    assert set(np.unique(out).tolist()) <= set(np.unique(label_vol).tolist()) | {0}


def test_realistic_gerber_ct_pair_roundtrip_recovers_known_transform():
    # Re-validates the round-trip against a realistic Gerber-voxelize ->
    # FBP-simulated-CT pair (not just toy phantoms), catching any interaction
    # between simulate_ct_volume's Radon/iradon artifacts and the resampling
    # math that the purely-synthetic tests above wouldn't exercise.
    pair = make_synthetic_pair()

    recovered = resample_volume(
        pair["misaligned_ct_hwz"],
        pair["fixed_spacing"],
        pair["aligned_ct_hwz"].shape,
        pair["gerber_spacing"],
        invert_transform(pair["known_transform"]),
        order=1,
    )

    # Note: don't trim the Z axis here -- the copper trace (the only
    # spatially-structured content) lives entirely in the first Z slice, so
    # excluding Z edges would compare only the near-featureless FR4 slab.
    interior = np.s_[6:-6, 6:-6, :]
    aligned_interior = pair["aligned_ct_hwz"][interior]
    recovered_interior = recovered[interior]

    # FBP reconstruction produces a handful of extreme ringing-artifact
    # voxels at high-contrast edges (values in the hundreds against a bulk
    # background around ~12) -- Pearson correlation is dominated by those
    # few outliers, so clip to the bulk's range before comparing.
    lo, hi = np.percentile(aligned_interior, [1, 99])
    clipped_aligned = np.clip(aligned_interior, lo, hi)
    clipped_recovered = np.clip(recovered_interior, lo, hi)

    corr = np.corrcoef(clipped_recovered.ravel(), clipped_aligned.ravel())[0, 1]
    assert corr > 0.8


def test_invert_transform_is_self_consistent():
    t = RigidScaleTransform(tx=2.0, ty=-1.0, tz=0.5, rot_z_deg=30.0, scale=1.5)
    t_inv = invert_transform(t)
    t_roundtrip = invert_transform(t_inv)

    assert np.isclose(t_roundtrip.tx, t.tx)
    assert np.isclose(t_roundtrip.ty, t.ty)
    assert np.isclose(t_roundtrip.tz, t.tz)
    assert np.isclose(t_roundtrip.rot_z_deg, t.rot_z_deg)
    assert np.isclose(t_roundtrip.scale, t.scale)

import numpy as np
import pytest
from scipy.ndimage import binary_erosion
from scipy.stats import spearmanr

from xrayrecon.alvarez_macovski import (
    PE_REFERENCE_ENERGY_KEV,
    f1,
    simulate_am_volume,
    to_a1a2,
    to_rho_e,
    to_zeff,
)
from xrayrecon.materials import (
    MATERIAL_NAMES,
    MATERIALS_DB,
    material_labels_to_zeff_rhoe,
)
from xrayrecon.synthetic_registration import make_pcb_registration_showcase


@pytest.mark.unit
def test_material_labels_map_exactly_to_canonical_zeff_and_density():
    labels = np.arange(len(MATERIAL_NAMES), dtype=np.int16).reshape(2, -1)

    properties = material_labels_to_zeff_rhoe(labels)

    assert properties.shape == labels.shape + (2,)
    assert properties.dtype == np.float32
    for material_id, name in enumerate(MATERIAL_NAMES):
        row, col = np.argwhere(labels == material_id)[0]
        expected = MATERIALS_DB[name]
        assert properties[row, col, 0] == pytest.approx(expected.z_eff)
        assert properties[row, col, 1] == pytest.approx(expected.electron_density)


@pytest.mark.unit
def test_material_property_mapping_rejects_invalid_labels():
    with pytest.raises(TypeError, match="integer"):
        material_labels_to_zeff_rhoe(np.array([0.0], dtype=np.float32))
    with pytest.raises(ValueError, match="must be in"):
        material_labels_to_zeff_rhoe(np.array([-1], dtype=np.int16))
    with pytest.raises(ValueError, match="must be in"):
        material_labels_to_zeff_rhoe(
            np.array([len(MATERIAL_NAMES)], dtype=np.int16)
        )


@pytest.mark.unit
def test_showcase_produces_am_ready_physical_property_volume():
    showcase = make_pcb_registration_showcase()
    labels_zhw = np.moveaxis(showcase.moving_labels_hwz, -1, 0)

    am_volume = material_labels_to_zeff_rhoe(labels_zhw)

    assert am_volume.shape == (24, 96, 128, 2)
    assert np.isfinite(am_volume).all()
    assert am_volume[..., 0].min() >= 0
    assert am_volume[..., 1].min() >= 0


@pytest.mark.unit
def test_am_basis_round_trip_preserves_material_properties():
    zeff = np.asarray(
        [MATERIALS_DB[name].z_eff for name in MATERIAL_NAMES], dtype=np.float64
    )
    rhoe = np.asarray(
        [MATERIALS_DB[name].electron_density for name in MATERIAL_NAMES],
        dtype=np.float64,
    )

    a1, a2 = to_a1a2(zeff, rhoe)

    np.testing.assert_allclose(to_zeff(a1, a2), zeff, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(to_rho_e(a2), rhoe, rtol=0, atol=0)


@pytest.mark.unit
def test_am_reconstruction_preserves_rectangular_slice_shape():
    labels = np.full((1, 32, 48), MATERIAL_NAMES.index("FR4"), dtype=np.int16)
    labels[:, 9:23, 18:31] = MATERIAL_NAMES.index("COPPER")
    am_volume = material_labels_to_zeff_rhoe(labels)

    result = simulate_am_volume(am_volume, num_angles=32)

    assert {
        "zeff_ground_truth",
        "rhoe_ground_truth",
        "zeff_recovered",
        "rhoe_recovered",
        "zeff_regularized",
        "rhoe_regularized",
        "material_labels_recovered",
    } <= set(result)
    for volume in result.values():
        assert volume.shape == labels.shape
        assert np.isfinite(volume).all()
        assert np.all(volume >= 0)


@pytest.mark.unit
def test_photoelectric_basis_uses_shared_reference_normalization():
    assert f1(PE_REFERENCE_ENERGY_KEV) == pytest.approx(1.0)
    assert f1(PE_REFERENCE_ENERGY_KEV * 2) == pytest.approx(1 / 8)


@pytest.mark.unit
@pytest.mark.parametrize(
    "material_name",
    ("FR4", "ALUMINA", "ALUMINUM", "COPPER", "TIN", "TUNGSTEN", "GOLD"),
)
def test_representative_material_medians_recover_within_twenty_percent(
    material_name,
):
    labels = np.full(
        (1, 40, 40), MATERIAL_NAMES.index("AIR"), dtype=np.int16
    )
    labels[:, 8:32, 8:32] = MATERIAL_NAMES.index(material_name)
    am_volume = material_labels_to_zeff_rhoe(labels)

    result = simulate_am_volume(am_volume, num_angles=64)

    interior = (0, slice(12, 28), slice(12, 28))
    zeff_median = float(np.median(result["zeff_recovered"][interior]))
    rhoe_median = float(np.median(result["rhoe_recovered"][interior]))
    expected = MATERIALS_DB[material_name]
    assert abs(zeff_median - expected.z_eff) / expected.z_eff <= 0.20
    assert (
        abs(rhoe_median - expected.electron_density) / expected.electron_density
        <= 0.20
    )


@pytest.mark.unit
def test_showcase_slice_meets_approved_am_recovery_contract():
    showcase = make_pcb_registration_showcase()
    # Downsample and center-crop the component-rich top layer for a fast,
    # deterministic CI test that still contains seven showcase materials.
    labels = showcase.moving_labels_hwz[::2, 16:112:2, 19]
    am_volume = material_labels_to_zeff_rhoe(labels[None])

    result = simulate_am_volume(am_volume, num_angles=96)

    air_id = MATERIAL_NAMES.index("AIR")
    interior = np.zeros(labels.shape, dtype=bool)
    for material_id in np.unique(labels):
        if material_id == air_id:
            continue
        interior |= binary_erosion(labels == material_id, iterations=2)
    assert np.count_nonzero(interior) >= 500

    zeff_gt = result["zeff_ground_truth"][0][interior]
    rhoe_gt = result["rhoe_ground_truth"][0][interior]
    zeff_regularized = result["zeff_regularized"][0][interior]
    rhoe_raw = result["rhoe_recovered"][0][interior]
    recovered_labels = result["material_labels_recovered"][0][interior]

    assert spearmanr(zeff_gt, zeff_regularized).statistic >= 0.80
    assert np.corrcoef(rhoe_gt, rhoe_raw)[0, 1] >= 0.90
    assert np.mean(recovered_labels == labels[interior]) >= 0.70


@pytest.mark.slow
def test_full_rectangular_showcase_completes_am_decomposition():
    showcase = make_pcb_registration_showcase()
    labels_zhw = np.moveaxis(showcase.moving_labels_hwz, -1, 0)
    am_volume = material_labels_to_zeff_rhoe(labels_zhw)

    result = simulate_am_volume(am_volume, num_angles=64)

    expected_shape = (24, 96, 128)
    for volume in result.values():
        assert volume.shape == expected_shape
        assert np.isfinite(volume).all()

import numpy as np
import pytest
from time import perf_counter
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
    WATER_ELECTRON_DENSITY_E23_CM3,
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
def test_material_database_separates_mass_and_electron_density_units():
    assert WATER_ELECTRON_DENSITY_E23_CM3 == pytest.approx(3.343, rel=0.01)
    for material in MATERIALS_DB.values():
        assert material.mass_density_g_cm3 > 0
        assert material.relative_electron_density > 0
        assert material.electron_density == material.relative_electron_density
        assert material.electron_density_e23_cm3 == pytest.approx(
            material.relative_electron_density
            * WATER_ELECTRON_DENSITY_E23_CM3
        )

    # Mass density and relative electron density are physically distinct;
    # copper is a useful guard against accidentally restoring the old proxy.
    copper = MATERIALS_DB["COPPER"]
    assert copper.mass_density_g_cm3 == pytest.approx(8.96)
    assert copper.relative_electron_density == pytest.approx(7.38, rel=0.03)


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
        "projection_residual",
        "confidence",
        "rhoe_e23_cm3_ground_truth",
        "rhoe_e23_cm3_recovered",
        "rhoe_e23_cm3_regularized",
    } <= set(result)
    for volume in result.values():
        assert volume.shape == labels.shape
        assert np.isfinite(volume).all()
        assert np.all(volume >= 0)

    assert np.all(result["confidence"] <= 1)
    np.testing.assert_allclose(
        result["rhoe_e23_cm3_recovered"],
        result["rhoe_recovered"] * WATER_ELECTRON_DENSITY_E23_CM3,
        rtol=1e-6,
    )


@pytest.mark.unit
def test_noisy_am_recovery_is_seeded_and_reports_auditable_quality_maps():
    labels = np.full((1, 40, 48), MATERIAL_NAMES.index("AIR"), dtype=np.int16)
    labels[:, 7:33, 9:39] = MATERIAL_NAMES.index("ALUMINA")
    labels[:, 15:26, 20:30] = MATERIAL_NAMES.index("COPPER")
    volume = material_labels_to_zeff_rhoe(labels)

    first = simulate_am_volume(
        volume, num_angles=32, photon_count=200_000, random_seed=71
    )
    second = simulate_am_volume(
        volume, num_angles=32, photon_count=200_000, random_seed=71
    )

    np.testing.assert_array_equal(first["zeff_recovered"], second["zeff_recovered"])
    np.testing.assert_array_equal(first["confidence"], second["confidence"])
    assert np.isfinite(first["projection_residual"]).all()
    assert np.all((first["confidence"] >= 0) & (first["confidence"] <= 1))
    assert float(np.ptp(first["confidence"])) > 0.1


@pytest.mark.unit
def test_am_noise_contract_rejects_ambiguous_parameters():
    labels = np.full((1, 12, 12), MATERIAL_NAMES.index("FR4"), dtype=np.int16)
    volume = material_labels_to_zeff_rhoe(labels)

    with pytest.raises(ValueError, match="photon_count"):
        simulate_am_volume(volume, num_angles=12, photon_count=0)
    with pytest.raises(ValueError, match="random_seed"):
        simulate_am_volume(volume, num_angles=12, photon_count=100_000)
    with pytest.raises(ValueError, match="random_seed"):
        simulate_am_volume(volume, num_angles=12, random_seed=1)


@pytest.mark.unit
def test_photoelectric_basis_uses_shared_reference_normalization():
    assert f1(PE_REFERENCE_ENERGY_KEV) == pytest.approx(1.0)
    assert f1(PE_REFERENCE_ENERGY_KEV * 2) == pytest.approx(1 / 8)


@pytest.mark.unit
@pytest.mark.parametrize(
    "material_name",
    ("FR4", "ALUMINA", "ALUMINUM", "COPPER", "TIN", "TUNGSTEN", "GOLD"),
)
def test_noiseless_material_medians_recover_within_five_percent(
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
    assert abs(zeff_median - expected.z_eff) / expected.z_eff <= 0.05
    assert (
        abs(rhoe_median - expected.electron_density) / expected.electron_density
        <= 0.05
    )


@pytest.mark.unit
@pytest.mark.parametrize("material_name", ("TIN", "TUNGSTEN", "GOLD"))
def test_noisy_high_z_materials_meet_approved_error_thresholds(material_name):
    labels = np.full((1, 40, 40), MATERIAL_NAMES.index("AIR"), dtype=np.int16)
    labels[:, 8:32, 8:32] = MATERIAL_NAMES.index(material_name)
    result = simulate_am_volume(
        material_labels_to_zeff_rhoe(labels),
        num_angles=64,
        photon_count=2_000_000,
        random_seed=19,
    )

    interior = (0, slice(12, 28), slice(12, 28))
    expected = MATERIALS_DB[material_name]
    zeff_error = abs(
        float(np.median(result["zeff_recovered"][interior])) - expected.z_eff
    ) / expected.z_eff
    rhoe_error = abs(
        float(np.median(result["rhoe_recovered"][interior]))
        - expected.relative_electron_density
    ) / expected.relative_electron_density
    assert zeff_error <= 0.25
    assert rhoe_error <= 0.15


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


@pytest.mark.unit
def test_noisy_showcase_meets_density_classification_and_confidence_contract():
    showcase = make_pcb_registration_showcase()
    labels = showcase.moving_labels_hwz[::2, 16:112:2, 19]
    result = simulate_am_volume(
        material_labels_to_zeff_rhoe(labels[None]),
        num_angles=96,
        photon_count=2_000_000,
        random_seed=71,
    )

    air_id = MATERIAL_NAMES.index("AIR")
    interior = np.zeros(labels.shape, dtype=bool)
    for material_id in np.unique(labels):
        if material_id != air_id:
            interior |= binary_erosion(labels == material_id, iterations=2)

    zeff_gt = result["zeff_ground_truth"][0]
    rhoe_gt = result["rhoe_ground_truth"][0]
    zeff_error_regularized = np.abs(
        result["zeff_regularized"][0][interior] - zeff_gt[interior]
    ) / zeff_gt[interior]
    rhoe_error_raw = np.abs(
        result["rhoe_recovered"][0][interior] - rhoe_gt[interior]
    ) / rhoe_gt[interior]
    recovered_labels = result["material_labels_recovered"][0]

    assert np.median(zeff_error_regularized) <= 0.20
    assert np.median(rhoe_error_raw) <= 0.15
    assert np.mean(recovered_labels[interior] == labels[interior]) >= 0.85

    combined_raw_error = 0.5 * np.abs(
        result["zeff_recovered"][0] - zeff_gt
    ) / np.maximum(zeff_gt, 1.0) + 0.5 * np.abs(
        result["rhoe_recovered"][0] - rhoe_gt
    ) / np.maximum(rhoe_gt, 0.05)
    confidence = result["confidence"][0]
    interior_confidence = confidence[interior]
    low_cutoff, high_cutoff = np.quantile(interior_confidence, (0.25, 0.75))
    low_error = np.median(combined_raw_error[interior & (confidence <= low_cutoff)])
    high_error = np.median(combined_raw_error[interior & (confidence >= high_cutoff)])
    assert low_error > high_error


@pytest.mark.unit
def test_reduced_angle_showcase_retains_interpretable_material_accuracy():
    showcase = make_pcb_registration_showcase()
    labels = showcase.moving_labels_hwz[::2, 16:112:2, 19]
    result = simulate_am_volume(
        material_labels_to_zeff_rhoe(labels[None]), num_angles=48
    )

    air_id = MATERIAL_NAMES.index("AIR")
    interior = np.zeros(labels.shape, dtype=bool)
    for material_id in np.unique(labels):
        if material_id != air_id:
            interior |= binary_erosion(labels == material_id, iterations=2)
    accuracy = np.mean(
        result["material_labels_recovered"][0][interior] == labels[interior]
    )
    assert accuracy >= 0.85


@pytest.mark.slow
def test_full_rectangular_showcase_completes_am_decomposition():
    showcase = make_pcb_registration_showcase()
    labels_zhw = np.moveaxis(showcase.moving_labels_hwz, -1, 0)
    am_volume = material_labels_to_zeff_rhoe(labels_zhw)

    started = perf_counter()
    result = simulate_am_volume(am_volume, num_angles=64)
    elapsed = perf_counter() - started

    expected_shape = (24, 96, 128)
    for volume in result.values():
        assert volume.shape == expected_shape
        assert np.isfinite(volume).all()
    assert elapsed <= 10.0

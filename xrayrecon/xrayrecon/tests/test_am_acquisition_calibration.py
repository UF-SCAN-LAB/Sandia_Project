import json

import numpy as np
import pytest

from xrayrecon.am_acquisition import (
    AMAcquisitionConfig,
    AMSpectrum,
    PairedProjectionData,
    load_paired_projection_files,
)
from xrayrecon.alvarez_macovski import (
    reconstruct_am_projections,
    simulate_am_projections,
    simulate_am_volume,
)
from xrayrecon.am_calibration import (
    AMCalibrationSample,
    AMCalibrationModel,
    evaluate_calibrated_result,
    fit_calibration_from_result,
    fit_material_calibration,
    load_calibration_model,
    make_am_calibration_phantom,
    material_validation_mask,
    save_calibration_model,
)
from xrayrecon.materials import MATERIAL_NAMES, material_labels_to_zeff_rhoe


@pytest.mark.unit
def test_default_acquisition_configuration_round_trips_as_json():
    config = AMAcquisitionConfig.industrial_default(
        num_angles=64,
        photon_count=2_000_000,
        random_seed=71,
        calibration_id="pcb-reference-v1",
    )

    restored = AMAcquisitionConfig.from_json(config.to_json())

    assert restored == config
    payload = json.loads(config.to_json())
    assert payload["schema_version"] == 1
    assert payload["source_kind"] == "simulated"
    assert payload["electron_density_unit"] == "relative_to_water"
    assert sum(restored.low_spectrum.weights) == pytest.approx(1.0)
    assert sum(restored.high_spectrum.weights) == pytest.approx(1.0)
    assert restored.calibration_fingerprint == config.calibration_fingerprint


@pytest.mark.unit
def test_calibration_fingerprint_tracks_physics_not_sampling_or_provenance():
    reference = AMAcquisitionConfig.industrial_default(
        num_angles=96,
        photon_count=2_000_000,
        random_seed=71,
        calibration_id="reference-v1",
    )
    different_sampling = AMAcquisitionConfig.from_dict(
        {
            **reference.to_dict(),
            "num_angles": 64,
            "photon_count": None,
            "random_seed": None,
            "source_kind": "measured",
            "calibration_id": "scanner-run-v2",
            "low_spectrum": {
                **reference.low_spectrum.to_dict(),
                "label": "operator label does not affect physics",
            },
        }
    )
    different_pixel_size = AMAcquisitionConfig.from_dict(
        {**reference.to_dict(), "pixel_size_cm": 0.02}
    )
    different_spectrum = AMAcquisitionConfig.from_dict(
        {
            **reference.to_dict(),
            "low_spectrum": AMSpectrum.triangular(
                70.0, 160.0, label="different filtration"
            ).to_dict(),
        }
    )

    assert reference.calibration_fingerprint.startswith("am-physics-v1:")
    assert (
        different_sampling.calibration_fingerprint
        == reference.calibration_fingerprint
    )
    assert different_pixel_size.calibration_fingerprint != reference.calibration_fingerprint
    assert different_spectrum.calibration_fingerprint != reference.calibration_fingerprint


@pytest.mark.unit
def test_acquisition_contract_rejects_ambiguous_physics():
    with pytest.raises(ValueError, match="strictly increasing"):
        AMSpectrum((60.0, 50.0), (0.5, 0.5), "bad")
    with pytest.raises(ValueError, match="same length"):
        AMSpectrum((50.0, 60.0), (1.0,), "bad")
    with pytest.raises(ValueError, match="positive total"):
        AMSpectrum((50.0, 60.0), (0.0, 0.0), "bad")

    default = AMAcquisitionConfig.industrial_default(num_angles=32)
    with pytest.raises(ValueError, match="random_seed"):
        AMAcquisitionConfig(
            low_spectrum=default.low_spectrum,
            high_spectrum=default.high_spectrum,
            num_angles=32,
            photon_count=100_000,
        )
    with pytest.raises(ValueError, match="source_kind"):
        AMAcquisitionConfig(
            low_spectrum=default.low_spectrum,
            high_spectrum=default.high_spectrum,
            num_angles=32,
            source_kind="unknown",
        )


@pytest.mark.unit
def test_paired_projection_contract_validates_geometry_and_provenance():
    config = AMAcquisitionConfig.industrial_default(num_angles=24)
    low = np.ones((2, 48, 24), dtype=np.float32)
    high = np.ones_like(low) * 0.7
    theta = np.linspace(0.0, 180.0, 24, endpoint=False)

    pair = PairedProjectionData(low, high, theta, config)

    assert pair.shape == (2, 48, 24)
    assert pair.source_kind == "simulated"
    assert pair.is_measured is False
    with pytest.raises(ValueError, match="matching shapes"):
        PairedProjectionData(low, high[:, :-1], theta, config)
    with pytest.raises(ValueError, match="angle axis"):
        PairedProjectionData(low, high, theta[:-1], config)
    invalid = low.copy()
    invalid[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        PairedProjectionData(invalid, high, theta, config)
    negative = low.copy()
    negative[0, 0, 0] = -0.1
    with pytest.raises(ValueError, match="nonnegative"):
        PairedProjectionData(negative, high, theta, config)


@pytest.mark.unit
def test_measured_projection_config_never_claims_simulated_provenance():
    simulated = AMAcquisitionConfig.industrial_default(num_angles=16)
    measured = AMAcquisitionConfig.from_dict(
        {**simulated.to_dict(), "source_kind": "measured", "random_seed": None,
         "photon_count": None, "calibration_id": "scanner-cal-2026-08"}
    )
    low = np.ones((32, 16), dtype=np.float32)
    pair = PairedProjectionData(
        low, low.copy(), np.arange(16, dtype=float), measured
    )

    assert pair.is_measured is True
    assert pair.source_kind == "measured"
    assert pair.config.calibration_id == "scanner-cal-2026-08"


@pytest.mark.unit
def test_measured_projection_file_import_is_strict_and_validated(tmp_path):
    simulated = AMAcquisitionConfig.industrial_default(num_angles=12)
    measured = AMAcquisitionConfig.from_dict(
        {
            **simulated.to_dict(),
            "source_kind": "measured",
            "calibration_id": "scanner-reference-v2",
        }
    )
    low_path = tmp_path / "low.npy"
    high_path = tmp_path / "high.npy"
    theta_path = tmp_path / "theta.npy"
    np.save(low_path, np.ones((20, 12), dtype=np.float32))
    np.save(high_path, np.ones((20, 12), dtype=np.float32) * 0.8)
    np.save(theta_path, measured.theta_deg)

    pair = load_paired_projection_files(
        low_path, high_path, theta_path, measured
    )

    assert pair.is_measured
    assert pair.shape == (20, 12)
    with pytest.raises(ValueError, match="source_kind"):
        load_paired_projection_files(
            low_path, high_path, theta_path, simulated
        )


def _small_am_volume():
    labels = np.full((1, 24, 32), MATERIAL_NAMES.index("AIR"), dtype=np.int16)
    labels[:, 5:19, 7:25] = MATERIAL_NAMES.index("ALUMINA")
    labels[:, 10:15, 13:20] = MATERIAL_NAMES.index("COPPER")
    return material_labels_to_zeff_rhoe(labels)


@pytest.mark.unit
def test_configured_simulation_preserves_legacy_default_behavior():
    volume = _small_am_volume()
    config = AMAcquisitionConfig.industrial_default(num_angles=32)

    legacy = simulate_am_volume(volume, num_angles=32)
    configured = simulate_am_volume(volume, config=config)

    for key in legacy:
        np.testing.assert_array_equal(configured[key], legacy[key])
    assert configured.acquisition_config == config
    assert configured.source_kind == "simulated"
    assert configured.has_ground_truth is True


@pytest.mark.unit
def test_projection_simulation_and_reconstruction_are_separable():
    volume = _small_am_volume()
    config = AMAcquisitionConfig.industrial_default(
        num_angles=32, photon_count=500_000, random_seed=13
    )

    projections = simulate_am_projections(volume, config)
    result = reconstruct_am_projections(projections, output_shape=(24, 32))

    assert projections.shape[0] == 1
    assert projections.shape[-1] == 32
    assert result.acquisition_config == config
    assert result.has_ground_truth is False
    assert "zeff_ground_truth" not in result
    for array in result.values():
        assert array.shape == (1, 24, 32)
        assert np.isfinite(array).all()


@pytest.mark.unit
def test_measured_projection_reconstruction_retains_honest_provenance():
    volume = _small_am_volume()
    simulated_config = AMAcquisitionConfig.industrial_default(num_angles=24)
    simulated = simulate_am_projections(volume, simulated_config)
    measured_config = AMAcquisitionConfig.from_dict(
        {
            **simulated_config.to_dict(),
            "source_kind": "measured",
            "calibration_id": "bench-phantom-v1",
        }
    )
    measured = PairedProjectionData(
        simulated.low_log,
        simulated.high_log,
        simulated.theta_deg,
        measured_config,
        simulated.a1_max_bound,
        simulated.a2_max_bound,
    )

    result = reconstruct_am_projections(measured, output_shape=(24, 32))

    assert result.source_kind == "measured"
    assert result.has_ground_truth is False
    assert result.calibration_id == "bench-phantom-v1"


@pytest.mark.unit
def test_material_calibration_corrects_known_systematic_response():
    samples = tuple(
        AMCalibrationSample(
            material_name=name,
            true_zeff=true_z,
            true_rhoe=true_r,
            recovered_zeff=true_z ** 1.12 * 0.82,
            recovered_rhoe=true_r * 0.91 + 0.08,
            confidence=0.8,
        )
        for name, true_z, true_r in (
            ("FR4", 6.5, 1.70),
            ("ALUMINUM", 13.0, 2.35),
            ("COPPER", 29.0, 7.38),
            ("TIN", 50.0, 5.56),
        )
    )

    config = AMAcquisitionConfig.industrial_default(num_angles=32)
    model = fit_material_calibration(
        samples,
        calibration_id="unit-cal",
        acquisition_fingerprint=config.calibration_fingerprint,
    )

    recovered_z = np.asarray([sample.recovered_zeff for sample in samples])
    recovered_r = np.asarray([sample.recovered_rhoe for sample in samples])
    np.testing.assert_allclose(
        model.calibrate_zeff(recovered_z),
        [sample.true_zeff for sample in samples],
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        model.calibrate_rhoe(recovered_r),
        [sample.true_rhoe for sample in samples],
        rtol=1e-5,
    )


@pytest.mark.unit
def test_calibration_bundle_round_trips_and_prevents_accidental_overwrite(tmp_path):
    config = AMAcquisitionConfig.industrial_default(num_angles=32)
    model = AMCalibrationModel(
        calibration_id="scanner-reference-v3",
        training_materials=("FR4", "ALUMINUM", "COPPER"),
        acquisition_fingerprint=config.calibration_fingerprint,
        zeff_log_slope=0.91,
        zeff_log_intercept=0.12,
        rhoe_slope=1.03,
        rhoe_intercept=-0.02,
    )
    path = tmp_path / "scanner-reference-v3.json"

    restored_json = AMCalibrationModel.from_json(model.to_json())
    saved_path = save_calibration_model(model, path)
    restored_file = load_calibration_model(saved_path)

    assert restored_json == model
    assert restored_file == model
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 1
    with pytest.raises(FileExistsError, match="already exists"):
        save_calibration_model(model, path)
    with pytest.raises(ValueError, match="schema_version"):
        AMCalibrationModel.from_dict({**model.to_dict(), "schema_version": 2})
    with pytest.raises(ValueError, match="artifact kind"):
        AMCalibrationModel.from_dict(
            {**model.to_dict(), "artifact_kind": "different.artifact"}
        )


@pytest.mark.unit
def test_calibration_phantom_is_cross_validated_on_held_out_materials():
    all_materials = (
        "FR4",
        "ALUMINA",
        "ALUMINUM",
        "COPPER",
        "TIN",
        "TUNGSTEN",
        "GOLD",
    )
    training = ("FR4", "ALUMINA", "COPPER", "TIN", "TUNGSTEN")
    held_out = ("ALUMINUM", "GOLD")
    labels = make_am_calibration_phantom(all_materials, size=72)
    volume = material_labels_to_zeff_rhoe(labels)
    training_result = simulate_am_volume(
        volume,
        config=AMAcquisitionConfig.industrial_default(
            num_angles=96, photon_count=2_000_000, random_seed=71
        ),
    )
    model = fit_calibration_from_result(
        training_result,
        labels,
        training,
        calibration_id="synthetic-cross-validation-v1",
    )
    validation_result = simulate_am_volume(
        volume,
        config=AMAcquisitionConfig.industrial_default(
            num_angles=96, photon_count=2_000_000, random_seed=73
        ),
    )
    calibrated = model.apply(validation_result)

    aluminum_metrics = evaluate_calibrated_result(
        calibrated, material_validation_mask(labels, ("ALUMINUM",))
    )
    gold_metrics = evaluate_calibrated_result(
        calibrated, material_validation_mask(labels, ("GOLD",))
    )
    held_out_metrics = evaluate_calibrated_result(
        calibrated, material_validation_mask(labels, held_out)
    )

    assert aluminum_metrics["zeff_median_relative_error"] <= 0.15
    assert gold_metrics["zeff_median_relative_error"] <= 0.20
    assert held_out_metrics["rhoe_median_relative_error"] <= 0.10
    assert 0.80 <= held_out_metrics["zeff_coverage"] <= 0.97
    assert 0.80 <= held_out_metrics["rhoe_coverage"] <= 0.97
    assert held_out_metrics["confidence_error_rank_correlation"] >= 0.60

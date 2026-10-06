import numpy as np
import pytest
from skimage.metrics import structural_similarity

from xrayrecon.reconstruction_metrics import am_reconstruction_metrics, reconstruction_fidelity


def _reference():
    return np.linspace(1, 20, 120).reshape(10, 12)


def test_exact_nonconstant_recovery():
    truth = _reference()
    metrics = reconstruction_fidelity(truth, truth.copy())
    assert metrics["snr_db"] == float("inf")
    assert metrics["ssi"] == pytest.approx(1)


def test_snr_has_known_error_and_ssi_uses_reference_range():
    truth = _reference()
    recovered = truth + 2
    metrics = reconstruction_fidelity(truth, recovered)
    expected_snr = 10 * np.log10(np.sum(truth ** 2) / (truth.size * 4))
    assert metrics["snr_db"] == pytest.approx(expected_snr)
    assert metrics["ssi_data_range"] == 19
    assert metrics["ssi"] == pytest.approx(structural_similarity(truth, recovered, data_range=19))
    assert metrics["ssi"] < 1  # A scale/offset error must not be normalized away.


def test_metrics_decrease_with_larger_noise():
    truth = _reference()
    noise = np.random.default_rng(3).normal(size=truth.shape)
    mild = reconstruction_fidelity(truth, truth + noise * 0.1)
    strong = reconstruction_fidelity(truth, truth + noise * 3)
    assert mild["snr_db"] > strong["snr_db"]
    assert mild["ssi"] > strong["ssi"]


def test_volume_snr_uses_all_slices_and_ssi_averages_slices():
    truth = np.stack((_reference(), _reference() * 2))
    recovered = truth.copy()
    recovered[1] += 3
    metrics = reconstruction_fidelity(truth, recovered)
    assert metrics["snr_db"] == pytest.approx(10 * np.log10(np.sum(truth ** 2) / np.sum((truth - recovered) ** 2)))
    assert metrics["ssi"] == pytest.approx(np.mean([
        structural_similarity(t, r, data_range=39) for t, r in zip(truth, recovered)
    ]))


def test_tiny_constant_and_zero_references_report_unavailable_metrics():
    zero = reconstruction_fidelity(np.zeros((8, 8)), np.zeros((8, 8)))
    assert zero["snr_db"] is None
    assert zero["ssi"] is None
    assert zero["ssi_status"] == "constant_reference"
    constant = reconstruction_fidelity(np.ones((8, 8)), np.ones((8, 8)))
    assert constant["snr_db"] == float("inf")
    assert constant["ssi"] is None
    tiny = reconstruction_fidelity(np.arange(4).reshape(2, 2), np.ones((2, 2)))
    assert tiny["ssi_status"] == "image_too_small"
    small = reconstruction_fidelity(np.arange(20).reshape(4, 5), np.arange(20).reshape(4, 5))
    assert small["ssi_window_size"] == 3
    assert small["ssi"] == 1


@pytest.mark.parametrize("truth,recovered", [(np.ones((2, 3)), np.ones((3, 2))), (np.ones(8), np.ones(8)), (np.ones((0, 3, 3)), np.ones((0, 3, 3))), (np.full((3, 3), np.nan), np.ones((3, 3)))])
def test_invalid_images_are_rejected(truth, recovered):
    with pytest.raises(ValueError):
        reconstruction_fidelity(truth, recovered)


def test_missing_ground_truth_does_not_produce_fidelity():
    assert am_reconstruction_metrics({"zeff_recovered": _reference()[None]}) == {}


@pytest.mark.gui
def test_analysis_panel_metrics_follow_selected_slice_and_mode(qapp):
    from xrayrecon.gui.am_plot_window import AMAnalysisPanel

    truth = np.stack((_reference(), _reference() * 2)).astype(np.float32)
    result = {
        "zeff_ground_truth": truth, "rhoe_ground_truth": truth / 10,
        "zeff_recovered": truth + 2, "rhoe_recovered": truth / 10 + 0.2,
        "zeff_regularized": truth.copy(), "rhoe_regularized": truth / 10,
        "confidence": np.ones_like(truth), "projection_residual": np.zeros_like(truth),
    }
    panel = AMAnalysisPanel()
    try:
        panel.setResult(result)
        assert "SNR inf dB" in panel.metricSummary.text()
        assert "SSI (SSIM) 1.0000" in panel.metricSummary.text()
        panel.modeCombo.setCurrentIndex(panel.modeCombo.findData("raw"))
        panel.sliceSlider.setValue(0)
        first_summary = panel.metricSummary.text()
        panel.sliceSlider.setValue(1)
        assert panel.metricSummary.text() != first_summary
        result.pop("zeff_ground_truth")
        result.pop("rhoe_ground_truth")
        panel.setResult(result)
        assert "SNR and SSI unavailable: no ground-truth reference" in panel.metricSummary.text()
    finally:
        panel.close()

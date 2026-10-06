import re
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from xrayrecon import simulation_metrics
from xrayrecon.alvarez_macovski import AMResult, simulate_am_volume
from xrayrecon.ct_simulation import simulate_ct_volume


def _timing(capsys):
    lines = capsys.readouterr().out.splitlines()
    summaries = [line for line in lines if " | Total: " in line]
    assert len(summaries) == 1
    summary = summaries[0]
    assert re.findall(r"\d+\.\d+ s", summary)
    assert all(re.fullmatch(r"\d+\.\d{2} s", value)
               for value in re.findall(r"\d+\.\d+ s", summary))
    assert "[simulation]" not in summary
    return summary


def _volume():
    volume = np.ones((1, 8, 10, 2), dtype=np.float32)
    volume[..., 0] = 6
    volume[:, 2:6, 3:7] = (29, 7)
    return volume


def test_ct_prints_only_action_processor_and_step_times(capsys):
    result = simulate_ct_volume(
        _volume(), num_angles=8, sample_file="board stackup.json",
        voxel_spacing_mm=(0.2, 0.2, 0.05),
    )
    summary = _timing(capsys)
    assert result.shape == (1, 8, 10)
    assert summary.startswith("[CT simulation] CPU | Total: ")
    assert " | Projection: " in summary
    assert " | Backprojection: " in summary
    assert "board stackup.json" not in summary
    assert "shape" not in summary
    assert "voxel" not in summary


def test_am_prints_step_times_and_preserves_detailed_array_only_mapping(capsys):
    result = simulate_am_volume(
        _volume(), num_angles=8, pixel_size_cm=0.02,
        sample_file="board.json", voxel_spacing_mm=(0.2, 0.2, 0.05),
        photon_count=100_000, random_seed=3,
    )
    summary = _timing(capsys)
    record = result.run_metrics
    assert summary.startswith("[AM simulation] CPU | Total: ")
    assert " | Acquisition: " in summary
    assert " | Reconstruction: " in summary
    assert "Basis BP: " in summary and "Residual BP: " in summary
    assert " | Metrics: " in summary
    assert "board.json" not in summary
    assert "snr" not in summary and "ssi" not in summary
    assert record["model"] == "am"
    assert record["elapsed_s"] > 0
    assert record["sample_file"] == "board.json"
    assert record["pixel_size_cm"] == 0.02
    assert record["photon_count"] == 100_000
    assert record["random_seed"] == 3
    assert record["spectrum_backend"] == "cpu"
    assert all(isinstance(value, np.ndarray) for value in result.values())
    copied = AMResult(result, acquisition_config=result.acquisition_config,
                      has_ground_truth=True)
    assert copied.run_metrics == result.run_metrics


def test_times_are_truncated_and_total_includes_metrics(capsys):
    simulation_metrics.print_simulation_timing({
        "action_name": "Show AM Volume in 3D", "processor": "cuda",
        "elapsed_s": 1.239, "metrics_elapsed_s": 0.025,
        "acquisition_s": 0.0199, "reconstruction_s": 1.15,
        "gpu": "test device", "sample_file": "board.json",
    })
    assert _timing(capsys) == (
        "[Show AM Volume in 3D] GPU | Total: 1.26 s | Acquisition: 0.01 s"
        " | Reconstruction: 1.15 s | Metrics: 0.02 s"
    )


def test_failed_run_does_not_report_completion(capsys):
    with pytest.raises(ValueError, match="projection_backend"):
        simulate_ct_volume(_volume(), projection_backend="automatic")
    assert "Total:" not in capsys.readouterr().out


def test_record_uses_elapsed_clock_and_actual_cuda_device(monkeypatch):
    monkeypatch.setattr(simulation_metrics, "perf_counter", lambda: 13.5)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        current_device=lambda: 2, get_device_name=lambda index: f"test GPU {index}",
    )))
    record = simulation_metrics.simulation_record(
        10, model="am", shape_zhw=(1, 8, 10), num_angles=8,
        projection_backend="cpu", spectrum_backend="cuda",
    )
    assert record["elapsed_s"] == 3.5
    assert record["processor"] == "cuda"
    assert record["gpu"] == "test GPU 2"
    assert record["projection_backend"] == record["backprojection_backend"] == "cpu"


@pytest.mark.gui
def test_editor_passes_sample_spacing_and_selected_backend(qapp, monkeypatch, capsys, tmp_path):
    from pyqtgraph.Qt import QtWidgets
    from xrayrecon.gui.app import MainWin
    from xrayrecon.stackup import StackupConfig

    win = MainWin(dataCube=np.ones((8, 10, 2), dtype=np.float32))
    win._gerber_voxel_spacing_mm = (0.2, 0.2, 0.05)
    monkeypatch.setattr(win, "_voxelize_stackup", lambda *args: (_volume(), [], [], []))
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *args: None)
    monkeypatch.setattr(win.model, "setCube", lambda *args: None)
    monkeypatch.setattr(win.model, "setSpacing", lambda *args: None)
    monkeypatch.setattr(win.viewer_3d_left, "setDataCube", lambda *args: None)
    monkeypatch.setattr(win.viewer_3d_left, "resetCamera", lambda: None)
    monkeypatch.setattr(win.amAnalysisPanel, "setResult", lambda *args: None)
    monkeypatch.setattr(win, "_showAMAnalysis", lambda: None)
    monkeypatch.setattr(win, "_renderAMVolume", lambda: None)
    capsys.readouterr()
    config = StackupConfig(resolution_xy=0.2, resolution_z=0.05)
    path = tmp_path / "board.json"
    try:
        win._simulateXrayFromEditor(config, path, "cpu")
        assert _timing(capsys).startswith("[Simulate X-ray -> Left Pane] CPU | Total: ")
        win._showAlvarezMacovskiPlot(config, path, "cpu")
        assert _timing(capsys).startswith("[Show Z_eff / rho_e Plot] CPU | Total: ")
        result = win._am_result
        assert result is not None
        am_record = result.run_metrics
        assert am_record["sample_file"] == str(path)
        assert am_record["voxel_spacing_mm_xyz"] == [0.2, 0.2, 0.05]
        assert am_record["projection_backend"] == am_record["spectrum_backend"] == "cpu"
        win._showAMVolumeIn3D(config, path, "cpu")
        assert _timing(capsys).startswith("[Show AM Volume in 3D] CPU | Total: ")
    finally:
        win.close()

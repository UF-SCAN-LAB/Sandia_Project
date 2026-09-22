import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import xrayrecon.gui.app as app_module
from xrayrecon.alvarez_macovski import AMResult
from xrayrecon.gui.app import MainWin
from xrayrecon.gui.am_plot_window import AMAnalysisPanel
from xrayrecon.gui.registration import RegistrationPanel
from xrayrecon.gui.stackup_editor import StackupEditorWindow
from xrayrecon.gui.helpers import phantomSpheres
from xrayrecon.stackup import StackupConfig, StackupLayer


def _silence_messages(monkeypatch):
    from pyqtgraph.Qt import QtWidgets

    seen = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "information",
        staticmethod(lambda _parent, title, message, *args, **kwargs: seen.append((title, message))),
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "critical",
        staticmethod(lambda *args, **kwargs: None),
    )
    return seen


def _registration_ready_mainwin(monkeypatch):
    seen = _silence_messages(monkeypatch)
    win = MainWin(dataCube=phantomSpheres())
    win._loadSyntheticRegistrationDemo()
    win.openRegistrationTool()
    return win, seen


@pytest.mark.gui
def test_stackup_am_processor_selection_reaches_both_actions(qapp):
    calls = []
    config = StackupConfig(
        layers=[StackupLayer(file="layer.gbr", name="L1", thickness=0.1)]
    )
    editor = StackupEditorWindow(
        lambda *_args: None,
        lambda _config, _path, processor: calls.append(("ct", processor)),
        lambda _config, _path, processor: calls.append(("plot", processor)),
        lambda _config, _path, processor: calls.append(("volume", processor)),
    )
    editor.setConfig(config, Path.cwd())

    assert editor.processorCombo.currentData() == "cpu"
    editor._onShowAMPlot()
    editor.processorCombo.setCurrentIndex(editor.processorCombo.findData("cuda"))
    editor._onSimulate()
    editor._onShowAMVolume()

    assert calls == [("plot", "cpu"), ("ct", "cuda"), ("volume", "cuda")]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows DLL-order regression")
def test_optional_torch_dll_preparation_allows_qt_then_torch_import():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("optional PyTorch is not installed")
    code = """
import sys
from xrayrecon.windows_cuda import prepare_optional_torch_dlls
assert prepare_optional_torch_dlls()
assert "torch" not in sys.modules
import pyqtgraph
import torch
"""
    subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
    )


def test_gui_startup_does_not_require_optional_torch():
    code = """
import importlib.abc
import sys
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ImportError("blocked optional torch")
sys.meta_path.insert(0, NoTorch())
from xrayrecon.gui import MainWin
assert MainWin is not None
assert "torch" not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
    )


@pytest.mark.gui
def test_stackup_am_cuda_selection_drives_both_backends_and_physical_spacing(
    qapp, monkeypatch
):
    win = MainWin(dataCube=phantomSpheres())
    _silence_messages(monkeypatch)
    volume = np.zeros((1, 6, 8, 2), dtype=np.float32)
    config = StackupConfig(layers=[], resolution_xy=0.25, resolution_z=0.05)
    captured = {}
    expected = {"complete": True}

    monkeypatch.setattr(
        win,
        "_voxelize_stackup",
        lambda *_args, **_kwargs: (volume, [], [], []),
    )

    def fake_simulation(actual_volume, **kwargs):
        captured.update(volume=actual_volume, kwargs=kwargs)
        return expected

    monkeypatch.setattr(app_module, "simulate_am_volume", fake_simulation)

    actual = win._runAMDecomposition(config, Path.cwd(), "cuda")

    assert actual is expected
    assert captured["volume"] is volume
    assert captured["kwargs"] == {
        "pixel_size_cm": 0.025,
        "backend": "cuda",
        "projection_backend": "cuda",
    }


@pytest.mark.gui
def test_stackup_ct_cuda_selection_drives_projection_backend(qapp, monkeypatch):
    win = MainWin(dataCube=phantomSpheres())
    _silence_messages(monkeypatch)
    volume = np.zeros((1, 6, 8, 2), dtype=np.float32)
    config = StackupConfig(layers=[], resolution_xy=0.25, resolution_z=0.05)
    captured = {}

    monkeypatch.setattr(
        win,
        "_voxelize_stackup",
        lambda *_args, **_kwargs: (volume, [], [], []),
    )

    def fake_ct(actual_volume, **kwargs):
        captured.update(volume=actual_volume, kwargs=kwargs)
        return np.zeros(volume.shape[:3], dtype=np.float32)

    monkeypatch.setattr(app_module, "simulate_ct_volume", fake_ct)
    monkeypatch.setattr(win.viewer_3d_left, "setDataCube", lambda *_args: None)
    monkeypatch.setattr(win.viewer_3d_left, "resetCamera", lambda: None)

    win._simulateXrayFromEditor(config, Path.cwd(), "cuda")

    assert captured["volume"] is volume
    assert captured["kwargs"] == {
        "num_angles": 8,
        "projection_backend": "cuda",
    }


@pytest.mark.gui
@pytest.mark.integration
def test_registration_workspace_is_embedded_in_main_window(qapp, monkeypatch):
    win, _seen = _registration_ready_mainwin(monkeypatch)

    assert isinstance(win.registrationPanel, RegistrationPanel)
    assert not win.registrationPanel.isWindow()
    assert win.registrationPanel.window() is win
    assert win._workspaceStack.currentWidget() is win.registrationPanel
    assert not hasattr(win, "_registrationWindow")
    assert "Gerber = red" in win.registrationPanel.overlayGuide.text()
    assert win.registrationPanel.topViewBtn.isEnabled()
    assert win.registrationPanel.isoViewBtn.isEnabled()


@pytest.mark.gui
@pytest.mark.integration
def test_showcase_loads_material_view_and_named_layer_bands(qapp, monkeypatch):
    win, seen = _registration_ready_mainwin(monkeypatch)

    assert win._current_rgba_base_right.shape == win._gerber_intensity_hwz.shape + (4,)
    assert win._current_rgba_base_right.dtype == np.uint8
    assert win._gerber_material_id_vol_hwz.shape == (96, 128, 24)
    assert win._showcase_am_volume_zhw2.shape == (24, 96, 128, 2)
    assert win.model.layerDf["Layer Name"].tolist() == [
        "Bottom copper routing",
        "FR4 core and plated vias",
        "Top copper and gold finish",
        "Mounted components",
    ]
    assert any(title == "PCB Registration Showcase Loaded" for title, _ in seen)


@pytest.mark.gui
@pytest.mark.integration
def test_showcase_am_action_populates_all_visualization_modes(qapp, monkeypatch):
    win, seen = _registration_ready_mainwin(monkeypatch)
    expected_shape = win._showcase_am_volume_zhw2.shape[:3]
    called = {}

    def fake_decomposition(volume, *, config):
        called["shape"] = volume.shape
        called["angles"] = config.num_angles
        zeff = volume[..., 0].astype(np.float32)
        rhoe = volume[..., 1].astype(np.float32)
        labels = np.moveaxis(win._gerber_material_id_vol_hwz, -1, 0)
        return AMResult(
            {
                "zeff_ground_truth": zeff,
                "rhoe_ground_truth": rhoe,
                "zeff_recovered": zeff,
                "rhoe_recovered": rhoe,
                "zeff_regularized": zeff,
                "rhoe_regularized": rhoe,
                "material_labels_recovered": labels,
                "projection_residual": np.zeros(expected_shape, dtype=np.float32),
                "confidence": np.ones(expected_shape, dtype=np.float32),
                "rhoe_e23_cm3_ground_truth": rhoe * 3.343,
                "rhoe_e23_cm3_recovered": rhoe * 3.343,
                "rhoe_e23_cm3_regularized": rhoe * 3.343,
            },
            acquisition_config=config,
            has_ground_truth=True,
        )

    class FakeCalibration:
        calibration_id = "fake-synthetic-calibration"
        nominal_coverage = 0.90
        calibration_source = "synthetic_reference"

        def apply(self, result):
            result.update(
                zeff_calibrated=result["zeff_regularized"],
                rhoe_calibrated=result["rhoe_recovered"],
                zeff_interval_lower=result["zeff_regularized"] * 0.9,
                zeff_interval_upper=result["zeff_regularized"] * 1.1,
                rhoe_interval_lower=result["rhoe_recovered"] * 0.9,
                rhoe_interval_upper=result["rhoe_recovered"] * 1.1,
                confidence_calibrated=result["confidence"],
            )
            result.calibration_model = self
            result.calibration_id = self.calibration_id
            result.calibration_compatibility = "verified"
            return result

    monkeypatch.setattr(app_module, "simulate_am_volume", fake_decomposition)
    monkeypatch.setattr(
        win, "_buildShowcaseAMCalibration", lambda: FakeCalibration()
    )

    win.runPcbShowcaseAM()

    assert called == {"shape": (24, 96, 128, 2), "angles": 64}
    assert win._am_result["zeff_regularized"].shape == expected_shape
    assert isinstance(win.amAnalysisPanel, AMAnalysisPanel)
    assert not win.amAnalysisPanel.isWindow()
    assert win.amAnalysisPanel.window() is win
    assert win._workspaceStack.currentWidget() is win.amAnalysisPanel
    assert win.amAnalysisPanel.result is win._am_result
    assert "synthetic reference calibration" in win.amAnalysisPanel.calibrationStatus.text()
    assert "acquisition physics verified" in win.amAnalysisPanel.calibrationStatus.text()
    assert win.amAnalysisPanel.modeCombo.currentData() == "calibrated"
    assert not hasattr(win, "_amPlotWindow")
    assert win.layerPanel.isHidden()
    assert win._plotDock.isHidden()
    assert win._current_rgba_base_left.shape == (96, 128, 24, 4)
    assert any(title == "Showcase AM Decomposition Complete" for title, _ in seen)

    win.amColorRhoeBtn.click()
    assert win._current_rgba_base_left.shape == (96, 128, 24, 4)
    win.amColorMaterialBtn.click()
    assert win._current_rgba_base_left.shape == (96, 128, 24, 4)

    win.amAnalysisPanel.backBtn.click()
    assert win._workspaceStack.currentWidget() is win._mainWorkspacePage
    assert not win.layerPanel.isHidden()
    assert not win._plotDock.isHidden()


@pytest.mark.gui
@pytest.mark.integration
def test_registration_edit_survives_return_to_main_workspace(qapp, monkeypatch):
    win, _seen = _registration_ready_mainwin(monkeypatch)
    win.registrationPanel.txSbox.setValue(0.35)
    win.registrationPanel._updatePreview()

    win.registrationPanel.backBtn.click()
    assert win._workspaceStack.currentWidget() is win._mainWorkspacePage
    win.openRegistrationTool()

    assert win._workspaceStack.currentWidget() is win.registrationPanel
    assert np.isclose(win.registrationPanel.currentTransform().tx, 0.35)
    assert np.isclose(win._registration_session.current_transform.tx, 0.35)


@pytest.mark.gui
@pytest.mark.integration
def test_suggestion_neither_accepts_nor_transfers_labels(qapp, monkeypatch):
    win, _seen = _registration_ready_mainwin(monkeypatch)
    before_rgba = win._current_rgba_base_left

    win.registrationPanel.suggestBtn.click()

    assert win._registration_session.proposal is not None
    assert win._registration_session.accepted_transform is None
    assert win._current_rgba_base_left is before_rgba


@pytest.mark.gui
@pytest.mark.integration
def test_same_shape_ct_replacement_invalidates_accepted_registration(qapp, monkeypatch):
    win, seen = _registration_ready_mainwin(monkeypatch)
    win.registrationPanel.acceptBtn.click()
    seen.clear()
    replacement = win.model.dataCube.copy()
    spacing = win.model.spacing_xyz

    win.model.setCube(replacement)
    win.model.setSpacing(*spacing)
    win.transferGerberLabels()

    assert any(title == "Registration Out Of Date" for title, _message in seen)


@pytest.mark.gui
@pytest.mark.integration
def test_same_shape_gerber_replacement_invalidates_accepted_registration(qapp, monkeypatch):
    win, seen = _registration_ready_mainwin(monkeypatch)
    win.registrationPanel.acceptBtn.click()
    seen.clear()

    win._gerber_intensity_hwz = win._gerber_intensity_hwz.copy()
    win.transferGerberLabels()

    assert any(title == "Registration Out Of Date" for title, _message in seen)

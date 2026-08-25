import numpy as np
import pytest

import xrayrecon.gui.app as app_module
from xrayrecon.gui.app import MainWin
from xrayrecon.gui.am_plot_window import AMAnalysisPanel
from xrayrecon.gui.registration import RegistrationPanel
from xrayrecon.gui.helpers import phantomSpheres


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

    def fake_decomposition(volume, num_angles):
        called["shape"] = volume.shape
        called["angles"] = num_angles
        zeff = volume[..., 0].astype(np.float32)
        rhoe = volume[..., 1].astype(np.float32)
        labels = np.moveaxis(win._gerber_material_id_vol_hwz, -1, 0)
        return {
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
        }

    monkeypatch.setattr(app_module, "simulate_am_volume", fake_decomposition)

    win.runPcbShowcaseAM()

    assert called == {"shape": (24, 96, 128, 2), "angles": 64}
    assert win._am_result["zeff_regularized"].shape == expected_shape
    assert isinstance(win.amAnalysisPanel, AMAnalysisPanel)
    assert not win.amAnalysisPanel.isWindow()
    assert win.amAnalysisPanel.window() is win
    assert win._workspaceStack.currentWidget() is win.amAnalysisPanel
    assert win.amAnalysisPanel.result is win._am_result
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

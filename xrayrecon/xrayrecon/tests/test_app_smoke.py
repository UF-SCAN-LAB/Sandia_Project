import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets

from xrayrecon.gui.app import MainWin
from xrayrecon.gui.am_plot_window import AMAnalysisPanel, MaterialLegendWidget
from xrayrecon.gui.helpers import phantomSpheres

NEW_PHASE2_ACTIONS = (
    "Set Real CT Voxel Spacing...",
    "Register Gerber to Real CT...",
    "Transfer Gerber Labels to Real CT",
    "Debug: Load Synthetic Registration Demo",
    "Regenerate Z_eff / rho_e Plot",
    "Load PCB Registration Showcase",
    "Run Showcase Z-effective / Electron Density",
    "Load Real PCB Demo (Gerber + CT)",
)


def test_mainwin_builds_with_phantom_data(qapp):
    win = MainWin(dataCube=phantomSpheres())

    assert win.viewer_3d_left is not None
    assert win.viewer_3d_right is not None
    assert win.model.dataCube is not None
    assert win.registrationPanel.parent() is win._workspaceStack
    assert win.registrationPanel.window() is win
    assert win._workspaceStack.currentWidget() is win._mainWorkspacePage


def test_mainwin_registers_phase2_actions(qapp):
    win = MainWin(dataCube=phantomSpheres())

    registered = set(win.toolsEditor.nameFunctionMap.keys())
    for action_name in NEW_PHASE2_ACTIONS:
        assert action_name in registered


def test_material_legend_stays_compact(qapp):
    legend = MaterialLegendWidget()

    legend.resize(720, legend.sizeHint().height())
    legend.show()
    qapp.processEvents()

    assert legend.maximumWidth() > 10000
    assert legend.maximumHeight() <= 96
    assert legend.sizeHint().height() <= 96
    assert legend._columnCount > 1
    assert legend.findChild(QtWidgets.QScrollArea) is not None


def test_view_menu_restores_closed_plot_and_tools_docks(qapp):
    win = MainWin(dataCube=phantomSpheres())

    assert not hasattr(win, "topRibbon")

    win._toolsDock.close()
    win._plotDock.close()
    qapp.processEvents()

    assert win._toolsDock.isHidden()
    assert win._plotDock.isHidden()

    win.showToolsAction.trigger()
    win.showPlotDockAction.trigger()
    qapp.processEvents()

    assert not win._toolsDock.isHidden()
    assert not win._plotDock.isHidden()


def _cached_am_result():
    data = np.arange(32, dtype=np.float32).reshape(2, 4, 4) / 32 + 1
    return {
        "zeff_ground_truth": data * 6,
        "rhoe_ground_truth": data,
        "zeff_recovered": data * 5.5,
        "rhoe_recovered": data * 0.9,
        "zeff_regularized": data * 5.75,
        "rhoe_regularized": data * 0.95,
        "projection_residual": data * 0.01,
        "confidence": np.ones_like(data),
    }


def _cached_am_action(win, menu_name):
    name = "Regenerate Z_eff / rho_e Plot"
    if menu_name == "Tools":
        return win.toolsEditor.rootParameter.child(name).activate
    menu = next(
        action.menu() for action in win.menuBar().actions() if action.text() == menu_name
    )
    return next(action for action in menu.actions() if action.text() == name).trigger


@pytest.mark.parametrize("menu_name", ["Tools", "View"])
def test_cached_am_action_opens_analysis_and_returns_to_viewer(qapp, monkeypatch, menu_name):
    win = MainWin(dataCube=phantomSpheres())
    win._am_result = _cached_am_result()
    original = {key: value.copy() for key, value in win._am_result.items()}
    viewer_cube = win.model.dataCube
    panel = win.amAnalysisPanel
    open_analysis = _cached_am_action(win, menu_name)

    def unexpected_simulation(*args, **kwargs):
        pytest.fail("Opening cached analysis must not rerun the decomposition")

    monkeypatch.setattr(win, "_runAMDecomposition", unexpected_simulation)
    open_analysis()
    qapp.processEvents()

    assert isinstance(panel, AMAnalysisPanel)
    assert win._workspaceStack.currentWidget() is panel
    assert panel.result is win._am_result
    assert not panel.isWindow()
    assert not hasattr(win, "_amPlotWindow")
    assert win.layerPanel.isHidden()
    assert win._plotDock.isHidden()
    assert "SNR" in panel.metricSummary.text()
    assert panel.backBtn.text() == "Return to Viewer"

    panel.backBtn.click()
    qapp.processEvents()

    assert win._workspaceStack.currentWidget() is win._mainWorkspacePage
    assert not win.layerPanel.isHidden()
    assert not win._plotDock.isHidden()
    assert win.model.dataCube is viewer_cube

    open_analysis()
    qapp.processEvents()
    assert win._workspaceStack.currentWidget() is panel
    assert win.amAnalysisPanel is panel
    for key, value in original.items():
        np.testing.assert_array_equal(win._am_result[key], value)
    win.close()


@pytest.mark.parametrize("menu_name", ["Tools", "View"])
def test_cached_am_action_without_result_keeps_viewer(qapp, monkeypatch, menu_name):
    win = MainWin(dataCube=phantomSpheres())
    messages = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "information",
        lambda parent, title, text: messages.append((title, text)),
    )

    _cached_am_action(win, menu_name)()
    qapp.processEvents()

    assert messages == [("No AM Result", "Run an Alvarez-Macovski decomposition first.")]
    assert win._workspaceStack.currentWidget() is win._mainWorkspacePage
    assert win.amAnalysisPanel.result is None
    win.close()

import numpy as np
from pyqtgraph.Qt import QtWidgets

from xrayrecon.gui.app import MainWin
from xrayrecon.gui.am_plot_window import AMPlotWindow, MaterialLegendWidget
from xrayrecon.gui.helpers import phantomSpheres

NEW_PHASE2_ACTIONS = (
    "Set Real CT Voxel Spacing...",
    "Register Gerber to Real CT...",
    "Transfer Gerber Labels to Real CT",
    "Debug: Load Synthetic Registration Demo",
    "Regenerate Z_eff / rho_e Plot",
)


def test_mainwin_builds_with_phantom_data(qapp):
    win = MainWin(dataCube=phantomSpheres())

    assert win.viewer_3d_left is not None
    assert win.viewer_3d_right is not None
    assert win.model.dataCube is not None


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


def test_view_menu_regenerates_cached_am_plot(qapp):
    win = MainWin(dataCube=phantomSpheres())
    data = np.ones((2, 4, 4), dtype=np.float32)
    win._am_result = {
        "zeff_ground_truth": data * 6,
        "rhoe_ground_truth": data,
        "zeff_recovered": data * 5.5,
        "rhoe_recovered": data * 0.9,
    }

    win.regenerateAMPlotAction.trigger()
    qapp.processEvents()

    firstWindow = win._amPlotWindow
    assert isinstance(firstWindow, AMPlotWindow)

    win.regenerateAMPlotAction.trigger()
    qapp.processEvents()

    assert isinstance(win._amPlotWindow, AMPlotWindow)
    assert win._amPlotWindow is not firstWindow

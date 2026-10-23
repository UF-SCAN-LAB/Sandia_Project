"""Regressions for the integrated window, not just standalone analysis panels."""
import numpy as np
import pytest
import os
from pathlib import Path
from pyqtgraph.Qt import QtCore, QtGui, QtTest

from xrayrecon.gui.app import MainWin
from xrayrecon.gui.am_plot_window import AMAnalysisPanel


@pytest.fixture(autouse=True)
def desktop_font(qapp):
    # Windows' offscreen plugin otherwise uses missing Helvetica glyphs with
    # inflated advances ("Return to Viewer" measures 280 px instead of 105).
    # Load a real local font so pixel-size checks reflect a supported desktop.
    font_file = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/segoeui.ttf"
    original = qapp.font()
    font_id = -1
    if font_file.exists():
        font_id = QtGui.QFontDatabase.addApplicationFont(str(font_file))
        families = QtGui.QFontDatabase.applicationFontFamilies(font_id)
        if families:
            qapp.setFont(QtGui.QFont(families[0], 9))
    yield
    qapp.setFont(original)
    if font_id >= 0:
        QtGui.QFontDatabase.removeApplicationFont(font_id)


def _result():
    data = np.ones((12, 20, 24), dtype=np.float32)
    for index in range(len(data)):
        data[index] *= 10 ** (index / 2 - 3)
    return dict(zeff_ground_truth=data, rhoe_ground_truth=data,
                zeff_recovered=data * .9, rhoe_recovered=data * .9,
                zeff_regularized=data, rhoe_regularized=data,
                confidence=np.ones_like(data), projection_residual=data / data.max())


def _settle(qapp):
    for _ in range(3):
        qapp.processEvents()
        QtTest.QTest.qWait(10)


@pytest.mark.gui
def test_analysis_uses_available_window_and_restores_docks(qapp):
    win = MainWin(dataCube=np.ones((12, 14, 3), dtype=np.float32))
    win.setAttribute(QtCore.Qt.WA_DontShowOnScreen, True)
    win.addDockWidget(QtCore.Qt.RightDockWidgetArea, win._toolsDock)
    win._plotDock.hide()  # preserve a user-closed dock on returning
    win._am_result = _result()
    win.resize(1500, 950)
    win.show()
    try:
        win.showCachedAlvarezMacovskiPlot()
        _settle(qapp)
        assert win._toolsDock.isHidden()
        assert win._plotDock.isHidden()
        assert win.gerberLegendLabel.isHidden()
        for size in ((1500, 950), (900, 650), (1600, 1000)):
            win.resize(*size)
            _settle(qapp)
            assert (win.width(), win.height()) == size, {
                name: (widget.minimumSizeHint().width(), widget.minimumWidth())
                for name, widget in (("stack", win._workspaceStack), ("overview", win.amAnalysisPanel),
                                     ("volume", win._mainWorkspacePage), ("registration", win.registrationPanel),
                                     ("back", win.amAnalysisPanel.backBtn), ("mode", win.amAnalysisPanel.modeCombo),
                                     ("quality", win.amAnalysisPanel.qualityCombo), ("canvas", win.amAnalysisPanel.canvas))
            }
            assert abs(win._workspaceStack.width() - win.width()) <= 2
            assert win.amAnalysisPanel.size() == win._workspaceStack.size()
            assert win.amAnalysisPanel.canvas.width() >= win.width() - 30
        win.gerberLegendLabel.setText("Gerber legend " + "board-COPPER rgba(255,140,0,200); " * 100)
        win.showErrorInspection()
        _settle(qapp)
        assert win.errorInspectionPanel.size() == win._workspaceStack.size()
        assert abs(win._workspaceStack.width() - win.width()) <= 2
        win._showMainWorkspace()
        _settle(qapp)
        assert not win._toolsDock.isHidden()
        assert win._plotDock.isHidden()
        assert not win.gerberLegendLabel.isHidden()
        assert win.gerberLegendLabel.toolTip() == win.gerberLegendLabel.text()
    finally:
        if win.errorInspectionPanel is not None:
            win.errorInspectionPanel.cancel()
            win.errorInspectionPanel._pool.waitForDone(5000)
        win.close()


@pytest.mark.gui
def test_slice_and_variant_changes_preserve_axes_and_colorbar_geometry(qapp):
    panel = AMAnalysisPanel()
    panel.resize(1250, 850)
    panel.show()
    result = _result()
    original = {key: value.copy() for key, value in result.items()}
    try:
        panel.setResult(result)
        panel.sliceSlider.setValue(0)
        _settle(qapp)
        panel.canvas.draw()
        axes = tuple(panel.figure.axes)
        bounds = [ax.get_position().bounds for ax in axes]
        canvas_geometry = panel.canvas.geometry().getRect()
        for mode in ("raw", "regularized", "raw"):
            panel.modeCombo.setCurrentIndex(panel.modeCombo.findData(mode))
            for index in (0, 5, 11, 0):
                panel.sliceSlider.setValue(index)
                _settle(qapp)
                panel.canvas.draw()
                assert tuple(panel.figure.axes) == axes
                np.testing.assert_allclose([ax.get_position().bounds for ax in axes], bounds, atol=1e-10)
                assert panel.canvas.geometry().getRect() == canvas_geometry
                np.testing.assert_array_equal(panel.figure.axes[0].images[0].get_array(), result["zeff_ground_truth"][index])
        panel.resize(900, 650)
        _settle(qapp)
        panel.canvas.draw()
        # Labels must stay within the canvas after an actual Qt resize.
        renderer = panel.canvas.get_renderer()
        for axis in panel.figure.axes:
            box = axis.get_tightbbox(renderer)
            assert box.x0 >= -2 and box.x1 <= panel.figure.bbox.width + 2
            assert box.y0 >= -2 and box.y1 <= panel.figure.bbox.height + 2
        for key, value in original.items():
            np.testing.assert_array_equal(result[key], value)
    finally:
        panel.close()


@pytest.mark.gui
def test_replacing_result_with_fewer_slices_does_not_draw_stale_index(qapp):
    panel = AMAnalysisPanel()
    result = _result()
    panel.setResult(result)
    panel.sliceSlider.setValue(11)
    panel.modeCombo.setCurrentIndex(panel.modeCombo.findData("raw"))
    panel.setResult({key: value[:1] for key, value in result.items()})
    assert panel.sliceSlider.value() == 0
    panel.close()

from dataclasses import replace
import threading
import time

import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtTest

from xrayrecon.error_inspection import ComparisonContext, ComparisonGrid
from xrayrecon.gui.error_inspection import ErrorInspectionPanel


def wait_for(qapp, condition, timeout=5):
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        qapp.processEvents()
        QtTest.QTest.qWait(5)
    assert condition(), "Timed out waiting for inspection worker"


@pytest.fixture
def panel(qapp):
    truth = np.indices((3, 5, 7))[2].astype(float)
    labels = np.ones(truth.shape, dtype=np.int16)
    labels[:, 0] = 0
    ctx = ComparisonContext(ComparisonGrid(truth.shape, (.2, .5, 2.), (10., 20., 30.),
                           ((0., -1., 0.), (1., 0., 0.), (0., 0., 1.)), "mm"),
                           {"zeff": truth, "rhoe": truth / 10},
                           {("zeff", "raw"): truth + 2, ("zeff", "regularized"): truth + 1,
                            ("rhoe", "raw"): truth / 10 + .1}, labels=labels,
                           label_names={0: "Air", 1: "Test material"})
    widget = ErrorInspectionPanel()
    widget.resize(1400, 900)
    widget.setContext(ctx)
    wait_for(qapp, lambda: widget.evaluation is not None)
    yield widget
    widget.cancel()
    widget._pool.waitForDone(5000)
    widget.close()
    widget.deleteLater()
    qapp.processEvents()


@pytest.mark.gui
def test_probe_plane_mapping_linked_crosshairs_and_3d(panel, qapp):
    # QOpenGLWidget.context() must remain callable for native GL initialization.
    assert callable(panel.volume.context)
    panel.selectMapPosition((1.5, .4))
    world = panel.context.grid.voxel_to_world((1, 2, 3))
    np.testing.assert_allclose(panel.state.cursor_world, world)
    for view in panel.maps:
        assert view.cross[0].value() == pytest.approx(1.5)
        assert view.cross[1].value() == pytest.approx(.4)
    np.testing.assert_allclose(panel.volume.marker.pos, [world])
    panel.planeCombo.setCurrentText("XZ")
    wait_for(qapp, lambda: panel.evaluation is not None)
    assert panel.state.index == 3
    np.testing.assert_allclose(panel.state.cursor_world, world)
    panel.volume.sigWorldPicked.emit(panel.context.grid.voxel_to_world((2, 1, 4)))
    wait_for(qapp, lambda: panel.evaluation is not None)
    assert panel.state.index == 4
    np.testing.assert_allclose(panel.context.grid.world_to_voxel(panel.state.cursor_world), (2, 1, 4))


@pytest.mark.gui
def test_roi_line_histogram_and_shared_3d_selection(panel, qapp):
    panel.toolCombo.setCurrentIndex(panel.toolCombo.findData("roi"))
    panel.selectMapPosition((.5, .2)); panel.selectMapPosition((2., .6))
    wait_for(qapp, lambda: panel.evaluation is not None)
    assert panel.evaluation["metrics"]["valid_count"] == 12
    assert panel.evaluation["metrics"]["bias"] == 2
    panel.toolCombo.setCurrentIndex(panel.toolCombo.findData("line"))
    panel.selectMapPosition((0., .2)); panel.selectMapPosition((3., .2))
    wait_for(qapp, lambda: panel.evaluation is not None)
    assert panel.evaluation["line"]["distance"][-1] == pytest.approx(3.)
    profile = panel.maps[1].profile
    profile.movePoint(profile.getHandles()[1], (2., .2))
    wait_for(qapp, lambda: panel.evaluation is not None)
    assert panel.evaluation["line"]["distance"][-1] == pytest.approx(2.)
    panel.analysisCombo.setCurrentIndex(0)
    panel.brushItem.setRegion((1., 3.))
    wait_for(qapp, lambda: panel.evaluation is not None)
    assert panel.state.brush == (1., 3.)
    np.testing.assert_array_equal(panel.volume.highlighted_sample, panel.evaluation["highlighted"])
    assert panel.evaluation["metrics"]["valid_count"] == 12
    panel.brushItem.setRegion((3., 4.))
    wait_for(qapp, lambda: panel.evaluation is not None)
    assert panel.evaluation["brushed_metrics"]["valid_count"] == 0
    assert panel.evaluation["metrics"]["valid_count"] == 12
    panel.clearRoiBtn.click(); panel.clearBrushBtn.click()
    wait_for(qapp, lambda: panel.evaluation is not None)
    assert panel.evaluation["metrics"]["valid_count"] == 35


@pytest.mark.gui
def test_variant_quantity_ranges_and_display_do_not_mutate_sources(panel, qapp):
    original = panel.context.outputs["zeff", "raw"].copy()
    field = panel.evaluation["field"]
    initial_limits = panel.state.intensity_limits
    camera = dict(panel.volume.opts)
    panel.toleranceSpin.setValue(2.)
    wait_for(qapp, lambda: panel.evaluation is not None)
    assert panel.evaluation["metrics"]["above_tolerance"] == 0
    assert panel.evaluation["field"] is field
    panel.minSpin.setValue(-5)
    panel.clipCheck.setChecked(True)
    panel.opacitySlider.setValue(70)
    assert panel.evaluation["field"] is field
    assert panel.evaluation["metrics"]["bias"] == 2
    panel.modeCombo.setCurrentIndex(panel.modeCombo.findData("absolute"))
    assert panel.evaluation["metrics"]["bias"] == 2
    assert panel.volume.opts["distance"] == camera["distance"]
    panel.variantCombo.setCurrentIndex(panel.variantCombo.findData("regularized"))
    assert panel.evaluation is None and not panel.exportBtn.isEnabled()
    wait_for(qapp, lambda: panel.evaluation is not None)
    assert panel.evaluation["metrics"]["bias"] == 1
    assert panel.state.intensity_limits[0] == -5
    panel.quantityCombo.setCurrentIndex(1)
    wait_for(qapp, lambda: panel.evaluation is not None)
    assert panel.evaluation["metrics"]["bias"] == pytest.approx(.1)
    panel.toleranceSpin.setValue(.03)
    wait_for(qapp, lambda: panel.evaluation is not None)
    panel.quantityCombo.setCurrentIndex(0)
    wait_for(qapp, lambda: panel.evaluation is not None)
    assert panel.state.tolerance == 2.
    np.testing.assert_array_equal(original, panel.context.outputs["zeff", "raw"])


@pytest.mark.gui
def test_delayed_results_cannot_replace_new_state_and_event_loop_runs(panel, qapp, monkeypatch):
    import xrayrecon.gui.error_inspection as module
    original = module.evaluate
    entered, release = threading.Event(), threading.Event()

    def delayed(*args, **kwargs):
        entered.set()
        release.wait(3)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "evaluate", delayed)
    panel._schedule()
    wait_for(qapp, entered.is_set)
    ticks = []
    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: ticks.append(1)); timer.start(5)
    QtTest.QTest.qWait(40)
    assert ticks
    old_token = panel._token
    panel.variantCombo.setCurrentIndex(panel.variantCombo.findData("regularized"))
    release.set()
    wait_for(qapp, lambda: panel.evaluation is not None)
    assert panel._token > old_token
    assert panel.evaluation["metrics"]["bias"] == 1
    # Explicit late callback is discarded, including on cancellation.
    panel._finished(old_token, (replace(panel.state, variant="raw"), {}), None)
    assert panel.state.variant == "regularized"
    panel.cancel()
    panel._finished(old_token, None, "late error")
    assert "cancelled" in panel.status.text()
    timer.stop()


@pytest.mark.gui
def test_absent_reference_unavailable_and_small_layout(panel, qapp):
    panel.show()
    panel.resize(900, 600)
    qapp.processEvents()
    assert not panel.inspectorToggle.isChecked()
    assert not panel.analysisToggle.isChecked()
    assert panel.maps[0].width() >= 140
    ctx = replace(panel.context, references={}, labels=None, run_id="no-reference")
    panel.setContext(ctx)
    wait_for(qapp, lambda: "unavailable" in panel.status.text())
    assert panel.evaluation is None
    assert not panel.materialCombo.isEnabled()
    assert not panel.exportBtn.isEnabled()

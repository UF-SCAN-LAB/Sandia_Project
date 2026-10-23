"""Connected PyQt error inspection, using the application's existing GL renderer."""
from __future__ import annotations

from copy import deepcopy
import threading

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from ..error_inspection import (
    AXIS_NAMES, PLANES, UNITS, ComparisonContext, ErrorEvaluator,
    InspectionCancelled, InspectionState, context_from_am, evaluate,
    export_inspection,
)
from .view3d import PCBViewer3D


class _Signals(QtCore.QObject):
    finished = QtCore.Signal(int, object, object)


class _Work(QtCore.QRunnable):
    def __init__(self, token, operation, signals):
        super().__init__()
        self.token, self.operation, self.signals = token, operation, signals

    def run(self):
        try:
            self.signals.finished.emit(self.token, self.operation(), None)
        except InspectionCancelled:
            pass
        except Exception as exc:
            self.signals.finished.emit(self.token, None, str(exc))


class InspectionVolumeView(PCBViewer3D):
    """HWZ texture -> declared world coordinates; no automatic camera reset.

    The existing renderer has no world picking API. ``sigWorldPicked`` is the
    integration adapter for a future depth/ray picker, consumed by the panel.
    """
    sigWorldPicked = QtCore.Signal(object)

    def __init__(self):
        super().__init__()
        self.marker = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), color=(1., .85, .1, 1.), size=10)
        self.planeOutline = gl.GLLinePlotItem(pos=np.zeros((5, 3)), color=(.2, .8, 1., .9), width=2)
        self.roiOutline = gl.GLLinePlotItem(pos=np.zeros((5, 3)), color=(1., .8, .1, 1.), width=2)
        self.lineItem = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=(.3, 1., .7, 1.), width=2)
        for item in (self.marker, self.planeOutline, self.roiOutline, self.lineItem):
            self.addItem(item)
        self.setGridVisible(False)
        self.setAxisVisible(False)
        self.comparison_context = None
        self.highlighted_sample = None

    @staticmethod
    def rectangle(context, state, roi=None):
        normal, row, col = PLANES[state.plane]
        lo = np.full(3, -.5)
        hi = np.asarray(context.grid.shape, dtype=float) - .5
        if roi is not None:
            lo, hi = np.asarray(roi, dtype=float)
            lo, hi = lo - .5, hi + .5
        points = []
        for a, b in ((lo[row], lo[col]), (lo[row], hi[col]), (hi[row], hi[col]), (hi[row], lo[col]), (lo[row], lo[col])):
            p = np.zeros(3)
            p[normal], p[row], p[col] = state.index, a, b
            points.append(p)
        return context.grid.voxel_to_world(points)

    def updateSelection(self, context, state):
        self.marker.setData(pos=np.asarray([state.cursor_world]))
        self.planeOutline.setData(pos=self.rectangle(context, state))
        self.roiOutline.setVisible(state.roi is not None)
        if state.roi is not None:
            self.roiOutline.setData(pos=self.rectangle(context, state, state.roi))
        self.lineItem.setVisible(state.line is not None)
        if state.line:
            self.lineItem.setData(pos=np.asarray(state.line))

    def updateField(self, context, state, evaluation, opacity=.2, clip=False):
        first = self.comparison_context is not context
        self.comparison_context = context
        shape = np.asarray(context.grid.shape)
        strides = np.maximum(1, np.ceil(shape / 128).astype(int))
        slices = tuple(slice(None, None, int(s)) for s in strides)
        source = context.outputs[state.quantity, state.variant][slices]
        low, high = state.intensity_limits
        intensity = np.clip((np.nan_to_num(source.astype(float), nan=low, posinf=high, neginf=low) - low) / max(high - low, 1e-12), 0, 1)
        rgba = np.zeros(source.shape + (4,), dtype=np.uint8)
        rgba[..., :3] = (intensity[..., None] * 190 + 30).astype(np.uint8)
        rgba[..., 3] = (intensity * opacity * 180).astype(np.uint8)
        selected = evaluation["highlighted"][slices]
        # Same numerical selection in 2D/3D; only this texture is decimated.
        self.highlighted_sample = selected.copy()
        errors = evaluation["field"].signed[slices]
        negative, positive = selected & (errors < 0), selected & (errors > 0)
        if state.display_mode == "above tolerance":
            negative &= np.abs(errors) > state.tolerance
            positive &= np.abs(errors) > state.tolerance
        rgba[negative, :3] = (50, 130, 220)
        rgba[positive, :3] = (235, 100, 40)
        if state.display_mode == "absolute":
            magnitude = np.clip(np.abs(np.nan_to_num(errors, nan=0)) / state.error_limit, 0, 1)
            colors = pg.colormap.get("viridis").map(magnitude, mode="byte")
            rgba[selected, :3] = colors[selected, :3]
        rgba[negative | positive, 3] = max(30, int(opacity * 255))
        if state.brush is not None:
            rgba[selected, :3] = (255, 220, 45)
            rgba[selected, 3] = max(30, int(opacity * 255))
        rgba[~evaluation["field"].valid[slices]] = (160, 60, 170, int(opacity * 180))
        if clip:
            normal = PLANES[state.plane][0]
            cut = [slice(None)] * 3
            cut[normal] = slice(int(state.index // strides[normal]) + 1, None)
            rgba[tuple(cut)][..., 3] = 0
        self.setRGBADataCube(np.moveaxis(rgba, 0, 2), downsampleFactor=1)
        # GLVolumeItem texture cells are [i,i+1]; comparison indices are centers.
        matrix = np.eye(4)
        basis = np.asarray(context.grid.direction) @ np.diag(context.grid.spacing_hwz)
        scale = strides[[1, 2, 0]]
        matrix[:3, :3] = basis @ np.diag(scale)
        matrix[:3, 3] = np.asarray(context.grid.origin) - basis @ (scale / 2)
        self.volume.setTransform(QtGui.QMatrix4x4(*matrix.ravel().tolist()))
        self.updateSelection(context, state)
        if first:
            center = context.grid.voxel_to_world((shape - 1) / 2)
            self.setCameraPosition(pos=pg.Vector(*center), distance=float(max(shape * context.grid.spacing_zhw) * 2), elevation=30, azimuth=45)


class _Map(QtWidgets.QWidget):
    clicked = QtCore.Signal(object)

    def __init__(self, title):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        self.title = QtWidgets.QLabel(title)
        self.title.setWordWrap(True)
        self.title.setStyleSheet("font-weight: bold")
        layout.addWidget(self.title)
        self.plot = pg.PlotWidget(background="#20252c")
        self.plot.setMinimumSize(140, 110)
        self.plot.setAspectLocked(True)
        self.plot.showGrid(x=False, y=False)
        layout.addWidget(self.plot, 1)
        self.image = pg.ImageItem(axisOrder="row-major")
        self.overlay = pg.ImageItem(axisOrder="row-major")
        self.plot.addItem(self.image)
        self.plot.addItem(self.overlay)
        self.cross = [pg.InfiniteLine(angle=90, pen=pg.mkPen("#f4d84f")), pg.InfiniteLine(angle=0, pen=pg.mkPen("#f4d84f"))]
        for line in self.cross:
            self.plot.addItem(line, ignoreBounds=True)
        self.roi = self.plot.plot(pen=pg.mkPen("#f4d84f", width=2))
        self.profile = pg.LineSegmentROI([(0, 0), (1, 1)], pen=pg.mkPen("#50e8ac", width=2), movable=False)
        self.plot.addItem(self.profile, ignoreBounds=True)
        self.profile.hide()
        self.plot.scene().sigMouseClicked.connect(self._clicked)
        self.legend = QtWidgets.QLabel("")
        self.legend.setWordWrap(True)
        layout.addWidget(self.legend)

    def _clicked(self, event):
        if event.button() == QtCore.Qt.LeftButton and self.plot.getViewBox().sceneBoundingRect().contains(event.scenePos()):
            point = self.plot.getViewBox().mapSceneToView(event.scenePos())
            self.clicked.emit((point.x(), point.y()))

    def showImage(self, array, rect, limits, lut, overlay):
        self.image.setImage(array, autoLevels=False, levels=limits, lut=lut)
        self.image.setRect(rect)
        self.overlay.setImage(overlay, autoLevels=False)
        self.overlay.setRect(rect)


class ErrorInspectionPanel(QtWidgets.QWidget):
    sigCloseRequested = QtCore.Signal()
    sigVolumeRequested = QtCore.Signal()
    sigWorldSelected = QtCore.Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.context = None
        self.contexts = []
        self.state = InspectionState()
        self.evaluation = None
        self.evaluated_state = None
        self.evaluator = ErrorEvaluator()
        self._quantity_settings = {}
        self._token = 0
        self._cancel = threading.Event()
        self._pool = QtCore.QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._exportPool = QtCore.QThreadPool(self)
        self._exportPool.setMaxThreadCount(1)
        self._exporting = False
        self._signals = _Signals(self)
        self._signals.finished.connect(self._finished)
        self._exportSignals = _Signals(self)
        self._exportSignals.finished.connect(self._exportFinished)
        self._debounce = QtCore.QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self._start)
        self._first_point = None
        self._view_key = None
        self._build()

    @staticmethod
    def combo(items):
        widget = QtWidgets.QComboBox()
        for label, value in items:
            widget.addItem(label, value)
        widget.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        widget.setMinimumContentsLength(8)
        return widget

    @staticmethod
    def spin(value=0, minimum=0, maximum=1e9):
        box = QtWidgets.QDoubleSpinBox()
        box.setDecimals(5)
        box.setRange(minimum, maximum)
        box.setValue(value)
        box.setKeyboardTracking(False)
        return box

    def _build(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self.setMinimumSize(780, 520)
        layout.setContentsMargins(6, 6, 6, 6)
        header = QtWidgets.QHBoxLayout()
        self.backBtn = QtWidgets.QPushButton("← Overview")
        self.backBtn.clicked.connect(self.sigCloseRequested)
        header.addWidget(self.backBtn)
        self.volumeBtn = QtWidgets.QPushButton("Volume workspace")
        self.volumeBtn.clicked.connect(self.sigVolumeRequested)
        header.addWidget(self.volumeBtn)
        title = QtWidgets.QLabel("Error inspection")
        title.setStyleSheet("font-size: 17px; font-weight: bold")
        header.addWidget(title)
        header.addStretch()
        self.fitBtn = QtWidgets.QPushButton("Fit views")
        self.fitBtn.clicked.connect(self._fitViews)
        header.addWidget(self.fitBtn)
        self.inspectorToggle = QtWidgets.QCheckBox("Inspector")
        self.inspectorToggle.setChecked(True)
        self.analysisToggle = QtWidgets.QCheckBox("Analysis")
        self.analysisToggle.setChecked(True)
        header.addWidget(self.inspectorToggle)
        header.addWidget(self.analysisToggle)
        self.exportBtn = QtWidgets.QPushButton("Export data + state…")
        self.exportBtn.clicked.connect(self._export)
        header.addWidget(self.exportBtn)
        layout.addLayout(header)

        context = QtWidgets.QGridLayout()
        self.referenceCombo = self.combo([("Synthetic truth", "synthetic")])
        self.runCombo = self.combo([])
        self.variantCombo = self.combo([])
        self.quantityCombo = self.combo([("Z-effective", "zeff"), ("Relative electron density", "rhoe")])
        for col, (label, control) in enumerate((("Reference", self.referenceCombo), ("Run", self.runCombo), ("Output", self.variantCombo), ("Quantity", self.quantityCombo))):
            context.addWidget(QtWidgets.QLabel(label), 0, col)
            context.addWidget(control, 1, col)
        layout.addLayout(context)
        self.status = QtWidgets.QLabel("Open an AM simulation to inspect numerical errors.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        nav = QtWidgets.QHBoxLayout()
        self.planeCombo = self.combo([(p, p) for p in PLANES])
        self.sliceSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sliceLabel = QtWidgets.QLabel("Slice —")
        self.toolCombo = self.combo([("Probe", "probe"), ("Line", "line"), ("ROI", "roi"), ("Pan / zoom", "pan")])
        self.toolCombo.setToolTip("Probe: click a voxel. Line: click two endpoints. ROI: click two opposite corners. Pan / zoom: drag or use the wheel.")
        self.clearRoiBtn = QtWidgets.QPushButton("Clear ROI")
        self.clearLineBtn = QtWidgets.QPushButton("Clear line")
        self.cancelBtn = QtWidgets.QPushButton("Cancel")
        for widget in (self.planeCombo, self.sliceLabel, self.sliceSlider, self.toolCombo, self.clearRoiBtn, self.clearLineBtn, self.cancelBtn):
            nav.addWidget(widget, 1 if widget is self.sliceSlider else 0)
        layout.addLayout(nav)

        self.outerSplit = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.leftSplit = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.mapSplit = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        top = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        bottom = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.topMapSplit, self.bottomMapSplit = top, bottom
        self.maps = [_Map("Reference"), _Map("Reconstruction"), _Map("Signed error · R − T")]
        top.addWidget(self.maps[0]); top.addWidget(self.maps[1])
        bottom.addWidget(self.maps[2])
        volumeBox = QtWidgets.QWidget()
        vl = QtWidgets.QVBoxLayout(volumeBox)
        vl.setContentsMargins(2, 2, 2, 2)
        vl.addWidget(QtWidgets.QLabel("3D context · existing volume renderer"))
        self.volume = InspectionVolumeView()
        self.volume.setMinimumSize(150, 100)
        vl.addWidget(self.volume, 1)
        threeControls = QtWidgets.QHBoxLayout()
        threeControls.addWidget(QtWidgets.QLabel("Opacity"))
        self.opacitySlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.opacitySlider.setRange(0, 100); self.opacitySlider.setValue(25)
        self.clipCheck = QtWidgets.QCheckBox("Clip at slice")
        threeControls.addWidget(self.opacitySlider, 1); threeControls.addWidget(self.clipCheck)
        vl.addLayout(threeControls)
        note = QtWidgets.QLabel("2D → 3D linked · texture ≤128³ · rotate/zoom freely")
        note.setWordWrap(True); vl.addWidget(note)
        bottom.addWidget(volumeBox)
        for splitter in (top, bottom):
            splitter.setStretchFactor(0, 1)
            splitter.setStretchFactor(1, 1)
            splitter.setSizes([500, 500])
        top.splitterMoved.connect(lambda pos, _: bottom.setSizes(top.sizes()))
        bottom.splitterMoved.connect(lambda pos, _: top.setSizes(bottom.sizes()))
        self.mapSplit.addWidget(top); self.mapSplit.addWidget(bottom)
        self.leftSplit.addWidget(self.mapSplit)

        self.analysisBox = QtWidgets.QWidget()
        al = QtWidgets.QVBoxLayout(self.analysisBox); al.setContentsMargins(0, 0, 0, 0)
        analysisControls = QtWidgets.QHBoxLayout()
        self.analysisCombo = self.combo([("Signed-error histogram", "histogram"), ("Line profile", "profile"), ("Line signed difference", "difference")])
        self.analysisCombo.setMinimumContentsLength(23)
        self.brushLabel = QtWidgets.QLabel("No histogram selection")
        self.brushLabel.setWordWrap(True)
        self.clearBrushBtn = QtWidgets.QPushButton("Clear histogram selection")
        analysisControls.addWidget(self.analysisCombo); analysisControls.addWidget(self.brushLabel, 1); analysisControls.addWidget(self.clearBrushBtn)
        al.addLayout(analysisControls)
        self.analysisPlot = pg.PlotWidget(background="#20252c")
        self.analysisPlot.setMinimumHeight(110)
        al.addWidget(self.analysisPlot, 1)
        self.brushItem = pg.LinearRegionItem(values=(-.5, .5), brush=(80, 180, 230, 45))
        self.brushItem.sigRegionChangeFinished.connect(self._brushChanged)
        self.analysisPlot.scene().sigMouseClicked.connect(self._histogramClicked)
        self.leftSplit.addWidget(self.analysisBox)
        self.leftSplit.setSizes([580, 190])
        self.outerSplit.addWidget(self.leftSplit)

        self.inspector = QtWidgets.QScrollArea()
        self.inspector.setWidgetResizable(True)
        self.inspector.setMinimumWidth(220)
        content = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(content)
        form.setRowWrapPolicy(QtWidgets.QFormLayout.WrapLongRows)
        self.scopeCombo = self.combo([("Current slice (+ ROI)", "slice"), ("Entire volume (+ ROI)", "volume")])
        self.materialCombo = self.combo([("All reference materials", None)])
        self.regionCombo = self.combo([("All regions", "all"), ("Material interiors", "interior"), ("Boundary band", "boundary"), ("Background", "background")])
        self.boundarySpin = self.spin(1)
        for label, widget in (("Scope", self.scopeCombo), ("Material", self.materialCombo), ("Region", self.regionCombo), ("Boundary width", self.boundarySpin)):
            form.addRow(label, widget)
        self.selectionLabel = QtWidgets.QLabel("Selected location —")
        self.metricsLabel = QtWidgets.QLabel("Measurements —")
        self.subsetLabel = QtWidgets.QLabel("")
        for widget in (self.selectionLabel, self.metricsLabel, self.subsetLabel):
            widget.setWordWrap(True); widget.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            form.addRow(widget)
        self.toleranceSpin = self.spin()
        self.toleranceSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.toleranceSlider.setRange(0, 1000)
        form.addRow("Exploratory tolerance", self.toleranceSpin)
        form.addRow(self.toleranceSlider)
        self.modeCombo = self.combo([("Signed error", "signed"), ("Absolute error", "absolute"), ("Above tolerance", "above tolerance")])
        form.addRow("Error map", self.modeCombo)
        self.minSpin, self.maxSpin = self.spin(0, -1e9), self.spin(1, -1e9)
        self.errorSpin = self.spin(1, .00001)
        form.addRow("Shared window min", self.minSpin)
        form.addRow("Shared window max", self.maxSpin)
        form.addRow("Error scale ±", self.errorSpin)
        self.resetRangeBtn = QtWidgets.QPushButton("Reset ranges to this run")
        form.addRow(self.resetRangeBtn)
        self.provenanceLabel = QtWidgets.QLabel("")
        self.provenanceLabel.setWordWrap(True)
        form.addRow(self.provenanceLabel)
        self.inspector.setWidget(content)
        self.outerSplit.addWidget(self.inspector)
        self.outerSplit.setSizes([1000, 290])
        layout.addWidget(self.outerSplit, 1)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 0); self.progress.setMaximumHeight(5); self.progress.hide()
        layout.addWidget(self.progress)
        for map_view in self.maps:
            map_view.clicked.connect(self.selectMapPosition)
            map_view.profile.sigRegionChangeFinished.connect(lambda _roi, view=map_view: self._lineMoved(view))
        for map_view in self.maps[1:]:
            map_view.plot.setXLink(self.maps[0].plot)
            map_view.plot.setYLink(self.maps[0].plot)
        self.volume.sigWorldPicked.connect(self.selectWorld)
        self.runCombo.currentIndexChanged.connect(self._runChanged)
        self.quantityCombo.currentIndexChanged.connect(self._quantityChanged)
        self.variantCombo.currentIndexChanged.connect(self._variantChanged)
        self.planeCombo.currentIndexChanged.connect(self._planeChanged)
        self.sliceSlider.valueChanged.connect(self._sliceChanged)
        self.toolCombo.currentIndexChanged.connect(self._toolChanged)
        for widget in (self.scopeCombo, self.materialCombo, self.regionCombo):
            widget.currentIndexChanged.connect(self._filtersChanged)
        self.boundarySpin.valueChanged.connect(self._filtersChanged)
        self.toleranceSpin.valueChanged.connect(self._toleranceChanged)
        self.toleranceSlider.valueChanged.connect(lambda v: self.toleranceSpin.setValue(v * self.state.error_limit / 1000))
        self.modeCombo.currentIndexChanged.connect(self._displayChanged)
        for widget in (self.minSpin, self.maxSpin, self.errorSpin):
            widget.valueChanged.connect(self._displayChanged)
        self.analysisCombo.currentIndexChanged.connect(self._drawAnalysis)
        self.clearRoiBtn.clicked.connect(lambda: self._clear("roi"))
        self.clearLineBtn.clicked.connect(lambda: self._clear("line"))
        self.clearBrushBtn.clicked.connect(lambda: self._clear("brush"))
        self.resetRangeBtn.clicked.connect(self._resetRanges)
        self.cancelBtn.clicked.connect(self.cancel)
        self.opacitySlider.valueChanged.connect(self._drawVolume)
        self.clipCheck.toggled.connect(self._drawVolume)
        self.inspectorToggle.toggled.connect(self.inspector.setVisible)
        self.analysisToggle.toggled.connect(self.analysisBox.setVisible)
        self._toolChanged()

    def setResult(self, result):
        self.setContext(context_from_am(result))

    def setContext(self, context: ComparisonContext):
        # Retain up to three explicit artifacts for comparing recent runs.
        self.contexts = [c for c in self.contexts if c.run_id != context.run_id] + [context]
        self.contexts = self.contexts[-3:]
        self.runCombo.blockSignals(True); self.runCombo.clear()
        for index, item in enumerate(self.contexts):
            self.runCombo.addItem(f"Run {index + 1} · {item.run_id[:8]}", item.run_id)
        self.runCombo.setCurrentIndex(len(self.contexts) - 1); self.runCombo.blockSignals(False)
        self._runChanged()

    def _runChanged(self, *_):
        if not self.contexts:
            return
        self.context = self.contexts[self.runCombo.currentIndex()]
        center = (np.asarray(self.context.grid.shape) - 1) // 2
        self.state.cursor_world = tuple(self.context.grid.voxel_to_world(center))
        self.state.roi = self.state.line = self.state.brush = None
        self._first_point = None
        self.referenceCombo.clear()
        self.referenceCombo.addItem(self.context.reference_type + " · " + self.context.reference_id)
        self.referenceCombo.setToolTip("Only explicitly supplied, aligned reference artifacts are listed.")
        self.materialCombo.blockSignals(True); self.materialCombo.clear()
        self.materialCombo.addItem("All reference materials", None)
        if self.context.labels is not None:
            # Canonical names, not an expensive full-volume unique on the UI thread.
            for label, name in self.context.label_names.items():
                self.materialCombo.addItem(str(name), label)
        self.materialCombo.blockSignals(False)
        has_labels = self.context.labels is not None
        self.materialCombo.setEnabled(has_labels); self.regionCombo.setEnabled(has_labels)
        self.regionCombo.blockSignals(True); self.regionCombo.setCurrentIndex(0); self.regionCombo.blockSignals(False)
        self.state.material, self.state.region = None, "all"
        self.state.boundary_width = min(self.context.grid.spacing_hwz)
        self.boundarySpin.blockSignals(True); self.boundarySpin.setValue(self.state.boundary_width); self.boundarySpin.blockSignals(False)
        self.boundarySpin.setSuffix(" " + self.context.grid.unit)
        self.provenanceLabel.setText(
            f"Grid: Z,H,W {self.context.grid.shape}\nSpacing H,W,Z: {self.context.grid.spacing_hwz} {self.context.grid.unit}\n"
            f"Origin: {self.context.grid.origin}\nDirection: {self.context.grid.direction}\n"
            f"Alignment: {self.context.alignment}\nSampling: {self.context.resampling}\n"
            + ("Reference material IDs supplied.\n" if has_labels else "Reference material IDs unavailable; filters disabled.\n")
            + "Boundary width: distance from interface-adjacent voxel centers, both sides.\n"
            "Invalid: magenta. ROI: current-plane rectangle. Histogram: inclusive endpoints.\n"
            "Quality is not probability. Projection residual stays in Overview and is backprojected inconsistency.\n"
            "3D picking unavailable; world-selection adapter provided. Nominal CAD deviations alone do not establish defects."
        )
        self._quantityChanged()
        self._planeChanged()

    def _quantityChanged(self, *_):
        if self.context is None:
            return
        self.state.quantity = self.quantityCombo.currentData()
        self.state.brush = None
        variants = [("Raw inversion", "raw"), ("Material-regularized", "regularized"), ("Calibrated", "calibrated")]
        prior = self.state.variant
        self.variantCombo.blockSignals(True); self.variantCombo.clear()
        for label, key in variants:
            if (self.state.quantity, key) in self.context.outputs:
                self.variantCombo.addItem(label, key)
        idx = self.variantCombo.findData(prior)
        self.variantCombo.setCurrentIndex(max(0, idx)); self.variantCombo.blockSignals(False)
        settings = self._quantity_settings.get(self.state.quantity)
        if settings:
            self.state.intensity_limits, self.state.error_limit, self.state.tolerance = settings
        else:
            self.state.tolerance = 0.
        self._variantChanged()

    def _variantChanged(self, *_):
        self.state.variant = self.variantCombo.currentData()
        self._schedule()

    def _planeChanged(self, *_):
        if self.context is None:
            return
        self.state.plane = self.planeCombo.currentData()
        normal = PLANES[self.state.plane][0]
        voxel = self.context.grid.world_to_voxel(self.state.cursor_world, clamp=True)
        self.state.index = int(np.rint(voxel[normal]))
        # Slice-local ROI and line cannot be reinterpreted as a new-plane region.
        self.state.roi = self.state.line = None
        self._first_point = None
        self.sliceSlider.blockSignals(True)
        self.sliceSlider.setRange(0, self.context.grid.shape[normal] - 1)
        self.sliceSlider.setValue(self.state.index); self.sliceSlider.blockSignals(False)
        self._schedule()

    def _sliceChanged(self, value):
        if self.context is None:
            return
        voxel = self.context.grid.world_to_voxel(self.state.cursor_world, clamp=True)
        voxel[PLANES[self.state.plane][0]] = value
        self.state.index = value
        self.state.cursor_world = tuple(self.context.grid.voxel_to_world(voxel))
        self.state.roi = self.state.line = None
        self._first_point = None
        self._schedule()

    def _filtersChanged(self, *_):
        self.state.scope = self.scopeCombo.currentData()
        self.state.material = self.materialCombo.currentData()
        self.state.region = self.regionCombo.currentData()
        self.state.boundary_width = self.boundarySpin.value()
        self._schedule()

    def _toleranceChanged(self, value):
        self.state.tolerance = value
        self._saveSettings()
        self._schedule()

    def _saveSettings(self):
        self._quantity_settings[self.state.quantity] = (self.state.intensity_limits, self.state.error_limit, self.state.tolerance)

    def _displayChanged(self, *_):
        if self.minSpin.value() >= self.maxSpin.value():
            self.status.setText("Shared window maximum must exceed minimum.")
            return
        changed_scale = self.state.error_limit != self.errorSpin.value()
        self.state.intensity_limits = (self.minSpin.value(), self.maxSpin.value())
        self.state.error_limit = self.errorSpin.value()
        self.state.display_mode = self.modeCombo.currentData()
        self._saveSettings()
        if changed_scale:
            self._schedule()
        elif self.evaluation is not None:
            self.evaluated_state = deepcopy(self.state)
            self._draw()
        else:
            self._schedule()

    def _resetRanges(self):
        self._quantity_settings.pop(self.state.quantity, None)
        self._schedule()

    def _toolChanged(self, *_):
        self._first_point = None
        pan = self.toolCombo.currentData() == "pan"
        for view in self.maps:
            view.plot.setMouseEnabled(x=pan, y=pan)
            view.plot.setCursor(QtCore.Qt.OpenHandCursor if pan else QtCore.Qt.CrossCursor)
        if self.context is not None and self.toolCombo.currentData() in ("line", "roi"):
            self.status.setText("Click two endpoints in any map for a line profile." if self.toolCombo.currentData() == "line"
                                else "Click two opposite corners in any map for a current-slice ROI.")

    def _clear(self, name):
        setattr(self.state, name, None)
        self._first_point = None
        self._schedule()

    def _schedule(self):
        if self.context is None:
            return
        self._cancel.set(); self._pool.clear()
        self._cancel = threading.Event()
        self._token += 1
        self.evaluation = None
        self.exportBtn.setEnabled(False)
        self.progress.show(); self.cancelBtn.setEnabled(True)
        self.status.setText("Computing full-resolution metrics… display ranges remain locked across slices and runs.")
        self.metricsLabel.setText("Measurements pending…")
        self.subsetLabel.clear(); self.selectionLabel.setText("Selected location pending…")
        for view in self.maps:
            view.image.clear(); view.overlay.clear(); view.profile.hide()
        self.volume.hide()
        self.analysisPlot.clear()
        self._debounce.start(60)

    def cancel(self):
        self._cancel.set(); self._pool.clear(); self._debounce.stop(); self._token += 1
        self.progress.hide(); self.cancelBtn.setEnabled(False)
        self.status.setText("Inspection cancelled. Change a selection to recompute.")
        if self.evaluation is None:
            self.metricsLabel.setText("N/A — inspection cancelled")

    def _start(self):
        ctx, state, cancel = self.context, deepcopy(self.state), self._cancel
        reset = state.quantity not in self._quantity_settings
        evaluator = self.evaluator

        def operation():
            if reset:
                low, high, error_max = np.inf, -np.inf, 0.
                truth = ctx.references.get(state.quantity)
                if truth is None:
                    raise ValueError("Reference properties unavailable. Geometry remains in the Volume workspace; assign compatible AM material properties before comparing.")
                for (quantity, _variant), array in ctx.outputs.items():
                    if quantity != state.quantity:
                        continue
                    for z in range(ctx.grid.shape[0]):
                        if cancel.is_set():
                            raise InspectionCancelled()
                        for values in (array[z], truth[z]):
                            finite = values[np.isfinite(values)]
                            if finite.size:
                                low, high = min(low, float(finite.min())), max(high, float(finite.max()))
                        valid = np.isfinite(array[z]) & np.isfinite(truth[z])
                        if valid.any():
                            error_max = max(error_max, float(np.max(np.abs(array[z][valid].astype(float) - truth[z][valid]))))
                if not np.isfinite(low):
                    low, high = 0., 1.
                state.intensity_limits = (low, high if high > low else low + 1.)
                state.error_limit = max(error_max, 1e-5)
            return (state, evaluate(ctx, state, evaluator, cancel.is_set))

        self._pool.start(_Work(self._token, operation, self._signals))

    def _finished(self, token, payload, error):
        if token != self._token:
            return
        self.progress.hide(); self.cancelBtn.setEnabled(False)
        if error:
            self.status.setText(error)
            self.metricsLabel.setText("N/A — " + error)
            return
        # A probe move does not change metrics; preserve it if made during work.
        cursor_world = self.state.cursor_world
        self.state, self.evaluation = payload
        self.state.cursor_world = cursor_world
        self.evaluated_state = deepcopy(self.state)
        self._saveSettings()
        self.exportBtn.setEnabled(not self._exporting)
        self.volume.show()
        self.status.setText(f"{self.context.alignment} · {UNITS[self.state.quantity]} · invalid = magenta · pan/zoom linked")
        for widget, value in ((self.minSpin, self.state.intensity_limits[0]), (self.maxSpin, self.state.intensity_limits[1]),
                              (self.errorSpin, self.state.error_limit), (self.toleranceSpin, self.state.tolerance),
                              (self.toleranceSlider, int(min(1000, self.state.tolerance / self.state.error_limit * 1000)))):
            widget.blockSignals(True); widget.setValue(value); widget.blockSignals(False)
        self.toleranceSpin.setToolTip(UNITS[self.state.quantity] + "; exploratory, no pass/fail specification")
        self._draw()

    def selectMapPosition(self, position):
        if self.context is None or self.evaluation is None or self.toolCombo.currentData() == "pan":
            return
        normal, row, col = PLANES[self.state.plane]
        voxel = np.zeros(3)
        voxel[normal] = self.state.index
        voxel[row], voxel[col] = position[1] / self.context.grid.spacing_zhw[row], position[0] / self.context.grid.spacing_zhw[col]
        voxel = np.clip(np.rint(voxel), 0, np.asarray(self.context.grid.shape) - 1).astype(int)
        world = tuple(self.context.grid.voxel_to_world(voxel))
        tool = self.toolCombo.currentData()
        if tool in ("line", "roi"):
            if self._first_point is None:
                self._first_point = tuple(voxel)
                self.status.setText("First point selected. Click the second endpoint." if tool == "line" else "First corner selected. Click the opposite ROI corner.")
            else:
                if tool == "roi":
                    self.state.roi = (tuple(np.minimum(self._first_point, voxel)), tuple(np.maximum(self._first_point, voxel)))
                else:
                    self.state.line = (tuple(self.context.grid.voxel_to_world(self._first_point)), world)
                    self.analysisCombo.setCurrentIndex(1)
                self._first_point = None
                self.state.cursor_world = world
                self._schedule()
                return
        self.selectWorld(world)

    def selectWorld(self, world):
        """World-picking adapter: round to nearest voxel center, clamp to grid."""
        if self.context is None:
            return
        voxel = np.rint(self.context.grid.world_to_voxel(world, clamp=True)).astype(int)
        normal = PLANES[self.state.plane][0]
        self.state.cursor_world = tuple(self.context.grid.voxel_to_world(voxel))
        if voxel[normal] != self.state.index:
            self.sliceSlider.setValue(int(voxel[normal]))
        else:
            if self.evaluated_state is not None:
                self.evaluated_state.cursor_world = self.state.cursor_world
            self._drawSelection()
        self.sigWorldSelected.emit(self.state.cursor_world)

    def _lineMoved(self, view):
        if self.evaluation is None or self.state.line is None:
            return
        normal, row, col = PLANES[self.state.plane]
        endpoints = []
        for handle in view.profile.getHandles():
            point = view.profile.mapToParent(handle.pos())
            voxel = np.zeros(3)
            voxel[normal] = self.state.index
            voxel[row] = point.y() / self.context.grid.spacing_zhw[row]
            voxel[col] = point.x() / self.context.grid.spacing_zhw[col]
            voxel = np.clip(voxel, 0, np.asarray(self.context.grid.shape) - 1)
            endpoints.append(tuple(self.context.grid.voxel_to_world(voxel)))
        self.state.line = tuple(endpoints)
        self._schedule()

    def _slice(self, array):
        return np.take(array, self.state.index, axis=PLANES[self.state.plane][0])

    def _fitViews(self):
        if self.context is None:
            return
        _, row, col = PLANES[self.state.plane]
        sy, sx = self.context.grid.spacing_zhw[[row, col]]
        height, width = self.context.grid.shape[row], self.context.grid.shape[col]
        self.maps[0].plot.setRange(QtCore.QRectF(-sx / 2, -sy / 2, width * sx, height * sy), padding=.02)

    def _draw(self):
        if self.evaluation is None:
            return
        state, ctx, out = self.state, self.context, self.evaluation
        normal, row, col = PLANES[state.plane]
        sy, sx = ctx.grid.spacing_zhw[[row, col]]
        height, width = ctx.grid.shape[row], ctx.grid.shape[col]
        rect = QtCore.QRectF(-sx / 2, -sy / 2, width * sx, height * sy)
        field = out["field"]
        error = self._slice(field.signed)
        valid = self._slice(field.valid)
        mode = state.display_mode
        shown_error = np.abs(error) if mode == "absolute" else error
        if mode == "above tolerance":
            shown_error = np.where(valid, np.abs(error) > state.tolerance, np.nan)
        errorscale = (-state.error_limit, state.error_limit) if mode == "signed" else (0, 1 if mode == "above tolerance" else state.error_limit)
        overlay = np.zeros(valid.shape + (4,), dtype=np.uint8)
        # Spatial stipple makes invalid data distinct from both zero and selection.
        yy, xx = np.indices(valid.shape)
        overlay[~valid] = (175, 70, 175, 180)
        overlay[~valid & ((xx + yy) % 4 == 0)] = (245, 170, 240, 230)
        if state.brush is not None:
            overlay[self._slice(out["highlighted"])] = (255, 220, 45, 100)
        lut = pg.colormap.get("viridis").getLookupTable(nPts=256)
        diverging = pg.ColorMap([0, .5, 1], ["#2878c8", "#f4f4f4", "#d65b22"]).getLookupTable(nPts=256)
        arrays = (self._slice(ctx.references[state.quantity]), self._slice(ctx.outputs[state.quantity, state.variant]), shown_error)
        for i, (view, array) in enumerate(zip(self.maps, arrays)):
            limits = state.intensity_limits if i < 2 else errorscale
            view.showImage(array, rect, limits, lut if i < 2 or mode != "signed" else diverging, overlay)
            view.plot.setLabel("bottom", AXIS_NAMES[col], units=ctx.grid.unit)
            view.plot.setLabel("left", AXIS_NAMES[row], units=ctx.grid.unit)
            saturated = int(np.count_nonzero(np.isfinite(array) & ((array < limits[0]) | (array > limits[1]))))
            view.legend.setText(f"{limits[0]:.4g} → {limits[1]:.4g} {'0/1' if i == 2 and mode == 'above tolerance' else UNITS[state.quantity]} · {saturated} saturated")
        self.maps[0].title.setText("Reference · " + ctx.reference_type)
        self.maps[1].title.setText("Reconstruction · " + str(state.variant))
        self.maps[2].title.setText({"signed": "Signed error · R − T (blue under / orange over)", "absolute": "Absolute error · |R − T|", "above tolerance": "Above exploratory tolerance · |R − T| > τ"}[mode])
        self.sliceLabel.setText(f"{state.index + 1}/{ctx.grid.shape[normal]} · {state.index * ctx.grid.spacing_zhw[normal]:.4g} {ctx.grid.unit}")
        self.sliceLabel.setToolTip("Distance from grid origin along " + AXIS_NAMES[normal] + "; world position in inspector")
        view_key = (str(ctx.grid.metadata()), state.plane)
        if self._view_key != view_key:
            self.maps[0].plot.setRange(rect, padding=.02)
            self._view_key = view_key
        self.metricsLabel.setText(self._metricsText("Base region (unbrushed)", out["metrics"]))
        self.subsetLabel.setText(self._metricsText("Histogram subset", out["brushed_metrics"]) if out["brushed_metrics"] else "")
        self._drawSelection(); self._drawAnalysis(); self._drawVolume()

    @staticmethod
    def _metricsText(title, metrics):
        if metrics is None:
            return ""
        def fmt(value):
            return "N/A" if value is None else f"{value:.5g}"
        coverage = "N/A" if metrics["coverage"] is None else f"{metrics['coverage']:.1%}"
        above = "N/A" if metrics["above_tolerance"] is None else f"{metrics['above_tolerance']:.2%}"
        return (f"{title}\nValid coverage: {metrics['valid_count']} / {metrics['expected_count']} ({coverage})\n"
                f"Excluded non-overlap: {metrics['excluded_nonoverlap']}\n"
                f"Bias: {fmt(metrics['bias'])}\nMAE: {fmt(metrics['mae'])}\nRMSE: {fmt(metrics['rmse'])}\n"
                f"P95 |error|: {fmt(metrics['p95'])}\nAbove tolerance: {above}\n" + (metrics["reason"] or ""))

    def _drawSelection(self):
        if self.context is None:
            return
        ctx, state = self.context, self.state
        normal, row, col = PLANES[state.plane]
        voxel = np.rint(ctx.grid.world_to_voxel(state.cursor_world, clamp=True)).astype(int)
        sy, sx = ctx.grid.spacing_zhw[[row, col]]
        for view in self.maps:
            view.cross[0].setValue(voxel[col] * sx); view.cross[1].setValue(voxel[row] * sy)
            if state.roi:
                lo, hi = np.asarray(state.roi, dtype=float)
                view.roi.setData((np.array([lo[col], hi[col] + 1, hi[col] + 1, lo[col], lo[col]]) - .5) * sx,
                                 (np.array([lo[row], lo[row], hi[row] + 1, hi[row] + 1, lo[row]]) - .5) * sy)
            else:
                view.roi.setData([], [])
            if state.line:
                points = ctx.grid.world_to_voxel(state.line)
                view.profile.blockSignals(True)
                for handle, point in zip(view.profile.getHandles(), points):
                    view.profile.movePoint(handle, pg.Point(point[col] * sx, point[row] * sy))
                view.profile.blockSignals(False)
                view.profile.show()
            else:
                view.profile.hide()
        idx = tuple(voxel)
        def fmt(array):
            return "N/A" if array is None or not np.isfinite(array[idx]) else f"{array[idx]:.6g}"
        material = "unavailable" if ctx.labels is None else ctx.label_names.get(int(ctx.labels[idx]), str(ctx.labels[idx]))
        error = self.evaluation["field"].signed if self.evaluation else None
        self.selectionLabel.setText(
            f"Selected world H,W,Z ({ctx.grid.unit})\n" + ", ".join(f"{v:.5g}" for v in state.cursor_world)
            + f"\nVoxel Z,H,W: {idx}\nReference: {fmt(ctx.references.get(state.quantity))}\n"
            f"Reconstruction: {fmt(ctx.outputs.get((state.quantity, state.variant)))}\n"
            f"Signed error: {fmt(error)}\nMaterial: {material}\n"
            + ("ROI: current-plane rectangle\n" if state.roi else "ROI: none\n") + UNITS[state.quantity]
        )
        self.volume.updateSelection(ctx, state)

    def _drawVolume(self, *_):
        self.state.volume_opacity = self.opacitySlider.value() / 100
        self.state.clip_at_slice = self.clipCheck.isChecked()
        if self.evaluation is not None:
            self.volume.updateField(self.context, self.state, self.evaluation, self.state.volume_opacity, self.state.clip_at_slice)

    def _drawAnalysis(self, *_):
        self.analysisPlot.clear()
        if self.evaluation is None:
            return
        out = self.evaluation
        if self.analysisCombo.currentData() == "histogram":
            self.analysisPlot.setLabel("bottom", "Signed error", units=UNITS[self.state.quantity])
            self.analysisPlot.setLabel("left", "Voxel count")
            self.analysisPlot.plot(out["histogram_edges"], out["histogram"], stepMode="center", fillLevel=0, brush=(70, 160, 205, 90), pen="#54b4df")
            self.brushItem.blockSignals(True)
            self.brushItem.setRegion(self.state.brush or (-self.state.error_limit / 4, self.state.error_limit / 4))
            self.brushItem.blockSignals(False)
            self.analysisPlot.addItem(self.brushItem)
            endpoints = "Drag band edges or click histogram to select" if self.state.brush is None else f"[{self.state.brush[0]:.5g}, {self.state.brush[1]:.5g}] {UNITS[self.state.quantity]}"
            self.brushLabel.setText(endpoints + f" · {out['histogram_outside']} outside displayed bins")
        else:
            self.brushLabel.setText("Click two endpoints in any map · invalid samples are gaps")
            self.analysisPlot.setLabel("bottom", "Distance along line", units=self.context.grid.unit)
            self.analysisPlot.setLabel("left", "Signed difference" if self.analysisCombo.currentData() == "difference" else UNITS[self.state.quantity])
            line = out["line"]
            if line is not None:
                if self.analysisCombo.currentData() == "difference":
                    self.analysisPlot.plot(line["distance"], line["signed_error"], pen="#e99545", connect="finite")
                else:
                    self.analysisPlot.plot(line["distance"], line["reference"], pen=pg.mkPen("#f3f3f3", width=2), connect="finite")
                    self.analysisPlot.plot(line["distance"], line["reconstruction"], pen=pg.mkPen("#5db9ed", width=2, style=QtCore.Qt.DashLine), connect="finite")
                    self.brushLabel.setText("Reference: white · reconstruction: dashed blue · invalid = gaps")

    def _histogramClicked(self, event):
        if self.analysisCombo.currentData() != "histogram" or self.evaluation is None:
            return
        if event.button() == QtCore.Qt.LeftButton and self.analysisPlot.getViewBox().sceneBoundingRect().contains(event.scenePos()):
            center = self.analysisPlot.getViewBox().mapSceneToView(event.scenePos()).x()
            half = self.state.error_limit / 10
            self.brushItem.setRegion((center - half, center + half))

    def _brushChanged(self):
        if self.evaluation is None:
            return
        self.state.brush = tuple(sorted(self.brushItem.getRegion()))
        self._schedule()

    def _export(self):
        if self.evaluation is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export numerical inspection (new file)", "error-inspection.npz", "NumPy archive (*.npz)")
        if not path:
            return
        if not path.lower().endswith(".npz"):
            path += ".npz"
        self.state.view_ranges = tuple(map(tuple, self.maps[0].plot.getViewBox().viewRange()))
        ctx, state, evaluation = self.context, deepcopy(self.state), self.evaluation
        # Export snapshot remains valid even if the interactive state changes.
        self.exportBtn.setEnabled(False)
        self._exporting = True
        self.status.setText("Exporting numerical snapshot…")
        self._exportPool.start(_Work(-1, lambda: export_inspection(path, ctx, state, evaluation), self._exportSignals))

    def _exportFinished(self, token, path, error):
        self._exporting = False
        self.exportBtn.setEnabled(self.evaluation is not None)
        self.status.setText("Export failed: " + error if error else "Exported numerical arrays + JSON provenance: " + str(path))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QtCore.QTimer.singleShot(0, lambda: self.bottomMapSplit.setSizes(self.topMapSplit.sizes()))
        if event.size().width() < 1000 and self.inspectorToggle.isChecked():
            self.inspectorToggle.setChecked(False)
        if event.size().height() < 650 and self.analysisToggle.isChecked():
            self.analysisToggle.setChecked(False)

    def closeEvent(self, event):
        self.cancel()
        super().closeEvent(event)

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Dict

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from qtextras import (
    ButtonCollection,
    EasyWidget,
    OptionsDict,
    ParameterEditor,
    attemptFileLoad,
    bindInteractorOptions as bind,
    safeSpawnDevConsole,
)
from s3a.plugins.file import ProjectData
from s3a.processing import ImagePipeline

from xrayrecon.cubecorrections.rotation import RotateWindow

from .helpers import FRPgParamDelegate, phantomSpheres
from .layermodel import LayerModel
from .layerview import LayerView
from .am_plot_window import AMPlotWindow, MaterialLegendWidget
from .stackup_editor import StackupEditorWindow
from .spacing_dialog import SpacingDialog
from .registration import RegistrationPanel
from .view3d import PCBViewer3D

# Gerber Integration Imports
from ..pygerber_integration import PyGerberRasterizer
from ..gerber_utils import GerberLayer
from ..voxelizer import Voxelizer
from ..materials import MATERIALS_DB, get_material, material_labels_to_zeff_rhoe
from ..stackup import StackupConfig
from ..stackup_voxelize import voxelize_stackup_config
from ..multimaterial import MaterialEntry, MultiMaterialConfig
from ..io_spectral import load_ed_data
from ..material_seg import normalize_volume, voxel_to_material_id, label_map_to_rgba
from ..ct_simulation import simulate_ct_volume
from ..alvarez_macovski import (
    simulate_am_volume,
    colorize_volume,
    colorize_by_material,
    colorize_material_labels,
)
from ..registration import RegistrationSession, RigidScaleTransform, resample_labels

pg.setConfigOption("imageAxisOrder", "row-major")

# def find_circles(image: np.ndarray, fgVerts, radius=10, param2='test'):


def find_circles(func: Callable, image=None):

    # out = np.zeros(image.shape[:2], 'uint8')
    # for vert in fgVerts:
    #   cv.circle(out, tuple(vert), radius, 1, -1)
    # return ProcessIO(image=out > 0)

    def findCirclesProcessor():
        proc = ImagePipeline(name=func.__name__)
        proc.addStage(func)
        return proc

    # procCtor =  ImageProcess.fromFunction(find_circles)
    # PRJ_SINGLETON.imgProcClctn.addProcessCtor(findCirclesProcessor)


class MainWin(QtWidgets.QMainWindow):
    def __init__(
        self,
        parent: QtWidgets.QWidget = None,
        layerInfo: Dict[str, list] = None,
        dataCube=None,
    ):
        super().__init__(parent)
        self.toolsEditor = ParameterEditor(name="Tools")

        from .algorithms import inputImage

        find_circles(inputImage)

        self.model = LayerModel(dataCube=dataCube, layerInfo=layerInfo)
        # Reassign dataCube in case it was modified
        self.dataCube = self.model.dataCube
        self.viewer = LayerView(self.model)
        # Left pane: real CT / multi-material data. Right pane: Gerber-derived model.
        self.viewer_3d_left = PCBViewer3D(dataCube)
        self.viewer_3d_right = PCBViewer3D()
        self._gerber_layer_blocks = []
        self._gerber_layer_legend = []
        self._gerber_gap_slices = 20
        self._gerber_intensity_hwz: np.ndarray | None = None
        self._current_rgba_base_left: np.ndarray | None = None  # None = grayscale mode
        self._current_rgba_base_right: np.ndarray | None = None
        self._stackupEditor: StackupEditorWindow | None = None
        self._am_result: dict | None = None
        self._showcase_am_volume_zhw2: np.ndarray | None = None
        self._showcase_material_labels_hwz: np.ndarray | None = None
        # mm/voxel (x, y, z) captured at voxelization time -- discarded by
        # Voxelizer itself, so this is the only place it survives for reuse
        # (e.g. by registration).
        self._gerber_voxel_spacing_mm: tuple[float, float, float] | None = None
        self._gerber_material_id_vol_hwz: np.ndarray | None = None
        self._registration_session = RegistrationSession()
        # self.model.sigCubeChanged.connect(
        #     lambda: self.viewer_3d_left.setDataCube(self.model.dataCube)
        # )

        pgDict = dict(type="list", limits={"True": True, "False": False})
        self._tblDelegate = FRPgParamDelegate(pgDict)

        self.tbl = QtWidgets.QTableView()
        self.tbl.setModel(self.model.tblModel)
        self.tbl.selectionModel().selectionChanged.connect(self.updateTblViews)
        self.tbl.setItemDelegateForColumn(0, self._tblDelegate)
        self.tbl.setSelectionBehavior(self.tbl.SelectionBehavior.SelectRows)

        self.plt = pg.PlotWidget()
        self.viewer.itemParent = self.plt
        for item in self.viewer.items:
            self.plt.addItem(item)
        vb = self.plt.getViewBox()
        vb.setAspectLocked(True)

        self.toolsEditor.registerFunction(self.loadCubeFromFile)
        self.toolsEditor.registerFunction(
            self.fixRotation, name="Interactive Rotation Fix"
        )
        self.toolsEditor.registerFunction(self.showDevConsole, name="Dev Console")
        self.toolsEditor.registerFunction(
            self.createS3AProject, name="Create S3A Project"
        )
        highlightProc = self.toolsEditor.registerFunction(
            self.highlight3DSlices, name="Show Selection in 3D"
        )
        liveUpdateParameter = self.toolsEditor.registerParameter(
            OptionsDict("Live Update 3D", False)
        )
        # self.toolsEditor.registerFunc(self.viewer_3d.resetZoom())
        def maybeUpdate():
            if liveUpdateParameter.value():
                self.highlight3DSlices(highlightProc.parameterCache["downsampleFactor"])

        self._maybeUpdate = maybeUpdate
        self.tbl.selectionModel().selectionChanged.connect(maybeUpdate)

        def maybeShowMetric():
            if self.model.metricInfo is not self.model.NO_METRIC:
                self.model.metricInfo.show()
            else:
                QtWidgets.QMessageBox.information(
                    self, "No Metric", "No metric to show."
                )

        self.toolsEditor.registerFunction(
            maybeShowMetric, name="Show Layer Slice Metrics"
        )
        self.toolsEditor.registerFunction(self.loadGerber, name="Load Gerber")
        self.toolsEditor.registerFunction(
            self.loadGerberStackup, name="Load Gerber Stackup"
        )
        self.toolsEditor.registerFunction(
            self.editGerberStackup, name="Edit Gerber Stackup"
        )
        self.toolsEditor.registerFunction(
            self.showGerberColorView, name="Show Gerber Color 3D"
        )
        self.toolsEditor.registerFunction(
            self.loadMultiMaterial, name="Load Multi-Material Config"
        )
        self.toolsEditor.registerFunction(self.loadEDData, name="Load ED Data")
        self.toolsEditor.registerFunction(self.loadPhantomDemo, name="Load Phantom Demo")
        self.toolsEditor.registerFunction(
            self.setRealCTSpacing, name="Set Real CT Voxel Spacing..."
        )
        self.toolsEditor.registerFunction(
            self.openRegistrationTool, name="Register Gerber to Real CT..."
        )
        self.toolsEditor.registerFunction(
            self.transferGerberLabels, name="Transfer Gerber Labels to Real CT"
        )
        self.toolsEditor.registerFunction(
            self.loadPcbRegistrationShowcase,
            name="Load PCB Registration Showcase",
        )
        self.toolsEditor.registerFunction(
            self.runPcbShowcaseAM,
            name="Run Showcase Z-effective / Electron Density",
        )
        self.toolsEditor.registerFunction(
            self.loadSophisticatedPcbDemo, name="Load Sophisticated PCB Demo",
        )
        self.toolsEditor.registerFunction(
            self.loadRealPcbCtDemo, name="Load Real PCB Demo (Gerber + CT)",
        )
        self._buildUi()

    def _build3DPane(
        self,
        viewer: PCBViewer3D,
        suffix: str,
        onControlChanged: Callable,
        extra_widgets: list | None = None,
    ) -> QtWidgets.QWidget:
        """Build a camera/grid/threshold/opacity control stack for *viewer*.

        Stores the created widgets as self.<name>_<suffix> (e.g. threshSlider_left)
        so per-pane state can be looked up by other methods.
        """
        cameraOpts = [OptionsDict(n, pType="action") for n in ["Top", "Front", "Side", "Reset"]]
        cameraFns = [
            viewer.setCameraTop,
            viewer.setCameraFront,
            viewer.setCameraSide,
            viewer.resetCamera,
        ]
        cameraCtrl = ButtonCollection(
            None, "Camera", cameraOpts, cameraFns, exclusive=False, asToolButton=False
        )
        setattr(self, f"cameraCtrl_{suffix}", cameraCtrl)

        gridToggleBtn = QtWidgets.QPushButton("Grid")
        gridToggleBtn.setCheckable(True)
        gridToggleBtn.setChecked(True)
        axisToggleBtn = QtWidgets.QPushButton("Axis")
        axisToggleBtn.setCheckable(True)
        axisToggleBtn.setChecked(True)
        gridToggleBtn.toggled.connect(viewer.setGridVisible)
        axisToggleBtn.toggled.connect(viewer.setAxisVisible)
        for btn in [gridToggleBtn, axisToggleBtn]:
            btn.setMaximumHeight(22)
        setattr(self, f"gridToggleBtn_{suffix}", gridToggleBtn)
        setattr(self, f"axisToggleBtn_{suffix}", axisToggleBtn)
        viewToggleRow = QtWidgets.QWidget()
        viewToggleLayout = QtWidgets.QHBoxLayout(viewToggleRow)
        viewToggleLayout.setContentsMargins(0, 0, 0, 0)
        viewToggleLayout.addWidget(gridToggleBtn)
        viewToggleLayout.addWidget(axisToggleBtn)

        threshSlider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        threshSlider.setRange(0, 50)
        threshSlider.setValue(10)
        threshSlider.setMaximumHeight(18)
        threshSlider.setToolTip("3D threshold: voxels below this intensity are hidden (0–50%)")

        opacitySlider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        opacitySlider.setRange(1, 10)
        opacitySlider.setValue(4)
        opacitySlider.setMaximumHeight(18)
        opacitySlider.setToolTip("3D opacity exponent: higher = more transparent mid-range voxels")

        threshRow = QtWidgets.QWidget()
        threshLayout = QtWidgets.QHBoxLayout(threshRow)
        threshLayout.setContentsMargins(0, 0, 0, 0)
        threshLayout.addWidget(QtWidgets.QLabel("3D Thresh:"))
        threshLayout.addWidget(threshSlider)

        opacRow = QtWidgets.QWidget()
        opacLayout = QtWidgets.QHBoxLayout(opacRow)
        opacLayout.setContentsMargins(0, 0, 0, 0)
        opacLayout.addWidget(QtWidgets.QLabel("3D Opacity:"))
        opacLayout.addWidget(opacitySlider)

        threshSlider.valueChanged.connect(onControlChanged)
        opacitySlider.valueChanged.connect(onControlChanged)
        setattr(self, f"threshSlider_{suffix}", threshSlider)
        setattr(self, f"opacitySlider_{suffix}", opacitySlider)

        controlsWidget = QtWidgets.QWidget()
        ccl = QtWidgets.QVBoxLayout(controlsWidget)
        ccl.setContentsMargins(0, 0, 0, 0)
        ccl.setSpacing(2)
        ccl.addWidget(cameraCtrl)
        ccl.addWidget(viewToggleRow)
        ccl.addWidget(threshRow)
        ccl.addWidget(opacRow)
        for widget in extra_widgets or []:
            ccl.addWidget(widget)
        ccl.addStretch()

        paneSplitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        paneSplitter.addWidget(controlsWidget)
        paneSplitter.addWidget(viewer)
        paneSplitter.setStretchFactor(0, 0)
        paneSplitter.setStretchFactor(1, 1)
        return paneSplitter

    def _buildUi(self):
        # --- Menu bar ---
        menu = QtWidgets.QMenuBar(self)
        self.setMenuBar(menu)
        fileMenu = menu.addMenu("File")
        self.recentMenu = fileMenu.addMenu("Recent Files")
        self._settings = QtCore.QSettings("XrayRecon", "MainWin")
        self._populateRecentMenu()

        # Tools panel as a real dock (resizable/floatable/closable) instead of
        # a fixed embed -- also gets a populated "Tools" menu-bar menu for free.
        self._toolsDock, _ = self.toolsEditor.createWindowDock(
            self, title="Tools", dockArea=QtCore.Qt.DockWidgetArea.LeftDockWidgetArea,
        )
        self._toolsDock.show()

        # --- Left panel: Select All/Deselect All + layer filter + table ---
        self.selectAllBtn = QtWidgets.QPushButton("Select All")
        self.deselectAllBtn = QtWidgets.QPushButton("Deselect All")
        self.moveUpBtn = QtWidgets.QPushButton("↑")
        self.moveDownBtn = QtWidgets.QPushButton("↓")
        self.selectAllBtn.clicked.connect(self._selectAllLayers)
        self.deselectAllBtn.clicked.connect(self._deselectAllLayers)
        self.moveUpBtn.clicked.connect(self._moveLayerUp)
        self.moveDownBtn.clicked.connect(self._moveLayerDown)
        tblBtnRow = QtWidgets.QWidget()
        tblBtnLayout = QtWidgets.QHBoxLayout(tblBtnRow)
        tblBtnLayout.setContentsMargins(0, 0, 0, 0)
        for btn in [self.selectAllBtn, self.deselectAllBtn, self.moveUpBtn, self.moveDownBtn]:
            tblBtnLayout.addWidget(btn)

        self.layerFilterEdit = QtWidgets.QLineEdit()
        self.layerFilterEdit.setPlaceholderText("Filter layers...")
        self.proxyModel = QtCore.QSortFilterProxyModel()
        self.proxyModel.setSourceModel(self.model.tblModel)
        self.proxyModel.setFilterKeyColumn(2)
        self.proxyModel.setFilterCaseSensitivity(QtCore.Qt.CaseSensitivity.CaseInsensitive)
        self.tbl.setModel(self.proxyModel)
        self.tbl.setItemDelegateForColumn(0, self._tblDelegate)
        # Re-connect selection model after model swap (setModel replaces the selectionModel)
        self.tbl.selectionModel().selectionChanged.connect(self.updateTblViews)
        self.tbl.selectionModel().selectionChanged.connect(self._maybeUpdate)
        self.layerFilterEdit.textChanged.connect(self.proxyModel.setFilterFixedString)

        leftPanel = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(leftPanel)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(tblBtnRow)
        ll.addWidget(self.layerFilterEdit)
        ll.addWidget(self.tbl)

        # --- Center panel: LEFT (real data) / RIGHT (Gerber model) 3D panes ---
        self.amColorZeffBtn = QtWidgets.QPushButton("Z_eff")
        self.amColorRhoeBtn = QtWidgets.QPushButton("ρ_e")
        self.amColorMaterialBtn = QtWidgets.QPushButton("Material")
        for btn in (self.amColorZeffBtn, self.amColorRhoeBtn, self.amColorMaterialBtn):
            btn.setCheckable(True)
            btn.setMaximumHeight(22)
        self.amColorZeffBtn.setChecked(True)
        self._amColorGroup = QtWidgets.QButtonGroup(self)
        self._amColorGroup.setExclusive(True)
        self._amColorGroup.addButton(self.amColorZeffBtn)
        self._amColorGroup.addButton(self.amColorRhoeBtn)
        self._amColorGroup.addButton(self.amColorMaterialBtn)
        self.amColorZeffBtn.toggled.connect(lambda checked: checked and self._renderAMVolume())
        self.amColorRhoeBtn.toggled.connect(lambda checked: checked and self._renderAMVolume())
        self.amColorMaterialBtn.toggled.connect(lambda checked: checked and self._renderAMVolume())

        self.materialLegendWidget = MaterialLegendWidget()
        self.materialLegendWidget.setVisible(False)

        self.amLegendBtn = QtWidgets.QPushButton("Legend")
        self.amLegendBtn.setMaximumHeight(22)
        self.amLegendBtn.setCheckable(True)
        self.amLegendBtn.toggled.connect(self.materialLegendWidget.setVisible)

        amColorRow = QtWidgets.QWidget()
        amColorLayout = QtWidgets.QHBoxLayout(amColorRow)
        amColorLayout.setContentsMargins(0, 0, 0, 0)
        amColorLayout.addWidget(QtWidgets.QLabel("AM Color:"))
        amColorLayout.addWidget(self.amColorZeffBtn)
        amColorLayout.addWidget(self.amColorRhoeBtn)
        amColorLayout.addWidget(self.amColorMaterialBtn)
        amColorLayout.addWidget(self.amLegendBtn)

        leftPane = self._build3DPane(
            self.viewer_3d_left, "left", self._on3DControlChangedLeft,
            extra_widgets=[amColorRow],
        )
        rightPane = self._build3DPane(self.viewer_3d_right, "right", self._on3DControlChangedRight)

        centerSplitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        centerSplitter.addWidget(leftPane)
        centerSplitter.addWidget(rightPane)
        centerSplitter.setStretchFactor(0, 1)
        centerSplitter.setStretchFactor(1, 1)

        # Right-click context menu on both 3D panes, built fresh from every
        # currently-registered Tools action on each click.
        for _viewer in (self.viewer_3d_left, self.viewer_3d_right):
            _viewer.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
            _viewer.customContextMenuRequested.connect(
                lambda pos, v=_viewer: self.toolsEditor.createActionsFromFunctions().exec_(
                    v.mapToGlobal(pos)
                )
            )

        self._mainWorkspacePage = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(self._mainWorkspacePage)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.addWidget(centerSplitter)

        self.registrationPanel = RegistrationPanel(self._registration_session, parent=self)
        self.registrationPanel.sigTransformAccepted.connect(self._onRegistrationAccepted)
        self.registrationPanel.sigCloseRequested.connect(self._showMainWorkspace)
        self._workspaceStack = QtWidgets.QStackedWidget()
        self._workspaceStack.setObjectName("MainWorkspaceStack")
        self._workspaceStack.addWidget(self._mainWorkspacePage)
        self._workspaceStack.addWidget(self.registrationPanel)
        self._workspaceStack.setCurrentWidget(self._mainWorkspacePage)

        # --- Right panel: axis buttons + brightness/contrast + 2D plot ---
        self.minSlider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.maxSlider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.minSlider.setRange(0, 255)
        self.minSlider.setValue(0)
        self.maxSlider.setRange(0, 255)
        self.maxSlider.setValue(255)
        self.minSlider.setToolTip("Minimum display level (brightness)")
        self.maxSlider.setToolTip("Maximum display level (contrast)")

        def onLevelChanged():
            self.viewer.setLevels(self.minSlider.value(), self.maxSlider.value())

        self.minSlider.valueChanged.connect(onLevelChanged)
        self.maxSlider.valueChanged.connect(onLevelChanged)

        minRow = QtWidgets.QWidget()
        minLayout = QtWidgets.QHBoxLayout(minRow)
        minLayout.setContentsMargins(0, 0, 0, 0)
        minLayout.addWidget(QtWidgets.QLabel("Min:"))
        minLayout.addWidget(self.minSlider)

        maxRow = QtWidgets.QWidget()
        maxLayout = QtWidgets.QHBoxLayout(maxRow)
        maxLayout.setContentsMargins(0, 0, 0, 0)
        maxLayout.addWidget(QtWidgets.QLabel("Max:"))
        maxLayout.addWidget(self.maxSlider)

        rightPanel = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(rightPanel)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self.viewer.xyzCtrl)
        rl.addWidget(minRow)
        rl.addWidget(maxRow)
        rl.addWidget(self.plt)

        # Dock the Z_eff/rho_e plot panel too, instead of a fixed splitter
        # pane, so it can be resized/floated/closed like the Tools dock.
        self._plotDock = QtWidgets.QDockWidget("Z_eff / ρ_e Plot", self)
        self._plotDock.setObjectName("PlotDock")
        self._plotDock.setWidget(rightPanel)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self._plotDock)

        children = EasyWidget(
            [leftPanel, self._workspaceStack],
            useSplitter=True,
            layout="H",
        )
        # Extra layer of nesting allows margin to look a bit better.
        # materialLegendWidget docks below everything else, hidden by
        # default, shown via the "Legend" toggle button.
        EasyWidget.buildMainWindow(
            [children, self.materialLegendWidget], layout="V", window=self
        )
        self.setWindowTitle("X-ray Reconstruction")

        # --- Status bar ---
        self.cubeInfoLabel = QtWidgets.QLabel("No data loaded")
        self.pixelValueLabel = QtWidgets.QLabel("Pixel: —")
        self.statusBar().addWidget(self.cubeInfoLabel)
        self.statusBar().addWidget(self.pixelValueLabel)
        self.gerberLegendLabel = QtWidgets.QLabel("Gerber 3D legend: none loaded")
        self.statusBar().addPermanentWidget(self.gerberLegendLabel)

        self.model.sigCubeChanged.connect(self._updateStatusBar)
        self.model.sigCubeChanged.connect(self._resetLevelSliders)
        self.model.tblModel.sigDataChanged.connect(self._updateStatusBar)

        self.plt.scene().sigMouseMoved.connect(self._onMouseMovedInPlot)

        # --- Keyboard shortcuts: arrow keys navigate table rows ---
        shortcutDown = QtWidgets.QShortcut(
            QtCore.Qt.Key.Key_Down, self, context=QtCore.Qt.ShortcutContext.ApplicationShortcut
        )
        shortcutUp = QtWidgets.QShortcut(
            QtCore.Qt.Key.Key_Up, self, context=QtCore.Qt.ShortcutContext.ApplicationShortcut
        )
        shortcutDown.activated.connect(self._selectNextLayer)
        shortcutUp.activated.connect(self._selectPrevLayer)

        # Enable double-click rename on Layer Name column
        self.tbl.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked
        )

    def _apply_rgba_controls(
        self,
        rgba_base: np.ndarray,
        threshold: float,
        transparency_factor: float,
        intensity_cube: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return a copy of *rgba_base* with threshold masking and opacity applied.

        threshold: 0.0–0.5 normalised intensity — voxels below this become transparent.
        transparency_factor: 0.5–5.0 — higher means more transparent (matches grayscale slider).
        intensity_cube: (H, W, Z) per-voxel intensity used for threshold masking.
        Defaults to model.dataCube (the left/real-data pane) when not given. Must
        match rgba_base's (H, W, Z) shape or masking is skipped — e.g. the Gerber
        color view's Z axis is a visually-compressed "gap slices" count that does
        not correspond 1:1 to the physical voxel depth.
        """
        rgba = rgba_base.copy()

        if intensity_cube is None:
            intensity_cube = self.model.dataCube

        if (
            threshold > 0
            and intensity_cube is not None
            and intensity_cube.size > 0
            and intensity_cube.shape[:3] == rgba.shape[:3]
        ):
            cube = intensity_cube.astype(np.float32)
            cmin, cmax = float(cube.min()), float(cube.max())
            norm_hwz = (cube - cmin) / (cmax - cmin) if cmax > cmin else cube
            rgba[norm_hwz < threshold, 3] = 0

        alpha_scale = 1.0 / max(0.5, transparency_factor)
        if alpha_scale < 1.0:
            rgba[..., 3] = np.clip(
                rgba[..., 3].astype(np.float32) * alpha_scale, 0, 255
            ).astype(np.uint8)

        return rgba

    def _material_or_error(self, name: str):
        if name.upper() not in MATERIALS_DB:
            available = ", ".join(sorted(MATERIALS_DB.keys()))
            raise ValueError(
                f"Unknown material '{name}'. Available materials: {available}"
            )
        return get_material(name)

    def _build_rgba_layer_blocks(self, data_hwz, visual_entries):
        h_px, w_px, _ = data_hwz.shape
        blocks = []
        z_start = 0
        for entry in visual_entries:
            num_slices = max(1, int(entry["num_slices"]))
            color = entry["color"]
            rgba = np.zeros((h_px, w_px, num_slices, 4), dtype=np.uint8)

            if entry["is_substrate"]:
                rgba[:, :, :, 0] = color[0]
                rgba[:, :, :, 1] = color[1]
                rgba[:, :, :, 2] = color[2]
                rgba[:, :, :, 3] = color[3]
            else:
                for z_local in range(num_slices):
                    z_global = z_start + z_local
                    if z_global >= data_hwz.shape[2]:
                        break
                    feature_mask = data_hwz[:, :, z_global] > 15
                    air_mask = ~feature_mask

                    rgba[:, :, z_local, 0][feature_mask] = color[0]
                    rgba[:, :, z_local, 1][feature_mask] = color[1]
                    rgba[:, :, z_local, 2][feature_mask] = color[2]
                    rgba[:, :, z_local, 3][feature_mask] = color[3]

                    rgba[:, :, z_local, 0][air_mask] = 50
                    rgba[:, :, z_local, 1][air_mask] = 50
                    rgba[:, :, z_local, 2][air_mask] = 200
                    rgba[:, :, z_local, 3][air_mask] = 5

            blocks.append((rgba, entry["is_substrate"]))
            z_start += num_slices

        return blocks

    def _render_gerber_color_volume(self):
        if not self._gerber_layer_blocks:
            return

        base_h, base_w = self._gerber_layer_blocks[0][0].shape[:2]
        spacer = np.zeros((base_h, base_w, max(self._gerber_gap_slices, 0), 4), dtype=np.uint8)

        parts = []
        for block, is_substrate in self._gerber_layer_blocks:
            if is_substrate:
                parts.append(spacer)
            else:
                parts.append(block)

        if not parts:
            return

        rgba_volume = np.concatenate(parts, axis=2)
        self._current_rgba_base_right = rgba_volume
        threshold = self.threshSlider_right.value() / 100.0
        tf = self.opacitySlider_right.value() / 2.0
        self.viewer_3d_right.setRGBADataCube(
            self._apply_rgba_controls(
                rgba_volume, threshold, tf, intensity_cube=self._gerber_intensity_hwz
            )
        )
        self.viewer_3d_right.resetCamera()

        legend_parts = []
        for name, color, is_substrate in self._gerber_layer_legend:
            role = "substrate" if is_substrate else "copper"
            legend_parts.append(
                f"{name}:{role} rgba({color[0]},{color[1]},{color[2]},{color[3]})"
            )
        if legend_parts:
            summary = "; ".join(legend_parts[:4])
            if len(legend_parts) > 4:
                summary += "; ..."
            self.gerberLegendLabel.setText(f"Gerber 3D legend: {summary}")

    def _apply_gerber_volume(
        self,
        vol,
        layer_names,
        layer_slices,
        visual_entries,
        default_gap_slices,
    ):
        data_zyx = vol[..., 0]
        data_hwz = np.moveaxis(data_zyx, 0, -1)
        self._gerber_intensity_hwz = data_hwz

        self._gerber_layer_blocks = self._build_rgba_layer_blocks(data_hwz, visual_entries)
        self._gerber_layer_legend = [
            (entry["name"], entry["color"], entry["is_substrate"])
            for entry in visual_entries
        ]
        self._gerber_gap_slices = max(0, int(default_gap_slices))
        self._render_gerber_color_volume()

    @bind(file=dict(type="file", nameFilter="*.npy", value=""))
    def loadCubeFromFile(self, file: str | Path):
        if not file:
            return
        data = np.load(file)
        self._current_rgba_base_left = None
        self.model.setCube(data)
        self._addToRecentFiles(str(file))

    @bind(file=dict(type="file", nameFilter="Gerber Files (*.gbr *.GBR);;All Files (*)", value=""))
    def loadGerber(self, file: str | Path):
        if not file:
            return
        
        file_path = Path(file)
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Rasterize
            # Use high DPI for quality, but keep memory in check? 300 DPI? 
            # Demo used 100 for speed. Let's try 300.
            rasterizer = PyGerberRasterizer(dpi=300)
            mask = rasterizer.rasterize(content)
            
            if np.max(mask) == 0:
                QtWidgets.QMessageBox.warning(self, "Empty Gerber", "The loaded Gerber file produced an empty mask.")
                return

            # Package
            layer = GerberLayer(file_path.name)
            layer.raster_mask = mask
            
            # Assume arbitrary 10x10mm size if we can't get it from backend?
            # Ideally we check mask aspect ratio.
            # PyGerber output mask size depends on content + DPI.
            # physical_width = px_width / (dpi/25.4)
            h_px, w_px = mask.shape
            width_mm = w_px / (300 / 25.4)
            height_mm = h_px / (300 / 25.4)
            
            layer.update_bounds(0, 0)
            layer.update_bounds(width_mm, height_mm)
            
            mat_cu = self._material_or_error("COPPER")
            mat_air = self._material_or_error("AIR")
            stackup = [(layer, 0.035, mat_cu, mat_air)]

            original_res = 25.4 / 300
            vox = Voxelizer(resolution_xy=original_res, resolution_z=0.01)
            vol = vox.voxelize_stackup(stackup, mat_air)

            layer_names = [file_path.stem]
            layer_slices = [np.arange(max(1, int(np.round(0.035 / 0.01))))]
            visual_entries = [
                {
                    "name": file_path.stem,
                    "is_substrate": False,
                    "num_slices": max(1, int(np.round(0.035 / 0.01))),
                    "color": (255, 140, 0, 200),
                }
            ]
            self._apply_gerber_volume(
                vol,
                layer_names,
                layer_slices,
                visual_entries,
                default_gap_slices=0,
            )
            
            QtWidgets.QMessageBox.information(self, "Success", f"Loaded Gerber: {file_path.name}\nSize: {width_mm:.2f}x{height_mm:.2f}mm")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load Gerber:\n{e}")

    def _voxelize_stackup(
        self, config: StackupConfig, stackup_path: Path, return_material_ids: bool = False
    ):
        """Rasterize + voxelize every layer in *config*.

        Thin MainWin-side wrapper around the pure
        `stackup_voxelize.voxelize_stackup_config` (no Qt dependency, directly
        testable). Returns (vol, layer_names, layer_slices, visual_entries) --
        the same shape `_apply_gerber_volume` expects -- and stashes the mm/voxel
        spacing (and, if requested, the ground-truth material-ID volume) on
        `self` for reuse by registration/label-transfer. Shared by
        `loadGerberStackup` (JSON file), the Gerber Stackup Editor, and the
        simulated-X-ray/AM-decomposition demo paths.
        """
        vol, layer_names, layer_slices, visual_entries, spacing_mm, material_ids = (
            voxelize_stackup_config(
                config, stackup_path, return_material_ids=return_material_ids
            )
        )
        self._gerber_voxel_spacing_mm = spacing_mm
        if return_material_ids:
            # Voxelizer returns (Z, H, W) -- move to (H, W, Z) to match every
            # other "_hwz"-suffixed Gerber array in this class.
            self._gerber_material_id_vol_hwz = np.moveaxis(material_ids, 0, -1)
        # Any previously-accepted registration was fit to the OLD Gerber
        # geometry -- invalidate it so a stale transform can't silently be
        # applied to newly-(re)voxelized data via transferGerberLabels.
        self._registration_session.invalidate("Gerber geometry changed")
        return vol, layer_names, layer_slices, visual_entries

    @bind(file=dict(type="file", nameFilter="Stackup Files (*.json);;All Files (*)", value=""))
    def loadGerberStackup(self, file: str | Path):
        if not file:
            return

        stackup_path = Path(file)
        try:
            config = StackupConfig.from_json(stackup_path)
            vol, layer_names, layer_slices, visual_entries = self._voxelize_stackup(
                config, stackup_path, return_material_ids=True
            )
            self._apply_gerber_volume(
                vol,
                layer_names,
                layer_slices,
                visual_entries,
                default_gap_slices=config.default_gap_slices,
            )
            self._openStackupEditor(config, stackup_path)

            QtWidgets.QMessageBox.information(
                self,
                "Stackup Loaded",
                f"Loaded {len(config.layers)} copper layers from {stackup_path.name}",
            )

        except Exception as e:
            import traceback

            traceback.print_exc()
            QtWidgets.QMessageBox.critical(
                self, "Error", f"Failed to load stackup:\n{e}"
            )

    @bind(gapSlices=dict(limits=(0, 200), value=20))
    def showGerberColorView(self, gapSlices=20):
        if not self._gerber_layer_blocks:
            QtWidgets.QMessageBox.information(
                self,
                "No Gerber Color Data",
                "Load a Gerber or stackup first to enable color view.",
            )
            return

        self._gerber_gap_slices = int(gapSlices)
        self._render_gerber_color_volume()

    def editGerberStackup(self):
        self._openStackupEditor()

    def _openStackupEditor(
        self, config: StackupConfig | None = None, stackup_path: Path | None = None
    ):
        if self._stackupEditor is None:
            self._stackupEditor = StackupEditorWindow(
                self._voxelizeStackupFromEditor,
                self._simulateXrayFromEditor,
                self._showAlvarezMacovskiPlot,
                self._showAMVolumeIn3D,
                parent=self,
            )
        if config is not None:
            self._stackupEditor.setConfig(config, stackup_path or Path.cwd())
        self._stackupEditor.show()
        self._stackupEditor.raise_()
        self._stackupEditor.activateWindow()

    def _voxelizeStackupFromEditor(self, config: StackupConfig, stackup_path: Path):
        try:
            vol, layer_names, layer_slices, visual_entries = self._voxelize_stackup(
                config, stackup_path, return_material_ids=True
            )
            self._apply_gerber_volume(
                vol,
                layer_names,
                layer_slices,
                visual_entries,
                default_gap_slices=config.default_gap_slices,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to voxelize stackup:\n{e}")

    def _simulateXrayFromEditor(self, config: StackupConfig, stackup_path: Path):
        """Simulate a CT reconstruction of *config* and show it in the LEFT pane."""
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            # Not return_material_ids=True: this path only ever feeds the LEFT
            # pane (simulated "real CT" preview) via model.setCube below, never
            # `_apply_gerber_volume` -- so it must NOT touch
            # `_gerber_material_id_vol_hwz`, or that label volume would silently
            # describe a different config than whatever `_gerber_intensity_hwz`
            # (the RIGHT-pane Gerber ground truth) currently holds.
            vol, _layer_names, _layer_slices, _visual_entries = self._voxelize_stackup(
                config, stackup_path
            )
            sim_zhw = simulate_ct_volume(vol)
            sim_hwz = np.moveaxis(sim_zhw, 0, -1)

            self._current_rgba_base_left = None
            self.model.setCube(sim_hwz)
            self.model.setSpacing(*self._gerber_voxel_spacing_mm)
            self.viewer_3d_left.setDataCube(sim_hwz)
            self.viewer_3d_left.resetCamera()
        except Exception as e:
            import traceback
            traceback.print_exc()
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to simulate X-ray:\n{e}")
            return
        QtWidgets.QApplication.restoreOverrideCursor()
        QtWidgets.QMessageBox.information(
            self, "Simulated X-ray",
            f"Simulated CT volume: {sim_hwz.shape} (FBP reconstruction via Radon transform)"
        )

    def _runAMDecomposition(self, config: StackupConfig, stackup_path: Path) -> dict | None:
        """Voxelize *config* and run the Alvarez-Macovski dual-energy
        decomposition across every slice. Returns None (after showing an
        error dialog) on failure."""
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            # Not return_material_ids=True -- same reasoning as
            # _simulateXrayFromEditor: this path never touches
            # `_apply_gerber_volume`/`_gerber_intensity_hwz`, so it must not
            # touch `_gerber_material_id_vol_hwz` either.
            vol, _layer_names, _layer_slices, _visual_entries = self._voxelize_stackup(
                config, stackup_path
            )
            result = simulate_am_volume(vol)
        except Exception as e:
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to run AM decomposition:\n{e}")
            return None
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        return result

    def _showAlvarezMacovskiPlot(self, config: StackupConfig, stackup_path: Path):
        """Run the Alvarez-Macovski dual-energy decomposition on *config* and
        show a ground-truth-vs-recovered Z_eff/electron-density comparison."""
        result = self._runAMDecomposition(config, stackup_path)
        if result is None:
            return
        self._am_result = result
        self._amPlotWindow = AMPlotWindow(result, parent=self)
        self._amPlotWindow.show()

    def _showAMVolumeIn3D(self, config: StackupConfig, stackup_path: Path):
        """Run the Alvarez-Macovski decomposition across every slice of the
        stackup and render the recovered volume in the LEFT 3D pane,
        colorized by whichever quantity (Z_eff / rho_e) is currently
        toggled via amColorZeffBtn/amColorRhoeBtn."""
        result = self._runAMDecomposition(config, stackup_path)
        if result is None:
            return
        self._am_result = result
        self._renderAMVolume()

    def _renderAMVolume(self):
        """Colorize the cached AM result by the current toggle state and push
        it into the LEFT 3D pane. No-op if no AM decomposition has run yet."""
        if self._am_result is None:
            return
        zeff_vol = self._am_result.get(
            "zeff_regularized", self._am_result["zeff_recovered"]
        )
        rhoe_vol = self._am_result.get(
            "rhoe_regularized", self._am_result["rhoe_recovered"]
        )

        if self.amColorZeffBtn.isChecked():
            rgba = colorize_volume(zeff_vol, "plasma")
        elif self.amColorRhoeBtn.isChecked():
            rgba = colorize_volume(rhoe_vol, "cividis")
        else:
            labels = self._am_result.get("material_labels_recovered")
            rgba = (
                colorize_material_labels(labels)
                if labels is not None
                else colorize_by_material(zeff_vol, rhoe_vol)
            )

        scalar_hwz = np.moveaxis(rhoe_vol, 0, -1)

        self._current_rgba_base_left = rgba
        self.model.setRGBACube(scalar_hwz, rgba)
        self.viewer_3d_left.setRGBADataCube(rgba)
        self.viewer_3d_left.resetCamera()

    @bind(location=dict(type="file", fileMode="Directory", value=""))
    @bind(alert=dict(value=True, ignore=True))
    def createS3AProject(self, location: str | Path = None, alert=False):
        if not location:
            return
        here = Path(__file__).resolve().parent
        prj = ProjectData.create(
            name=location, config=attemptFileLoad(here.parent / "projcfg.s3aprj")
        )
        layerDf = self.model.layerDf
        for ii, name in enumerate(layerDf["Layer Name"]):
            imgName = prj.imagesPath / f"{name}.png"
            if imgName not in prj.images:
                selectedSlices = layerDf.loc[ii, "Slices"]
                img = self.model.dataCube[..., selectedSlices].mean(2).astype("uint8")
                prj.addImage(imgName, data=img, copyToProject=True, allowOverwrite=True)
        if alert:
            msgBox = QtWidgets.QMessageBox(self)
            msgBox.setWindowTitle("Project Created")
            msgBox.setText(
                f"<qt>Created S3A project. To make annotations, open it in S3A using "
                f"the terminal command<br><br>"
                f"<code>s3a-gui --project {prj.configPath}</code></qt>"
            )
            msgBox.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )
            msgBox.setModal(True)
            msgBox.exec()

    @staticmethod
    def _default_ed_config() -> MultiMaterialConfig:
        return MultiMaterialConfig(
            name="ED Data (default)",
            channels=["fbp"],
            materials=[
                MaterialEntry("AIR",    color=(0,   0,   200, 5),   threshold=(0.0,  0.05)),
                MaterialEntry("FR4",    color=(80,  80,  80,  80),  threshold=(0.05, 0.3)),
                MaterialEntry("COPPER", color=(255, 140, 0,   200), threshold=(0.3,  1.0)),
            ],
        )

    def _apply_multimaterial_volume(
        self, vol: np.ndarray, config: MultiMaterialConfig
    ) -> None:
        """Segment *vol* per *config*, render RGBA in 3D, update 2D model."""
        norm = normalize_volume(vol)
        labels = voxel_to_material_id(norm, config)
        rgba = label_map_to_rgba(labels, config)

        vol_3d = norm[0] if norm.ndim == 4 else norm    # (Z, H, W)
        vol_hwz = np.moveaxis(vol_3d, 0, -1)             # (H, W, Z)

        # setRGBACube sets both atomically so LayerView.resetItems sees RGBA on first draw
        self._current_rgba_base_left = rgba
        self.model.setRGBACube(vol_hwz, rgba)
        threshold = self.threshSlider_left.value() / 100.0
        tf = self.opacitySlider_left.value() / 2.0
        self.viewer_3d_left.setRGBADataCube(self._apply_rgba_controls(rgba, threshold, tf))
        self.viewer_3d_left.resetCamera()

    @bind(file=dict(type="file", nameFilter="Multi-Material Config (*.json);;All Files (*)", value=""))
    def loadMultiMaterial(self, file: str | Path):
        if not file:
            return
        config_path = Path(file)
        try:
            config = MultiMaterialConfig.from_json(config_path)
            data_dir = config.resolve_data_dir(config_path)
            if not data_dir.exists():
                QtWidgets.QMessageBox.critical(
                    self, "Error",
                    f"Data directory not found: {data_dir}\n"
                    "Set 'data_dir' in the config JSON or place data next to the config file."
                )
                return
            vol = load_ed_data(data_dir)
            self._apply_multimaterial_volume(vol, config)
            QtWidgets.QMessageBox.information(
                self, "Loaded",
                f"Multi-material config: {config.name}\n"
                f"Volume shape: {vol.shape}  ({len(config.materials)} materials)"
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load multi-material data:\n{e}")

    @bind(directory=dict(type="file", fileMode="Directory", value=""))
    def loadEDData(self, directory: str | Path):
        if not directory:
            return
        try:
            vol = load_ed_data(directory)
            config = self._default_ed_config()
            self._apply_multimaterial_volume(vol, config)
            QtWidgets.QMessageBox.information(
                self, "ED Data Loaded",
                f"Volume shape: {vol.shape}\n"
                "Using default AIR / FR4 / COPPER thresholds.\n"
                "Load a Multi-Material Config JSON to customise."
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load ED data:\n{e}")

    def loadPhantomDemo(self):
        """Load a self-contained synthetic multi-material phantom (no real data required)."""
        try:
            config_path = Path(__file__).resolve().parents[2] / "configs" / "phantom_demo.json"
            config = MultiMaterialConfig.from_json(config_path)
            vol = phantomSpheres(n=128)
            self._apply_multimaterial_volume(vol, config)
            QtWidgets.QMessageBox.information(
                self, "Phantom Demo Loaded",
                f"Synthetic phantom: {vol.shape}  ({len(config.materials)} materials)"
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load phantom demo:\n{e}")

    def loadSophisticatedPcbDemo(self):
        """Voxelize a richer synthetic multi-material PCB stackup (ENIG + HASL
        finishes) and run it through the simulated-CT / AM-decomposition
        pipeline -- no external files touched."""
        from ..demo_pcb import build_sophisticated_pcb_stackup
        try:
            spacing = (0.05, 0.05, 0.02)
            stackup, air, layer_names, layer_slices = build_sophisticated_pcb_stackup(
                resolution_z=spacing[2]
            )
            vox = Voxelizer(resolution_xy=spacing[0], resolution_z=spacing[2])
            vol, material_ids_zhw = vox.voxelize_stackup(stackup, air, return_material_ids=True)

            sim_zhw = simulate_ct_volume(vol)
            sim_hwz = np.moveaxis(sim_zhw, 0, -1)
            gt_rgba = colorize_material_labels(material_ids_zhw)

            self._current_rgba_base_left = None
            self.model.setCube(sim_hwz)
            # setCube's own auto-layer-detection is tuned for real CT scans
            # with hundreds of slices and collapses to one degenerate row on
            # a volume this small -- overwrite with the real, known layer
            # boundaries from build_sophisticated_pcb_stackup instead.
            self.model.setupTblRows(layer_names, layer_slices)
            self.model.setSpacing(*spacing)
            self.viewer_3d_left.setDataCube(sim_hwz)
            self.viewer_3d_left.resetCamera()
            self.viewer_3d_right.setRGBADataCube(gt_rgba)
            self.viewer_3d_right.resetCamera()

            self._am_result = simulate_am_volume(vol)
            QtWidgets.QMessageBox.information(
                self, "Sophisticated PCB Demo Loaded",
                f"Voxelized: {vol.shape[:3]}  |  AM decomposition complete -- "
                "use 'Show Alvarez-Macovski Plot' / AM Color buttons to inspect.",
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load PCB demo:\n{e}")

    def loadRealPcbCtDemo(self):
        """Load genuinely real data on both sides: a real X-ray CT scan
        (E:\\Data\\20250701_PCB2, downsampled) into the left pane, and a real
        Gerber PCB design (E:\\git\\AltiumGerberX2, board 'PCB1') into the
        right pane.

        NOT a confirmed registered pair: the "PCB1"/"PCB2" naming does not by
        itself establish these are the same physical board, so this loads
        them side by side as independent real data rather than claiming an
        alignment that hasn't been verified.
        """
        from ..real_pcb_demo import load_real_pcb_ct

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        self.statusBar().showMessage(
            "Loading real CT scan (downsampling ~1375 slices, this can take a while)..."
        )
        QtWidgets.QApplication.processEvents()
        try:
            ct_zhw = load_real_pcb_ct()
            ct_hwz = np.moveaxis(ct_zhw, 0, -1)

            self._current_rgba_base_left = None
            self.model.setCube(ct_hwz)
            self.viewer_3d_left.setDataCube(ct_hwz)
            self.viewer_3d_left.resetCamera()

            self.statusBar().showMessage("Loading real Gerber PCB stackup...")
            QtWidgets.QApplication.processEvents()
            gerber_cfg = (
                Path(__file__).resolve().parents[2] / "configs" / "altium_gerberx2_stackup.json"
            )
            self.loadGerberStackup(gerber_cfg)

            self.statusBar().showMessage("Real PCB demo loaded.", 5000)
            QtWidgets.QMessageBox.information(
                self, "Real PCB Demo Loaded",
                f"Left: real CT scan {ct_hwz.shape} (E:\\Data\\20250701_PCB2, downsampled).\n"
                "Right: real Gerber PCB (E:\\git\\AltiumGerberX2, board 'PCB1').\n\n"
                "Note: these are NOT a confirmed registered pair -- shown side "
                "by side as independent real data, not aligned. Real voxel "
                "spacing for the CT scan is also unknown -- set it via "
                "'Set Real CT Voxel Spacing...' before registering.",
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.statusBar().clearMessage()
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load real PCB demo:\n{e}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def setRealCTSpacing(self):
        dlg = SpacingDialog(self.model.spacing_xyz, parent=self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self.model.setSpacing(*dlg.spacing())

    def openRegistrationTool(self):
        if self._gerber_intensity_hwz is None or self._gerber_voxel_spacing_mm is None:
            QtWidgets.QMessageBox.information(
                self, "No Gerber Volume",
                "Load/voxelize a Gerber stackup first to enable registration.",
            )
            return
        if self.model.dataCube is None:
            QtWidgets.QMessageBox.information(
                self, "No Real CT Data",
                "Load real CT data first to enable registration.",
            )
            return
        if not self.model._spacing_confirmed:
            proceed = QtWidgets.QMessageBox.question(
                self, "Real CT Spacing Not Set",
                "Real CT voxel spacing hasn't been confirmed for the "
                "currently loaded data -- it's still using the "
                "1.0mm/voxel placeholder default. Registration will be "
                "physically meaningless until you set it via "
                "'Set Real CT Voxel Spacing...'.\n\nProceed anyway?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if proceed != QtWidgets.QMessageBox.Yes:
                return

        self.registrationPanel.setVolumes(
            self._gerber_intensity_hwz,
            self._gerber_voxel_spacing_mm,
            self.model.dataCube,
            self.model.spacing_xyz,
        )
        self._workspaceStack.setCurrentWidget(self.registrationPanel)
        self.statusBar().showMessage("Registration workspace active", 3000)

    def _showMainWorkspace(self):
        self._workspaceStack.setCurrentWidget(self._mainWorkspacePage)
        self.statusBar().showMessage("Main workspace active", 3000)

    def _onRegistrationAccepted(self, transform: RigidScaleTransform):
        QtWidgets.QMessageBox.information(
            self, "Registration Accepted",
            f"Transform stored: tx={transform.tx:.3f}mm ty={transform.ty:.3f}mm "
            f"tz={transform.tz:.3f}mm rot_z={transform.rot_z_deg:.2f}deg "
            f"scale={transform.scale:.3f}",
        )

    def transferGerberLabels(self):
        if self._registration_session.accepted_transform is None:
            QtWidgets.QMessageBox.information(
                self, "No Registration",
                "Use 'Register Gerber to Real CT...' and accept a transform first.",
            )
            return
        if self._gerber_material_id_vol_hwz is None:
            QtWidgets.QMessageBox.information(
                self, "No Material Labels",
                "Re-voxelize the Gerber stackup with material-ID labels first "
                "(e.g. via the PCB registration showcase, or a future "
                "'voxelize with labels' action).",
            )
            return
        if self.model.dataCube is None:
            QtWidgets.QMessageBox.information(
                self, "No Real CT Data", "Load real CT data first."
            )
            return
        if not self._registration_session.is_compatible(
            self._gerber_intensity_hwz,
            self._gerber_voxel_spacing_mm,
            self.model.dataCube,
            self.model.spacing_xyz,
        ):
            QtWidgets.QMessageBox.information(
                self, "Registration Out Of Date",
                "The real CT data or its voxel spacing changed since this "
                "transform was accepted. Use 'Register Gerber to Real CT...' "
                "again before transferring labels.",
            )
            return

        labels_hwz = resample_labels(
            self._gerber_material_id_vol_hwz,
            self._gerber_voxel_spacing_mm,
            self.model.dataCube.shape,
            self.model.spacing_xyz,
            self._registration_session.accepted_transform,
        )
        rgba = colorize_material_labels(np.moveaxis(labels_hwz, -1, 0))

        self._current_rgba_base_left = rgba
        self.model.setRGBACube(self.model.dataCube, rgba)
        threshold = self.threshSlider_left.value() / 100.0
        tf = self.opacitySlider_left.value() / 2.0
        self.viewer_3d_left.setRGBADataCube(self._apply_rgba_controls(rgba, threshold, tf))
        self.viewer_3d_left.resetCamera()
        self.materialLegendWidget.setVisible(True)
        self.amLegendBtn.setChecked(True)

    def loadPcbRegistrationShowcase(self):
        """Load a recognizable, known-ground-truth PCB registration scene.

        The left pane is a blurred/noisy CT-like acquisition. The right pane
        is the clean, material-colored Gerber ground truth. Both are generated
        from the same production fixture used by regression tests.
        """
        from ..synthetic_registration import make_pcb_registration_showcase

        pair = make_pcb_registration_showcase()

        self._current_rgba_base_left = None
        self.model.setCube(pair.fixed_hwz)
        self.model.setupTblRows(
            pair.layer_names,
            [np.asarray(slices, dtype=int) for slices in pair.layer_slices],
        )
        self.model.setSpacing(*pair.fixed_spacing)
        self.viewer_3d_left.setDataCube(pair.fixed_hwz)
        self.viewer_3d_left.setCameraTop()

        self._gerber_intensity_hwz = pair.moving_hwz
        self._gerber_voxel_spacing_mm = pair.moving_spacing
        self._gerber_material_id_vol_hwz = pair.moving_labels_hwz
        self._showcase_material_labels_hwz = pair.moving_labels_hwz
        self._showcase_am_volume_zhw2 = material_labels_to_zeff_rhoe(
            np.moveaxis(pair.moving_labels_hwz, -1, 0)
        )
        self._am_result = None
        self._registration_session.invalidate("PCB showcase reloaded")
        ground_truth_rgba = colorize_material_labels(
            np.moveaxis(pair.moving_labels_hwz, -1, 0)
        )
        self._current_rgba_base_right = ground_truth_rgba
        self.viewer_3d_right.setRGBADataCube(ground_truth_rgba)
        self.viewer_3d_right.setCameraTop()

        material_summary = ", ".join(name.title() for name in pair.material_names[1:])
        self.gerberLegendLabel.setText(
            f"PCB showcase materials: {material_summary}"
        )
        self.materialLegendWidget.setVisible(True)
        self.amLegendBtn.setChecked(True)
        self.statusBar().showMessage(
            "PCB registration showcase loaded: CT left, material Gerber right",
            8000,
        )

        t = pair.known_transform
        QtWidgets.QMessageBox.information(
            self, "PCB Registration Showcase Loaded",
            "LEFT: blurred/noisy CT-like scan\n"
            "RIGHT: clean material-colored Gerber model\n\n"
            "Recognizable landmarks include a QFP controller, a smaller IC, "
            "connector shell, gold edge fingers, routed traces, vias, "
            "fiducials, passives, and four mounting holes.\n\n"
            "Known answer (aligns RIGHT onto LEFT):\n"
            f"tx={t.tx:.3f}mm  ty={t.ty:.3f}mm  tz={t.tz:.3f}mm\n"
            f"rot_z={t.rot_z_deg:.2f}deg  scale={t.scale:.3f}\n\n"
            "Both panes start in Top view for readability. Open "
            "'Register Gerber to Real CT...' to compare the red Gerber and "
            "green CT overlay, run Suggest, review, and explicitly Accept.\n\n"
            "Choose 'Run Showcase Z-effective / Electron Density' to run the "
            "Alvarez-Macovski component on the same material ground truth.",
        )

    def runPcbShowcaseAM(self):
        """Run AM decomposition on the active interpretable PCB showcase."""
        if (
            self._showcase_am_volume_zhw2 is None
            or self._showcase_material_labels_hwz is None
            or self._gerber_material_id_vol_hwz
            is not self._showcase_material_labels_hwz
        ):
            QtWidgets.QMessageBox.information(
                self,
                "PCB Showcase Not Loaded",
                "Load 'PCB Registration Showcase' before running its "
                "Z-effective / electron-density decomposition.",
            )
            return

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        self.statusBar().showMessage(
            "Running showcase Alvarez-Macovski decomposition..."
        )
        QtWidgets.QApplication.processEvents()
        try:
            self._am_result = simulate_am_volume(
                self._showcase_am_volume_zhw2,
                num_angles=64,
            )
            self._renderAMVolume()
            self.viewer_3d_left.setCameraTop()
            self.materialLegendWidget.setVisible(True)
            self.amLegendBtn.setChecked(True)
            self._amPlotWindow = AMPlotWindow(self._am_result, parent=self)
            self._amPlotWindow.show()
            self.statusBar().showMessage(
                "Showcase AM decomposition ready: choose Z_eff, rho_e, or Material",
                10000,
            )
            QtWidgets.QMessageBox.information(
                self,
                "Showcase AM Decomposition Complete",
                "Recovered Z-effective and electron-density volumes are ready.\n\n"
                "The main-window AM Color buttons switch among Z_eff, rho_e, "
                "and approximate material views. The comparison window shows "
                "ground truth beside material-regularized recovery; raw "
                "decomposition arrays remain preserved for quantitative audit.",
            )
        except Exception as error:
            import traceback

            traceback.print_exc()
            self.statusBar().clearMessage()
            QtWidgets.QMessageBox.critical(
                self,
                "Showcase AM Decomposition Failed",
                f"The decomposition could not complete:\n{error}",
            )
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _loadSyntheticRegistrationDemo(self):
        """Backward-compatible entry point retained for scripts/tests."""
        self.loadPcbRegistrationShowcase()

    def fixRotation(self):
        win = RotateWindow(self.model.dataCube)
        win.show()

        def _accept_rotation():
            self._current_rgba_base_left = None
            self.model.setCube(win.cube)

        win.sigChangesAccepted.connect(_accept_rotation)

    def showDevConsole(self):
        safeSpawnDevConsole(self)

    def updateTblViews(
        self, selected: QtCore.QItemSelection, deselected: QtCore.QItemSelection
    ):
        def toSourceRows(itemSelection):
            rows = []
            for idx in itemSelection.indexes():
                sourceIdx = self.proxyModel.mapToSource(idx)
                rows.append(sourceIdx.row())
            return np.unique(rows)

        selRows = toSourceRows(selected)
        deselRows = toSourceRows(deselected)
        for r in deselRows:
            if 0 <= r < len(self.model.tblModel.dataDf):
                self.model.tblModel.dataDf.iloc[r, self.model.tblModel.dataDf.columns.get_loc("View")] = False
        for r in selRows:
            if 0 <= r < len(self.model.tblModel.dataDf):
                self.model.tblModel.dataDf.iloc[r, self.model.tblModel.dataDf.columns.get_loc("View")] = True
        self.model.tblModel.sigDataChanged.emit()

    def _updateStatusBar(self):
        cube = self.model.dataCube
        if cube is None or cube.size == 0:
            self.cubeInfoLabel.setText("No data loaded")
            return
        h, w, d = cube.shape
        viewed = self.model.layerDf.loc[self.model.layerDf["View"], "Layer Name"].tolist()
        layerText = ", ".join(str(n) for n in viewed) if viewed else "none"
        self.cubeInfoLabel.setText(f"Cube: {h}×{w}×{d}  |  Visible: {layerText}")

    def _resetLevelSliders(self):
        cube = self.model.dataCube
        if cube is None or cube.size == 0:
            return
        maxVal = int(cube.max())
        self.minSlider.setRange(0, maxVal)
        self.maxSlider.setRange(0, maxVal)
        self.minSlider.setValue(0)
        self.maxSlider.setValue(maxVal)

    def _onMouseMovedInPlot(self, pos):
        if not self.plt.sceneBoundingRect().contains(pos):
            self.pixelValueLabel.setText("Pixel: —")
            return
        mousePoint = self.plt.getViewBox().mapSceneToView(pos)
        x, y = int(mousePoint.x()), int(mousePoint.y())
        visibleItems = [item for item in self.viewer.items if item.isVisible()]
        if visibleItems:
            topItem = visibleItems[-1]
            img = topItem.image
            if img is not None and 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                val = img[y, x]
                self.pixelValueLabel.setText(f"Pixel: ({x}, {y}) = {val:.1f}")
                return
        self.pixelValueLabel.setText("Pixel: —")

    def _selectAllLayers(self):
        self.tbl.selectAll()

    def _deselectAllLayers(self):
        self.tbl.clearSelection()

    def _moveLayerUp(self):
        cur = self.tbl.currentIndex().row()
        srcRow = self.proxyModel.mapToSource(self.tbl.currentIndex()).row()
        if srcRow > 0:
            self.model.tblModel.swapRows(srcRow, srcRow - 1)
            self.tbl.selectRow(max(cur - 1, 0))

    def _moveLayerDown(self):
        cur = self.tbl.currentIndex().row()
        srcRow = self.proxyModel.mapToSource(self.tbl.currentIndex()).row()
        maxRow = self.model.tblModel.rowCount() - 1
        if srcRow < maxRow:
            self.model.tblModel.swapRows(srcRow, srcRow + 1)
            self.tbl.selectRow(min(cur + 1, self.proxyModel.rowCount() - 1))

    def _selectNextLayer(self):
        cur = self.tbl.currentIndex().row()
        self.tbl.selectRow(min(cur + 1, self.proxyModel.rowCount() - 1))

    def _selectPrevLayer(self):
        cur = self.tbl.currentIndex().row()
        self.tbl.selectRow(max(cur - 1, 0))

    def _populateRecentMenu(self):
        self.recentMenu.clear()
        recent = self._settings.value("recentFiles", []) or []
        if isinstance(recent, str):
            recent = [recent]
        for path in recent[:10]:
            act = self.recentMenu.addAction(Path(path).name)
            act.setToolTip(path)
            act.triggered.connect(lambda checked, p=path: self._loadRecentFile(p))
        if not recent:
            self.recentMenu.addAction("(none)").setEnabled(False)

    def _loadRecentFile(self, path: str):
        try:
            data = np.load(path)
            self._current_rgba_base_left = None
            self.model.setCube(data)
            self._addToRecentFiles(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load file:\n{e}")

    def _addToRecentFiles(self, path: str):
        recent = self._settings.value("recentFiles", []) or []
        if isinstance(recent, str):
            recent = [recent]
        path = str(path)
        if path in recent:
            recent.remove(path)
        recent.insert(0, path)
        self._settings.setValue("recentFiles", recent[:10])
        self._populateRecentMenu()

    def _on3DControlChangedLeft(self):
        if self.model.dataCube is None or self.model.dataCube.size == 0:
            return
        threshold = self.threshSlider_left.value() / 100.0
        tf = self.opacitySlider_left.value() / 2.0
        if self._current_rgba_base_left is not None:
            rgba = self._apply_rgba_controls(self._current_rgba_base_left, threshold, tf)
            self.viewer_3d_left.setRGBADataCube(rgba)
        else:
            self.viewer_3d_left.setDataCube(
                self.model.dataCube,
                transparencyFactor=tf,
                threshold=threshold,
            )

    def _on3DControlChangedRight(self):
        if self._current_rgba_base_right is None:
            return
        threshold = self.threshSlider_right.value() / 100.0
        tf = self.opacitySlider_right.value() / 2.0
        rgba = self._apply_rgba_controls(
            self._current_rgba_base_right, threshold, tf,
            intensity_cube=self._gerber_intensity_hwz,
        )
        self.viewer_3d_right.setRGBADataCube(rgba)

    @bind(downsampleFactor=dict(limits=(1, 10)), transparency=dict(limits=(0, None)))
    def highlight3DSlices(self, downsampleFactor=3, transparencyFactor=None):
        """
        Parameters
        ----------
        downsampleFactor
            How much to downsample the 3D view by. 1 means no downsampling.
        transparencyFactor
            Opacity exponent. If None, uses the 3D Opacity slider value.
        """
        if transparencyFactor is None:
            transparencyFactor = self.opacitySlider_left.value() / 2.0
        threshold = self.threshSlider_left.value() / 100.0
        selectedRows = self.model.layerDf["View"]
        if np.any(selectedRows):
            selectedSlices = np.concatenate(
                self.model.layerDf.loc[selectedRows, "Slices"].values
            )
        else:
            selectedSlices = None

        if self._current_rgba_base_left is not None:
            rgba = self._apply_rgba_controls(self._current_rgba_base_left, threshold, transparencyFactor)
            if selectedSlices is not None:
                all_z = np.arange(rgba.shape[2])
                non_sel = np.setdiff1d(all_z, selectedSlices)
                if non_sel.size:
                    rgba[:, :, non_sel, 3] = (
                        rgba[:, :, non_sel, 3].astype(np.float32) * 0.05
                    ).astype(np.uint8)
            self.viewer_3d_left.setRGBADataCube(rgba)
        else:
            self.viewer_3d_left.setDataCube(
                self.model.dataCube, selectedSlices, downsampleFactor, transparencyFactor,
                threshold=threshold
            )

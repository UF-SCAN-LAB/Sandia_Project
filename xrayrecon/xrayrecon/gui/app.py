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

from .helpers import FRPgParamDelegate
from .layermodel import LayerModel
from .layerview import LayerView
from .view3d import PCBViewer3D

# Gerber Integration Imports
from ..pygerber_integration import PyGerberRasterizer
from ..gerber_utils import GerberLayer
from ..voxelizer import Voxelizer
from ..materials import MATERIALS_DB, get_material
from ..stackup import StackupConfig

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
        self.viewer_3d = PCBViewer3D(dataCube)
        self._gerber_layer_blocks = []
        self._gerber_layer_legend = []
        self._gerber_gap_slices = 20
        # self.model.sigCubeChanged.connect(
        #     lambda: self.viewer_3d.setDataCube(self.model.dataCube)
        # )

        pgDict = dict(type="list", values={"True": True, "False": False})
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
            self.showGerberColorView, name="Show Gerber Color 3D"
        )
        self._buildUi()

    def _buildUi(self):
        # --- Menu bar ---
        menu = QtWidgets.QMenuBar(self)
        self.setMenuBar(menu)
        fileMenu = menu.addMenu("File")
        self.recentMenu = fileMenu.addMenu("Recent Files")
        self._settings = QtCore.QSettings("XrayRecon", "MainWin")
        self._populateRecentMenu()

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
        ll.addWidget(self.toolsEditor)
        ll.addWidget(tblBtnRow)
        ll.addWidget(self.layerFilterEdit)
        ll.addWidget(self.tbl)

        # --- Center panel: camera buttons + view toggles + 3D viewer ---
        cameraOpts = [OptionsDict(n, pType="action") for n in ["Top", "Front", "Side", "Reset"]]
        cameraFns = [
            self.viewer_3d.setCameraTop,
            self.viewer_3d.setCameraFront,
            self.viewer_3d.setCameraSide,
            self.viewer_3d.resetCamera,
        ]
        self.cameraCtrl = ButtonCollection(
            None, "Camera", cameraOpts, cameraFns, exclusive=False, asToolButton=False
        )

        self.gridToggleBtn = QtWidgets.QPushButton("Grid")
        self.gridToggleBtn.setCheckable(True)
        self.gridToggleBtn.setChecked(True)
        self.axisToggleBtn = QtWidgets.QPushButton("Axis")
        self.axisToggleBtn.setCheckable(True)
        self.axisToggleBtn.setChecked(True)
        self.gridToggleBtn.toggled.connect(self.viewer_3d.setGridVisible)
        self.axisToggleBtn.toggled.connect(self.viewer_3d.setAxisVisible)
        for btn in [self.gridToggleBtn, self.axisToggleBtn]:
            btn.setMaximumHeight(22)
        viewToggleRow = QtWidgets.QWidget()
        viewToggleLayout = QtWidgets.QHBoxLayout(viewToggleRow)
        viewToggleLayout.setContentsMargins(0, 0, 0, 0)
        viewToggleLayout.addWidget(self.gridToggleBtn)
        viewToggleLayout.addWidget(self.axisToggleBtn)

        # --- 3D volume controls ---
        self.threshSlider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.threshSlider.setRange(0, 50)
        self.threshSlider.setValue(10)
        self.threshSlider.setMaximumHeight(18)
        self.threshSlider.setToolTip("3D threshold: voxels below this intensity are hidden (0–50%)")

        self.opacitySlider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.opacitySlider.setRange(1, 10)
        self.opacitySlider.setValue(4)
        self.opacitySlider.setMaximumHeight(18)
        self.opacitySlider.setToolTip("3D opacity exponent: higher = more transparent mid-range voxels")

        threshRow = QtWidgets.QWidget()
        threshLayout = QtWidgets.QHBoxLayout(threshRow)
        threshLayout.setContentsMargins(0, 0, 0, 0)
        threshLayout.addWidget(QtWidgets.QLabel("3D Thresh:"))
        threshLayout.addWidget(self.threshSlider)

        opacRow = QtWidgets.QWidget()
        opacLayout = QtWidgets.QHBoxLayout(opacRow)
        opacLayout.setContentsMargins(0, 0, 0, 0)
        opacLayout.addWidget(QtWidgets.QLabel("3D Opacity:"))
        opacLayout.addWidget(self.opacitySlider)

        self.threshSlider.valueChanged.connect(self._on3DControlChanged)
        self.opacitySlider.valueChanged.connect(self._on3DControlChanged)

        cameraControlsWidget = QtWidgets.QWidget()
        ccl = QtWidgets.QVBoxLayout(cameraControlsWidget)
        ccl.setContentsMargins(0, 0, 0, 0)
        ccl.setSpacing(2)
        ccl.addWidget(self.cameraCtrl)
        ccl.addWidget(viewToggleRow)
        ccl.addWidget(threshRow)
        ccl.addWidget(opacRow)
        ccl.addStretch()

        centerSplitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        centerSplitter.addWidget(cameraControlsWidget)
        centerSplitter.addWidget(self.viewer_3d)
        centerSplitter.setStretchFactor(0, 0)
        centerSplitter.setStretchFactor(1, 1)

        centerPanel = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(centerPanel)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.addWidget(centerSplitter)

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

        children = EasyWidget(
            [leftPanel, centerPanel, rightPanel],
            useSplitter=True,
            layout="H",
        )
        # Extra layer of nesting allows margin to look a bit better
        EasyWidget.buildMainWindow([children], window=self)
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
        self.viewer_3d.setRGBADataCube(rgba_volume)

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
        self.model.setCube(data_hwz)
        self.model.setupTblRows(layer_names, layer_slices)

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

    @bind(file=dict(type="file", nameFilter="Stackup Files (*.json);;All Files (*)", value=""))
    def loadGerberStackup(self, file: str | Path):
        if not file:
            return

        stackup_path = Path(file)
        try:
            config = StackupConfig.from_json(stackup_path)
            if config.resolution_xy <= 0 or config.resolution_z <= 0:
                raise ValueError("resolution_xy and resolution_z must both be > 0")

            rasterizer = PyGerberRasterizer(dpi=config.dpi)
            substrate_mat = self._material_or_error(config.substrate_material)
            air_mat = self._material_or_error("AIR")

            stackup = []
            visual_entries = []
            layer_names = []
            layer_slices = []
            z_cursor = 0

            for layer_cfg in config.layers:
                layer_path = config.resolve_layer_path(stackup_path, layer_cfg.file)
                if not layer_path.exists():
                    raise FileNotFoundError(
                        f"Gerber file for layer '{layer_cfg.name}' not found: {layer_path}"
                    )

                with open(layer_path, "r") as f:
                    content = f.read()
                mask = rasterizer.rasterize(content)
                if np.max(mask) == 0:
                    raise ValueError(
                        f"Layer '{layer_cfg.name}' produced an empty raster mask."
                    )

                layer = GerberLayer(layer_cfg.name)
                layer.raster_mask = mask

                h_px, w_px = mask.shape
                width_mm = w_px / (config.dpi / 25.4)
                height_mm = h_px / (config.dpi / 25.4)
                layer.update_bounds(0, 0)
                layer.update_bounds(width_mm, height_mm)

                feature_mat = self._material_or_error(layer_cfg.feature_material)
                bg_mat = self._material_or_error(layer_cfg.background_material)
                stackup.append((layer, layer_cfg.thickness, feature_mat, bg_mat))

                layer_num_slices = max(
                    1, int(np.round(layer_cfg.thickness / config.resolution_z))
                )
                layer_names.append(layer_cfg.name)
                layer_slices.append(np.arange(z_cursor, z_cursor + layer_num_slices))
                z_cursor += layer_num_slices
                visual_entries.append(
                    {
                        "name": layer_cfg.name,
                        "is_substrate": False,
                        "num_slices": layer_num_slices,
                        "color": layer_cfg.color,
                    }
                )

                if layer_cfg.gap > 0:
                    substrate_name = f"{layer_cfg.name}_substrate"
                    substrate_layer = GerberLayer(substrate_name)
                    substrate_layer.update_bounds(0, 0)
                    substrate_layer.update_bounds(width_mm, height_mm)
                    stackup.append(
                        (substrate_layer, layer_cfg.gap, substrate_mat, substrate_mat)
                    )

                    gap_slices = max(1, int(np.round(layer_cfg.gap / config.resolution_z)))
                    layer_names.append(substrate_name)
                    layer_slices.append(np.arange(z_cursor, z_cursor + gap_slices))
                    z_cursor += gap_slices
                    visual_entries.append(
                        {
                            "name": substrate_name,
                            "is_substrate": True,
                            "num_slices": gap_slices,
                            "color": (80, 80, 80, 10),
                        }
                    )

            vox = Voxelizer(
                resolution_xy=config.resolution_xy,
                resolution_z=config.resolution_z,
            )
            vol = vox.voxelize_stackup(stackup, air_mat)
            self._apply_gerber_volume(
                vol,
                layer_names,
                layer_slices,
                visual_entries,
                default_gap_slices=config.default_gap_slices,
            )

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

    def fixRotation(self):
        win = RotateWindow(self.model.dataCube)
        win.show()
        win.sigChangesAccepted.connect(lambda: self.model.setCube(win.cube))

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

    def _on3DControlChanged(self):
        if self.model.dataCube is None or self.model.dataCube.size == 0:
            return
        threshold = self.threshSlider.value() / 100.0
        tf = self.opacitySlider.value() / 2.0
        self.viewer_3d.setDataCube(
            self.model.dataCube,
            transparencyFactor=tf,
            threshold=threshold,
        )

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
            transparencyFactor = self.opacitySlider.value() / 2.0
        threshold = self.threshSlider.value() / 100.0
        selectedRows = self.model.layerDf["View"]
        if np.any(selectedRows):
            selectedSlices = np.concatenate(
                self.model.layerDf.loc[selectedRows, "Slices"].values
            )
        else:
            selectedSlices = None
        self.viewer_3d.setDataCube(
            self.model.dataCube, selectedSlices, downsampleFactor, transparencyFactor,
            threshold=threshold
        )

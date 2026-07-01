from __future__ import annotations

from pathlib import Path
from typing import Callable

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from ..materials import MATERIALS_DB, MATERIAL_COLORS as _DEFAULT_MATERIAL_COLORS
from ..stackup import StackupConfig, StackupLayer

_MATERIAL_NAMES = sorted(MATERIALS_DB.keys())

(
    _COL_NAME,
    _COL_FILE,
    _COL_THICKNESS,
    _COL_GAP,
    _COL_FEATURE,
    _COL_BACKGROUND,
    _COL_COLOR,
) = range(7)
_HEADERS = [
    "Layer Name",
    "Gerber File",
    "Thickness (mm)",
    "Gap (mm)",
    "Feature Material",
    "Background Material",
    "Color",
]


class StackupEditorWindow(QtWidgets.QMainWindow):
    """Non-modal window for assigning per-layer materials to a Gerber stackup.

    Lets the user add Gerber files, pick a feature/background material per
    layer from a dropdown (instead of hand-editing a stackup JSON), then
    either voxelize the result into the main window's right-hand 3D pane
    (*voxelizeCallback*), simulate a CT reconstruction of the same stackup
    into the left-hand pane (*simulateCallback*), for a
    ground-truth-vs-simulated-scan comparison of the identical object, run
    an Alvarez-Macovski dual-energy decomposition and plot the recovered
    effective atomic number / electron density for one representative slice
    (*amPlotCallback*), or render that same decomposition as a colorized
    volume across every slice in the left-hand 3D pane (*amVolumeCallback*).
    """

    def __init__(
        self,
        voxelizeCallback: Callable[[StackupConfig, Path], None],
        simulateCallback: Callable[[StackupConfig, Path], None],
        amPlotCallback: Callable[[StackupConfig, Path], None],
        amVolumeCallback: Callable[[StackupConfig, Path], None],
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Gerber Stackup Editor")
        self._voxelizeCallback = voxelizeCallback
        self._simulateCallback = simulateCallback
        self._amPlotCallback = amPlotCallback
        self._amVolumeCallback = amVolumeCallback
        self._lastStackupPath: Path = Path.cwd()
        self._configExtras: dict = {}

        self.table = QtWidgets.QTableWidget(0, len(_HEADERS))
        self.table.setHorizontalHeaderLabels(_HEADERS)
        self.table.horizontalHeader().setSectionResizeMode(
            _COL_FILE, QtWidgets.QHeaderView.ResizeMode.Stretch
        )

        addBtn = QtWidgets.QPushButton("Add Layer(s)...")
        removeBtn = QtWidgets.QPushButton("Remove Selected")
        loadJsonBtn = QtWidgets.QPushButton("Load Stackup JSON...")
        saveJsonBtn = QtWidgets.QPushButton("Save Stackup JSON...")
        simulateBtn = QtWidgets.QPushButton("Simulate X-ray → Left Pane")
        amPlotBtn = QtWidgets.QPushButton("Show Z_eff / ρ_e Plot")
        amVolumeBtn = QtWidgets.QPushButton("Show AM Volume in 3D")
        voxelizeBtn = QtWidgets.QPushButton("Voxelize && Show")
        voxelizeBtn.setStyleSheet("font-weight: bold;")

        addBtn.clicked.connect(self._onAddLayers)
        removeBtn.clicked.connect(self._onRemoveSelected)
        loadJsonBtn.clicked.connect(self._onLoadJson)
        saveJsonBtn.clicked.connect(self._onSaveJson)
        simulateBtn.clicked.connect(self._onSimulate)
        amPlotBtn.clicked.connect(self._onShowAMPlot)
        amVolumeBtn.clicked.connect(self._onShowAMVolume)
        voxelizeBtn.clicked.connect(self._onVoxelize)

        btnRow = QtWidgets.QWidget()
        btnLayout = QtWidgets.QHBoxLayout(btnRow)
        btnLayout.setContentsMargins(0, 0, 0, 0)
        for btn in [addBtn, removeBtn, loadJsonBtn, saveJsonBtn]:
            btnLayout.addWidget(btn)
        btnLayout.addStretch()
        btnLayout.addWidget(simulateBtn)
        btnLayout.addWidget(amPlotBtn)
        btnLayout.addWidget(amVolumeBtn)
        btnLayout.addWidget(voxelizeBtn)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.addWidget(btnRow)
        layout.addWidget(self.table)
        self.setCentralWidget(central)
        self.resize(900, 400)

    # --- row helpers ---------------------------------------------------

    def _addRow(self, layer: StackupLayer):
        row = self.table.rowCount()
        self.table.insertRow(row)

        self.table.setItem(row, _COL_NAME, QtWidgets.QTableWidgetItem(layer.name))
        fileItem = QtWidgets.QTableWidgetItem(layer.file)
        fileItem.setFlags(fileItem.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
        self.table.setItem(row, _COL_FILE, fileItem)

        thicknessSpin = QtWidgets.QDoubleSpinBox()
        thicknessSpin.setRange(0.001, 10.0)
        thicknessSpin.setDecimals(4)
        thicknessSpin.setValue(layer.thickness)
        self.table.setCellWidget(row, _COL_THICKNESS, thicknessSpin)

        gapSpin = QtWidgets.QDoubleSpinBox()
        gapSpin.setRange(0.0, 10.0)
        gapSpin.setDecimals(4)
        gapSpin.setValue(layer.gap)
        self.table.setCellWidget(row, _COL_GAP, gapSpin)

        featureCombo = QtWidgets.QComboBox()
        featureCombo.addItems(_MATERIAL_NAMES)
        featureCombo.setCurrentText(layer.feature_material.upper())
        self.table.setCellWidget(row, _COL_FEATURE, featureCombo)

        bgCombo = QtWidgets.QComboBox()
        bgCombo.addItems(_MATERIAL_NAMES)
        bgCombo.setCurrentText(layer.background_material.upper())
        self.table.setCellWidget(row, _COL_BACKGROUND, bgCombo)

        colorBtn = QtWidgets.QPushButton()
        colorBtn.setProperty("rgba", tuple(layer.color))
        self._refreshColorBtn(colorBtn)
        colorBtn.clicked.connect(lambda: self._onPickColor(colorBtn))
        featureCombo.currentTextChanged.connect(
            lambda name, btn=colorBtn: self._onFeatureMaterialChanged(name, btn)
        )
        self.table.setCellWidget(row, _COL_COLOR, colorBtn)

    def _onFeatureMaterialChanged(self, materialName: str, colorBtn: QtWidgets.QPushButton):
        color = _DEFAULT_MATERIAL_COLORS.get(materialName.upper(), (255, 140, 0, 200))
        colorBtn.setProperty("rgba", color)
        self._refreshColorBtn(colorBtn)

    def _onPickColor(self, colorBtn: QtWidgets.QPushButton):
        current = colorBtn.property("rgba") or (255, 140, 0, 200)
        qColor = QtGui.QColor(*current)
        picked = QtWidgets.QColorDialog.getColor(
            qColor,
            self,
            "Pick Layer Color",
            QtWidgets.QColorDialog.ColorDialogOption.ShowAlphaChannel,
        )
        if picked.isValid():
            colorBtn.setProperty(
                "rgba", (picked.red(), picked.green(), picked.blue(), picked.alpha())
            )
            self._refreshColorBtn(colorBtn)

    @staticmethod
    def _refreshColorBtn(colorBtn: QtWidgets.QPushButton):
        r, g, b, a = colorBtn.property("rgba")
        colorBtn.setStyleSheet(f"background-color: rgba({r},{g},{b},{a}); border: 1px solid #444;")
        colorBtn.setText(f"{r},{g},{b},{a}")

    # --- toolbar actions -------------------------------------------------

    def _onAddLayers(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Add Gerber Layer(s)", "", "Gerber Files (*.gbr *.GBR);;All Files (*)"
        )
        for f in files:
            path = Path(f)
            self._addRow(
                StackupLayer(
                    file=str(path),
                    name=path.stem,
                    thickness=0.035,
                    gap=0.1,
                    feature_material="COPPER",
                    background_material="AIR",
                    color=_DEFAULT_MATERIAL_COLORS["COPPER"],
                )
            )

    def _onRemoveSelected(self):
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.table.removeRow(row)

    def _onLoadJson(self):
        file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Stackup JSON", "", "Stackup Files (*.json);;All Files (*)"
        )
        if not file:
            return
        try:
            config = StackupConfig.from_json(file)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load stackup:\n{e}")
            return
        self.setConfig(config, Path(file))

    def _onSaveJson(self):
        file, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Stackup JSON", "", "Stackup Files (*.json);;All Files (*)"
        )
        if not file:
            return
        try:
            config = self._buildConfig()
            config.to_json(file)
            self._lastStackupPath = Path(file)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save stackup:\n{e}")

    def _buildValidConfig(self) -> StackupConfig | None:
        """Build a StackupConfig from the table, showing an error/info dialog
        and returning None if the table is invalid or empty."""
        try:
            config = self._buildConfig()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Invalid stackup:\n{e}")
            return None
        if not config.layers:
            QtWidgets.QMessageBox.information(
                self, "No Layers", "Add at least one Gerber layer first."
            )
            return None
        return config

    def _onVoxelize(self):
        config = self._buildValidConfig()
        if config is not None:
            self._voxelizeCallback(config, self._lastStackupPath)

    def _onSimulate(self):
        config = self._buildValidConfig()
        if config is not None:
            self._simulateCallback(config, self._lastStackupPath)

    def _onShowAMPlot(self):
        config = self._buildValidConfig()
        if config is not None:
            self._amPlotCallback(config, self._lastStackupPath)

    def _onShowAMVolume(self):
        config = self._buildValidConfig()
        if config is not None:
            self._amVolumeCallback(config, self._lastStackupPath)

    # --- config <-> table --------------------------------------------------

    def setConfig(self, config: StackupConfig, stackup_path: Path):
        self._lastStackupPath = stackup_path
        self.table.setRowCount(0)
        for layer in config.layers:
            self._addRow(layer)
        self._configExtras = dict(
            dpi=config.dpi,
            resolution_xy=config.resolution_xy,
            resolution_z=config.resolution_z,
            substrate_material=config.substrate_material,
            default_gap_slices=config.default_gap_slices,
            gerber_dir=config.gerber_dir,
        )

    def _buildConfig(self) -> StackupConfig:
        layers = []
        for row in range(self.table.rowCount()):
            name = self.table.item(row, _COL_NAME).text()
            file = self.table.item(row, _COL_FILE).text()
            thickness = self.table.cellWidget(row, _COL_THICKNESS).value()
            gap = self.table.cellWidget(row, _COL_GAP).value()
            feature = self.table.cellWidget(row, _COL_FEATURE).currentText()
            background = self.table.cellWidget(row, _COL_BACKGROUND).currentText()
            color = self.table.cellWidget(row, _COL_COLOR).property("rgba")
            layers.append(
                StackupLayer(
                    file=file,
                    name=name,
                    thickness=thickness,
                    gap=gap,
                    feature_material=feature,
                    background_material=background,
                    color=tuple(color),
                )
            )
        extras = self._configExtras
        return StackupConfig(
            layers=layers,
            gerber_dir=extras.get("gerber_dir"),
            dpi=extras.get("dpi", 300),
            resolution_xy=extras.get("resolution_xy", 0.1),
            resolution_z=extras.get("resolution_z", 0.01),
            substrate_material=extras.get("substrate_material", "FR4"),
            default_gap_slices=extras.get("default_gap_slices", 20),
        )

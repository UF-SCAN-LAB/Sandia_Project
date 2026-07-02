from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Union

import numpy as np
import pandas as pd
from pyqtgraph.Qt import QtCore, QtWidgets
from qtextras import OptionsDict

from xrayrecon.cubecorrections import performAllCorrections
from xrayrecon.gui.table import PandasModel
from xrayrecon.layerseg import layerInfoForWindow


class LayerModel(QtCore.QObject):
    sigLayerUpdated = QtCore.Signal(int, object)  # row number, resulting 2d image
    sigCubeChanged = QtCore.Signal()
    sigSpacingChanged = QtCore.Signal()
    NO_METRIC = None

    def __init__(
        self,
        dataCube: np.ndarray = None,
        layerInfo: Dict[str, list] = None,
        applyCubeCorrections=False,
        parent=None,
    ):
        super().__init__(parent)
        self.defaultValues = {"View": False, "#": -1, "Layer Name": "", "Slices": -1}
        self.tblModel = PandasModel(self.defaultValues, indexCol="#")
        if applyCubeCorrections:
            dataCube = performAllCorrections(dataCube)
        self.dataCube = self.forceDepthToLastAx(dataCube)
        self.rgbaCube: np.ndarray | None = None  # (H, W, Z, 4) uint8, or None for grayscale
        # mm/voxel (x, y, z); (1.0, 1.0, 1.0) means "unitless voxel space" --
        # real CT loaders have no way to know true physical spacing, so this
        # must be set explicitly (e.g. via SpacingDialog) before it's meaningful.
        self.spacing_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0)
        # True only once setSpacing() has been called for the CURRENTLY loaded
        # cube -- setCube/setRGBACube reset this whenever a new cube comes in,
        # so callers (e.g. registration) can tell "genuinely confirmed 1mm
        # spacing" apart from "never confirmed, still the placeholder default".
        self._spacing_confirmed: bool = False
        self.metricInfo: QtWidgets.QWidget | None = self.NO_METRIC
        # cube depth (Z slice count) -> (layerNames, layerSlices) for every
        # depth explicit layer info was set via setupTblRows(); see
        # _resetLayerInfoIfDepthChanged.
        self._explicitLayerInfoByDepth: dict[int, tuple[list, list]] = {}

        if layerInfo is None:
            layerInfo = self.getAutoLayerInfo(self.dataCube, returnMetric=True)
        if isinstance(layerInfo, dict):
            self.tblModel.addDfRows(layerInfo)
            self.sigCubeChanged.emit()
        elif isinstance(layerInfo, Path) or isinstance(layerInfo, str):
            layerInfo = Path(layerInfo)
            if "grouped" in layerInfo.name:
                self.parseGroupSliceCsv(layerInfo)
            else:
                self.parseIndSliceCsv(layerInfo)

        self.layerDf = self.tblModel.dataDf
        self.axMapping = {"x": 0, "y": 1, "z": 2}

    def changeAx(self, whichAx: OptionsDict):
        whichAx = whichAx.name.lower()
        axs = whichAx.split("-")[:2]
        if self.dataCube is None or any([ax not in self.axMapping for ax in axs]):
            return
        axNums = [self.axMapping[ax] for ax in axs]
        missingAx = str(np.setdiff1d(list(self.axMapping.keys()), axs)[0])
        axNums.append(self.axMapping[missingAx])
        if axNums == [0, 1, 2]:
            return
        for k, v in zip(axs + [missingAx], range(3)):
            self.axMapping[k] = v
        new_cube = self.dataCube.transpose(axNums)
        # Permute spacing_xyz the same way as the data: new axis i is old
        # axis axNums[i], so spacing must follow the same relabeling or it
        # silently describes the wrong axis for anisotropic voxel spacing.
        self.spacing_xyz = tuple(self.spacing_xyz[a] for a in axNums)
        if self.rgbaCube is not None:
            # Keep rgbaCube in its original (H, W, Z, 4) orientation so that
            # layerDf Z-slices always reference the original depth axis correctly.
            self.setRGBACube(new_cube, self.rgbaCube, preserve_spacing=True)
        else:
            self.setCube(new_cube, preserve_spacing=True)

    def parseIndSliceCsv(self, csvFname: Union[Path, str]):
        layerDf = pd.read_csv(csvFname)
        layerNames = pd.unique(layerDf["Layer Name"])
        layerSlices = []
        for name in layerNames:
            layerSlices.append(
                layerDf.loc[layerDf["Layer Name"] == name, "Slices"].values
            )
        self.setupTblRows(layerNames, layerSlices)

    def setupTblRows(self, layerNames, layerSlices):
        self.tblModel.removeLayers()
        self.tblModel.addDfRows({"Layer Name": layerNames, "Slices": layerSlices})
        # addDfRows reassigns tblModel.dataDf to a new DataFrame (via
        # pd.concat) rather than mutating in place, so this cached reference
        # must be refreshed too -- mirrors _resetLayerInfoIfDepthChanged,
        # which already does this correctly.
        self.layerDf = self.tblModel.dataDf
        # Remember this explicit (non-auto-detected) layer info by cube
        # depth, so a later axis toggle (changeAx -> setCube) that changes
        # depth away and back doesn't silently discard it in favor of the
        # generic auto-detector -- see _resetLayerInfoIfDepthChanged.
        if self.dataCube is not None:
            self._explicitLayerInfoByDepth[self.dataCube.shape[2]] = (
                list(layerNames), list(layerSlices)
            )
        self.sigCubeChanged.emit()

    def parseGroupSliceCsv(self, csvFname: Union[Path, str]):
        layerDf = pd.read_csv(csvFname)
        layerNames = pd.unique(layerDf["Layer Name"])
        layerSlices = []
        for name in layerNames:
            groups = layerDf.loc[layerDf["Layer Name"] == name, "Slices"].values
            layerSlices.append(np.arange(groups[0], groups[1] + 1))
        self.setupTblRows(layerNames, layerSlices)

    def getAutoLayerInfo(self, cube, method="templateScores", returnMetric=False):
        if method == "templateScores":
            # layerInfo = getAutoLayerInfo_templateScores(cube, returnMetric)
            layerInfo = layerInfoForWindow(cube, cube.shape[:2], 50, returnMetric=True)
            if returnMetric:
                layerInfo, self.metricInfo = layerInfo
            else:
                self.metricInfo = self.NO_METRIC
        else:
            raise ValueError(
                f"Method {method} for auto layer info retrieval is not implemented."
            )
        return layerInfo

    def setCube(self, dataCube: np.ndarray, preserve_spacing: bool = False):
        self.rgbaCube = None
        self._resetLayerInfoIfDepthChanged(dataCube)
        self.dataCube = dataCube
        if not preserve_spacing:
            self._spacing_confirmed = False
        self.sigCubeChanged.emit()

    def _resetLayerInfoIfDepthChanged(self, newCube: np.ndarray):
        """Recompute layerDf/tblModel when the new cube's Z-depth differs from
        the current one. Without this, stale per-layer slice ranges from a
        differently-sized cube index out of bounds in applyProcess/LayerView
        (e.g. loading a 34-slice volume after a 256-slice one).

        If explicit (non-auto-detected) layer info was previously set via
        setupTblRows() for this exact depth -- e.g. toggling the axis
        selector away and back to an orientation whose layer names/slices
        are already known -- restore that instead of re-running the generic
        auto-detector, which has no memory of prior state and would
        otherwise silently discard known-good layer info."""
        oldDepth = self.dataCube.shape[2] if self.dataCube is not None else None
        newDepth = newCube.shape[2] if newCube is not None else None
        if newDepth is None or newDepth == oldDepth:
            return
        self.tblModel.removeLayers()
        if newDepth in self._explicitLayerInfoByDepth:
            names, slices = self._explicitLayerInfoByDepth[newDepth]
            self.tblModel.addDfRows({"Layer Name": names, "Slices": slices})
        else:
            layerInfo = self.getAutoLayerInfo(newCube, returnMetric=True)
            self.tblModel.addDfRows(layerInfo)
        self.layerDf = self.tblModel.dataDf

    def setSpacing(self, dx: float, dy: float, dz: float):
        self.spacing_xyz = (dx, dy, dz)
        self._spacing_confirmed = True
        self.sigSpacingChanged.emit()

    def setRGBACube(
        self, dataCube: np.ndarray, rgbaCube: np.ndarray, preserve_spacing: bool = False
    ):
        """Set grayscale + RGBA cubes atomically so resetItems sees both."""
        self._resetLayerInfoIfDepthChanged(dataCube)
        self.dataCube = dataCube
        self.rgbaCube = rgbaCube
        if not preserve_spacing:
            self._spacing_confirmed = False
        self.sigCubeChanged.emit()

    def applyProcess(
        self,
        layerIdx: int,
        process: Callable[[np.ndarray], np.ndarray] = None,
        emit=True,
    ):
        slices = self.layerDf.loc[layerIdx, "Slices"]
        if self.rgbaCube is not None:
            h, w = self.rgbaCube.shape[:2]
            if len(slices) == 0:
                img = np.zeros((h, w, 4), dtype=np.uint8)
            else:
                # Middle-slice projection: O(1) memory, shows material cross-section
                # at the representative depth of each layer.
                mid = slices[len(slices) // 2]
                img = self.rgbaCube[:, :, mid, :].copy()
        else:
            if process is None:
                process = lambda _img: np.mean(_img, 2)
            img = process(self.dataCube[:, :, slices])
        if emit:
            self.sigLayerUpdated.emit(layerIdx, img)
        return img

    @staticmethod
    def makeDummyLayers(
        numCubeSlices=256, numLayers=10, layerNames: Union[str, List[str]] = None
    ):
        if layerNames is None:
            layerNames = [f"Layer {ii}" for ii in range(numLayers)]
        if isinstance(layerNames, str):
            layerNames = [layerNames]
        if len(layerNames) == 1:
            layerNames = layerNames * numLayers
        ranges = []
        stops = np.linspace(0, numCubeSlices, numLayers + 1)
        for start, stop in zip(stops, stops[1:]):
            ranges.append(np.arange(start, stop, dtype=int))
        dummyLayerInfo = {"Layer Name": layerNames, "Slices": ranges}
        return dummyLayerInfo

    @staticmethod
    def forceDepthToLastAx(cube: np.ndarray):
        minAx = np.argmin(cube.shape)
        if minAx != 2:
            neworder = np.setdiff1d([0, 1, 2], minAx).tolist() + [minAx]
            cube = cube.transpose(neworder)
        return cube

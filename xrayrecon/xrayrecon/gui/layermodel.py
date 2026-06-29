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
        self.metricInfo: QtWidgets.QWidget | None = self.NO_METRIC

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
        self.setCube(self.dataCube.transpose(axNums))

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

    def setCube(self, dataCube: np.ndarray):
        self.dataCube = dataCube
        self.sigCubeChanged.emit()

    def applyProcess(
        self,
        layerIdx: int,
        process: Callable[[np.ndarray], np.ndarray] = None,
        emit=True,
    ):
        if process is None:
            process = lambda _img: np.mean(_img, 2)
        img = process(self.dataCube[:, :, self.layerDf.loc[layerIdx, "Slices"]])
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

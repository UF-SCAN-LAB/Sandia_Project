from typing import Callable, List, Optional

import numpy as np
import pyqtgraph as pg
from qtextras import ButtonCollection, OptionsDict

from xrayrecon.gui.layermodel import LayerModel


class LayerView:
    def __init__(
        self,
        model: LayerModel,
        process: Callable[[np.ndarray], np.ndarray] = None,
        parent: pg.PlotItem = None,
    ):
        self.items: List[pg.ImageItem] = []
        self.forceShowIdxs = []
        self.parentViewbox: Optional[pg.ViewBox] = None
        self.itemParent = parent
        self.sliceProcess = process

        self.model = model
        self.resetItems()
        model.sigLayerUpdated.connect(
            lambda rowIdx, img: self.items[rowIdx].setImage(img)
        )
        model.sigCubeChanged.connect(lambda: self.resetItems())

        model.tblModel.sigDataChanged.connect(lambda *args: self.showRelevantImages())
        self.showRelevantImages()

        btnGrp = [OptionsDict(name, pType="action") for name in ["X-Y", "Y-Z", "X-Z"]]

        self.xyzCtrl = ButtonCollection(
            parent,
            "Axis Selection",
            btnGrp,
            self.model.changeAx,
            exclusive=True,
            asToolButton=False,
            checkable=True,
        )
        next(iter(self.xyzCtrl.optionsButtonMap.values())).setChecked(True)

    @property
    def itemParent(self):
        return self._itemParent

    @itemParent.setter
    def itemParent(self, newParent: pg.PlotItem = None):
        self._itemParent = newParent
        if newParent is not None:
            self.parentViewbox = newParent.getViewBox()
            self.parentViewbox.invertY()

    def resetItems(self):
        if self.itemParent is not None:
            for item in self.items:
                self.itemParent.removeItem(item)
        self.items.clear()

        for idx, row in self.model.layerDf.iterrows():
            item = pg.ImageItem(
                self.model.applyProcess(idx, self.sliceProcess, emit=False)
            )
            item.setZValue(idx)
            self.items.append(item)

        if self.itemParent is not None:
            for item in self.items:
                self.itemParent.addItem(item)

    def showRelevantImages(self):
        showIdxs = np.flatnonzero(self.model.layerDf["View"].values)
        numToShow = len(showIdxs)
        imgOpacity = 1 / numToShow if numToShow > 0 else 0
        for ii, item in enumerate(self.items):
            if ii in showIdxs or ii in self.forceShowIdxs:
                item.setOpacity(imgOpacity)
                item.show()
            else:
                item.hide()
        if numToShow > 0 and self.parentViewbox is not None:
            self.parentViewbox.autoRange()

    def setLevels(self, minVal: float, maxVal: float):
        """Apply brightness/contrast levels to all visible image items."""
        for item in self.items:
            if item.isVisible():
                item.setLevels((minVal, maxVal))

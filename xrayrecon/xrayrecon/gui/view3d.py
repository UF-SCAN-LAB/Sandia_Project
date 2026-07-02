from pathlib import Path

import cv2 as cv
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtGui, QtWidgets
from qtextras import EasyWidget


class PCBViewer3D(gl.GLViewWidget):
    downsampFactor = 1

    def __init__(self, data=None):
        super().__init__()
        self.opts["distance"] = 200
        self.setWindowTitle("pyqtgraph example: GLVolumeItem")
        self.setBackgroundColor((30, 30, 34))
        self.gridItem = gl.GLGridItem()
        self.gridItem.setColor((150, 150, 150, 60))
        self.addItem(self.gridItem)
        self.curVolume = None
        self.axisItem = gl.GLAxisItem()
        self.axisItem.setGLOptions("translucent")
        self.addItem(self.axisItem)
        self.volume = gl.GLVolumeItem(np.zeros((1, 1), dtype="uint8"))
        self.addItem(self.volume)
        self._dataShape = (1, 1, 1)

        if data is not None:
            self.setDataCube(data)

    def setDataCube(
        self, data, highlightIdxs=None, downsampleFactor=None,
        transparencyFactor=2.0, threshold=0.1
    ):
        if downsampleFactor is None:
            downsampleFactor = self.downsampFactor
        self.downsampFactor = downsampleFactor
        if downsampleFactor > 1:
            data, highlightIdxs = self.downsampleData(
                data, highlightIdxs, downsampleFactor
            )

        # Normalize to [0, 1] regardless of dtype/range (handles uint16, float, etc.)
        data_f = data.astype(np.float32)
        lo, hi = data_f.min(), data_f.max()
        if hi > lo:
            data_norm = (data_f - lo) / (hi - lo)
        else:
            data_norm = np.zeros_like(data_f)

        self._setGridForShape(data.shape)
        d2 = np.empty(data.shape + (4,), dtype=np.ubyte)
        d2[..., 0:3] = (data_norm * 255).astype(np.ubyte)[..., None]

        # Threshold-based transfer function: voxels below threshold are transparent
        alpha_f = np.clip((data_norm - threshold) / max(1e-6, 1.0 - threshold), 0.0, 1.0)
        alpha_f = alpha_f ** transparencyFactor
        d2[..., 3] = (alpha_f * 255).astype(np.ubyte)

        if highlightIdxs is not None:
            d2[:, :, highlightIdxs, 0:2] = 0
            highlight_alpha = (alpha_f[:, :, highlightIdxs] * 1.2 * 255).clip(0, 255).astype(np.ubyte)
            d2[:, :, highlightIdxs, 3] = highlight_alpha

        self.volume.setData(d2)

    def setRGBADataCube(self, data, downsampleFactor=None):
        if data.ndim != 4 or data.shape[-1] != 4:
            raise ValueError("RGBA data must have shape (H, W, Z, 4).")

        if downsampleFactor is None:
            downsampleFactor = self.downsampFactor
        self.downsampFactor = downsampleFactor

        if downsampleFactor > 1:
            data = self.downsampleRgbaData(data, downsampleFactor)

        self._setGridForShape(data.shape)
        rgba = data.astype(np.ubyte, copy=False)

        self.volume.setData(rgba)

    def _setGridForShape(self, shape):
        self._dataShape = tuple(shape[:3])
        self.gridItem.resetTransform()
        self.gridItem.scale(*np.array(shape[:2][::-1]), shape[2])

    def setCameraTop(self):
        """Look straight down the Z axis (plan view)."""
        cx, cy, cz = [s / 2 for s in self._dataShape]
        dist = max(self._dataShape[:2]) * 1.5
        self.setCameraPosition(pos=pg.Vector(cx, cy, cz), distance=dist, elevation=90, azimuth=0)

    def setCameraFront(self):
        """Look along the Y axis (front elevation)."""
        cx, cy, cz = [s / 2 for s in self._dataShape]
        dist = max(self._dataShape[0], self._dataShape[2]) * 1.5
        self.setCameraPosition(pos=pg.Vector(cx, cy, cz), distance=dist, elevation=0, azimuth=0)

    def setCameraSide(self):
        """Look along the X axis (side elevation)."""
        cx, cy, cz = [s / 2 for s in self._dataShape]
        dist = max(self._dataShape[1], self._dataShape[2]) * 1.5
        self.setCameraPosition(pos=pg.Vector(cx, cy, cz), distance=dist, elevation=0, azimuth=90)

    def resetCamera(self):
        """Return to default isometric view."""
        cx, cy, cz = [s / 2 for s in self._dataShape]
        dist = max(self._dataShape) * 2.0
        self.setCameraPosition(pos=pg.Vector(cx, cy, cz), distance=dist, elevation=30, azimuth=45)

    def setGridVisible(self, visible: bool):
        """Show or hide the reference grid."""
        if visible and self.gridItem not in self.items:
            self.addItem(self.gridItem)
        elif not visible and self.gridItem in self.items:
            self.removeItem(self.gridItem)

    def setAxisVisible(self, visible: bool):
        """Show or hide the axis indicator."""
        if visible and self.axisItem not in self.items:
            self.addItem(self.axisItem)
        elif not visible and self.axisItem in self.items:
            self.removeItem(self.axisItem)

    def downsampleData(self, data, highlightIdxs, factor=None):
        if factor is None:
            factor = self.downsampFactor
        self.downsampFactor = factor
        newSize = np.array(data.shape)
        newSize //= [factor] * 3
        outCube = np.zeros(newSize, dtype="uint8")
        for ii, resizeIdx in enumerate(
            np.linspace(0, data.shape[-1] - 1, newSize[-1], dtype=int)
        ):
            outCube[..., ii] = cv.resize(data[..., resizeIdx], outCube.shape[:2][::-1])
        # np.save('/home/UFAD/njessurun/Documents/resized.npy', outCube)
        if highlightIdxs is not None:
            highlightIdxs = np.unique((highlightIdxs // factor))
            highlightIdxs = highlightIdxs[highlightIdxs < outCube.shape[2]]
        return outCube, highlightIdxs

    def downsampleRgbaData(self, data, factor=None):
        if factor is None:
            factor = self.downsampFactor
        self.downsampFactor = factor
        newSize = np.array(data.shape[:3])
        newSize //= [factor] * 3
        newSize = np.maximum(newSize, 1)

        outCube = np.zeros(tuple(newSize) + (4,), dtype="uint8")
        sample_idxs = np.linspace(0, data.shape[2] - 1, newSize[2], dtype=int)
        for ii, resizeIdx in enumerate(sample_idxs):
            outCube[..., ii, :] = cv.resize(
                data[:, :, resizeIdx, :],
                outCube.shape[:2][::-1],
                interpolation=cv.INTER_NEAREST,
            )
        return outCube


if __name__ == "__main__":
    inpath = Path("C:/Users/njessurun/downloads/data_iotech.npy")
    from xrayrecon.gui.layermodel import LayerModel

    data = np.load(inpath)
    data = LayerModel.forceDepthToLastAx(data)
    orig = data
    lbl = QtWidgets.QLabel("hello")
    app = QtGui.QApplication([])
    viewer = PCBViewer3D()
    viewer.setDataCube(data, downsampleFactor=5)
    w = EasyWidget.buildMainWindow([lbl, viewer], useSplitter=True)
    w.show()

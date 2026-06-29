from typing import Any, List

import numpy as np
import pyqtgraph as pg
from imutils import rotate_bound
from pyqtgraph.console import ConsoleWidget
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from qtextras import EasyWidget
from skimage import transform as trans
from skimage.filters import laplace
from skimage.morphology import convex_hull_image

from xrayrecon.cubecorrections.emptyslices import trimEmptySlices

pg.setConfigOption("imageAxisOrder", "row-major")


def _edgesImgToAngle(edgesImg: np.ndarray, testAngles_deg=None):
    """
    Get the angle that the foreground component should be rotated, assuming it has strong
    straight-line edges and is against a white background.
    :param testAngles_deg: Determines the resolution and angle space this algorithm searches
      through when looking for the rotation angles. If none is provided, a default space
      of np.linspace(-10,10,500) is used.
    :param applyRotation: If *True*, the extracted rotation angle will be applied to the
      model's image
    """
    # Use edges of largest comp as hough guides
    if testAngles_deg is None:
        testAngles_deg = np.linspace(87, 92, 500)
    testAngles = np.deg2rad(testAngles_deg)
    h, theta, d = trans.hough_line(edgesImg, theta=testAngles)
    maxR, maxC = np.unravel_index(np.argmax(h), h.shape)
    rotAngle: float = np.rad2deg(theta[maxC])
    return rotAngle - 90


def getOptimalRotation(cube: np.ndarray, ax: int):
    # Place axis under test at back of cube for proper reshape
    axOrder = np.roll([0, 1, 2], 2 - ax)
    testSlices = cube.transpose(axOrder)
    reshaped = testSlices.reshape(-1, testSlices.shape[2])

    # Use the slice with the largest number of 'on' pixels
    idxToUse = np.argmax(reshaped.sum(0))
    img = testSlices[:, :, idxToUse]

    # Get just the outline of data at this slice
    convHull = convex_hull_image(img)
    edges = laplace(convHull) > 0
    return _edgesImgToAngle(edges)


def rotateCube(cube: np.ndarray, angle, ax):
    outcube = []
    slices: Any = [slice(None)] * 3
    for ii in range(cube.shape[ax]):
        slices[ax] = ii
        toReshape = cube[tuple(slices)]
        outcube.append(rotate_bound(toReshape, angle))
    outcube = np.dstack(outcube)
    oldNewMapping = {0: [2, 0, 1], 1: [0, 2, 1], 2: [0, 1, 2]}
    resetOrdering = oldNewMapping[ax]
    # resetOrdering = [0,1,2]
    outCube = outcube.transpose(resetOrdering)
    return outCube


# Just for type helping
class RotateWindow(QtWidgets.QMainWindow):
    sigChangesAccepted = QtCore.Signal()

    def __init__(self, cube: np.ndarray, parent=None, angleSteps: np.ndarray = None):
        super().__init__(parent)
        self.cube = cube

        if angleSteps is None:
            angleSteps = np.linspace(-180, 180, 9000).round(2)

        def _spinbox_lbl(name: str, steps: np.ndarray):
            lbl = QtWidgets.QLabel(name)
            spinbox = pg.SpinBox(self)

            def updateSteps(steps):
                diff = steps[1] - steps[0]
                spinbox.setOpts(
                    bounds=(steps[0], steps[-1] + diff),
                    step=diff,
                    int=np.issubdtype(steps.dtype, np.integer),
                    wrapping=True,
                )

            updateSteps(steps)
            spinbox.updateSteps = updateSteps
            return lbl, spinbox

        angleLbl, angleSbox = _spinbox_lbl("Angle", angleSteps)
        angleSbox.setValue(angleSteps[len(angleSteps) // 2])
        saveAngleBtn = QtWidgets.QPushButton("Save Angle", self)
        self.saveAngleBtn = saveAngleBtn

        axSteps = np.arange(3)
        axLbl, axSbox = _spinbox_lbl("Axis", axSteps)

        sliceLbl, sliceSbox = _spinbox_lbl("Slice", np.arange(3, dtype=int))

        plt = pg.PlotWidget()
        plt.getViewBox().setAspectLocked(True)
        plt.getViewBox().invertY(True)
        imgItem = pg.ImageItem(border="r")
        plt.addItem(imgItem)
        spinInnerChildren = []
        angleChild = EasyWidget([angleLbl, angleSbox, saveAngleBtn], layout="H")
        for children in ([axLbl, axSbox], [sliceLbl, sliceSbox]):
            spinInnerChildren.append(EasyWidget(children, layout="H"))
        spinInnerChildren.insert(0, angleChild)
        spinChildren = EasyWidget(spinInnerChildren, layout="H")

        self.line1 = pg.InfiniteLine(angle=0, movable=True)
        self.line2 = pg.InfiniteLine(movable=True)
        chkChildren = []
        for ii, line in enumerate([self.line1, self.line2]):
            chk = QtWidgets.QCheckBox(f"Show line {ii}")
            chk.setChecked(True)

            def showChanged(*args, chk=chk, line=line):
                line.setVisible(chk.isChecked())

            chk.stateChanged.connect(showChanged)
            chk.setChecked(False)
            plt.addItem(line)
            chkChildren.append(chk)

        self.angleSbox = angleSbox
        self.axSbox = axSbox
        self.sliceSbox = sliceSbox
        self.imgItem = imgItem
        self.console = ConsoleWidget(self, namespace={"self": self})
        self.console.setWindowFlags(QtCore.Qt.Window)
        self.showConsoleBtn = QtWidgets.QPushButton("Show Console")
        self.showConsoleBtn.clicked.connect(lambda: self.console.show())

        self.acceptBtn = QtWidgets.QPushButton("Accept")
        self.acceptBtn.clicked.connect(self.sigChangesAccepted.emit)

        self.cursorColorLbl = QtWidgets.QLabel(self)
        oldMm = plt.mouseMoveEvent

        def newMM(ev: QtGui.QMouseEvent):
            oldMm(ev)
            posRelToImage = imgItem.mapFromScene(ev.pos())
            pxY = int(posRelToImage.y())
            pxX = int(posRelToImage.x())
            pxColor = None
            if (
                imgItem.image is not None
                and 0 < pxX < imgItem.image.shape[1]
                and 0 < pxY < imgItem.image.shape[0]
            ):
                pxColor = imgItem.image[pxY, pxX]
                if pxColor.ndim == 0:
                    pxColor = np.array([pxColor])
            if pxColor is not None:
                self.cursorColorLbl.setText(f"Cursor Color: {pxColor}")

        plt.mouseMoveEvent = newMM

        EasyWidget.buildMainWindow(
            [
                plt,
                spinChildren,
                [chkChildren, [self.showConsoleBtn, self.cursorColorLbl]],
                self.acceptBtn,
            ],
            window=self,
        )

        def onInfoChange():
            slices: List[Any] = [slice(None)] * 3
            slices[self.axSbox.value()] = self.sliceSbox.value()
            newImg = self.cube[tuple(slices)]
            newImg = rotate_bound(newImg, self.angleSbox.value())
            self.imgItem.setImage(newImg)

        def axChanged(val):
            self.sliceSbox.updateSteps(np.arange(self.cube.shape[int(val)]))
            self.sliceSbox.setValue(0)

        self.axSbox.sigValueChanging.connect(lambda _self, val: axChanged(val))

        def onSaveAngle():
            rotated = rotateCube(self.cube, self.angleSbox.value(), self.axSbox.value())
            self.cube = trimEmptySlices(rotated)
            self.angleSbox.setValue(0)

        self.saveAngleBtn.clicked.connect(onSaveAngle)

        for sbox in self.angleSbox, self.axSbox, self.sliceSbox:
            sbox.sigValueChanging.connect(lambda *args: onInfoChange())
        self.axSbox.setValue(2)
        self.angleSbox.setValue(0)
        onInfoChange()

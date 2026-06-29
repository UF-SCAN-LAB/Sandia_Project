from __future__ import division

from typing import Union

import numpy as np
from pyqtgraph.parametertree import Parameter
from pyqtgraph.parametertree.Parameter import PARAM_TYPES
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets


def phantomSpheres(n=128, numSpheres=5, minRadius=None, maxRadius=None):
    out = np.zeros((n, n, n))
    if minRadius is None:
        minRadius = n // 10
    if maxRadius is None:
        maxRadius = n // 6
    origins = np.random.randint(0, n, size=[numSpheres, 3])
    radii = np.random.randint(minRadius, maxRadius, numSpheres)

    xx, yy, zz = np.mgrid[0:n, 0:n, 0:n]

    for (o_x, o_y, o_z), r in zip(origins, radii):
        cur = np.sqrt((xx - o_x) ** 2 + (yy - o_y) ** 2 + (zz - o_z) ** 2)
        cur[cur > r] = 0
        out += cur
    return ((out / out.max()) * 255).astype("uint8")


def phantom3d(phantom="modified-shepp-logan", n=64):
    """Three-dimensional Shepp-Logan phantom

    Can be used to test 3-D reconstruction algorithms.

    Parameters
    ==========
    phantom: str
        One of {'modified-shepp-logan', 'shepp_logan', 'yu_ye_wang'},
        The type of phantom to draw.
    n : int, optional
        The grid size of the phantom

    Notes
    =====
    For any given voxel in the output image, the voxel's value is equal to the
    sum of the additive intensity values of all ellipsoids that the voxel is a
    part of.  If a voxel is not part of any ellipsoid, its value is 0.

    The additive intensity value A for an ellipsoid can be positive or
    negative;  if it is negative, the ellipsoid will be darker than the
    surrounding pixels.
    Note that, depending on the values of A, some voxels may have values
    outside the range [0,1].

    Copyright
    =========
    BSD License
    Copyright 2006 Matthias Christian Schabel (matthias @ stanfordalumni . org)
    University of Utah Department of Radiology
    Utah Center for Advanced Imaging Research
    729 Arapeen Drive

    """
    if phantom == "modified-shepp-logan":
        ellipse = modified_shepp_logan()
    elif phantom == "shepp_logan":
        ellipse = shepp_logan()
    elif phantom == "yu_ye_wang":
        ellipse = yu_ye_wang()
    else:
        raise TypeError('phantom type "%s" not recognized' % phantom)

    p = np.zeros(n**3)
    rng = ((np.arange(0, n - 1)) - (n - 1) / 2) / ((n - 1) / 2)
    x, y, z = np.meshgrid(rng, rng, rng)
    coord = np.vstack((x.flatten(), y.flatten(), z.flatten()))
    pi = np.pi

    for k in np.arange(ellipse.shape[0]):
        A = ellipse[k, 0]  # Amplitude change for this ellipsoid
        asq = ellipse[k, 1] ** 2  # a^2
        bsq = ellipse[k, 2] ** 2  # b^2
        csq = ellipse[k, 3] ** 2  # c^2
        x0 = ellipse[k, 4]  # x offset
        y0 = ellipse[k, 5]  # y offset
        z0 = ellipse[k, 6]  # z offset
        phi = ellipse[k, 7] * pi / 180  # first Euler angle in radians
        theta = ellipse[k, 8] * pi / 180  # second Euler angle in radians
        psi = ellipse[k, 9] * pi / 180  # third Euler angle in radians

        cphi = np.cos(phi)
        sphi = np.sin(phi)
        ctheta = np.cos(theta)
        stheta = np.sin(theta)
        cpsi = np.cos(psi)
        spsi = np.sin(psi)

        # Euler rotation matrix
        alpha = np.array(
            [
                [
                    cpsi * cphi - ctheta * sphi * spsi,
                    cpsi * sphi + ctheta * cphi * spsi,
                    spsi * stheta,
                ],
                [
                    -spsi * cphi - ctheta * sphi * cpsi,
                    -spsi * sphi + ctheta * cphi * cpsi,
                    cpsi * stheta,
                ],
                [stheta * sphi, -stheta * cphi, ctheta],
            ]
        )

        # rotated ellipsoid coordinates
        coordp = np.dot(alpha, coord)
        idx = np.nonzero(
            (coordp[0, :] - x0) ** 2.0 / asq
            + (coordp[1, :] - y0) ** 2.0 / bsq
            + (coordp[2, :] - z0) ** 2.0 / csq
            <= 1
        )[0]
        p[idx] = p[idx] + A

    return p.reshape((n, n, n))


def shepp_logan():
    arr = modified_shepp_logan()
    arr[:, 0] = np.array([1, -0.98, -0.02, -0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    return arr


def modified_shepp_logan():
    """
    This head phantom is the same as the Shepp-Logan except
    the intensities are changed to yield higher contrast in
    the image.  Taken from Toft, 199-200.

          A      a     b     c     x0      y0      z0    phi  theta    psi
         -----------------------------------------------------------------
    """
    mat = """
          1  .6900  .920  .810      0       0       0      0      0      0
        -.8  .6624  .874  .780      0  -.0184       0      0      0      0
        -.2  .1100  .310  .220    .22       0       0    -18      0     10
        -.2  .1600  .410  .280   -.22       0       0     18      0     10
         .1  .2100  .250  .410      0     .35    -.15      0      0      0
         .1  .0460  .046  .050      0      .1     .25      0      0      0
         .1  .0460  .046  .050      0     -.1     .25      0      0      0
         .1  .0460  .023  .050   -.08   -.605       0      0      0      0
         .1  .0230  .023  .020      0   -.606       0      0      0      0
         .1  .0230  .046  .020    .06   -.605       0      0      0      0"""
    return np.asarray(np.matrix(mat))


def yu_ye_wang():
    """
    Yu H, Ye Y, Wang G, Katsevich-Type Algorithms for
    Variable Radius Spiral Cone-Beam CT

          A      a     b     c     x0      y0      z0    phi  theta    psi
         -----------------------------------------------------------------
    """
    mat = """
          1  .6900  .920  .900      0       0       0      0      0      0
        -.8  .6624  .874  .880      0       0       0      0      0      0
        -.2  .4100  .160  .210   -.22       0    -.25    108      0      0
        -.2  .3100  .110  .220    .22       0    -.25     72      0      0
         .2  .2100  .250  .500      0     .35    -.25      0      0      0
         .2  .0460  .046  .046      0      .1    -.25      0      0      0
         .1  .0460  .023  .020   -.08    -.65    -.25      0      0      0
         .1  .0460  .023  .020    .06    -.65    -.25     90      0      0
         .2  .0560  .040  .100    .06   -.105    .625     90      0      0
        -.2  .0560  .056  .100      0    .100    .625      0      0      0"""
    return np.asarray(np.matrix(mat))


class FRPgParamDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, paramDict: dict, parent=None):
        super().__init__(parent)
        errMsg = f"{self.__class__} can only create parameter editors from"
        " registered pg widgets that implement makeWidget()"

        if paramDict["type"] not in PARAM_TYPES:
            raise Exception(errMsg)
        paramDict.update(name="dummy")
        param = Parameter.create(**paramDict)
        if hasattr(param.itemClass, "makeWidget"):
            self.item = param.itemClass(param, 0)
        else:
            raise Exception(errMsg)

    def createEditor(self, parent, option, index: QtCore.QModelIndex):
        editor = self.item.makeWidget()
        editor.setParent(parent)
        editor.setMaximumSize(option.rect.width(), option.rect.height())
        return editor

    def setModelData(
        self,
        editor: QtWidgets.QWidget,
        model: QtCore.QAbstractTableModel,
        index: QtCore.QModelIndex,
    ):
        model.setData(index, editor.value())

    def setEditorData(self, editor: QtWidgets.QWidget, index):
        value = index.data(QtCore.Qt.EditRole)
        editor.setValue(value)

    def updateEditorGeometry(
        self,
        editor: QtWidgets.QWidget,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ):
        editor.setGeometry(option.rect)


def create_addMenuAct(
    mainWin: QtWidgets.QWidget, parentMenu: QtWidgets.QMenu, title: str, asMenu=False
) -> Union[QtWidgets.QMenu, QtGui.QAction]:
    menu = None
    if asMenu:
        menu = QtWidgets.QMenu(title, mainWin)
        act = menu.menuAction()
    else:
        act = QtGui.QAction(title)
    parentMenu.addAction(act)
    if asMenu:
        return menu
    else:
        return act

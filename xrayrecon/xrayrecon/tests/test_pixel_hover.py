"""Exercise the actual hover handler without constructing the full application."""

from types import SimpleNamespace

import numpy as np
import pyqtgraph as pg
import pytest
from pyqtgraph.Qt import QtCore, QtWidgets

from xrayrecon.gui.app import MainWin


def hover_view(image, *, visible=True, scene_contains=True):
    item = pg.ImageItem(image)
    item.setVisible(visible)
    label = QtWidgets.QLabel()
    plot = SimpleNamespace(
        sceneBoundingRect=lambda: SimpleNamespace(contains=lambda pos: scene_contains),
        getViewBox=lambda: SimpleNamespace(mapSceneToView=lambda pos: pos),
    )
    return SimpleNamespace(
        plt=plot, viewer=SimpleNamespace(items=[item]), pixelValueLabel=label
    )


@pytest.mark.parametrize(
    "pixel, expected",
    [
        (np.float32(1.234), "1.2"),
        (np.uint16(1024), "1024.0"),
        (np.array([10, 20, 30], dtype=np.uint8), "RGB (10.0, 20.0, 30.0)"),
        (np.array([10, 20, 30, 255], dtype=np.uint8), "RGBA (10.0, 20.0, 30.0, 255.0)"),
        (np.array([0.125, 0.5, 0.875]), "RGB (0.1, 0.5, 0.9)"),
    ],
)
def test_hover_formats_scalars_and_color_without_mutation(qapp, pixel, expected):
    image = np.array([[pixel]])
    original = image.copy()
    view = hover_view(image)

    MainWin._onMouseMovedInPlot(view, QtCore.QPointF(0, 0))

    assert view.pixelValueLabel.text() == f"Pixel: (0, 0) = {expected}"
    np.testing.assert_array_equal(image, original)
    np.testing.assert_array_equal(view.viewer.items[0].image, original)


@pytest.mark.parametrize(
    "pos, visible, scene_contains",
    [((2, 0), True, True), ((0, 2), True, True), ((-1, 0), True, True),
     ((0, 0), False, True), ((0, 0), True, False)],
)
def test_hover_clears_label_outside_visible_image(qapp, pos, visible, scene_contains):
    image = np.array([[[10, 20, 30, 255]]], dtype=np.uint8)
    view = hover_view(image, visible=visible, scene_contains=scene_contains)
    view.pixelValueLabel.setText("stale pixel")

    MainWin._onMouseMovedInPlot(view, QtCore.QPointF(*pos))

    assert view.pixelValueLabel.text() == "Pixel: —"


def test_hover_reads_top_visible_item_in_row_major_order(qapp):
    view = hover_view(np.ones((2, 3)))
    image = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    view.viewer.items.append(pg.ImageItem(image))
    hidden = pg.ImageItem(np.zeros((2, 3, 4), dtype=np.uint8))
    hidden.hide()
    view.viewer.items.append(hidden)

    MainWin._onMouseMovedInPlot(view, QtCore.QPointF(2, 1))

    assert view.pixelValueLabel.text() == "Pixel: (2, 1) = RGBA (20.0, 21.0, 22.0, 23.0)"

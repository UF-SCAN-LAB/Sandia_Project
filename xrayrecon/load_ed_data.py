"""Load and visualize the ED Data (part1 FBP) with multi-material coloring.

Run from the xrayrecon/ directory:
    python load_ed_data.py
"""
from xrayrecon.windows_cuda import prepare_optional_torch_dlls

prepare_optional_torch_dlls()

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

from xrayrecon.gui import MainWin
from xrayrecon.io_spectral import load_ed_data
from xrayrecon.material_seg import normalize_volume, voxel_to_material_id, label_map_to_rgba
from xrayrecon.multimaterial import MaterialEntry, MultiMaterialConfig

ED_DATA_DIR = r"E:\Data\ED Data\part1\part1"

config = MultiMaterialConfig(
    name="ED Part1 FBP",
    channels=["fbp"],
    materials=[
        MaterialEntry("AIR",    color=(0,   0,   200, 5),   threshold=(0.0,  0.05)),
        MaterialEntry("FR4",    color=(80,  80,  80,  80),  threshold=(0.05, 0.3)),
        MaterialEntry("COPPER", color=(255, 140, 0,   200), threshold=(0.3,  1.0)),
    ],
)


def main():
    print(f"Loading ED data from: {ED_DATA_DIR}")
    vol = load_ed_data(ED_DATA_DIR)          # (1, Z, H, W) float32
    print(f"Volume shape: {vol.shape}")

    print("Normalizing and segmenting...")
    norm = normalize_volume(vol)
    labels = voxel_to_material_id(norm, config)
    rgba = label_map_to_rgba(labels, config)

    counts = {m.name: int(np.sum(labels == i)) for i, m in enumerate(config.materials, 1)}
    print("Voxel counts:", counts)
    print(f"RGBA volume shape: {rgba.shape}")

    # Use first channel as the 2D grayscale cube for layer view / table
    vol_hwz = np.moveaxis(norm[0], 0, -1)   # (Z,H,W) -> (H,W,Z)

    app = pg.mkQApp()
    win = MainWin(None, None, vol_hwz)
    def _show_rgba():
        win._current_rgba_base = rgba
        win.model.rgbaCube = rgba
        win.viewer.resetItems()          # refresh 2D thumbnails with RGBA colors
        win.viewer_3d.setRGBADataCube(rgba)
        win.viewer_3d.resetCamera()     # position camera to see full volume

    QtCore.QTimer.singleShot(0, win.showMaximized)
    QtCore.QTimer.singleShot(200, _show_rgba)
    app.exec_()


if __name__ == "__main__":
    main()

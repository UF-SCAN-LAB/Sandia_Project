"""Standalone multi-material visualization demo using a synthetic phantom.

Run from the xrayrecon/ directory:
    python demo_multimaterial.py

No real data required. The phantom is generated procedurally. This is a thin
launcher around MainWin.loadPhantomDemo(), the same action reachable from the
"Load Phantom Demo" toolbar button in the running app.
"""
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

from xrayrecon.gui import MainWin
from xrayrecon.gui.helpers import phantomSpheres


def main():
    app = pg.mkQApp()
    # Seed MainWin with a valid initial grayscale cube (matches xrayrecon.main's
    # no-datafile default); loadPhantomDemo() replaces it with the RGBA volume.
    win = MainWin(None, None, phantomSpheres(n=128))
    QtCore.QTimer.singleShot(0, win.showMaximized)
    # Delay until after showMaximized processes all initialization events
    # (slider signals, resize, etc.) that would otherwise call setDataCube and overwrite
    QtCore.QTimer.singleShot(200, win.loadPhantomDemo)
    app.exec_()


if __name__ == "__main__":
    main()

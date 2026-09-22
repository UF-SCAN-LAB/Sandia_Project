from xrayrecon.windows_cuda import prepare_optional_torch_dlls

prepare_optional_torch_dlls()

import numpy as np
import pyqtgraph as pg
from fire import Fire
from pyqtgraph.Qt import QtCore

from xrayrecon.gui import MainWin
from xrayrecon.gui.helpers import phantomSpheres


from xrayrecon.io import load_data

def main(datafile: str = None):
    if datafile is None:
        print("No data file specified. Generating Phantom Data...")
        # Use dummy data
        np.random.seed(42)
        dataCube = phantomSpheres(n=256)
    else:
        print(f"Loading data from: {datafile}")
        try:
            dataCube = load_data(datafile)
        except Exception as e:
            print(f"Error loading data: {e}")
            return

    app = pg.mkQApp()
    layerInfo = None
    win = MainWin(None, layerInfo, dataCube)
    QtCore.QTimer.singleShot(0, win.showMaximized)
    app.exec_()


def main_script():
    Fire(main)


if __name__ == "__main__":
    main_script()

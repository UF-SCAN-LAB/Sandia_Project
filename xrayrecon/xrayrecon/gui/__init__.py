from ..windows_cuda import prepare_optional_torch_dlls

prepare_optional_torch_dlls()

import pyqtgraph as pg

from .app import MainWin

pg.setConfigOption("imageAxisOrder", "row-major")

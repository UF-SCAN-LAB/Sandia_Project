import os
import sys
from pathlib import Path

# This package's editable install metadata still points at a stale sibling
# clone (a namespace-package hazard), so `xrayrecon` only resolves reliably
# to *this* repo when its root is first on sys.path. Do this before any
# `import xrayrecon...` in the test suite.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Must be set before the first PyQt5/pyqtgraph import in the process.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest


@pytest.fixture(scope="session")
def qapp():
    import pyqtgraph as pg

    return pg.mkQApp()

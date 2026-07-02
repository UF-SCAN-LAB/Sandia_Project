import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from qtextras import EasyWidget


class SpacingDialog(QtWidgets.QDialog):
    """Small modal dialog for entering physical voxel spacing (mm/voxel).

    Real CT loaders in this app carry no spacing metadata at all, so this is
    the only way that information enters the pipeline -- the user must supply
    it (e.g. from scanner/reconstruction settings) if physical-unit features
    like registration are to be meaningful.
    """

    def __init__(self, spacing_xyz=(1.0, 1.0, 1.0), parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set Voxel Spacing (mm/voxel)")

        def _spinbox(value):
            sbox = pg.SpinBox(value=value, bounds=(1e-6, None), dec=True, step=0.01)
            return sbox

        self.xSbox = _spinbox(spacing_xyz[0])
        self.ySbox = _spinbox(spacing_xyz[1])
        self.zSbox = _spinbox(spacing_xyz[2])

        self.isoChk = QtWidgets.QCheckBox("Isotropic XY")
        self.isoChk.setChecked(spacing_xyz[0] == spacing_xyz[1])

        def onXChanged():
            if self.isoChk.isChecked():
                self.ySbox.setValue(self.xSbox.value())

        self.xSbox.sigValueChanging.connect(lambda *_: onXChanged())

        okBtn = QtWidgets.QPushButton("OK")
        cancelBtn = QtWidgets.QPushButton("Cancel")
        okBtn.clicked.connect(self.accept)
        cancelBtn.clicked.connect(self.reject)

        content = EasyWidget.buildWidget(
            [
                [QtWidgets.QLabel("X (mm/voxel)"), self.xSbox],
                [QtWidgets.QLabel("Y (mm/voxel)"), self.ySbox],
                [QtWidgets.QLabel("Z (mm/voxel)"), self.zSbox],
                self.isoChk,
                [okBtn, cancelBtn],
            ],
            layout="V",
        )
        outerLayout = QtWidgets.QVBoxLayout(self)
        outerLayout.addWidget(content)

    def spacing(self) -> tuple[float, float, float]:
        return (self.xSbox.value(), self.ySbox.value(), self.zSbox.value())

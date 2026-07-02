import tempfile
from pathlib import Path

import numpy as np
from pyqtgraph.Qt import QtWidgets

from xrayrecon.gui.app import MainWin
from xrayrecon.gui.helpers import phantomSpheres
from xrayrecon.pygerber_integration import PyGerberRasterizer
from xrayrecon.registration import RigidScaleTransform
from xrayrecon.stackup import StackupConfig, StackupLayer
from xrayrecon.tests.fixtures import make_synthetic_pair


def _silence_message_boxes(monkeypatch, question_answer=QtWidgets.QMessageBox.Yes):
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "question", staticmethod(lambda *a, **k: question_answer)
    )


def _registered_mainwin(monkeypatch):
    """MainWin with the synthetic demo loaded and a transform accepted --
    the state transferGerberLabels needs to succeed."""
    _silence_message_boxes(monkeypatch)
    win = MainWin(dataCube=phantomSpheres())
    win._loadSyntheticRegistrationDemo()
    win.openRegistrationTool()
    win._registrationWindow.acceptBtn.click()
    return win


def test_transfer_gerber_labels_succeeds_when_state_is_consistent(qapp, monkeypatch):
    win = _registered_mainwin(monkeypatch)
    assert win._registration_transform is not None

    win.transferGerberLabels()

    assert win._current_rgba_base_left is not None


def test_transfer_gerber_labels_blocks_when_ct_data_changed_since_accept(qapp, monkeypatch):
    win = _registered_mainwin(monkeypatch)
    before = win._current_rgba_base_left

    # Load a different-shaped real CT dataset without re-registering.
    different_pair = make_synthetic_pair()
    bigger_ct = np.pad(different_pair["misaligned_ct_hwz"], ((0, 4), (0, 4), (0, 0)))
    win.model.setCube(bigger_ct)
    win.model.setSpacing(*different_pair["fixed_spacing"])

    win.transferGerberLabels()

    # Blocked: no new rgba was computed from the mismatched state.
    assert win._current_rgba_base_left is before


def test_transfer_gerber_labels_blocks_when_spacing_changed_since_accept(qapp, monkeypatch):
    win = _registered_mainwin(monkeypatch)
    before = win._current_rgba_base_left

    # Same shape, but re-confirm a different physical spacing.
    win.model.setSpacing(0.999, 0.999, 0.999)

    win.transferGerberLabels()

    assert win._current_rgba_base_left is before


def test_open_registration_tool_blocked_when_user_declines_unconfirmed_spacing(qapp, monkeypatch):
    _silence_message_boxes(monkeypatch, question_answer=QtWidgets.QMessageBox.No)
    win = MainWin(dataCube=phantomSpheres())
    win._loadSyntheticRegistrationDemo()
    # Reset to the unconfirmed state on purpose (the demo already confirms it).
    win.model._spacing_confirmed = False
    win._registrationWindow = None

    win.openRegistrationTool()

    assert win._registrationWindow is None


def test_open_registration_tool_proceeds_when_user_accepts_unconfirmed_spacing(qapp, monkeypatch):
    _silence_message_boxes(monkeypatch, question_answer=QtWidgets.QMessageBox.Yes)
    win = MainWin(dataCube=phantomSpheres())
    win._loadSyntheticRegistrationDemo()
    win.model._spacing_confirmed = False

    win.openRegistrationTool()

    assert win._registrationWindow is not None


def test_voxelize_stackup_invalidates_prior_registration(qapp, monkeypatch):
    _silence_message_boxes(monkeypatch)
    win = MainWin(dataCube=phantomSpheres())
    win._registration_transform = RigidScaleTransform(tx=1.0)
    win._registration_moving_spacing = (0.1, 0.1, 0.1)
    win._registration_fixed_spacing = (0.1, 0.1, 0.1)
    win._registration_fixed_shape = (10, 10, 10)

    def _fake_rasterize(self, content):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:4, 2:4] = 255
        return mask

    monkeypatch.setattr(PyGerberRasterizer, "rasterize", _fake_rasterize)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "layer.gbr").write_text("dummy")
        config = StackupConfig(
            layers=[StackupLayer(file="layer.gbr", name="L1", thickness=0.1)],
        )
        win._voxelize_stackup(config, tmp_path / "stackup.json")

    assert win._registration_transform is None
    assert win._registration_moving_spacing is None
    assert win._registration_fixed_spacing is None
    assert win._registration_fixed_shape is None

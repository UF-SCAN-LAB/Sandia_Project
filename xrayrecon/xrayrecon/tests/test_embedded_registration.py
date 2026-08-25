import numpy as np
import pytest

from xrayrecon.gui.app import MainWin
from xrayrecon.gui.registration import RegistrationPanel
from xrayrecon.gui.helpers import phantomSpheres


def _silence_messages(monkeypatch):
    from pyqtgraph.Qt import QtWidgets

    seen = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "information",
        staticmethod(lambda _parent, title, message, *args, **kwargs: seen.append((title, message))),
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "critical",
        staticmethod(lambda *args, **kwargs: None),
    )
    return seen


def _registration_ready_mainwin(monkeypatch):
    seen = _silence_messages(monkeypatch)
    win = MainWin(dataCube=phantomSpheres())
    win._loadSyntheticRegistrationDemo()
    win.openRegistrationTool()
    return win, seen


@pytest.mark.gui
@pytest.mark.integration
def test_registration_workspace_is_embedded_in_main_window(qapp, monkeypatch):
    win, _seen = _registration_ready_mainwin(monkeypatch)

    assert isinstance(win.registrationPanel, RegistrationPanel)
    assert not win.registrationPanel.isWindow()
    assert win.registrationPanel.window() is win
    assert win._workspaceStack.currentWidget() is win.registrationPanel
    assert not hasattr(win, "_registrationWindow")


@pytest.mark.gui
@pytest.mark.integration
def test_registration_edit_survives_return_to_main_workspace(qapp, monkeypatch):
    win, _seen = _registration_ready_mainwin(monkeypatch)
    win.registrationPanel.txSbox.setValue(0.35)
    win.registrationPanel._updatePreview()

    win.registrationPanel.backBtn.click()
    assert win._workspaceStack.currentWidget() is win._mainWorkspacePage
    win.openRegistrationTool()

    assert win._workspaceStack.currentWidget() is win.registrationPanel
    assert np.isclose(win.registrationPanel.currentTransform().tx, 0.35)
    assert np.isclose(win._registration_session.current_transform.tx, 0.35)


@pytest.mark.gui
@pytest.mark.integration
def test_suggestion_neither_accepts_nor_transfers_labels(qapp, monkeypatch):
    win, _seen = _registration_ready_mainwin(monkeypatch)
    before_rgba = win._current_rgba_base_left

    win.registrationPanel.suggestBtn.click()

    assert win._registration_session.proposal is not None
    assert win._registration_session.accepted_transform is None
    assert win._current_rgba_base_left is before_rgba


@pytest.mark.gui
@pytest.mark.integration
def test_same_shape_ct_replacement_invalidates_accepted_registration(qapp, monkeypatch):
    win, seen = _registration_ready_mainwin(monkeypatch)
    win.registrationPanel.acceptBtn.click()
    seen.clear()
    replacement = win.model.dataCube.copy()
    spacing = win.model.spacing_xyz

    win.model.setCube(replacement)
    win.model.setSpacing(*spacing)
    win.transferGerberLabels()

    assert any(title == "Registration Out Of Date" for title, _message in seen)


@pytest.mark.gui
@pytest.mark.integration
def test_same_shape_gerber_replacement_invalidates_accepted_registration(qapp, monkeypatch):
    win, seen = _registration_ready_mainwin(monkeypatch)
    win.registrationPanel.acceptBtn.click()
    seen.clear()

    win._gerber_intensity_hwz = win._gerber_intensity_hwz.copy()
    win.transferGerberLabels()

    assert any(title == "Registration Out Of Date" for title, _message in seen)

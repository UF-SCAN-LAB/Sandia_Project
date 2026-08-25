from xrayrecon.gui.app import MainWin
from xrayrecon.gui.helpers import phantomSpheres

NEW_PHASE2_ACTIONS = (
    "Set Real CT Voxel Spacing...",
    "Register Gerber to Real CT...",
    "Transfer Gerber Labels to Real CT",
    "Debug: Load Synthetic Registration Demo",
)


def test_mainwin_builds_with_phantom_data(qapp):
    win = MainWin(dataCube=phantomSpheres())

    assert win.viewer_3d_left is not None
    assert win.viewer_3d_right is not None
    assert win.model.dataCube is not None
    assert win.registrationPanel.parent() is win._workspaceStack
    assert win.registrationPanel.window() is win
    assert win._workspaceStack.currentWidget() is win._mainWorkspacePage


def test_mainwin_registers_phase2_actions(qapp):
    win = MainWin(dataCube=phantomSpheres())

    registered = set(win.toolsEditor.nameFunctionMap.keys())
    for action_name in NEW_PHASE2_ACTIONS:
        assert action_name in registered

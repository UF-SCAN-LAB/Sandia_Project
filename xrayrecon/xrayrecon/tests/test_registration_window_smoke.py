import numpy as np

from xrayrecon.gui.registration import RegistrationWindow
from xrayrecon.tests.fixtures import make_synthetic_pair


def test_registration_window_builds_and_recovers_known_transform(qapp):
    pair = make_synthetic_pair()

    win = RegistrationWindow(
        pair["aligned_ct_hwz"],
        pair["gerber_spacing"],
        pair["misaligned_ct_hwz"],
        pair["fixed_spacing"],
    )

    t = pair["known_transform"]
    win.txSbox.setValue(t.tx)
    win.tySbox.setValue(t.ty)
    win.tzSbox.setValue(t.tz)
    win.rotSbox.setValue(t.rot_z_deg)
    win.scaleSbox.setValue(t.scale)

    captured = {}
    win.sigTransformAccepted.connect(lambda transform: captured.setdefault("t", transform))
    win.acceptBtn.click()

    assert "t" in captured
    accepted = captured["t"]
    assert np.isclose(accepted.tx, t.tx)
    assert np.isclose(accepted.ty, t.ty)
    assert np.isclose(accepted.tz, t.tz)
    assert np.isclose(accepted.rot_z_deg, t.rot_z_deg)
    assert np.isclose(accepted.scale, t.scale)


def test_registration_window_reset_restores_identity(qapp):
    pair = make_synthetic_pair()
    win = RegistrationWindow(
        pair["aligned_ct_hwz"],
        pair["gerber_spacing"],
        pair["misaligned_ct_hwz"],
        pair["fixed_spacing"],
    )

    win.txSbox.setValue(1.0)
    win.rotSbox.setValue(10.0)
    win.resetBtn.click()

    t = win.currentTransform()
    assert t.tx == 0.0 and t.ty == 0.0 and t.tz == 0.0
    assert t.rot_z_deg == 0.0 and t.scale == 1.0


def test_registration_window_sliders_sweep_without_raising(qapp):
    pair = make_synthetic_pair()
    win = RegistrationWindow(
        pair["aligned_ct_hwz"],
        pair["gerber_spacing"],
        pair["misaligned_ct_hwz"],
        pair["fixed_spacing"],
    )

    for tx in (-1.0, 0.0, 1.0):
        win.txSbox.setValue(tx)
    for rot in (-180.0, 0.0, 180.0):
        win.rotSbox.setValue(rot)
    for scale in (0.2, 1.0, 5.0):
        win.scaleSbox.setValue(scale)

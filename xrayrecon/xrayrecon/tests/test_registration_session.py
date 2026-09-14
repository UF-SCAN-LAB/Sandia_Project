import numpy as np
import pytest

from xrayrecon.registration import (
    RegistrationProposal,
    RegistrationSession,
    RigidScaleTransform,
)


def _inputs():
    moving = np.zeros((12, 14, 5), dtype=np.float32)
    fixed = np.zeros((12, 14, 5), dtype=np.float32)
    spacing = (0.1, 0.1, 0.2)
    return moving, fixed, spacing


@pytest.mark.unit
def test_session_preserves_edits_for_identical_inputs():
    moving, fixed, spacing = _inputs()
    session = RegistrationSession()
    assert session.configure(moving, spacing, fixed, spacing)
    session.update_current(RigidScaleTransform(tx=0.4), score=0.7)

    assert not session.configure(moving, spacing, fixed, spacing)
    assert session.current_transform.tx == 0.4
    assert session.current_score == 0.7


@pytest.mark.unit
def test_session_resets_for_new_same_shape_volume():
    moving, fixed, spacing = _inputs()
    session = RegistrationSession()
    session.configure(moving, spacing, fixed, spacing)
    session.update_current(RigidScaleTransform(tx=0.4), score=0.7)
    session.accept()

    assert session.configure(moving, spacing, fixed.copy(), spacing)
    assert session.current_transform == RigidScaleTransform.identity()
    assert session.accepted_transform is None


@pytest.mark.unit
def test_session_resets_for_spacing_change():
    moving, fixed, spacing = _inputs()
    session = RegistrationSession()
    session.configure(moving, spacing, fixed, spacing)
    session.update_current(RigidScaleTransform(ty=-0.2), score=0.6)

    changed_spacing = (0.1, 0.1, 0.25)
    assert session.configure(moving, spacing, fixed, changed_spacing)
    assert session.current_score is None


@pytest.mark.unit
def test_acceptance_and_compatibility_are_explicit():
    moving, fixed, spacing = _inputs()
    session = RegistrationSession()
    session.configure(moving, spacing, fixed, spacing)
    session.update_current(RigidScaleTransform(rot_z_deg=3.0), score=0.8)

    assert not session.is_compatible(moving, spacing, fixed, spacing)
    accepted = session.accept()

    assert accepted.rot_z_deg == 3.0
    assert session.is_compatible(moving, spacing, fixed, spacing)
    assert not session.is_compatible(moving, spacing, fixed.copy(), spacing)


@pytest.mark.unit
def test_proposal_updates_current_but_does_not_accept():
    moving, fixed, spacing = _inputs()
    session = RegistrationSession()
    session.configure(moving, spacing, fixed, spacing)
    proposal = RegistrationProposal(
        transform=RigidScaleTransform(scale=1.05),
        initial_score=0.2,
        score=0.6,
        evaluations=20,
    )

    session.record_proposal(proposal)

    assert session.current_transform.scale == 1.05
    assert session.current_score == 0.6
    assert session.accepted_transform is None
    assert session.dirty


@pytest.mark.unit
def test_invalidate_clears_all_registration_state():
    moving, fixed, spacing = _inputs()
    session = RegistrationSession()
    session.configure(moving, spacing, fixed, spacing)
    session.update_current(RigidScaleTransform(tz=0.1), score=0.5)
    session.accept()

    session.invalidate("Gerber geometry changed")

    assert session.inputs is None
    assert session.current_transform == RigidScaleTransform.identity()
    assert session.accepted_transform is None
    assert session.invalid_reason == "Gerber geometry changed"


@pytest.mark.unit
@pytest.mark.parametrize(
    "spacing",
    [(0.1, 0.1), (0.1, 0.0, 0.1), (0.1, float("nan"), 0.1)],
)
def test_session_rejects_invalid_spacing(spacing):
    moving, fixed, _ = _inputs()
    session = RegistrationSession()

    with pytest.raises(ValueError, match="three positive finite"):
        session.configure(moving, spacing, fixed, (0.1, 0.1, 0.1))

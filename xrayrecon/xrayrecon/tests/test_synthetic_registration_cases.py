import numpy as np
import pytest

from xrayrecon.registration import (
    RigidScaleTransform,
    registration_quality,
    resample_volume,
)
from xrayrecon.synthetic_registration import (
    default_synthetic_registration_specs,
    make_synthetic_registration_case,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "spec", default_synthetic_registration_specs(), ids=lambda spec: spec.name
)
def test_synthetic_cases_are_deterministic(spec):
    first = make_synthetic_registration_case(spec)
    second = make_synthetic_registration_case(spec)

    np.testing.assert_array_equal(first.moving_hwz, second.moving_hwz)
    np.testing.assert_array_equal(first.fixed_hwz, second.fixed_hwz)
    np.testing.assert_array_equal(first.moving_labels_hwz, second.moving_labels_hwz)
    np.testing.assert_array_equal(first.fixed_labels_hwz, second.fixed_labels_hwz)


@pytest.mark.unit
@pytest.mark.parametrize(
    "spec",
    [spec for spec in default_synthetic_registration_specs() if spec.identifiable],
    ids=lambda spec: spec.name,
)
def test_known_transform_outscores_identity_for_identifiable_cases(spec):
    case = make_synthetic_registration_case(spec)
    identity = resample_volume(
        case.moving_hwz,
        case.moving_spacing,
        case.fixed_hwz.shape,
        case.fixed_spacing,
        RigidScaleTransform.identity(),
    )
    known = resample_volume(
        case.moving_hwz,
        case.moving_spacing,
        case.fixed_hwz.shape,
        case.fixed_spacing,
        case.known_transform,
    )

    identity_score = registration_quality(case.fixed_hwz, identity)
    known_score = registration_quality(case.fixed_hwz, known)

    if spec.name == "identity":
        assert known_score > 0.99
    else:
        assert known_score >= identity_score + spec.minimum_score_gain


@pytest.mark.unit
def test_low_information_case_does_not_claim_alignment_quality():
    spec = next(
        spec
        for spec in default_synthetic_registration_specs()
        if spec.name == "constant_low_information"
    )
    case = make_synthetic_registration_case(spec)

    assert registration_quality(case.fixed_hwz, case.moving_hwz) == 0.0


@pytest.mark.unit
def test_unknown_synthetic_geometry_fails_clearly():
    spec = default_synthetic_registration_specs()[0]
    invalid = type(spec)(name="invalid", geometry="not-a-geometry")

    with pytest.raises(ValueError, match="unknown synthetic geometry"):
        make_synthetic_registration_case(invalid)

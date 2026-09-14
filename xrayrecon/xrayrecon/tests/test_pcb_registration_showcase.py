import numpy as np
import pytest
from scipy.ndimage import label as connected_components

from xrayrecon.materials import MATERIAL_NAMES
from xrayrecon.registration import (
    RigidScaleTransform,
    propose_initial_transform,
    registration_quality,
    resample_volume,
)
from xrayrecon.synthetic_registration import (
    PCB_SHOWCASE_FEATURES,
    PCB_SHOWCASE_MATERIALS,
    make_pcb_registration_showcase,
)


@pytest.fixture(scope="module")
def showcase():
    return make_pcb_registration_showcase()


@pytest.mark.unit
def test_showcase_is_deterministic_and_visually_rich(showcase):
    repeated = make_pcb_registration_showcase()

    np.testing.assert_array_equal(showcase.moving_hwz, repeated.moving_hwz)
    np.testing.assert_array_equal(showcase.fixed_hwz, repeated.fixed_hwz)
    np.testing.assert_array_equal(
        showcase.moving_labels_hwz, repeated.moving_labels_hwz
    )
    assert showcase.moving_hwz.shape == (96, 128, 24)
    assert showcase.feature_names == PCB_SHOWCASE_FEATURES
    assert showcase.material_names == PCB_SHOWCASE_MATERIALS
    assert len(showcase.feature_names) == 10
    assert len(showcase.layer_names) == len(showcase.layer_slices) == 4


@pytest.mark.unit
def test_showcase_uses_canonical_material_ids_with_visible_populations(showcase):
    labels = showcase.moving_labels_hwz
    minimum_voxels = {
        "FR4": 100_000,
        "COPPER": 5_000,
        "GOLD": 1_000,
        "TIN": 1_000,
        "ALUMINUM": 1_000,
        "ALUMINA": 4_000,
        "TUNGSTEN": 300,
    }

    for name, minimum in minimum_voxels.items():
        material_id = MATERIAL_NAMES.index(name)
        assert np.count_nonzero(labels == material_id) >= minimum


@pytest.mark.unit
def test_showcase_contains_four_open_mounting_holes(showcase):
    labels = showcase.moving_labels_hwz
    air = MATERIAL_NAMES.index("AIR")
    copper = MATERIAL_NAMES.index("COPPER")
    h, w, depth = labels.shape
    centers = (
        (round(0.14 * (h - 1)), round(0.12 * (w - 1))),
        (round(0.14 * (h - 1)), round(0.72 * (w - 1))),
        (round(0.86 * (h - 1)), round(0.12 * (w - 1))),
        (round(0.86 * (h - 1)), round(0.88 * (w - 1))),
    )

    for row, col in centers:
        assert np.all(labels[row, col, :] == air)
        neighborhood = labels[row - 5 : row + 6, col - 5 : col + 6, depth - 7]
        assert np.count_nonzero(neighborhood == copper) >= 20


@pytest.mark.unit
def test_showcase_has_separate_gold_edge_fingers(showcase):
    labels = showcase.moving_labels_hwz
    gold = MATERIAL_NAMES.index("GOLD")
    gold_projection = np.any(labels == gold, axis=2)
    _components, count = connected_components(gold_projection)

    # Eight connector fingers plus two circular fiducials.
    assert count >= 10


@pytest.mark.unit
def test_showcase_ct_is_degraded_but_known_alignment_remains_strong(showcase):
    identity = resample_volume(
        showcase.moving_hwz,
        showcase.moving_spacing,
        showcase.fixed_hwz.shape,
        showcase.fixed_spacing,
        RigidScaleTransform.identity(),
    )
    known = resample_volume(
        showcase.moving_hwz,
        showcase.moving_spacing,
        showcase.fixed_hwz.shape,
        showcase.fixed_spacing,
        showcase.known_transform,
    )

    identity_score = registration_quality(showcase.fixed_hwz, identity)
    known_score = registration_quality(showcase.fixed_hwz, known)

    assert not np.array_equal(showcase.fixed_hwz, known)
    assert identity_score < 0.35
    assert known_score > 0.95
    assert known_score >= identity_score + 0.60


@pytest.mark.unit
def test_showcase_bounded_suggestion_is_a_useful_review_start(showcase):
    proposal = propose_initial_transform(
        showcase.moving_hwz,
        showcase.moving_spacing,
        showcase.fixed_hwz,
        showcase.fixed_spacing,
    )

    assert proposal.score > 0.80
    assert proposal.improvement > 0.50
    assert proposal.confidence in {"medium", "high"}

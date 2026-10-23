import json
from dataclasses import replace

import numpy as np
import pytest

from xrayrecon.error_inspection import (
    ComparisonContext, ComparisonGrid, ErrorEvaluator, InspectionCancelled,
    InspectionState, evaluate, export_inspection,
)


def context(truth, recon, **kwargs):
    return ComparisonContext(ComparisonGrid(truth.shape), {"zeff": truth},
                             {("zeff", "raw"): recon}, **kwargs)


@pytest.mark.parametrize("offset", [0., 2., -2.])
def test_offset_sign_metrics_and_strict_threshold(offset):
    truth = np.ones((2, 3, 4))
    ctx = context(truth, truth + offset)
    state = InspectionState(scope="volume", tolerance=abs(offset))
    result = evaluate(ctx, state)
    metrics = result["metrics"]
    assert metrics["bias"] == offset
    for name in ("mae", "rmse", "p95"):
        assert metrics[name] == abs(offset)
    assert metrics["above_tolerance"] == 0
    assert metrics["coverage"] == 1
    np.testing.assert_array_equal(result["field"].signed, offset)
    assert truth.flags.writeable
    assert not ctx.references["zeff"].flags.writeable


def test_cancellation_validity_coverage_empty_and_nonoverlap():
    truth = np.zeros((1, 2, 3))
    recon = np.array([[[-2., 2., np.nan], [9., 9., 9.]]])
    solver = np.array([[[True, True, True], [False, True, True]]])
    overlap = np.array([[[True, True, True], [True, False, False]]])
    ctx = context(truth, recon, validity={"solver_valid": solver}, overlap=overlap)
    state = InspectionState(scope="volume")
    out = evaluate(ctx, state)
    m = out["metrics"]
    assert m["bias"] == 0 and m["mae"] == m["rmse"] == 2
    assert m["coverage"] == 2 / 6 and m["excluded_nonoverlap"] == 2
    assert np.isnan(out["field"].signed[~out["field"].valid]).all()
    with pytest.raises(InspectionCancelled):
        evaluate(ctx, state, cancel=lambda: True)
    invalid = evaluate(context(truth, truth * np.nan), state)["metrics"]
    assert invalid["bias"] is None and invalid["reason"]
    empty = evaluate(replace(ctx, labels=np.zeros(truth.shape, dtype=int)), replace(state, material=9))["metrics"]
    assert empty["coverage"] is None and empty["mae"] is None


def test_oriented_anisotropic_grid_line_roi_and_brush():
    grid = ComparisonGrid((3, 4, 5), (.2, .5, 2.), (12., -3., 7.),
                          ((0., -1., 0.), (1., 0., 0.), (0., 0., 1.)), "mm")
    coords = np.array([[0, 0, 0], [2, 3, 4], [1, 2, 3]])
    np.testing.assert_allclose(grid.world_to_voxel(grid.voxel_to_world(coords)), coords)
    truth = np.indices(grid.shape)[2].astype(float)
    ctx = replace(context(truth, truth + 2), grid=grid)
    line = tuple(map(tuple, grid.voxel_to_world([[1, 1, 0], [1, 1, 4]])))
    state = InspectionState(index=1, line=line, roi=((1, 1, 1), (1, 2, 3)), brush=(1., 3.))
    out = evaluate(ctx, state)
    assert out["metrics"]["valid_count"] == 6
    assert out["highlighted"].sum() == 6
    assert out["brushed_metrics"]["bias"] == 2
    samples = out["line"]
    assert samples["distance"][-1] == pytest.approx(2.)
    np.testing.assert_allclose(samples["reference"], samples["distance"] * 2)
    np.testing.assert_allclose(samples["signed_error"], 2)
    np.testing.assert_allclose(grid.spacing_zhw, [2., .2, .5])


def test_line_preserves_invalid_gaps():
    truth = np.ones((1, 2, 5))
    recon = truth.copy()
    recon[:, :, 2] = np.nan
    ctx = context(truth, recon)
    state = InspectionState(line=((0., 0., 0.), (0., 4., 0.)))
    out = evaluate(ctx, state)
    assert (~out["line"]["valid"]).any()
    assert np.isnan(out["line"]["reference"][~out["line"]["valid"]]).all()


def test_reference_material_boundary_and_background_filters():
    truth = np.zeros((2, 7, 8))
    labels = np.zeros(truth.shape, dtype=int)
    labels[:, 2:5, 2:6] = 2
    ctx = replace(context(truth, truth + 1, labels=labels), grid=ComparisonGrid(truth.shape, (.5, .5, 2.), unit="mm"))
    state = InspectionState(scope="volume", region="background")
    assert evaluate(ctx, state)["metrics"]["valid_count"] == (labels == 0).sum()
    assert evaluate(ctx, replace(state, region="all", material=2))["metrics"]["valid_count"] == 24
    boundary = evaluate(ctx, replace(state, region="boundary", boundary_width=0))["selected"]
    interior = evaluate(ctx, replace(state, region="interior", boundary_width=0))["selected"]
    assert not (boundary & interior).any()
    assert interior.sum() == 4
    assert evaluate(ctx, replace(state, region="boundary", boundary_width=.5))["selected"].sum() > boundary.sum()
    with pytest.raises(ValueError, match="labels unavailable"):
        evaluate(replace(ctx, labels=None), state)


def test_cache_identity_and_display_independence():
    truth = np.ones((2, 3, 4))
    ctx = context(truth, truth + 2)
    evaluator = ErrorEvaluator()
    state = InspectionState(scope="volume")
    a = evaluate(ctx, state, evaluator)
    b = evaluate(ctx, replace(state, tolerance=4, intensity_limits=(-8, 20), error_limit=5), evaluator)
    assert a["field"] is b["field"]
    for name in ("bias", "mae", "rmse", "p95"):
        assert a["metrics"][name] == b["metrics"][name]
    assert b["metrics"]["above_tolerance"] == 0
    changed = replace(ctx, source_version="2")
    assert evaluator.field(changed, "zeff", "raw") is not a["field"]
    transform = np.eye(4); transform[0, 3] = 2
    changed = replace(ctx, transform=tuple(map(tuple, transform)))
    assert evaluator.field(changed, "zeff", "raw").key != a["field"].key
    changed = replace(ctx, outputs={**ctx.outputs, ("zeff", "regularized"): truth + 3})
    assert evaluate(changed, replace(state, variant="regularized"), evaluator)["metrics"]["bias"] == 3


def test_export_roundtrip_and_no_overwrite(tmp_path):
    truth = np.arange(24.).reshape(2, 3, 4)
    ctx = context(truth, truth + 1, labels=np.ones(truth.shape, dtype=int))
    state = InspectionState(scope="volume", tolerance=1, brush=(.5, 2), line=((0., 0., 0.), (0., 3., 0.)))
    out = evaluate(ctx, state)
    path = export_inspection(tmp_path / "inspection.npz", ctx, state, out)
    with np.load(path, allow_pickle=False) as stored:
        meta = json.loads(str(stored["metadata_json"]))
        restored_grid = ComparisonGrid(**{k: meta["grid"][k] for k in ("shape", "spacing_hwz", "origin", "direction", "unit")})
        np.testing.assert_array_equal(stored["signed_error"], out["field"].signed)
        np.testing.assert_array_equal(stored["valid_mask"], out["field"].valid)
        np.testing.assert_array_equal(stored["highlighted_mask"], out["highlighted"])
        assert not stored["above_tolerance_mask"].any()
        assert meta["state"]["tolerance"] == 1
        assert meta["metrics"] == out["metrics"]
        np.testing.assert_allclose(restored_grid.voxel_to_world([1, 2, 3]), ctx.grid.voxel_to_world([1, 2, 3]))
    with pytest.raises(FileExistsError):
        export_inspection(path, ctx, state, out)
    with pytest.raises(ValueError, match="stale selection"):
        export_inspection(tmp_path / "stale.npz", ctx, replace(state, tolerance=2), out)


def test_large_exact_percentile_buffer_and_numpy_roi_export(tmp_path):
    truth = np.zeros((2, 1000, 501), dtype=np.float32)
    recon = (np.arange(truth.size) % 17).reshape(truth.shape).astype(float)
    ctx = context(truth, recon)
    roi = tuple(map(tuple, np.array([[0, 0, 0], [1, 999, 500]], dtype=np.int64)))
    state = InspectionState(scope="volume", roi=roi)
    out = evaluate(ctx, state)
    assert out["metrics"]["p95"] == np.percentile(recon, 95, method="linear")
    path = export_inspection(tmp_path / "large.npz", ctx, state, out)
    with np.load(path, allow_pickle=False) as stored:
        np.testing.assert_array_equal(stored["absolute_error"], recon)
        assert json.loads(str(stored["metadata_json"]))["state"]["roi"] == [[0, 0, 0], [1, 999, 500]]


def test_linear_percentile_method():
    truth = np.zeros((1, 2, 2))
    out = evaluate(context(truth, np.array([[[0., -1.], [5., 11.]]])), InspectionState())
    assert out["metrics"]["p95"] == pytest.approx(10.1)


def test_nominal_cad_requires_explicit_alignment_and_properties():
    truth = np.ones((1, 2, 2))
    with pytest.raises(ValueError, match="alignment decision"):
        context(truth, truth, reference_type="nominal CAD")
    ctx = replace(context(truth, truth), references={})
    with pytest.raises(ValueError, match="no assigned reference"):
        evaluate(ctx, InspectionState())

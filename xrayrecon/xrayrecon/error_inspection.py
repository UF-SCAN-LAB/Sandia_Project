"""Rendering-independent, full-resolution AM comparison and export contract.

Source arrays are borrowed through read-only views, never modified. Callers must
replace contexts (and source_version) if an artifact changes. World coordinates
follow the application's HWZ registration convention, not image column/row XYZ.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import tempfile
from uuid import uuid4

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates

PLANES = {"XY": (0, 1, 2), "XZ": (2, 0, 1), "YZ": (1, 0, 2)}
AXIS_NAMES = ("Z", "X (H)", "Y (W)")
UNITS = {"zeff": "Z-effective (dimensionless)", "rhoe": "rho_e / rho_e,water"}
VARIANTS = {"raw": "recovered", "regularized": "regularized", "calibrated": "calibrated"}


@dataclass(frozen=True)
class ComparisonGrid:
    shape: tuple[int, int, int]
    spacing_hwz: tuple[float, float, float] = (1., 1., 1.)
    origin: tuple[float, float, float] = (0., 0., 0.)
    direction: tuple = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))
    unit: str = "voxel"

    def __post_init__(self):
        if len(self.shape) != 3 or any(int(s) != s or s < 1 for s in self.shape):
            raise ValueError("Comparison grid must have a positive (Z,H,W) shape")
        if len(self.spacing_hwz) != 3 or not np.all(np.isfinite(self.spacing_hwz)) or min(self.spacing_hwz) <= 0:
            raise ValueError("Spacing must be finite and positive in HWZ order")
        d = np.asarray(self.direction)
        if d.shape != (3, 3) or not np.allclose(d.T @ d, np.eye(3), atol=1e-8):
            raise ValueError("Direction must be orthonormal; shear grids require explicit resampling")
        if np.shape(self.origin) != (3,) or not np.isfinite(self.origin).all():
            raise ValueError("Origin must contain three finite coordinates")

    @property
    def spacing_zhw(self):
        return np.asarray(self.spacing_hwz)[[2, 0, 1]]

    def voxel_to_world(self, zhw):
        hwz = np.asarray(zhw, dtype=float)[..., [1, 2, 0]]
        return (hwz * self.spacing_hwz) @ np.asarray(self.direction).T + self.origin

    def world_to_voxel(self, world, clamp=False):
        hwz = ((np.asarray(world) - self.origin) @ np.asarray(self.direction)) / self.spacing_hwz
        zhw = hwz[..., [2, 0, 1]]
        return np.clip(zhw, 0, np.asarray(self.shape) - 1) if clamp else zhw

    def metadata(self):
        return {**asdict(self), "array_order": "Z,H,W", "world_basis": "H,W,Z",
                "voxel_location": "centers; origin is voxel [0,0,0]",
                "handedness": "right" if np.linalg.det(self.direction) > 0 else "left"}


def _readonly(array, shape):
    view = np.asarray(array).view()
    if view.shape != tuple(shape):
        raise ValueError(f"Field shape {view.shape} differs from comparison grid {shape}; resample explicitly")
    view.flags.writeable = False
    return view


@dataclass(frozen=True)
class ComparisonContext:
    grid: ComparisonGrid
    references: dict
    outputs: dict
    run_id: str = field(default_factory=lambda: str(uuid4()))
    source_version: str = "1"
    reference_id: str = "synthetic truth"
    reference_type: str = "synthetic"
    labels: np.ndarray | None = None
    label_names: dict = field(default_factory=dict)
    background_id: int = 0
    validity: dict = field(default_factory=dict)
    overlap: np.ndarray | None = None
    transform: tuple = tuple(tuple(float(i == j) for j in range(4)) for i in range(4))
    alignment: str = "known synthetic shared grid"
    resampling: str = "none; native simulation grid"
    reference_policy: str = "explicit AM property fields; native voxelized values, including partial volume"
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.reference_type not in ("synthetic", "nominal CAD", "measured"):
            raise ValueError("Declare synthetic, nominal CAD, or measured reference")
        if self.reference_type != "synthetic" and self.alignment == "known synthetic shared grid":
            raise ValueError("Independent reference requires an explicit alignment decision")
        if not self.alignment or np.shape(self.transform) != (4, 4) or not np.isfinite(self.transform).all():
            raise ValueError("Explicit finite alignment transform required")
        for name in ("references", "outputs", "validity"):
            object.__setattr__(self, name, {key: _readonly(value, self.grid.shape)
                                         for key, value in getattr(self, name).items()})
        for name in ("labels", "overlap"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, _readonly(getattr(self, name), self.grid.shape))
        if self.labels is not None and not np.issubdtype(self.labels.dtype, np.integer):
            raise ValueError("Reference material IDs must be categorical integers")

    def key(self, quantity, variant):
        return (self.run_id, self.source_version, quantity, variant,
                json.dumps(self.grid.metadata(), sort_keys=True), str(self.transform),
                self.alignment, self.resampling, id(self.references.get(quantity)),
                id(self.outputs.get((quantity, variant))), id(self.overlap),
                tuple((str(k), id(v)) for k, v in self.validity.items()))


def context_from_am(result: dict) -> ComparisonContext:
    """Adapt native AM artifacts; never infer truth labels from recovered fields."""
    metadata = getattr(result, "inspection_metadata", {})
    metrics = getattr(result, "run_metrics", None) or {}
    spacing = metadata.get("spacing_hwz") or metrics.get("voxel_spacing_mm_xyz")
    grid = ComparisonGrid(
        tuple(result["zeff_recovered"].shape), tuple(spacing or (1., 1., 1.)),
        tuple(metadata.get("origin", (0., 0., 0.))),
        tuple(tuple(row) for row in metadata.get("direction", np.eye(3))),
        "mm" if spacing is not None else "voxel",
    )
    from .materials import MATERIAL_NAMES
    return ComparisonContext(
        grid, {q: result[q + "_ground_truth"] for q in UNITS if q + "_ground_truth" in result},
        {(q, v): result[q + "_" + suffix] for q in UNITS for v, suffix in VARIANTS.items()
         if q + "_" + suffix in result},
        run_id=metadata.get("run_id", f"session-{id(result):x}"),
        labels=result.get("material_labels_ground_truth"),
        label_names=dict(enumerate(MATERIAL_NAMES)), background_id=MATERIAL_NAMES.index("AIR"),
        validity={k: v for k, v in result.items() if k == "solver_valid" or k.endswith("_valid")},
        overlap=result.get("comparison_overlap"),
        provenance={"acquisition": getattr(result, "acquisition_config", None).to_dict()
                    if getattr(result, "acquisition_config", None) else None,
                    "run": metrics, "coordinate_note": "HWZ world convention matches registration and PCBViewer3D",
                    "spacing_note": "physical spacing supplied" if spacing else "physical spacing unavailable; voxel distances only",
                    "effective_z_convention": "AM PE_EXPONENT=3.8; K_REF_Z=29"},
    )


@dataclass
class InspectionState:
    quantity: str = "zeff"
    variant: str = "raw"
    plane: str = "XY"
    index: int = 0
    cursor_world: tuple = (0., 0., 0.)
    scope: str = "slice"
    roi: tuple | None = None  # inclusive ZHW lower, upper; single-plane rectangle
    material: int | None = None
    region: str = "all"
    boundary_width: float = 1.
    tolerance: float = 0.
    brush: tuple | None = None
    line: tuple | None = None  # two world-space endpoints
    intensity_limits: tuple = (0., 1.)
    error_limit: float = 1.
    display_mode: str = "signed"
    view_ranges: tuple | None = None
    volume_opacity: float = .25
    clip_at_slice: bool = False


class InspectionCancelled(Exception):
    pass


def _check(cancel):
    if cancel is not None and cancel():
        raise InspectionCancelled()


def _json_native(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot encode {type(value).__name__} in inspection metadata")


def evaluation_key(state):
    """Exclude presentation and probe changes; include every numerical policy."""
    names = ("quantity", "variant", "plane", "index", "scope", "roi", "material",
             "region", "boundary_width", "tolerance", "brush", "line", "error_limit")
    return json.dumps({name: getattr(state, name) for name in names}, sort_keys=True, default=_json_native)


@dataclass
class ErrorField:
    signed: np.ndarray
    valid: np.ndarray
    overlap: np.ndarray | None
    key: tuple


class ErrorEvaluator:
    """One-field cache; tolerance and presentation changes never subtract again."""
    def __init__(self):
        self.cached = None
        self.boundary_key = None
        self.boundary = None

    def field(self, context, quantity, variant, cancel=None):
        key = context.key(quantity, variant)
        if self.cached is not None and self.cached.key == key:
            return self.cached
        if quantity not in context.references:
            raise ValueError("Property comparison unavailable: no assigned reference property field")
        if (quantity, variant) not in context.outputs:
            raise ValueError("Requested output variant is unavailable")
        truth, recon = context.references[quantity], context.outputs[quantity, variant]
        error = np.empty(context.grid.shape, dtype=np.float64)
        valid = np.empty(context.grid.shape, dtype=bool)
        for z in range(context.grid.shape[0]):
            _check(cancel)
            # Non-finite inputs are invalid, never substituted with zero.
            with np.errstate(invalid="ignore", over="ignore"):
                np.subtract(recon[z], truth[z], out=error[z], dtype=np.float64)
            good = np.isfinite(truth[z]) & np.isfinite(recon[z]) & np.isfinite(error[z])
            for name in ("solver_valid", f"{quantity}_valid", f"{quantity}_{variant}_valid", f"{quantity}_reference_valid"):
                if name in context.validity:
                    flag = context.validity[name][z]
                    good &= np.isfinite(flag) & (flag != 0)
            if context.overlap is not None:
                good &= np.isfinite(context.overlap[z]) & (context.overlap[z] != 0)
            valid[z] = good
            error[z][~good] = np.nan
        _check(cancel)
        self.cached = ErrorField(error, valid, context.overlap, key)
        return self.cached

    def region_mask(self, context, state, cancel=None):
        shape = context.grid.shape
        mask = np.ones(shape, dtype=bool)
        if state.scope == "slice":
            mask[:] = False
            sl = [slice(None)] * 3
            sl[PLANES[state.plane][0]] = state.index
            mask[tuple(sl)] = True
        elif state.scope != "volume":
            raise ValueError("Unknown evaluation scope")
        if state.roi is not None:
            roi_mask = np.zeros(shape, dtype=bool)
            lo, hi = np.clip(state.roi, 0, np.asarray(shape) - 1).astype(int)
            roi_mask[tuple(slice(a, b + 1) for a, b in zip(lo, hi))] = True
            mask &= roi_mask
        labels = context.labels
        if (state.material is not None or state.region != "all") and labels is None:
            raise ValueError("Reference material labels unavailable; material/region filtering cannot be inferred")
        if state.material is not None:
            mask &= labels == state.material
        if state.region == "background":
            mask &= labels == context.background_id
        elif state.region in ("boundary", "interior"):
            if not np.isfinite(state.boundary_width) or state.boundary_width < 0:
                raise ValueError("Boundary width must be finite and nonnegative")
            key = (id(labels), str(context.grid.metadata()), state.boundary_width)
            if key != self.boundary_key:
                _check(cancel)
                seeds = np.zeros(shape, dtype=bool)
                for axis in range(3):
                    a, b = [slice(None)] * 3, [slice(None)] * 3
                    a[axis], b[axis] = slice(None, -1), slice(1, None)
                    changed = labels[tuple(a)] != labels[tuple(b)]
                    seeds[tuple(a)] |= changed
                    seeds[tuple(b)] |= changed
                self.boundary = (distance_transform_edt(~seeds, sampling=context.grid.spacing_zhw)
                                 <= state.boundary_width) if seeds.any() else seeds
                self.boundary_key = key
                _check(cancel)
            mask &= self.boundary if state.region == "boundary" else (~self.boundary & (labels != context.background_id))
        elif state.region != "all" and state.region != "background":
            raise ValueError("Unknown region")
        return mask


def summarize(field, selected, tolerance, cancel=None):
    """Exact metrics; chunked reductions and disk-backed exact linear P95.

    The percentile buffer holds absolute errors only, is partitioned in place,
    and spills to a temporary file above 8 MB. No sampled metrics are reported.
    """
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Tolerance must be finite and nonnegative")
    expected = int(np.count_nonzero(selected))
    count = sum(int(np.count_nonzero(s & v)) for s, v in zip(selected, field.valid))
    nonoverlap = 0 if field.overlap is None else int(np.count_nonzero(selected & ~(np.isfinite(field.overlap) & (field.overlap != 0))))
    metrics = dict(expected_count=expected, valid_count=count, coverage=count / expected if expected else None,
                   excluded_nonoverlap=nonoverlap, bias=None, mae=None, rmse=None, p95=None,
                   above_tolerance=None, reason=None)
    if not count:
        metrics["reason"] = "Empty selected region" if not expected else "No valid comparison voxels"
        return metrics
    with tempfile.TemporaryFile() as backing:
        values = np.memmap(backing, dtype="float64", mode="w+", shape=count) if count > 1_000_000 else np.empty(count)
        offset, total, absolute_total, squares, above = 0, 0., 0., 0., 0
        try:
            for z in range(selected.shape[0]):
                _check(cancel)
                signed = field.signed[z][selected[z] & field.valid[z]]
                absolute = np.abs(signed)
                values[offset:offset + len(signed)] = absolute
                offset += len(signed)
                total += float(np.sum(signed, dtype=np.float64))
                absolute_total += float(np.sum(absolute, dtype=np.float64))
                squares += float(np.dot(signed, signed))
                above += int(np.count_nonzero(absolute > tolerance))
            p = .95 * (count - 1)
            low, high = int(np.floor(p)), int(np.ceil(p))
            values.partition((low, high))
            p95 = float(values[low] + (p - low) * (values[high] - values[low]))
            _check(cancel)
            metrics.update(bias=total / count, mae=absolute_total / count,
                           rmse=float(np.sqrt(squares / count)), p95=p95, above_tolerance=above / count)
        finally:
            if isinstance(values, np.memmap):
                values._mmap.close()
    return metrics


def evaluate(context, state, evaluator=None, cancel=None):
    evaluator = evaluator or ErrorEvaluator()
    field = evaluator.field(context, state.quantity, state.variant, cancel)
    selected = evaluator.region_mask(context, state, cancel)
    base = summarize(field, selected, state.tolerance, cancel)
    highlighted = selected & field.valid
    if state.brush is not None:
        highlighted &= (field.signed >= state.brush[0]) & (field.signed <= state.brush[1])
    brushed = summarize(field, highlighted, state.tolerance, cancel) if state.brush is not None else None
    # Fixed bins derive from the locked error display scale. Out-of-range counts
    # are reported separately; exports/metrics retain all actual values.
    edges = np.linspace(-state.error_limit, state.error_limit, 81)
    histogram = np.zeros(80, dtype=np.int64)
    for z in range(context.grid.shape[0]):
        _check(cancel)
        vals = field.signed[z][selected[z] & field.valid[z]]
        histogram += np.histogram(vals, bins=edges)[0]
    return dict(field=field, selected=selected, highlighted=highlighted, metrics=base,
                state_key=evaluation_key(state),
                brushed_metrics=brushed, histogram=histogram, histogram_edges=edges,
                histogram_outside=int(base["valid_count"] - histogram.sum()),
                line=sample_line(context, state, field) if state.line else None)


def sample_line(context, state, field):
    endpoints = np.asarray(state.line)
    length = float(np.linalg.norm(endpoints[1] - endpoints[0]))
    n = min(10000, max(2, int(np.ceil(length / min(context.grid.spacing_hwz) * 2)) + 1))
    world = np.linspace(endpoints[0], endpoints[1], n)
    coords = context.grid.world_to_voxel(world).T
    valid = map_coordinates(field.valid, coords, output=np.float64, order=1, mode="constant", cval=0., prefilter=False) >= 1 - 1e-12
    truth = map_coordinates(context.references[state.quantity], coords, output=np.float64, order=1, mode="constant", cval=np.nan, prefilter=False)
    recon = map_coordinates(context.outputs[state.quantity, state.variant], coords, output=np.float64, order=1, mode="constant", cval=np.nan, prefilter=False)
    valid &= np.isfinite(truth) & np.isfinite(recon)
    truth[~valid], recon[~valid] = np.nan, np.nan
    return dict(world=world, distance=np.linspace(0, length, n), reference=truth,
                reconstruction=recon, signed_error=recon - truth, valid=valid)


def inspection_metadata(context, state, evaluation):
    return dict(schema_version=1, run_id=context.run_id, source_version=context.source_version,
                reference_id=context.reference_id, reference_type=context.reference_type,
                quantity=state.quantity, units=UNITS[state.quantity], variant=state.variant,
                grid=context.grid.metadata(), transform=context.transform, alignment=context.alignment,
                resampling=context.resampling, reference_policy=context.reference_policy,
                material_names=context.label_names, background_id=context.background_id,
                state=asdict(state), metrics=evaluation["metrics"], brushed_metrics=evaluation["brushed_metrics"],
                definitions={"signed_error": "reconstruction - reference", "bias": "mean(signed_error)",
                             "mae": "mean(abs(error))", "rmse": "sqrt(mean(error**2))",
                             "p95": "95th percentile absolute error; NumPy linear method",
                             "above_tolerance": "count(abs(error) > tolerance) / valid_count; strict >",
                             "coverage": "valid selected / expected selected on entire comparison grid",
                             "validity": "finite reference/reconstruction, solver flags, field-of-view overlap; no quality cutoff",
                             "boundary": "centers within boundary_width of either center adjacent to a differing 6-neighbor label; includes both sides; grid exterior is not a material",
                             "roi": "inclusive voxel-center box restricted to the plane on which rectangle was drawn",
                             "brush": "inclusive signed-error endpoints; additional subset; base metrics remain unbrushed",
                             "line": "trilinear fields; any invalid contributing neighbor makes a gap; <=10000 samples",
                             "tolerance": "exploratory, not a certified acceptance specification",
                             "percentage_error": "not used", "missing_values": "NaN plus valid mask"},
                provenance=context.provenance)


def export_inspection(path, context, state, evaluation, cancel=None):
    """Exclusive-create lossless NPZ with embedded JSON, avoiding source overwrite."""
    path = Path(path)
    field = evaluation["field"]
    if field.key != context.key(state.quantity, state.variant):
        raise ValueError("Cannot export stale evaluation")
    if evaluation["state_key"] != evaluation_key(state):
        raise ValueError("Cannot export stale selection or tolerance")
    metadata = inspection_metadata(context, state, evaluation)
    line = {"line_" + k: v for k, v in (evaluation["line"] or {}).items()}
    _check(cancel)
    with tempfile.TemporaryFile() as backing:
        absolute = (np.memmap(backing, dtype="float64", mode="w+", shape=field.signed.shape)
                    if field.signed.size > 1_000_000 else np.empty_like(field.signed))
        try:
            for z in range(field.signed.shape[0]):
                _check(cancel)
                np.abs(field.signed[z], out=absolute[z])
            with path.open("xb") as stream:
                try:
                    np.savez_compressed(stream, signed_error=field.signed, absolute_error=absolute,
                                valid_mask=field.valid, selected_region_mask=evaluation["selected"],
                                highlighted_mask=evaluation["highlighted"],
                                above_tolerance_mask=field.valid & (absolute > state.tolerance),
                                reference=context.references[state.quantity],
                                reconstruction=context.outputs[state.quantity, state.variant],
                                histogram=evaluation["histogram"], histogram_edges=evaluation["histogram_edges"],
                                **({"reference_labels": context.labels} if context.labels is not None else {}),
                                metadata_json=np.asarray(json.dumps(metadata, allow_nan=False, default=_json_native)), **line)
                except Exception:
                    stream.close()
                    path.unlink(missing_ok=True)  # only this newly-created incomplete export
                    raise
        finally:
            if isinstance(absolute, np.memmap):
                absolute._mmap.close()
    return path

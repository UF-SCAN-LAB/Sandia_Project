# CT / AM error inspection

Run an AM simulation, open **Quantitative Alvarez–Macovski Analysis**, and select
**Error inspection…**. The existing overview, calibrated intervals, quality
indicator, and backprojected projection inconsistency remain available through
**Overview**. **Volume workspace** returns to the existing CT/Gerber tools.
Inspection never reruns reconstruction, changes source arrays, registers data,
or transfers labels.

## Workflow

1. Choose a recent run (up to three retained in this workspace), raw,
   material-regularized, or calibrated output, and Z-effective or relative
   electron density. Only variants actually present are offered. Each run's
   explicitly supplied reference is shown; there is no fabricated CAD property
   reference. New runs and variants replace dependent results atomically.
2. Choose XY, XZ, or YZ and a slice. **Probe** selects the nearest voxel center
   in any map, moving all crosshairs and the 3D marker. The inspector shows world
   coordinates, source values, signed error, and reference material identity.
   **Pan / zoom** enables dragging and wheel zoom. **Fit views** fits the maps
   to the grid without changing the 3D camera.
3. Choose **Line**, then click two endpoints in any map. The analysis selector
   switches between both property profiles, a separate signed-difference plot,
   and a signed-error histogram. Values and errors have separate labeled axes.
   Invalid line samples appear as gaps. Drag either line endpoint in any map to
   update the linked profile. Choose **ROI** and click two corners
   for an inclusive voxel-center rectangle on the current plane. Clear actions
   restore the underlying scope. Plane/slice changes clear these slice-local
   selections instead of silently converting them to a 3D box.
4. Select current slice or entire volume, reference material, and all regions,
   interior, boundary, or background. An ROI additionally restricts that scope
   to its single-plane rectangle. A histogram brush adds an inclusive signed
   error interval. Base measurements remain unbrushed; subset measurements are
   shown separately. The same numerical mask drives slice and 3D highlighting.
5. Set an **exploratory tolerance** with the field or slider. This is not a
   certified acceptance limit. Signed, absolute, and above-tolerance maps use
   numerical fields, not screenshot colors. Reference/reconstruction share a
   window. Each quantity retains independent limits and tolerance across runs;
   **Reset ranges to this run** explicitly recalculates limits over truth and
   available variants. Legends report saturation. Invalid voxels are magenta
   with a stippled overlay, never zero error.
6. Rotate/zoom the embedded existing OpenGL renderer. Adjust opacity or clip at
   the inspection plane to expose internal features. The texture uses nearest
   sampled voxels with at most 128 samples per axis; statistics, exports, and
   the numerical selection always use full-resolution arrays.
7. **Export data + state…** writes a new lossless `.npz` file. Existing files
   are refused. A worker exports the selected snapshot even if you subsequently
   change the inspection. Inspectors and analysis panes can be toggled and all
   main panes resized. Below 1000 logical pixels wide the inspector collapses;
   below 650 high the analysis collapses. They can be reopened manually.

## Scientific contract

The pure implementation is `xrayrecon.error_inspection`. `ComparisonContext`
borrows read-only views of immutable source artifacts; replace the context and
source version if an artifact changes. `InspectionState` owns the numerical and
display selections. `ErrorEvaluator` caches a single field using source/run IDs,
versions, quantity, variant, grid, transform, and validity identities. The GUI
uses serial Qt background work, debouncing, cancellation, and generation tokens
to discard stale callbacks. Export work uses a separate pool so navigation cannot
discard an explicitly requested export.

- Signed error is reconstruction minus reference. Bias, MAE, RMSE, and P95 use
  valid selected voxels. P95 is exact, using NumPy's **linear** convention.
  Above-tolerance fraction uses strict `abs(error) > tolerance`; equality is not
  counted. No percentage error or near-zero denominator policy is introduced.
- Coverage is valid selected / expected selected on the full comparison grid.
  Non-overlap exclusions are reported separately. Empty/invalid selections have
  null metrics, a reason, and `N/A` in the interface. Validity combines finite
  fields, supplied solver masks, and field-of-view overlap. Heuristic quality
  never silently removes a voxel.
- Reference labels come from the voxelizer/showcase, never reconstructed material
  classification. Missing labels disable the material and region controls.
  Background remains inspectable, including extra reconstructed material there.
  The boundary width is a physical **radius from the voxel centers on either
  side of a six-neighbor label transition**. It is a voxel-grid band, not an
  exact subvoxel surface distance; the grid exterior is not an inferred material.
  Interiors exclude that band and background. No boundary is excluded by default.
- Native AM array ordering is **Z,H,W**. The existing registration/GL world basis
  is **H,W,Z** (called X,Y,Z by those tools). Spacing follows HWZ; the legacy
  `voxel_spacing_mm_xyz` record is interpreted in that application convention.
  World = origin + direction × (HWZ index × spacing). Origin locates the first
  voxel center. Direction can be rotated or reflected and must be orthonormal.
  Plane labels follow grid axes; the inspector/export records the world basis.
  Unknown spacing stays in **voxel** units; a 2D acquisition pixel size alone is
  not used to invent a physical slice pitch. Native simulations use origin zero
  and identity direction. Changing plane preserves the world cursor; changing
  slice changes its normal component. World selections round to the nearest
  center and clamp to the volume.
- Native synthetic truth/reconstruction retain their known alignment and native
  voxelized property assignments, including existing partial volume. No automatic
  resampling or alignment optimization is performed. Line profiles use trilinear
  sampling in world space, no more than 10,000 positions, and require all
  contributing neighbors to be valid. Labels are never interpolated.
- `ComparisonContext` can accept an explicitly prepared comparison grid with
  reference/output fields, categorical labels, validity, overlap, a recorded
  transform and resampling policy. Independent measured/nominal references must
  declare an alignment decision. This UI currently lists native AM artifacts;
  it does not add an independent measured/CAD import-and-registration workflow.
  Missing property assignments explain the unavailable comparison; the original
  volume workspace continues to provide geometry inspection. Nominal CAD deviation
  alone does not establish simulation errors or manufactured defects.
- Quality, projection inconsistency, and truth error remain separate. The existing
  projection map is a normalized, smoothed **backprojection of basis-sinogram
  mismatch**, not a voxelwise truth error or a unique localization of its cause.

## Export and batch use

The `.npz` contains `reference`, `reconstruction`, `signed_error`, `absolute_error`,
`valid_mask`, `selected_region_mask`, `highlighted_mask`, `above_tolerance_mask`,
histogram counts/edges, reference labels when supplied, and line coordinates,
distance, reference/reconstruction/error/validity arrays when selected. The
above-tolerance mask is defined on the full valid grid; intersect it with
`selected_region_mask` to reproduce the region's fraction.

`metadata_json` contains JSON text: source/run/version and reference identity,
quantity/units/variant, grid spacing/origin/direction/order/handedness, transform,
alignment/resampling/reference policies, material names, scope/ROI/brush/line,
window/error limits, tolerance, map view ranges, 3D opacity/clipping, exact metric
definitions, base/subset metrics, validity/boundary/denominator policies, and
available acquisition/run provenance. NaNs plus the explicit validity mask
preserve missing samples. No pickle is required.

```python
import json
import numpy as np
from xrayrecon.error_inspection import (
    ComparisonGrid, InspectionState, context_from_am, evaluate, export_inspection,
)

context = context_from_am(am_result)
state = InspectionState(quantity="zeff", variant="raw", scope="volume", tolerance=1.0)
evaluation = evaluate(context, state)
export_inspection("new-inspection.npz", context, state, evaluation)
with np.load("new-inspection.npz", allow_pickle=False) as saved:
    metadata = json.loads(str(saved["metadata_json"]))
    grid = ComparisonGrid(**{name: metadata["grid"][name] for name in
        ("shape", "spacing_hwz", "origin", "direction", "unit")})
    world_position = grid.voxel_to_world((0, 0, 0))
```

The example tolerance is an arbitrary exploratory value. The UI, exports, and
batch call the same numerical evaluator. Reductions are chunked by Z slice;
large exact-percentile and absolute-export buffers use temporary disk backing
above one million voxels. Error and selection volumes are still resident; this
is not an out-of-core multi-terabyte volume framework. Cancellation is checked
between slices/stages; a SciPy distance transform, NumPy partition, or archive
compression already in progress cannot be interrupted inside that library call.

## Integration limits and Phase 2 prerequisites

The embedded `InspectionVolumeView` subclasses the existing `PCBViewer3D`.
2D-to-3D plane, point, ROI, line, and error/brush linkage are implemented. The
existing renderer does not offer world picking. Future picking should emit
`InspectionVolumeView.sigWorldPicked(world_hwz)` or call
`ErrorInspectionPanel.selectWorld(world_hwz)`. The adapter is tested; actual
3D-to-2D mouse picking is not claimed. OpenGL requires a functioning desktop
context; headless Qt tests also verify data linkage independently of painting.

Phase 2 remains a documented backlog, with no placeholder production controls:

- Bidirectional surface deviation and missing/extra/wrong-material components:
  require independently supported reconstructed labels/surfaces, a reliable
  inside/outside convention, physical component connectivity, and feature IDs.
- Connectivity: require reference net/feature topology or a documented 3D graph,
  resolved geometry, and tests separating trace breaks from cross-layer overlap.
- Z-effective/density scatter and quality calibration: require joint validity,
  reproducible voxel selections, known reference properties, and held-out repeated
  simulations/noise modeling before calibrated uncertainty can be claimed.
- Projection-ray diagnostics and slice trends: require retained per-energy
  measurement-domain residuals and acquisition geometry; normalized residuals
  additionally need a justified noise covariance model. Slice trends must reuse
  identical masks/limits/definitions across slices and output variants.

## Delivery and verification

Objective: a connected, reproducible Phase 1 inspection workspace matching the
wireframe's interaction structure while preserving the numerical pipeline.

| Goal | Dependencies | Acceptance and narrow verification | Non-goals | Status |
| --- | --- | --- | --- | --- |
| G1: numerical evaluation/provenance | Native AM fields and grid conventions | Deterministic offset/sign/threshold, invalid/empty/coverage, oriented anisotropic grid, line/ROI/material/boundary, cache, exact percentile, lossless export tests | Reconstruction changes; inferred material properties | Complete |
| G2: interactive workspace | G1; PyQt and existing GL renderer | Linked crosshairs/marker, line/ROI/brush, output/quantity changes, stale-work rejection and responsive event loop; GUI tests | Renderer replacement; actual 3D mouse picking | Complete |
| G3: integration/usability | G2; simulation/showcase | Existing overview entry point; regression checks; representative 24×96×128 PCB CPU simulation; native desktop and compact screenshots | Phase 2 diagnostics; production acceptance claims | Complete |

Baseline: 28 existing tests passed. A broader AM/calibration/showcase/application
regression run passed 87 tests. Final focused checks passed 34 tests, including
the additional export, native GL context-method regression, and navigation
changes. Existing dependency deprecation warnings remain.

Native hidden-window review at 1500×980 and 900×600 logical pixels produced
screenshots in `build/error-inspection-review/` (generated, not versioned).
The 294,912-voxel representative evaluation completed in approximately 0.18 seconds
including debounce/render update, with 11 event-loop timer ticks and a valid
OpenGL context. This is one local smoke check, not a general performance guarantee.
Synthetic truth error is a comparison against that simulation's declared model,
not calibrated uncertainty or proof of physical manufacturing accuracy.

Changed files for this delivery:

- `xrayrecon/xrayrecon/error_inspection.py`: pure grid/state/evaluation/export API.
- `xrayrecon/xrayrecon/gui/error_inspection.py`: PyQt workspace and GL adapter.
- `xrayrecon/xrayrecon/alvarez_macovski.py`: preserve run/grid/reference-label provenance.
- `xrayrecon/xrayrecon/gui/app.py`: navigation and reference-label integration.
- `xrayrecon/xrayrecon/gui/am_plot_window.py`: inspection entry and accurate quality/residual labels.
- `xrayrecon/xrayrecon/tests/test_error_inspection.py`: numerical and export regression tests.
- `xrayrecon/xrayrecon/tests/test_error_inspection_gui.py`: linked interaction and async tests.
- `xrayrecon/xrayrecon/tests/test_embedded_registration.py`: application entry and provenance regression.
- `README.md`, `xrayrecon/ARCHITECTURE.md`, `xrayrecon/ERROR_INSPECTION.md`: entry point, architecture, workflow, scientific contract, and Phase 2 backlog.

Delivery was made on `gpu-test` without commits or remote actions. Task changes
are uncommitted. The pre-existing untracked `codex-error-inspection-prompt.md`,
`ct-error-inspection-wireframe.png`, and `xrayrecon/configs/jonathan_board_2.json`
were intentionally left untouched. Generated review screenshots, cached fixture,
and preview script are confined to the ignored `build/` directory.

## Window sizing and slice-layout correction (2026-10-21)

The analysis pages now temporarily hide both viewer docks (Tools and slice plot)
and the volume-specific status labels. Returning to the volume workspace restores
the docks' previous visibility without moving or redocking them. The long Gerber
status legend elides to available space; its full text remains in a tooltip.
Both analysis pages use the full available central window width.

The quantitative overview now retains its six image axes and six colorbars.
Slice/output/quality changes update artists in place; they no longer clear the
figure and repeatedly call `tight_layout`, which previously shifted the same
slice's plots on successive redraws. Fixed grid cells and aspect-preserving axes
dividers resize with the Qt canvas. Compact scientific tick formatting and wrapped
titles keep labels within the canvas. Slice navigation has its own row; the
metric-summary height stays stable and its tooltip includes the full text.
Reference ranges for the existing fidelity metrics are cached per loaded result,
avoiding full-volume min/max scans on every slice. Numerical definitions and source
arrays are unchanged.

- **G1 complete — usable window sizing.** Depends on the existing main-window
  stack and docking. Acceptance: overview and inspection fill available width
  at desktop/compact sizes and restore prior dock visibility. Verified by an
  integrated Qt regression and native hidden-window screenshots at 1500×950 and
  900×650 logical pixels. Viewer redesign is excluded.
- **G2 complete — stationary slice plots.** Depends on G1 and existing AM fields.
  Acceptance: repeated slice/variant changes preserve axes identity, geometry,
  and canvas placement; labels remain in bounds after resizing and sources are
  unchanged. Verified with deterministic multi-scale fields, result replacement,
  and native screenshots. Reconstruction and metric changes are excluded.

Changes for this correction are in `gui/app.py`, `gui/am_plot_window.py`,
`tests/test_analysis_layout.py`, and this document. The targeted application,
registration, inspection, numerical-metric, and new layout run passed **44 tests**.
Review screenshots are in the ignored `build/ui-layout-review/` directory. The
tests explicitly load a local desktop font on Windows because Qt's offscreen
plugin otherwise substitutes missing glyphs with inflated size estimates.

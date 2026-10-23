# Codex implementation prompt: CT / CAD / AM error inspection

Paste everything below the divider into Codex in the application's repository. Attach `ct-error-inspection-wireframe.png` and, if available, the current simulation screenshot. The wireframe is a layout specification, not a source of numerical data. This prompt implements the core first and includes a follow-on specification for advanced diagnostics.

---

You are working in an existing X-Ray CAD-CAM application for simulating and analyzing PCB X-ray CT, CAD models, and Alvarez–Macovski (AM) material decomposition. Implement an interactive **Error inspection** workspace integrated with the existing application.

## Objective and execution scope

Turn the finished simulation screen into a tool that answers: **where is the discrepancy, how large is it, what kind is it, and which physical feature does it affect?** Human inspection and automated evaluation must use the same numerical data and metric definitions.

The current screen includes ground-truth and raw-inversion maps of effective atomic number and relative electron density, a reconstruction-quality indicator, a projection-residual map, a slice slider, and a route back to a separate 3D volume workspace. Raw and material-regularized outputs may both exist. Preserve those capabilities and underlying scientific outputs.

**Implement Phase 1 end to end in this run.** Phase 2 below defines the subsequent expansion and the data hooks Phase 1 should leave available; do not expand this first change into an unfinished implementation of every diagnostic. If I explicitly request the full roadmap, complete Phase 1 and then implement Phase 2 in coherent increments.

Begin by reading applicable repository instructions and inspecting the current simulation UI, volume renderer, AM result containers, coordinate conventions, export paths, and tests. Identify the actual UI framework and reuse it. Do not assume a web application, introduce a replacement renderer, or migrate frameworks based on the wireframe. Give a short implementation plan and then carry out the work. Resolve routine design choices independently; ask only for information that materially blocks a scientifically valid implementation.

## Phase 1: connected inspection workspace

### 1. Layout and navigation

Follow the numbered regions in the attached wireframe while adapting styling and widgets to the existing application:

1. **Comparison context:** reference selector; reconstruction/run selector; raw versus regularized output selector; quantity selector; alignment/units/availability status.
2. **Navigation and scope:** plane, slice position, selection scope, material filter, boundary/interior/background filter, and probe/line/ROI interaction tools. Display slice position in physical units when available.
3. **Reference and reconstruction views:** shared intensity window, linked pan/zoom, synchronized crosshairs, orientation labels, and physical aspect ratio.
4. **Error view:** signed error by default, with absolute error and above-tolerance modes; fixed color scale; explicit units; validity overlay.
5. **Linked 3D context:** existing volume/mesh renderer showing the inspection plane, selected point or ROI, and an error overlay where supported. Use clipping and opacity controls to expose internal errors.
6. **Inspector:** selected voxel values, region/material identity, metric scope, valid coverage, bias, MAE, RMSE, 95th-percentile absolute error, and percentage above tolerance.
7. **Analysis panel:** switch between a line profile and signed-error histogram. A selected histogram range highlights corresponding voxels. The per-slice trend view is Phase 2 unless trivial to integrate after all core requirements pass.
8. **Exports and provenance:** export numerical arrays, masks, measurements, and the inspection state used to produce them.

Use resizable panes/docks. Keep images large enough for inspection; collapse the inspector or analysis panel on smaller windows. Keep detailed metadata in the inspector rather than long text across the images. Show (Z_eff) and relative electron density as separately selectable quantities with independent units, ranges, and tolerances. Reuse the existing overview if it is useful for simultaneously comparing both properties.

### 2. Linked interactions

- Selecting a voxel in any 2D view updates the other crosshairs, the inspector, and a world-space marker in 3D.
- Selecting a world-space location in the existing 3D renderer updates the relevant slice and crosshairs. If the renderer cannot provide picking without a major replacement, implement the 2D-to-3D direction and a documented integration adapter; report this specific limitation.
- Changing plane or slice preserves the world-space selection where meaningful, with documented clamping when it leaves the volume. Pan and zoom remain synchronized for aligned 2D views, but do not force the 3D camera to reset.
- A two-point line tool samples reference and reconstruction at the same world-space positions. Plot both values against physical distance and allow a separate signed-difference plot. Keep property values and residuals on clearly labeled axes. Preserve invalid samples as gaps.
- A rectangular ROI tool updates the map selection and inspector. Explicitly label whether an ROI covers the current slice or a 3D box. Clearing the ROI restores the selected scope.
- Material selection uses existing CAD/material labels where available. For quantitative per-material metrics, use the declared reference labels rather than silently switching masks based on the reconstruction.
- Histogram brushing creates an additional numerical selection with visible endpoints and units. Apply it consistently to slice and 3D highlighting. Expose a clear-selection action. Keep the base-region summary visible or clearly distinguish it from the brushed subset to avoid circular statistics.
- The tolerance control has a numerical field and slider, in the selected quantity's units. Changing tolerance updates highlighting and the above-tolerance fraction, not the underlying reconstruction or unthresholded error metrics.
- Raw/regularized toggles update all dependent views and metrics atomically. Discard stale asynchronous results if the user changes runs, quantities, masks, or transforms.

### 3. Scientific definitions

Use real numerical arrays, never colors or screenshots, for evaluation. For reconstruction R and reference T on a documented comparison grid:

    signed_error(x) = R(x) - T(x)
    absolute_error(x) = abs(signed_error(x))
    mean_bias = mean(signed_error over valid selected voxels)
    MAE = mean(absolute_error over valid selected voxels)
    RMSE = sqrt(mean(signed_error^2 over valid selected voxels))
    P95 = 95th percentile of absolute_error over valid selected voxels
    above_tolerance = count(abs(error) > tau) / count(valid selected voxels)

Define and record the percentile method and the strict `>` threshold convention. Return a valid count and coverage alongside every metric. Empty or entirely invalid selections produce `N/A` and a reason, never zero or an apparent pass. Report coverage as valid selected voxels / expected selected voxels within the declared evaluation domain; report excluded non-overlap separately when relevant.

- **Reference identity:** distinguish synthetic truth, nominal CAD, and measured reference. Nominal CAD deviations are not automatically proof of simulation error or manufacturing defects.
- **Material compatibility:** CAD geometry alone does not supply electron density, attenuation, or effective atomic number. Quantitative material comparison requires explicit material-property assignments compatible with the AM model and the same effective-Z convention. If missing, keep geometry inspection available and explain why a property comparison is unavailable. Do not invent material values.
- **Units and coordinates:** preserve spacing, origin, orientation/direction matrix, axis ordering, handedness, and physical units. Use the application's established voxel-to-world mapping. Do not assume isotropic voxels or `[x,y,z]` array storage.
- **Alignment:** preserve known synthetic alignment. Reuse an existing registration transform for independently acquired data, show its provenance, and require an explicit alignment decision when needed. Do not silently optimize alignment to make the errors look better; report nominal and registered comparisons separately when applicable.
- **Sampling:** use a declared comparison grid and explicit resampling policy. Never interpolate categorical material IDs as continuous values. Use an appropriate continuous-field resampler for scalar properties and preserve invalid regions. Respect physical aspect ratios in all planes. Account for CAD rasterization and partial volume when forming a reference; record the policy.
- **Regions:** support interiors, boundary bands, and background separately. Define boundary-band width in physical units and record it. Do not automatically discard all boundaries or background. Missing/extra-material analysis later needs coverage of the union of relevant supports, not only their intersection.
- **Validity:** combine field availability, finite values, field-of-view coverage, and existing solver-validity flags. Display invalid voxels using a distinct hatch/overlay, never as zero error. Do not hide poor-quality but numerically valid results through an undocumented quality threshold.
- **Percentage error:** optional, and only when its denominator policy is explicit and scientifically meaningful. Mask near-zero references using a documented, quantity-specific threshold and report excluded coverage. The default view uses absolute property units.
- **Display independence:** reference/reconstruction share intensity limits; signed errors have a symmetric zero-centered diverging scale. Lock ranges across slices and compared runs; an explicit reset-to-data-range action is acceptable. Clipping or display downsampling must never modify full-resolution statistics. Show when display values saturate.
- **Tolerance:** unless a verified acceptance specification is supplied, label the control as an exploratory tolerance rather than a certified pass/fail limit. Do not invent acceptable manufacturing tolerances.
- **Confidence:** preserve the distinction between heuristic quality indicators and calibrated uncertainty. Do not call a 0–1 indicator a probability or claim confidence intervals without a validated model. Error, quality, and projection residual are separate fields.
- **Projection residual:** preserve its actual data domain and definition. If the existing map is backprojected, label it explicitly. It indicates model/data inconsistency and is not a voxelwise ground-truth error map or a unique localization of its cause.

### 4. Data and state architecture

Adapt these concepts to existing code rather than adding a parallel framework:

- A comparison context references immutable source artifacts, run IDs, output variants, the transform, comparison-grid metadata, available quantities, units, reference material labels, and validity masks.
- A shared inspection state owns selected quantity, plane, world-space cursor, ROI, material/region filters, metric scope, scale limits, tolerance, histogram selection, and display mode.
- A pure numerical evaluation layer computes fields, metrics, line samples, and histogram bins independently of rendering. UI, exports, and batch evaluation call this same layer.
- Cache keys include source versions/IDs, quantity, output variant, transform/grid, and mask policy. A presentation-only change should not rerun the reconstruction. A tolerance change should reuse cached errors.
- Keep CPU evaluation supported. Reuse existing acceleration only where justified by measured workload. Avoid repeated host/device transfers and multiple full-volume copies. Process large-volume metrics in chunks or through the existing volume abstraction. Label any sampled or approximate metric rather than presenting it as exact.
- Run expensive work outside the UI thread using the existing scheduling mechanism. Support cancellation and prevent stale results from replacing current selections. Keep interactive display resolution separate from quantitative resolution.
- Avoid new heavy dependencies or training pipelines. Keep the first implementation reviewable and avoid unrelated refactors.

### 5. Export contract

Use the application's existing volume format where possible; otherwise choose an appropriate lossless numerical container and metadata sidecar. Export:

- Signed error, absolute error, valid mask, selected-region mask, and above-tolerance mask.
- Scalar metrics and region/material identifiers in machine-readable JSON and/or CSV.
- Line-profile coordinates and values when a line is present.
- Source/run IDs, synthetic versus nominal reference type, raw/regularized variant, quantity, units, spacing, origin, orientation, axis ordering, transform, resampling policy, ROI/scope, validity policy, boundary width, tolerance, denominator rules, and metric definitions.
- Optional annotated PNG for people, clearly separated from numerical arrays.

Exported arrays retain invalid samples through a validity mask and a consistent missing-value representation. Export metadata must be sufficient to reproduce the displayed result; a round trip must preserve coordinate interpretation. Do not overwrite source simulation results.

## Phase 2: subsequent diagnostic work

Keep these as a documented backlog with clear prerequisites. Do not add dead production controls or fill missing data with synthetic values. When a capability is unavailable, show a concise reason in the capability/diagnostic description.

### A. Geometry and missing/extra material

- Compare reference and reconstructed material masks or surfaces in physical units.
- Show signed surface deviation when a reliable inside/outside convention is available. Define its sign; check watertightness/orientation or fall back to unsigned distances with an explanation.
- Compute both reconstruction-to-reference and reference-to-reconstruction distances so omitted features are detectable. Preserve the direction of each metric. Report mean and percentile distances and a documented symmetric summary.
- Separately classify missing material, extra material, and wrong material. Handle empty masks and unmatched components explicitly. Show affected volume, bounding box, centroid, and nearest CAD feature where that association is reliable.
- Add a clickable discrepancy list linked to slices and 3D. Group in physical 3D space using a stated connectivity rule; make minimum size filtering explicit and retain raw detections in exports.
- Add trace width, hole diameter, or via continuity measurements only when the relevant geometry is resolved and the measurement method is defined. Record segmentation thresholds and sensitivity; do not claim sub-resolution certainty.

### B. Connectivity

- Use reference feature IDs/net information where available, or a documented geometric centerline/graph representation.
- Identify candidate breaks, unintended bridges, disconnected vias, and merges. Account for PCB layers and known interlayer connections; do not infer a bridge from overlap in a 2D projection.
- Use overlap/centerline scores as supporting evidence, not as substitutes for graph connectivity checks. Distinguish geometric candidate defects from verified electrical connectivity.
- Make segmentation and connectivity conventions explicit; avoid silently smoothing away the defect being evaluated.

### C. Material-property scatter and quality calibration

- Linked (Z_eff)-versus-relative-electron-density scatter/density plot with points or aggregates colored by reference material. Lasso selection maps back to full-resolution voxel IDs or a reproducible selection mask.
- Show expected material values when known and per-material bias, covariance, and confusion. Do not fabricate class centers or calibrated ellipses from a heuristic confidence score.
- Where synthetic truth exists, plot observed error versus the current quality indicator. Calibrated uncertainty requires appropriate repeated simulations/noise modeling and validation; it is not a relabeling of the quality map.

### D. Projection diagnostics and slice trends

- Preserve separate residual sinograms per energy, in the actual measurement domain: counts, intensity, or log projections. Define model minus data consistently.
- Link selected angle/detector positions to acquisition rays using known geometry. A ray highlight indicates contribution, not unique causation.
- Noise-normalized residuals require a justified variance/covariance model. Do not estimate uncertainty from a noise-free forward-model label.
- Plot error metrics against slice position using identical scale and mask definitions; clicking a position opens that slice. Support comparing raw and regularized outputs under the same reference and evaluation policy.

## Acceptance criteria and verification

Use small deterministic numerical fixtures for scientific correctness and the existing UI test framework where available. Prioritize tests that catch meaningful errors:

1. Identical fields produce zero residual, zero error metrics, and zero above-tolerance fraction on valid voxels.
2. A constant known offset produces the expected sign, bias, MAE, RMSE, percentile, and tolerance classification. Equality to the tolerance is not counted by the chosen strict convention.
3. Opposite-sign offsets can cancel in mean bias while MAE/RMSE remain nonzero.
4. NaNs, invalid solver output, and non-overlap reduce reported coverage without becoming zero error. Entirely invalid/empty selections return `N/A`.
5. An anisotropic, nontrivially oriented fixture verifies world/voxel round trips, slice mapping, physical aspect ratios, cross-view selection, and physical line distances.
6. Material and boundary filters select the intended voxels, while background and missed features remain inspectable in their appropriate modes.
7. Changing raw/regularized output, run, transform, or quantity invalidates dependent results. Delayed work cannot replace a newer state.
8. Windowing, zoom, clipping, display downsampling, and tolerance changes leave unthresholded quantitative errors unchanged.
9. Known line samples and ROI statistics agree with direct numerical evaluation. Histogram selection and 3D highlights use the same selection.
10. An export round trip preserves values, masks, coordinate metadata, and evaluation settings.

For Phase 2, additionally test a known physical surface displacement in both directions, a missing feature, an extra feature, an empty surface, a trace break, and a cross-layer near-overlap that must not be classified as a bridge.

Use one representative existing simulation for integration and responsiveness checks. Confirm that the UI stays responsive while volume metrics are computed, then stop additional benchmarking unless a specific risk remains. Inspect the real interface at the normal desktop size and a smaller supported window size for clipped controls and unreadable labels.

## Required completion report

Summarize what is working, how to open it, changed files, tests/checks and their results, any scientific or integration limitations, and the remaining Phase 2 backlog. Provide screenshots of the implemented workspace if the environment supports them. Do not claim full 3D linkage, calibrated uncertainty, physical measurement accuracy, or projection-ray linkage unless actually implemented and verified.

The attached wireframe uses schematic geometry and placeholder measurements. Match its interaction structure, not its fictional data. Keep the existing application's visual language and numerical pipeline intact.

# Hover values and Gerber diagnostics

Delivery objective: remove the color-pixel hover exception and reduce misleading
Gerber warning repetition without changing reconstruction inputs or precision.
The user approved implementation after a read-only investigation.

## Delivery goals

| Goal | Status | Dependencies | Acceptance and verification | Non-goals |
| --- | --- | --- | --- | --- |
| G1: Display scalar and color hover values safely | Complete | Existing plot and layer viewer | Headless checks of scalar, RGB, RGBA, hidden images and bounds; source arrays remain identical | Image conversion, numerical processing, reconstruction changes |
| G2: Report scoped Gerber diagnostics | Complete | Existing PyGerber adapter | Warnings carry source filename and DPI; redundant units and empty circles are distinguished from conflicts and lost nonzero features; masks, spacing and material labels equal the original renderer exactly; unrelated logging is preserved | Source rewriting, coordinate-format changes, automatic DPI changes, parser upgrades |

## Findings and constraints

The viewer can return a channel vector for RGB/RGBA images. Applying a scalar
format such as `:.1f` to that vector raises the reported exception. Formatting
each channel affects the label only.

In installed PyGerber 2.4.3, a repeated unit command produces an override warning
even when it declares the same units. An explicit zero-diameter aperture hole
also produces a raster warning despite having no pixels to draw. A nonzero
circle that collapses at the selected DPI is a different case and must retain a
warning. The source coordinate precision recommendation must also remain
visible: changing an FS header alone can rescale every coordinate.

The adapter must retain the existing DPI, parsing, bounding boxes, alpha masks,
physical spacing and voxel/material construction. No source files are rewritten.

## Validation record

Before implementation, the existing spacing, material-label and app-smoke checks
passed: 19 tests. Synthetic Gerber experiments showed identical masks after
removing only a redundant G71 declaration or an explicit zero-diameter hole.
Increasing DPI for a small positive aperture changed its pixels, so that action
is excluded from this change.

G1 verification: `test_pixel_hover.py` and `test_app_smoke.py`: **16 passed**.
The 37 dependency deprecation warnings match the existing baseline.

Final targeted run: **71 passed** (12.01 seconds), including hover, Gerber
diagnostics, spacing, material labels, app smoke and registration state. The
91 Python warnings were existing S3A/pyqtgraph dependency deprecations.

```powershell
python -m pytest xrayrecon/xrayrecon/tests/test_pygerber_diagnostics.py xrayrecon/xrayrecon/tests/test_pixel_hover.py xrayrecon/xrayrecon/tests/test_spacing_metadata.py xrayrecon/xrayrecon/tests/test_material_labels.py xrayrecon/xrayrecon/tests/test_app_smoke.py xrayrecon/xrayrecon/tests/test_registration_state.py -q
```

There are 21 bit-exact comparisons with the original installed PyGerber backend
and parser at 300, 508 and 1000 DPI. Fixtures cover legacy declarations, empty and
positive holes, tiny circles, traces, rectangular apertures, regions, imperial
units and conflicting declarations. The reference deliberately uses raw
`handle.result`, matching the existing adapter; `get_image()` flips vertically
and would change the application's coordinate convention.

The end-to-end two-layer stackup comparison verifies identical physical/material
volumes, material IDs, slice assignments, names, visual metadata and voxel
spacing, and confirms source-file bytes remain unchanged. Logging checks verify
unknown warnings, coordinate-format override warnings, errors, other threads,
nested renders and cleanup after failures. The targeted tracked-file
`git diff --check` passed. No configured lint/type checker was found; optional
Black/Ruff tools were not installed.

## Behavior and changed files

- `xrayrecon/gui/app.py`: scalar pixels retain one decimal place; RGB/RGBA pixels
  display each channel with one decimal place. Gerber imports provide filename
  context to the rasterizer.
- `xrayrecon/pygerber_integration.py`: per-render diagnostic summaries include
  filename, DPI and occurrence count. Precision recommendations, unit conflicts
  and nonzero circles lost at raster DPI stay at WARNING. Legacy unit commands,
  redundant declarations of the same units and empty circles are INFO findings.
  The normal WARNING logging threshold hides those benign notices. Findings are
  also available through `rasterizer.diagnostics` for the most recent render.
  The logging filter is restricted to known PyGerber warning messages within the
  active rendering context, and is always removed afterward. An empty circle's
  drawing command and bounding box remain intact; only its no-op draw is skipped.
- `xrayrecon/stackup_voxelize.py`: each layer supplies its resolved path as
  diagnostic context; the existing DPI and voxelization operations are retained.
- `xrayrecon/tests/test_pixel_hover.py` and
  `xrayrecon/tests/test_pygerber_diagnostics.py`: focused regression coverage.
- `GERBER_DIAGNOSTICS.md`: delivery goals, constraints and validation record.

Example warning:

```text
Gerber top.gbr (300 DPI): Nonzero-diameter circle components could not be drawn at this raster DPI. Review feature sizes and DPI; the raster DPI is preserved. (count=1)
```

Validation used synthetic Gerber fixtures and headless UI checks with installed
PyGerber 2.4.3. The user's actual board files were unavailable in the workspace,
so they were not tested. Source precision and positive features lost at low DPI
remain properties of the original data/rendering settings, with warnings to
support review. This change does not improve their geometry by changing inputs.

## Repository state

Active branch: `gpu-test`, starting HEAD `f42921d`. Earlier GPU, timing, AM metrics,
benchmark and registration changes were already uncommitted at approval and are
preserved. The six files listed above are this task's uncommitted work; only the
two relevant GUI hunks were added to the already modified `app.py`. Earlier
README, benchmark, packaging, simulation, calibration, plot, editor and
registration changes, along with their new CUDA/metrics files, remain intact.
No commits were created, no files were staged and no remote actions were taken.

# Xray Reconstruction and Layer Segmentation Tool
![Graphical interface](./xrayrecon/interface.jpg)

A tool for working with 3D X-ray CT volumes of Printed Circuit Boards (PCBs).
It loads real CT data and/or a Gerber PCB design, renders both in a
dual-pane 3D viewer, and can register the Gerber model's known geometry onto
real CT data to auto-label materials voxel-by-voxel. Individual layers can
also be annotated through [S3A](https://gitlab.com/s3a/s3a) to indicate
netlist-specific information.

## Documentation
- [Architecture Overview](./xrayrecon/ARCHITECTURE.md)

## Installation
Clone the repository and install the package and test dependencies in editable mode:
```bash
git clone <repo-url>
cd Sandia_Project
pip install -e ".[test]"
```

Dependencies: PyQt5, PyQtGraph, qtextras, S3A, NumPy, SciPy, scikit-image,
OpenCV, pandas, matplotlib, Pillow, pygerber, tifffile, fire.

## Running the App
```bash
# Launch with phantom data (no file required)
python -m xrayrecon.main

# Launch with a .npy volume file
python -m xrayrecon.main --datafile /path/to/volume.npy

# Launch with a directory of image slices
python -m xrayrecon.main --datafile /path/to/image_dir/
```

The toolbar's "Tools" panel also has demo actions that need no data of your
own: **Load Phantom Demo**, and **Debug: Load Synthetic Registration Demo**
(loads a known-misaligned Gerber/CT pair so the registration tool can be
exercised against a known-correct answer).

## Running Tests
```bash
python -m pytest
```

The canonical test configuration lives in `pyproject.toml`. Core algorithm
tests do not require a display; Qt tests set the offscreen platform in their
shared fixture. Use `python -m pytest -m "not slow"` for the normal development
regression loop.

## Key Features
- **Dual-pane 3D viewer** (`gui/app.py`): real CT data on the left
  (`viewer_3d_left`), Gerber-derived model on the right (`viewer_3d_right`),
  each with independent camera/threshold/opacity controls.
- **Gerber stackup + materials** (`gui/stackup_editor.py`, `stackup.py`,
  `voxelizer.py`, `materials.py`): assign a material to every Gerber layer
  and voxelize into a 3D volume.
- **Embedded registration** (`registration.py`, `gui/registration.py`): an
  in-application 5-DOF (XYZ translate + Z-rotation + uniform scale) workspace
  with a live red/green overlay, bounded initial-alignment suggestion, explicit
  acceptance, and stale-input protection.
- **Label transfer**: once registered, the Gerber model's known per-voxel
  materials are resampled onto the real CT grid to auto-label it.
- **Simulated X-ray / CT bridge** (`ct_simulation.py`): a true Radon +
  filtered-back-projection simulation of a Gerber stackup, for validating
  registration/label-transfer logic on a perfectly pre-registered pair
  before tackling real, misaligned CT data.
- **Alvarez-Macovski dual-energy decomposition** (`alvarez_macovski.py`,
  `gui/am_plot_window.py`): recovers Z_eff / electron-density per voxel from
  a simulated dual-energy scan, with volume colorization and material
  classification against the materials database.

## Package Structure
```
xrayrecon/
├── main.py                  # Entry point; generates phantom data if no file given
├── io.py                    # Load .npy files or image directories into 3D arrays
├── io_spectral.py           # Load spectral/energy-decomposed ("ED") CT data
├── layerseg.py              # Layer segmentation via template matching + peak detection
├── stackup.py                # PCB stackup representation (StackupConfig)
├── stackup_voxelize.py       # Pure (no-Qt) Gerber stackup -> voxel volume pipeline
├── voxelizer.py               # Convert 2D stackups (Gerber) to 3D voxel grids
├── convert.py                # Data conversion utilities
├── materials.py               # Material property definitions (Z_eff, density, colors)
├── material_seg.py            # Threshold-based material segmentation of a volume
├── multimaterial.py           # Multi-material config (per-threshold color/material)
├── registration.py            # Rigid+scale transform, mm-based volume/label resampling
├── synthetic_registration.py  # Deterministic registration demos/regression cases
├── ct_simulation.py           # Radon + FBP simulation of a real CT scan
├── alvarez_macovski.py        # Dual-energy decomposition, colorization, classification
├── gerber_utils.py            # Gerber file parsing helpers
├── pygerber_integration.py    # Gerber file rasterization via pygerber
├── cubecorrections/
│   ├── rotation.py            # Volume rotation utilities
│   └── emptyslices.py         # Empty-slice detection/removal
├── configs/                   # Example Gerber stackup / multi-material JSON configs
├── tests/                     # pytest suite (registration math/state, materials, spacing, smoke)
└── gui/
    ├── app.py                # MainWin — main window, dual-pane coordination hub
    ├── layermodel.py         # LayerModel — data cube, layer metadata, voxel spacing
    ├── layerview.py          # 2D slice visualization
    ├── view3d.py             # PCBViewer3D — OpenGL 3D volume rendering
    ├── table.py              # Layer selection table
    ├── stackup_editor.py     # Non-modal Gerber layer/material assignment table
    ├── spacing_dialog.py     # Dialog for entering real-CT voxel spacing (mm/voxel)
    ├── registration.py       # Embedded RegistrationPanel workspace
    ├── am_plot_window.py     # Alvarez-Macovski ground-truth-vs-recovered comparison plot
    └── helpers.py            # Shared GUI utilities and phantom data generation
```

## Confidence-Guided Registration

Open **Register Gerber to Real CT...** after loading both volumes and confirming
their voxel spacing. The main workspace switches to the embedded registration
panel, which continuously displays an alignment score from `0` to `1`.
**Suggest Initial Alignment** performs a deterministic,
bounded search over XYZ translation, in-plane rotation, and uniform scale, then
places the best result in the existing controls for manual review. It does not
accept the transform or transfer material labels.

The score correlates robustly normalized intensity and boundary features from
the fixed CT and resampled Gerber volumes. It is intended only to compare
candidate transforms for the same volume pair. The displayed low/moderate/high
confidence is a conservative summary of the score and its improvement over the
identity transform; it is not a probability, accuracy guarantee, or substitute
for inspecting the red/green overlay.

The interactive search is intentionally limited to translations within 25% of
the fixed-volume extent, rotations within 15 degrees, and scales from 0.8 to
1.2. Registrations outside those ranges, boards with repeated/symmetric
features, cropped data, weak contrast, or large Gerber/CT material-intensity
differences may produce a low-confidence or incorrect suggestion. Adjust the
five controls directly to override it, use **Reset** to return to identity, and
select **Accept** only after reviewing the overlay. Label transfer remains a
separate explicit action in the main window.

Use **Back to Main Workspace** to return to the normal dual-pane view. The
current edit is preserved while the CT/Gerber inputs and physical spacing remain
unchanged; replacing either dataset invalidates stale accepted registration.

See [`xrayrecon/REGRESSION_TESTING.md`](xrayrecon/REGRESSION_TESTING.md) for the
test tiers, synthetic-case contract, warning policy, and CI expectations.

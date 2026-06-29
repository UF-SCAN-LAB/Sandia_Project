# Xray Reconstruction and Layer Segmentation Tool
![Graphical interface](./xrayrecon/interface.jpg)

A tool for converting 3D X-ray PCB volumes into 2D layer renderings. Individual
layers can be annotated through [S3A](https://gitlab.com/s3a/s3a) to indicate
netlist-specific information.

## Documentation
- [Architecture Overview](./xrayrecon/ARCHITECTURE.md)

## Installation
Clone the repository and install the package in editable mode:
```bash
git clone <repo-url>
cd Sandia_Project
pip install -e ./xrayrecon
```

Dependencies: PyQt5, PyQtGraph, S3A, NumPy, SciPy, OpenCV, fire.

## Running the App
```bash
# Launch with phantom data (no file required)
python -m xrayrecon.main

# Launch with a .npy volume file
python -m xrayrecon.main --datafile /path/to/volume.npy

# Launch with a directory of image slices
python -m xrayrecon.main --datafile /path/to/image_dir/
```

## Package Structure
```
xrayrecon/
├── main.py                  # Entry point; generates phantom data if no file given
├── io.py                    # Load .npy files or image directories into 3D arrays
├── layerseg.py              # Layer segmentation via template matching + peak detection
├── stackup.py               # PCB stackup representation
├── voxelizer.py             # Convert 2D stackups (Gerber) to 3D voxel grids
├── convert.py               # Data conversion utilities
├── materials.py             # Material property definitions
├── rotation.py              # Volume rotation utilities
├── emptyslices.py           # Empty-slice detection/removal
├── gerber_utils.py          # Gerber file parsing helpers
├── pygerber_integration.py  # Gerber file rasterization via pygerber
└── gui/
    ├── app.py               # MainWin — main window and coordination hub
    ├── layermodel.py        # LayerModel — holds the data cube and layer metadata
    ├── layerview.py         # 2D slice visualization
    ├── view3d.py            # PCBViewer3D — OpenGL 3D volume rendering
    ├── table.py             # Layer selection table
    └── helpers.py           # Shared GUI utilities and phantom data generation
```

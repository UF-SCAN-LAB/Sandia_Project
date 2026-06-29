# Xray Reconstruction and Layer Segmentation Tool
![Graphical interface](./interface.jpg)

A tool for converting 3D X-ray PCB volumes into 2D layer renderings. These individual
layers can be annoated through [S3A](https://gitlab.com/s3a/s3a) to indicate 
layers can be annoated through [S3A](https://gitlab.com/s3a/s3a) to indicate 
netlist-specific information.

## Documentation
- [Architecture Overview](./ARCHITECTURE.md)


## Installation
Installation can be accomplished using pip:
```bash
git clone https://github.com/UF-SCAN-LAB/xray-recon.git
pip install -e ./xrayrecon
```

## Running the App
The app can be started by running `main.py`:
```bash
python ./xrayrecon/main.py [--datafile /path/to/datafile.npy]

# Or, if xrayrecon is installed:
python -m xrayrecon.main [--datafile /path/to/datafile.npy]
```

If you have a folder with PCB data and slice information, uncomment the lines that read
in those files and run `main.py`.

"""
Prepare lizard_recon.npz as a .npz cube + material LUT for the xray-recon app.
Applies ring/holder correction, normalises to uint8, and builds a per-voxel
RGBA colour table that maps:
  background  (< tissue_u8)  →  black / fully transparent
  soft tissue (tissue_u8 .. bone_u8) →  orange, semi-transparent
  bone/dense  (>= bone_u8)   →  ivory, nearly opaque
Output shape: (nx, ny, nz) so z (axial slices) is the last axis.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter

NPZ = r"E:\git\Spectral-Xray\lizard_recon.npz"
OUT = r"E:\backup\Sandia\xray-recon\lizard_anat.npz"

TISSUE_LO = 0.0035
BONE_LO   = 0.0130
HEAD_R    = 28.0


def ring_correct(raw):
    vol = median_filter(raw.astype(np.float64), size=3)
    vol = gaussian_filter(vol, sigma=0.8)
    nz, ny, nx = vol.shape
    cy, cx = ny/2-.5, nx/2-.5
    Y, X = np.mgrid[:ny, :nx]
    R = np.hypot(X-cx, Y-cy)
    Ri = np.round(R).astype(int)
    nb = Ri.max()+1
    cnt = np.maximum(np.bincount(Ri.ravel(), None, nb), 1)
    anat = np.empty_like(vol)
    for z in range(nz):
        rm = np.bincount(Ri.ravel(), vol[z].ravel(), nb) / cnt
        anat[z] = vol[z] - rm[Ri]
    anat = np.clip(anat, 0, None)
    anat[:, R > HEAD_R] = 0.0
    anat = gaussian_filter(anat, sigma=0.8)
    hi = np.percentile(anat[anat > 0], 99.5)
    return np.clip(anat / hi, 0, 1), hi


def make_lut(tissue_u8, bone_u8):
    lut = np.zeros((256, 4), dtype=np.uint8)
    # background: transparent
    lut[:tissue_u8] = [0, 0, 0, 0]
    # soft tissue: orange (#d98c6a), alpha ramps 60→180
    for v in range(tissue_u8, bone_u8):
        t = (v - tissue_u8) / max(bone_u8 - tissue_u8, 1)
        alpha = int(60 + t * 120)
        lut[v] = [217, 140, 106, alpha]
    # bone/dense: ivory (#f2ead6), alpha ramps 200→255
    for v in range(bone_u8, 256):
        t = (v - bone_u8) / max(255 - bone_u8, 1)
        alpha = int(200 + t * 55)
        lut[v] = [242, 234, 214, alpha]
    return lut


d = np.load(NPZ)
anat, anat_hi = ring_correct(d["vol_br"])   # (nz, ny, nx), normalised 0-1
cube = (anat * 255).astype(np.uint8)
cube = cube.transpose(2, 1, 0)              # -> (nx, ny, nz): z/axial is last axis

tissue_u8 = int(round(TISSUE_LO / anat_hi * 255))
bone_u8   = int(round(BONE_LO   / anat_hi * 255))
lut = make_lut(tissue_u8, bone_u8)

np.savez_compressed(OUT, vol=cube, lut=lut)
print(f"Saved {OUT}")
print(f"  vol shape={cube.shape}  dtype={cube.dtype}")
print(f"  tissue_u8={tissue_u8}  bone_u8={bone_u8}")
print(f"  lut shape={lut.shape}")

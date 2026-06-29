"""Export lizard CT data as base64-encoded uint8 volumes for the HTML widget."""
import numpy as np
import io, base64, json
from scipy.ndimage import gaussian_filter, median_filter

NPZ = r"E:\git\Spectral-Xray\lizard_recon.npz"
TISSUE_LO = 0.0035
BONE_LO   = 0.0130
HEAD_R    = 28.0


def ring_correct(vol_raw):
    vol = median_filter(vol_raw.astype(np.float64), size=3)
    vol = gaussian_filter(vol, sigma=0.8)
    nz, ny, nx = vol.shape
    cy, cx = ny / 2 - 0.5, nx / 2 - 0.5
    Y, X = np.mgrid[0:ny, 0:nx]
    R = np.hypot(X - cx, Y - cy)
    Ri = np.round(R).astype(int)
    nb = Ri.max() + 1
    counts = np.maximum(np.bincount(Ri.ravel(), None, nb), 1)
    anat = np.empty_like(vol)
    for z in range(nz):
        rmean = np.bincount(Ri.ravel(), vol[z].ravel(), nb) / counts
        anat[z] = vol[z] - rmean[Ri]
    anat = np.clip(anat, 0, None)
    anat[:, R > HEAD_R] = 0.0
    anat = gaussian_filter(anat, sigma=0.8)
    hi = np.percentile(anat[anat > 0], 99.5)
    anat = np.clip(anat, 0, hi)
    return anat, hi


def norm_u8(arr):
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - mn) / (mx - mn) * 255).astype(np.uint8)


def b64(arr):
    return base64.b64encode(arr.astype(np.uint8).tobytes()).decode()


d = np.load(NPZ)
vol_br = d["vol_br"].astype(np.float64)
vol_lo = d["vol_lo"].astype(np.float64)
vol_hi = d["vol_hi"].astype(np.float64)

anat, anat_hi = ring_correct(vol_br)

# Spectral ratio: high-energy / low-energy attenuation ratio
ratio = np.where(vol_lo > 1e-4, vol_hi / vol_lo, 0.0)

# Segmentation labels (0=bg, 1=tissue, 2=bone)
seg = np.zeros(anat.shape, dtype=np.uint8)
seg[anat >= TISSUE_LO] = 1
seg[anat >= BONE_LO]   = 2

nz, ny, nx = anat.shape

out = {
    "shape": [nz, ny, nx],
    "voxel_mm": float(d["voxel_mm"]),
    "tissue_lo_u8": int(TISSUE_LO / anat_hi * 255),
    "bone_lo_u8":   int(BONE_LO   / anat_hi * 255),
    "vols": {
        "anat":  b64(norm_u8(anat)),
        "broad": b64(norm_u8(vol_br)),
        "lo":    b64(norm_u8(vol_lo)),
        "hi":    b64(norm_u8(vol_hi)),
        "ratio": b64(norm_u8(ratio)),
        "seg":   b64(seg),
    }
}

print(json.dumps(out))

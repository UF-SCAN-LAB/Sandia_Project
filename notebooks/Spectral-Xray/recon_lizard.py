"""
Reconstruct the hyperspectral lizard-head cone-beam CT scan and segment it into
material regions for 3D visualization.

Data (E:\\Data\\hyperspectral_data\\4737468):
  lizard_head_sinogram_full.mat : (det_v=80, angle=180, det_u=80, energy=200) counts
  FF.mat        : flat field (80, 80, 200)  -- open-beam I0 per detector pixel/energy
  Energy_axis.mat : E_axis (200,) keV, 0.94 .. 56 keV

Geometry: source-sample 332 mm, sample-detector 270 mm, det pixel 0.25 mm, 180
angles x 2 deg. Cone/fan half-angle ~1 deg, so we reconstruct slice-by-slice with
parallel-beam FBP per detector row (good approximation at this cone angle).
"""
import os
import numpy as np
import scipy.io as sio
from skimage.transform import iradon

DATA = r"E:\Data\hyperspectral_data\4737468"
OUT = r"E:\git\Spectral-Xray"

# energy bands (keV)
LOW_BAND = (15.0, 28.0)
HIGH_BAND = (33.0, 52.0)
BROAD_BAND = (15.0, 52.0)


def load():
    E = sio.loadmat(os.path.join(DATA, "Energy_axis.mat"))["E_axis"].ravel()
    FF = sio.loadmat(os.path.join(DATA, "FF.mat"))["FF_lizard"]            # (v,u,E)
    sino = sio.loadmat(os.path.join(DATA, "lizard_head_sinogram_full.mat"))[
        "lizard_head_sinogram_full"]                                       # (v,ang,u,E)
    return E, FF, sino


def band_attenuation(sino, FF, E, band):
    """Sum counts over an energy band, normalise by flat field, take -log."""
    sel = (E >= band[0]) & (E <= band[1])
    I = sino[:, :, :, sel].sum(axis=3).astype(np.float64)     # (v,ang,u)
    I0 = FF[:, :, sel].sum(axis=2).astype(np.float64)          # (v,u)
    T = I / np.maximum(I0[:, None, :], 1e-9)                   # broadcast over angle
    T = np.clip(T, 1e-6, 1.0)
    return -np.log(T)                                          # (v,ang,u) line integrals


def reconstruct(atten, theta):
    """Parallel-beam FBP per detector row. atten: (v, ang, u) -> vol (v, u, u)."""
    nv, nang, nu = atten.shape
    vol = np.zeros((nv, nu, nu), dtype=np.float32)
    for r in range(nv):
        sslice = atten[r].T                                    # (u, ang)
        rec = iradon(sslice, theta=theta, filter_name="hann", circle=True)
        vol[r] = np.clip(rec, 0, None)
    return vol


def main():
    print("Loading .mat files (~1.6 GB)...", flush=True)
    E, FF, sino = load()
    print("  sino", sino.shape, "FF", FF.shape, flush=True)
    theta = np.arange(0, 360, 2.0)                             # 180 angles
    assert len(theta) == sino.shape[1]

    print("Building band attenuations + reconstructing...", flush=True)
    vol_lo = reconstruct(band_attenuation(sino, FF, E, LOW_BAND), theta)
    vol_hi = reconstruct(band_attenuation(sino, FF, E, HIGH_BAND), theta)
    vol_br = reconstruct(band_attenuation(sino, FF, E, BROAD_BAND), theta)
    print("  volume shape", vol_br.shape, flush=True)

    np.savez_compressed(os.path.join(OUT, "lizard_recon.npz"),
                        vol_lo=vol_lo, vol_hi=vol_hi, vol_br=vol_br,
                        voxel_mm=0.25 * 332 / (332 + 270))
    print("Saved lizard_recon.npz", flush=True)
    for name, v in [("low", vol_lo), ("high", vol_hi), ("broad", vol_br)]:
        nz = v[v > 0]
        print(f"  {name}: min0 {nz.min():.3f} max {v.max():.3f} "
              f"mean(>0) {nz.mean():.3f} p99 {np.percentile(v, 99):.3f}", flush=True)


if __name__ == "__main__":
    main()

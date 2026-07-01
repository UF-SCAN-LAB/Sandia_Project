"""Alvarez-Macovski dual-energy CT decomposition, adapted from the proven
implementation in notebooks/Spectral-Xray/{alvarez_macovski_demo.ipynb,
make_am_process_diagram.py} for use on Gerber-Voxelizer-derived volumes.

Model: mu(x, y, E) = a1(x,y)*f1(E) + a2(x,y)*f2(E), where f1(E) = E^-3
(photoelectric) and f2(E) = Klein-Nishina Compton cross-section (energy
only). a2 is proportional to electron density; a1/a2 gives an effective
atomic number via a calibrated power law (PE_EXPONENT).

The notebook's own PCB phantom deliberately uses compressed, non-physical
Z_eff values to keep materials numerically separable at its 80/140 kVp
energy pair (its own comments say so). Real materials.py values (Cu Z=29,
solder Z=60, gold Z=79) span a much wider range, so this module re-calibrates
K_REF_Z and the energy pair for a higher-energy industrial-CT-like regime,
and uses a log-spaced LUT for decomposition (a per-ray fsolve polish was
tried and dropped -- too slow across a full sinogram, ~5.5s/slice vs.
~0.17s/slice for the LUT alone, and prone to unphysical divergence). Z_eff
recovery is noisier for low-Z, weakly-absorbing materials like FR4 -- a
genuine dual-energy-CT limitation, not a bug.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import griddata
from scipy.optimize import fsolve

from .materials import MATERIALS_DB, MATERIAL_COLORS
from skimage.transform import iradon, radon

PE_EXPONENT = 3.8
K_REF_Z = 29.0  # copper as the reference Z -- keeps a1 ~ a2 order of magnitude there


def f1(E):
    """Photoelectric basis function."""
    return E ** (-3)


def klein_nishina(E):
    """Klein-Nishina Compton scattering cross-section (normalized). E in keV."""
    alpha = E / 511.0
    term1 = (1 + alpha) / alpha ** 3
    term2 = (2 * alpha * (1 + alpha)) / (1 + 2 * alpha) - np.log(1 + 2 * alpha)
    term3 = np.log(1 + 2 * alpha) / (2 * alpha)
    term4 = (1 + 3 * alpha) / (1 + 2 * alpha) ** 2
    return term1 * term2 + term3 - term4


_KN_REF = klein_nishina(50.0)


def f2(E):
    """Compton (Klein-Nishina) basis function, normalized to 1 at 50 keV."""
    return klein_nishina(E) / _KN_REF


def make_spectrum(E_min, E_max, n_bins=200):
    """Triangular bremsstrahlung spectrum approximation."""
    E = np.linspace(E_min, E_max, n_bins)
    phi = np.maximum(0, (E_max - E) / (E_max - E_min))
    phi /= np.trapz(phi, E)
    return E, phi


# Industrial PCB dual-energy CT range -- real Cu/solder/gold need much higher
# energy than the notebook's illustrative 80/140 kVp medical-CT-like pair.
E_LO, PHI_LO = make_spectrum(60, 160)
E_HI, PHI_HI = make_spectrum(60, 320)


def forward_arr(A1_arr, A2_arr, E, phi):
    """Vectorised polychromatic Beer-Lambert over arrays of (A1, A2)."""
    A1_arr = np.asarray(A1_arr, float).ravel()
    A2_arr = np.asarray(A2_arr, float).ravel()
    exponent = -np.outer(A1_arr, f1(E)) - np.outer(A2_arr, f2(E))
    transmitted = np.trapz(phi * np.exp(exponent), E, axis=1)
    return -np.log(np.clip(transmitted, 1e-12, None))


def to_a1a2(z_eff: np.ndarray, density: np.ndarray):
    """Ground-truth (z_eff, density) -> AM basis coefficients (a1, a2)."""
    a2 = density
    a1 = density * (z_eff / K_REF_Z) ** PE_EXPONENT
    return a1, a2


def to_zeff(a1: np.ndarray, a2: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """AM basis coefficients -> effective atomic number."""
    safe_a2 = np.where(a2 > eps, a2, 1.0)
    ratio = np.where(a2 > eps, a1 / safe_a2, 0.0)
    return np.where(a2 > eps, K_REF_Z * np.clip(ratio, 0, None) ** (1 / PE_EXPONENT), 0.0)


def to_rho_e(a2: np.ndarray) -> np.ndarray:
    """AM Compton coefficient -> electron density (a2 is defined as density directly)."""
    return a2


def _decompose_ray(l_lo, l_hi, a1_init, a2_init, a1_max):
    """Per-ray fsolve polish (mirrors the notebook's decompose_ray). Only used
    for the small, targeted refinement in refine_slice_voxels below -- too
    slow to run per-ray across a full sinogram (measured ~5.5s/slice for
    ~10,000 rays, vs. ~0.15s/slice for the pure LUT lookup)."""
    def residuals(p):
        a1, a2 = p
        if a1 < 0 or a2 < 0:
            return [1e6, 1e6]
        return [
            forward_arr([a1], [a2], E_LO, PHI_LO)[0] - l_lo,
            forward_arr([a1], [a2], E_HI, PHI_HI)[0] - l_hi,
        ]

    sol = fsolve(residuals, [a1_init, a2_init])
    return float(np.clip(sol[0], 0.0, a1_max)), float(max(sol[1], 0.0))


def _build_lut(a1_max: float, a2_max: float):
    """Log-spaced A1 grid: real photoelectric coefficients span ~5 orders of
    magnitude (air/FR4 vs. copper/solder/gold), which a linear grid cannot
    resolve."""
    a1_max = max(a1_max, 1e-6)
    a2_max = max(a2_max, 1e-6)
    A1g = np.concatenate([[0.0], np.logspace(np.log10(a1_max * 1e-4), np.log10(a1_max), 160)])
    A2g = np.linspace(0.0, a2_max, 100)
    AA1, AA2 = np.meshgrid(A1g, A2g, indexing="ij")
    Llo_grid = forward_arr(AA1, AA2, E_LO, PHI_LO)
    Lhi_grid = forward_arr(AA1, AA2, E_HI, PHI_HI)
    pts = np.column_stack([Llo_grid, Lhi_grid])
    return pts, AA1.ravel(), AA2.ravel()


def _decompose_sinogram(sino_Llo: np.ndarray, sino_Lhi: np.ndarray, a1_max: float, a2_max: float):
    """LUT-based (griddata) decomposition -- same approach the notebook's own
    full end-to-end pipeline uses for whole-sinogram decomposition (per-ray
    fsolve is reserved there for a handful of illustrative single rays, not
    the full sinogram, for the same performance reason noted above).

    Returns (A1_rec, A2_rec) sinograms same shape as the inputs.
    """
    shape = sino_Llo.shape
    pts, a1_flat, a2_flat = _build_lut(a1_max, a2_max)
    query = np.column_stack([sino_Llo.ravel(), sino_Lhi.ravel()])

    def invert(values):
        lin = griddata(pts, values, query, method="linear")
        nan = ~np.isfinite(lin)
        if nan.any():
            lin[nan] = griddata(pts, values, query[nan], method="nearest")
        return lin

    A1_rec = np.clip(invert(a1_flat), 0, None)
    A2_rec = np.clip(invert(a2_flat), 0, None)
    return A1_rec.reshape(shape), A2_rec.reshape(shape)


def simulate_am_volume(vol: np.ndarray, num_angles: int | None = None,
                        pixel_size_cm: float = 0.01) -> dict:
    """Simulate an Alvarez-Macovski dual-energy CT scan of a Voxelizer volume.

    vol: (Z, H, W, 2) = [z_eff, density], as returned by
    Voxelizer.voxelize_stackup(). For each Z slice: convert ground truth to
    (a1, a2), Radon-project at two energies via the polychromatic
    Beer-Lambert forward model, decompose back to (a1_hat, a2_hat) via a
    LUT (griddata), FBP-reconstruct (circle=False -- required to avoid
    corner-clipping of features that reach the layout edges, same
    consideration as ct_simulation.py), then convert to (Z_eff, rho_e).

    Recovered Z_eff/rho_e are clipped to a sane physical range: FBP ringing
    can leave a2 (density) reconstructed near-zero at isolated voxels while
    a1 doesn't hit zero at the same spot, and since Z_eff = K*(a1/a2)^(1/n)
    is a ratio, that produces wild, unphysical outliers (observed up to
    ~700 in testing, vs. a real max of ~79 for gold) that would otherwise
    dominate the plot's color scale.

    Returns a dict with ground-truth and recovered (Z, H, W) volumes for
    Z_eff and rho_e, ready for side-by-side comparison.
    """
    z_depth, h, w, _ = vol.shape
    if num_angles is None:
        num_angles = max(h, w)
    theta = np.linspace(0.0, 180.0, num_angles, endpoint=False)

    z_eff_vol = vol[..., 0]
    density_vol = vol[..., 1]
    a1_vol, a2_vol = to_a1a2(z_eff_vol, density_vol)

    a1_max_bound = a1_vol.max() * max(h, w) * pixel_size_cm * 1.5
    a2_max_bound = a2_vol.max() * max(h, w) * pixel_size_cm * 1.5

    zeff_gt = to_zeff(a1_vol, a2_vol)
    rhoe_gt = to_rho_e(a2_vol)
    zeff_clip = float(zeff_gt.max()) * 1.5 if zeff_gt.max() > 0 else K_REF_Z * 3
    rhoe_clip = float(rhoe_gt.max()) * 1.5 if rhoe_gt.max() > 0 else 1.0

    zeff_rec = np.zeros((z_depth, h, w), dtype=np.float32)
    rhoe_rec = np.zeros((z_depth, h, w), dtype=np.float32)

    for z in range(z_depth):
        a1_slice = a1_vol[z].astype(np.float32)
        a2_slice = a2_vol[z].astype(np.float32)

        sino_A1 = radon(a1_slice, theta=theta, circle=False) * pixel_size_cm
        sino_A2 = radon(a2_slice, theta=theta, circle=False) * pixel_size_cm

        sino_Llo = forward_arr(sino_A1, sino_A2, E_LO, PHI_LO).reshape(sino_A1.shape)
        sino_Lhi = forward_arr(sino_A1, sino_A2, E_HI, PHI_HI).reshape(sino_A1.shape)

        sino_A1_rec, sino_A2_rec = _decompose_sinogram(
            sino_Llo, sino_Lhi, a1_max_bound, a2_max_bound
        )

        recon_a1 = np.clip(
            iradon(sino_A1_rec, theta=theta, filter_name="ramp", circle=False) / pixel_size_cm,
            0, None,
        )
        recon_a2 = np.clip(
            iradon(sino_A2_rec, theta=theta, filter_name="ramp", circle=False) / pixel_size_cm,
            0, None,
        )

        zeff_rec[z] = np.clip(to_zeff(recon_a1, recon_a2), 0, zeff_clip)
        rhoe_rec[z] = np.clip(to_rho_e(recon_a2), 0, rhoe_clip)

    return dict(
        zeff_ground_truth=zeff_gt.astype(np.float32),
        rhoe_ground_truth=rhoe_gt.astype(np.float32),
        zeff_recovered=zeff_rec,
        rhoe_recovered=rhoe_rec,
    )


def colorize_volume(scalar_vol: np.ndarray, cmap_name: str, background_frac: float = 0.02) -> np.ndarray:
    """Map a (Z, H, W) scalar volume through a matplotlib colormap into an
    (H, W, Z, 4) uint8 RGBA volume, ready for PCBViewer3D.setRGBADataCube.

    Voxels at or below background_frac * vmax (e.g. air/background) are made
    nearly transparent so the coloured material stands out in 3D, matching
    the alpha conventions used elsewhere in this app's RGBA volumes.
    """
    import matplotlib as mpl

    vmax = float(scalar_vol.max())
    if vmax <= 0:
        vmax = 1.0
    norm = np.clip(scalar_vol / vmax, 0, 1)

    cmap = mpl.colormaps[cmap_name]
    rgba_zhw = (cmap(norm) * 255).astype(np.uint8)  # (Z, H, W, 4)

    is_background = scalar_vol <= (vmax * background_frac)
    rgba_zhw[..., 3] = np.where(is_background, 5, 200).astype(np.uint8)

    return np.moveaxis(rgba_zhw, 0, 2)  # -> (H, W, Z, 4)


# --- Material lookup table (legend + nearest-match classification) ---------
#
# materials.MATERIALS_DB already holds reference (Z_eff, electron_density)
# for AIR/COPPER/FR4/SOLDER/GOLD/SILVER -- physically standard textbook
# values consistent with NIST's tabulated element/compound data
# (https://www.nist.gov/pml/x-ray-mass-attenuation-coefficients). Rather than
# re-deriving attenuation curves from the raw NIST tables (a much bigger
# undertaking than this demo needs), we use those reference points directly
# as a lookup table: classify each voxel's noisy AM-recovered (Z_eff, rho_e)
# by nearest match, to get an interpretable approximate material label.

MATERIAL_NAMES: list[str] = sorted(MATERIALS_DB.keys())
_REFERENCE_ZEFF = np.array([MATERIALS_DB[n].z_eff for n in MATERIAL_NAMES])
_REFERENCE_RHOE = np.array([MATERIALS_DB[n].electron_density for n in MATERIAL_NAMES])
# Normalize each axis by its own dynamic range across the reference materials
# before computing distance -- Z_eff (~0-79) and density (~0-19.3 g/cm^3)
# are on different scales and would otherwise let one axis dominate.
_ZEFF_SCALE = max(float(_REFERENCE_ZEFF.max()), 1e-9)
_RHOE_SCALE = max(float(_REFERENCE_RHOE.max()), 1e-9)


def classify_material_volume(zeff_vol: np.ndarray, rhoe_vol: np.ndarray) -> np.ndarray:
    """Classify each voxel's (Z_eff, rho_e) to the nearest reference material
    in MATERIALS_DB (normalized Euclidean distance in (Z_eff, rho_e) space).

    Returns an integer label volume the same shape as zeff_vol, indexing into
    MATERIAL_NAMES.
    """
    zeff_flat = (zeff_vol.ravel() / _ZEFF_SCALE)[:, None]
    rhoe_flat = (rhoe_vol.ravel() / _RHOE_SCALE)[:, None]
    ref_zeff = (_REFERENCE_ZEFF / _ZEFF_SCALE)[None, :]
    ref_rhoe = (_REFERENCE_RHOE / _RHOE_SCALE)[None, :]

    dist2 = (zeff_flat - ref_zeff) ** 2 + (rhoe_flat - ref_rhoe) ** 2
    return np.argmin(dist2, axis=1).reshape(zeff_vol.shape)


def colorize_by_material(zeff_vol: np.ndarray, rhoe_vol: np.ndarray) -> np.ndarray:
    """Classify every voxel to its nearest reference material and colorize
    using that material's canonical color (materials.MATERIAL_COLORS) --
    an interpretable 'approximate material' view of a noisy AM decomposition,
    in contrast to the continuous Z_eff/rho_e colormaps.
    """
    labels = classify_material_volume(zeff_vol, rhoe_vol)
    z_depth, h, w = labels.shape
    rgba_zhw = np.zeros((z_depth, h, w, 4), dtype=np.uint8)
    for idx, name in enumerate(MATERIAL_NAMES):
        rgba_zhw[labels == idx] = MATERIAL_COLORS[name]
    return np.moveaxis(rgba_zhw, 0, 2)  # -> (H, W, Z, 4)

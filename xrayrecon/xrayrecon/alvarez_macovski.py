"""Alvarez-Macovski dual-energy CT decomposition, adapted from the proven
implementation in notebooks/Spectral-Xray/{alvarez_macovski_demo.ipynb,
make_am_process_diagram.py} for use on Gerber-Voxelizer-derived volumes.

Model: mu(x, y, E) = a1(x,y)*f1(E) + a2(x,y)*f2(E), where
f1(E) = (50 keV / E)^3
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
from scipy.ndimage import gaussian_filter, median_filter, sobel
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.optimize import fsolve

from .am_acquisition import AMAcquisitionConfig, AMSpectrum, PairedProjectionData
from .materials import (
    MATERIALS_DB,
    MATERIAL_COLORS,
    MATERIAL_NAMES,
    WATER_ELECTRON_DENSITY_E23_CM3,
)
from skimage.transform import iradon, radon

PE_EXPONENT = 3.8
K_REF_Z = 29.0  # copper as the reference Z -- keeps a1 ~ a2 order of magnitude there
PE_REFERENCE_ENERGY_KEV = 50.0


class AMResult(dict):
    """Array-only result mapping with acquisition provenance as attributes.

    Keeping metadata out of mapping values preserves compatibility with callers
    that iterate over every returned volume while still making provenance and
    calibration state impossible to lose in the supported API.
    """

    def __init__(
        self,
        *args,
        acquisition_config: AMAcquisitionConfig,
        has_ground_truth: bool,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.acquisition_config = acquisition_config
        self.source_kind = acquisition_config.source_kind
        self.calibration_id = acquisition_config.calibration_id
        self.has_ground_truth = bool(has_ground_truth)


def f1(E):
    """Photoelectric basis, normalized to 1 at the shared 50 keV reference."""
    return (PE_REFERENCE_ENERGY_KEV / np.asarray(E, dtype=float)) ** 3


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


def forward_spectrum(
    A1_arr: np.ndarray, A2_arr: np.ndarray, spectrum: AMSpectrum
) -> np.ndarray:
    """Vectorized Beer-Lambert projection for a discrete normalized spectrum."""
    a1 = np.asarray(A1_arr, dtype=float).ravel()
    a2 = np.asarray(A2_arr, dtype=float).ravel()
    energies = np.asarray(spectrum.energies_kev, dtype=float)
    weights = np.asarray(spectrum.weights, dtype=float)
    exponent = -np.outer(a1, f1(energies)) - np.outer(a2, f2(energies))
    transmitted = np.sum(weights * np.exp(exponent), axis=1)
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


def _build_lut(
    a1_max: float, a2_max: float, config: AMAcquisitionConfig | None = None
):
    """Log-spaced A1 grid: real photoelectric coefficients span ~5 orders of
    magnitude (air/FR4 vs. copper/solder/gold), which a linear grid cannot
    resolve."""
    a1_max = max(a1_max, 1e-6)
    a2_max = max(a2_max, 1e-6)
    A1g = np.concatenate([[0.0], np.logspace(np.log10(a1_max * 1e-4), np.log10(a1_max), 160)])
    A2g = np.linspace(0.0, a2_max, 100)
    AA1, AA2 = np.meshgrid(A1g, A2g, indexing="ij")
    if config is None:
        Llo_grid = forward_arr(AA1, AA2, E_LO, PHI_LO)
        Lhi_grid = forward_arr(AA1, AA2, E_HI, PHI_HI)
    else:
        Llo_grid = forward_spectrum(AA1, AA2, config.low_spectrum)
        Lhi_grid = forward_spectrum(AA1, AA2, config.high_spectrum)
    pts = np.column_stack([Llo_grid, Lhi_grid])
    return pts, AA1.ravel(), AA2.ravel()


def _build_lut_inverter(
    a1_max: float, a2_max: float, config: AMAcquisitionConfig | None = None
):
    """Build reusable joint interpolators for both AM basis coefficients."""
    points, a1_values, a2_values = _build_lut(a1_max, a2_max, config)
    values = np.column_stack([a1_values, a2_values])
    return (
        LinearNDInterpolator(points, values, fill_value=np.nan),
        NearestNDInterpolator(points, values),
    )


def _decompose_sinogram(
    sino_Llo: np.ndarray,
    sino_Lhi: np.ndarray,
    a1_max: float,
    a2_max: float,
    *,
    lut=None,
    config: AMAcquisitionConfig | None = None,
):
    """LUT-based (griddata) decomposition -- same approach the notebook's own
    full end-to-end pipeline uses for whole-sinogram decomposition (per-ray
    fsolve is reserved there for a handful of illustrative single rays, not
    the full sinogram, for the same performance reason noted above).

    Returns (A1_rec, A2_rec) sinograms same shape as the inputs.
    """
    shape = sino_Llo.shape
    linear, nearest = lut or _build_lut_inverter(a1_max, a2_max, config)
    query = np.column_stack([sino_Llo.ravel(), sino_Lhi.ravel()])
    recovered = linear(query)
    invalid = ~np.isfinite(recovered).all(axis=1)
    if invalid.any():
        recovered[invalid] = nearest(query[invalid])

    A1_rec = np.clip(recovered[:, 0], 0, None)
    A2_rec = np.clip(recovered[:, 1], 0, None)
    return A1_rec.reshape(shape), A2_rec.reshape(shape)


def _match_reconstruction_shape(
    reconstruction: np.ndarray, target_shape: tuple[int, int]
) -> np.ndarray:
    """Center-crop/pad square ``iradon`` output to the source slice shape.

    With ``circle=False``, scikit-image pads rectangular inputs to a square
    detector footprint. ``iradon`` consequently returns a square image even
    when the original slice was rectangular. The physical object remains
    centered, so a symmetric crop (or defensive pad for unusual versions) is
    the inverse geometry needed before stacking reconstructed slices.
    """
    target_h, target_w = (int(target_shape[0]), int(target_shape[1]))
    source_h, source_w = reconstruction.shape
    output = np.zeros((target_h, target_w), dtype=reconstruction.dtype)

    copy_h = min(source_h, target_h)
    copy_w = min(source_w, target_w)
    source_y = max(0, (source_h - copy_h) // 2)
    source_x = max(0, (source_w - copy_w) // 2)
    target_y = max(0, (target_h - copy_h) // 2)
    target_x = max(0, (target_w - copy_w) // 2)
    output[target_y : target_y + copy_h, target_x : target_x + copy_w] = (
        reconstruction[
            source_y : source_y + copy_h,
            source_x : source_x + copy_w,
        ]
    )
    return output


def _poisson_log_measurement(
    line_integral: np.ndarray,
    photon_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a seeded photon-counting measurement in negative-log space."""
    transmission = np.exp(-np.asarray(line_integral, dtype=np.float64))
    measured_counts = rng.poisson(transmission * photon_count)
    measured_transmission = np.clip(
        measured_counts / float(photon_count),
        1.0 / float(photon_count),
        1.0,
    )
    return -np.log(measured_transmission)


def _robust_unit_interval(values: np.ndarray) -> np.ndarray:
    """Robustly map a nonnegative image to [0, 1] without metadata state."""
    array = np.nan_to_num(np.asarray(values, dtype=np.float32), copy=False)
    lower, upper = np.percentile(array, (5.0, 95.0))
    if upper <= lower + 1e-9:
        maximum = float(array.max())
        if maximum <= 0:
            return np.zeros_like(array, dtype=np.float32)
        lower, upper = 0.0, maximum
    return np.clip((array - lower) / (upper - lower), 0.0, 1.0).astype(
        np.float32
    )


def _quality_maps(
    recon_a1: np.ndarray,
    recon_a2: np.ndarray,
    measured_a1: np.ndarray,
    measured_a2: np.ndarray,
    theta: np.ndarray,
    pixel_size_cm: float,
    target_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate projection inconsistency and reconstruction confidence.

    The residual is calculated by forward-projecting the reconstructed basis
    images and backprojecting their mismatch with the decomposed AM sinograms.
    This is equivalent to a reconstruction data-consistency check without
    redundantly evaluating the expensive polychromatic spectrum model twice.
    Confidence combines that term with local edge sensitivity and signal
    strength. It is an interpretable quality indicator, not a probability.
    """
    projected_a1 = radon(recon_a1, theta=theta, circle=False) * pixel_size_cm
    projected_a2 = radon(recon_a2, theta=theta, circle=False) * pixel_size_cm
    a1_scale = max(float(np.percentile(np.abs(measured_a1), 90)), 1e-6)
    a2_scale = max(float(np.percentile(np.abs(measured_a2), 90)), 1e-6)
    ray_residual = np.sqrt(
        0.5
        * (
            ((projected_a1 - measured_a1) / a1_scale) ** 2
            + ((projected_a2 - measured_a2) / a2_scale) ** 2
        )
    )
    residual_image = _match_reconstruction_shape(
        iradon(
            ray_residual,
            theta=theta,
            filter_name=None,
            circle=False,
        ),
        target_shape,
    )
    residual = _robust_unit_interval(gaussian_filter(residual_image, 1.0))

    zeff = to_zeff(recon_a1, recon_a2)
    zeff_scale = max(float(np.percentile(zeff, 95)), 1e-6)
    density_scale = max(float(np.percentile(recon_a2, 95)), 1e-6)
    edge_energy = np.hypot(
        sobel(zeff / zeff_scale, axis=0),
        sobel(zeff / zeff_scale, axis=1),
    ) + np.hypot(
        sobel(recon_a2 / density_scale, axis=0),
        sobel(recon_a2 / density_scale, axis=1),
    )
    edge_sensitivity = _robust_unit_interval(gaussian_filter(edge_energy, 0.75))
    photoelectric_scale = max(float(np.percentile(recon_a1, 95)), 1e-9)
    # Z_eff is recovered from a1/a2. Confidence therefore requires both basis
    # channels to carry signal; density alone would overstate confidence in
    # low-Z regions where the photoelectric coefficient is noise-dominated.
    signal = np.sqrt(
        np.clip(recon_a1 / photoelectric_scale, 0.0, 1.0)
        * np.clip(recon_a2 / density_scale, 0.0, 1.0)
    )
    confidence = (
        np.exp(-1.6 * residual - 0.8 * edge_sensitivity)
        * (0.2 + 0.8 * signal)
    )
    return residual, np.clip(confidence, 0.0, 1.0).astype(np.float32)


def simulate_am_projections(
    vol: np.ndarray, config: AMAcquisitionConfig
) -> PairedProjectionData:
    """Forward-project a property volume into validated low/high AM data."""
    vol = np.asarray(vol)
    if vol.ndim != 4 or vol.shape[-1] != 2:
        raise ValueError("AM volume must have shape (Z, H, W, 2)")
    if config.source_kind != "simulated":
        raise ValueError("forward simulation requires source_kind='simulated'")
    rng = (
        np.random.default_rng(config.random_seed)
        if config.photon_count is not None
        else None
    )
    z_depth, h, w, _ = vol.shape
    a1_vol, a2_vol = to_a1a2(vol[..., 0], vol[..., 1])
    a1_max_bound = max(
        float(a1_vol.max()) * max(h, w) * config.pixel_size_cm * 1.5, 1e-6
    )
    a2_max_bound = max(
        float(a2_vol.max()) * max(h, w) * config.pixel_size_cm * 1.5, 1e-6
    )
    theta = config.theta_deg
    low_slices = []
    high_slices = []
    for z in range(z_depth):
        sino_a1 = radon(a1_vol[z].astype(np.float32), theta=theta, circle=False)
        sino_a2 = radon(a2_vol[z].astype(np.float32), theta=theta, circle=False)
        sino_a1 *= config.pixel_size_cm
        sino_a2 *= config.pixel_size_cm
        low = forward_spectrum(sino_a1, sino_a2, config.low_spectrum).reshape(
            sino_a1.shape
        )
        high = forward_spectrum(sino_a1, sino_a2, config.high_spectrum).reshape(
            sino_a1.shape
        )
        if config.photon_count is not None:
            low = _poisson_log_measurement(low, config.photon_count, rng)
            high = _poisson_log_measurement(high, config.photon_count, rng)
        low_slices.append(low.astype(np.float32))
        high_slices.append(high.astype(np.float32))
    return PairedProjectionData(
        np.stack(low_slices),
        np.stack(high_slices),
        theta,
        config,
        a1_max_bound,
        a2_max_bound,
    )


def reconstruct_am_projections(
    projections: PairedProjectionData,
    *,
    output_shape: tuple[int, int],
    zeff_clip: float | None = None,
    rhoe_clip: float | None = None,
) -> AMResult:
    """Reconstruct simulated or measured paired negative-log projections."""
    h, w = (int(output_shape[0]), int(output_shape[1]))
    if h <= 0 or w <= 0:
        raise ValueError("output_shape values must be positive")
    low_zhw = projections.low_log_zhw
    high_zhw = projections.high_log_zhw
    z_depth = low_zhw.shape[0]
    config = projections.config
    theta = projections.theta_deg
    zeff_limit = (
        float(zeff_clip)
        if zeff_clip is not None
        else float(_REFERENCE_ZEFF.max()) * 1.5
    )
    rhoe_limit = (
        float(rhoe_clip)
        if rhoe_clip is not None
        else float(_REFERENCE_RHOE.max()) * 1.5
    )
    zeff_rec = np.zeros((z_depth, h, w), dtype=np.float32)
    rhoe_rec = np.zeros((z_depth, h, w), dtype=np.float32)
    projection_residual = np.zeros((z_depth, h, w), dtype=np.float32)
    confidence = np.zeros((z_depth, h, w), dtype=np.float32)
    decomposition_lut = _build_lut_inverter(
        projections.a1_max_bound, projections.a2_max_bound, config
    )
    for z in range(z_depth):
        sino_a1_rec, sino_a2_rec = _decompose_sinogram(
            low_zhw[z],
            high_zhw[z],
            projections.a1_max_bound,
            projections.a2_max_bound,
            lut=decomposition_lut,
            config=config,
        )
        recon_a1 = np.clip(
            _match_reconstruction_shape(
                iradon(sino_a1_rec, theta=theta, filter_name="ramp", circle=False),
                (h, w),
            )
            / config.pixel_size_cm,
            0,
            None,
        )
        recon_a2 = np.clip(
            _match_reconstruction_shape(
                iradon(sino_a2_rec, theta=theta, filter_name="ramp", circle=False),
                (h, w),
            )
            / config.pixel_size_cm,
            0,
            None,
        )
        zeff_rec[z] = np.clip(to_zeff(recon_a1, recon_a2), 0, zeff_limit)
        rhoe_rec[z] = np.clip(to_rho_e(recon_a2), 0, rhoe_limit)
        projection_residual[z], confidence[z] = _quality_maps(
            recon_a1,
            recon_a2,
            sino_a1_rec,
            sino_a2_rec,
            theta,
            config.pixel_size_cm,
            (h, w),
        )
    recovered_labels, zeff_regularized, rhoe_regularized = (
        regularize_recovered_material_properties(zeff_rec, rhoe_rec)
    )
    property_disagreement = (
        np.abs(zeff_rec - zeff_regularized) / max(_ZEFF_SCALE, 1e-9)
        + np.abs(rhoe_rec - rhoe_regularized) / max(_RHOE_SCALE, 1e-9)
    )
    disagreement_penalty = np.exp(
        -2.5 * _robust_unit_interval(property_disagreement)
    )
    confidence = np.clip(confidence * disagreement_penalty, 0.0, 1.0).astype(
        np.float32
    )
    return AMResult(
        zeff_recovered=zeff_rec,
        rhoe_recovered=rhoe_rec,
        zeff_regularized=zeff_regularized,
        rhoe_regularized=rhoe_regularized,
        material_labels_recovered=recovered_labels,
        projection_residual=projection_residual,
        confidence=confidence,
        rhoe_e23_cm3_recovered=(
            rhoe_rec * WATER_ELECTRON_DENSITY_E23_CM3
        ).astype(np.float32),
        rhoe_e23_cm3_regularized=(
            rhoe_regularized * WATER_ELECTRON_DENSITY_E23_CM3
        ).astype(np.float32),
        acquisition_config=config,
        has_ground_truth=False,
    )


def simulate_am_volume(
    vol: np.ndarray,
    num_angles: int | None = None,
    pixel_size_cm: float = 0.01,
    *,
    photon_count: int | None = None,
    random_seed: int | None = None,
    config: AMAcquisitionConfig | None = None,
) -> AMResult:
    """Simulate and reconstruct a property volume through the AM pipeline.

    The legacy scalar arguments remain supported. New code should pass a
    validated ``AMAcquisitionConfig``. Supplying both forms is rejected so the
    physical acquisition is never ambiguous.
    """
    volume = np.asarray(vol)
    if volume.ndim != 4 or volume.shape[-1] != 2:
        raise ValueError("AM volume must have shape (Z, H, W, 2)")
    z_depth, h, w, _ = volume.shape
    del z_depth
    if config is not None:
        if (
            num_angles is not None
            or pixel_size_cm != 0.01
            or photon_count is not None
            or random_seed is not None
        ):
            raise ValueError("config cannot be combined with legacy acquisition arguments")
        acquisition = config
    else:
        acquisition = AMAcquisitionConfig.industrial_default(
            num_angles=max(h, w) if num_angles is None else num_angles,
            photon_count=photon_count,
            random_seed=random_seed,
        )
        if pixel_size_cm != acquisition.pixel_size_cm:
            acquisition = AMAcquisitionConfig.from_dict(
                {**acquisition.to_dict(), "pixel_size_cm": pixel_size_cm}
            )
    if acquisition.source_kind != "simulated":
        raise ValueError("simulate_am_volume requires source_kind='simulated'")
    projections = simulate_am_projections(volume, acquisition)
    zeff_gt = volume[..., 0].astype(np.float32)
    rhoe_gt = volume[..., 1].astype(np.float32)
    result = reconstruct_am_projections(
        projections,
        output_shape=(h, w),
        zeff_clip=float(zeff_gt.max()) * 1.5 if zeff_gt.max() > 0 else K_REF_Z * 3,
        rhoe_clip=float(rhoe_gt.max()) * 1.5 if rhoe_gt.max() > 0 else 1.0,
    )
    result.update(
        zeff_ground_truth=zeff_gt,
        rhoe_ground_truth=rhoe_gt,
        rhoe_e23_cm3_ground_truth=(
            rhoe_gt * WATER_ELECTRON_DENSITY_E23_CM3
        ).astype(np.float32),
    )
    result.has_ground_truth = True
    return result


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

_REFERENCE_ZEFF = np.array([MATERIALS_DB[n].z_eff for n in MATERIAL_NAMES])
_REFERENCE_RHOE = np.array([MATERIALS_DB[n].electron_density for n in MATERIAL_NAMES])
# Normalize each axis by its own dynamic range across the reference materials
# before computing distance -- Z_eff (~0-79) and density (~0-19.3 g/cm^3)
# are on different scales and would otherwise let one axis dominate.
_ZEFF_SCALE = max(float(_REFERENCE_ZEFF.max()), 1e-9)
_RHOE_SCALE = max(float(_REFERENCE_RHOE.max()), 1e-9)


def classify_material_volume(
    zeff_vol: np.ndarray,
    rhoe_vol: np.ndarray,
    *,
    zeff_weight: float = 1.0,
    rhoe_weight: float = 1.0,
) -> np.ndarray:
    """Classify each voxel's (Z_eff, rho_e) to the nearest reference material
    in MATERIALS_DB (normalized Euclidean distance in (Z_eff, rho_e) space).

    Returns an integer label volume the same shape as zeff_vol, indexing into
    MATERIAL_NAMES.
    """
    zeff_flat = (zeff_vol.ravel() / _ZEFF_SCALE)[:, None]
    rhoe_flat = (rhoe_vol.ravel() / _RHOE_SCALE)[:, None]
    ref_zeff = (_REFERENCE_ZEFF / _ZEFF_SCALE)[None, :]
    ref_rhoe = (_REFERENCE_RHOE / _RHOE_SCALE)[None, :]

    if zeff_weight < 0 or rhoe_weight < 0 or (zeff_weight + rhoe_weight) <= 0:
        raise ValueError("classification weights must be nonnegative and not both zero")
    dist2 = (
        float(zeff_weight) * (zeff_flat - ref_zeff) ** 2
        + float(rhoe_weight) * (rhoe_flat - ref_rhoe) ** 2
    )
    return np.argmin(dist2, axis=1).reshape(zeff_vol.shape)


def regularize_recovered_material_properties(
    zeff_recovered: np.ndarray,
    rhoe_recovered: np.ndarray,
    *,
    median_size: int = 5,
    zeff_weight: float = 0.001,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create stable approximate-material properties without hiding raw AM data.

    FBP ringing from very high-Z gold/tungsten can dominate the much smaller
    low-Z photoelectric coefficient in adjacent FR4 voxels. A 5x5 in-plane
    median on electron density suppresses those isolated streaks. Density is
    the more stable recovered channel, so nearest-reference classification
    uses it as the primary signal and raw Z_eff only as a weak tie-breaker.

    Returns ``(labels, zeff_regularized, rhoe_regularized)`` in the same
    spatial shape as the inputs. Callers must continue to expose the raw
    recovered arrays when quantitative decomposition output is required.
    """
    zeff = np.asarray(zeff_recovered, dtype=np.float32)
    rhoe = np.asarray(rhoe_recovered, dtype=np.float32)
    if zeff.shape != rhoe.shape:
        raise ValueError("Z-effective and electron-density volumes must match")
    if median_size < 1 or median_size % 2 == 0:
        raise ValueError("median_size must be a positive odd integer")

    if zeff.ndim >= 3:
        filter_size = (1,) * (zeff.ndim - 2) + (median_size, median_size)
    else:
        filter_size = (median_size,) * zeff.ndim
    rhoe_filtered = median_filter(rhoe, size=filter_size, mode="nearest")
    labels = classify_material_volume(
        zeff,
        rhoe_filtered,
        zeff_weight=zeff_weight,
        rhoe_weight=1.0,
    )

    reference_zeff = _REFERENCE_ZEFF.astype(np.float32)
    reference_rhoe = _REFERENCE_RHOE.astype(np.float32)
    return labels, reference_zeff[labels], reference_rhoe[labels]


def colorize_by_material(zeff_vol: np.ndarray, rhoe_vol: np.ndarray) -> np.ndarray:
    """Classify every voxel to its nearest reference material and colorize
    using that material's canonical color (materials.MATERIAL_COLORS) --
    an interpretable 'approximate material' view of a noisy AM decomposition,
    in contrast to the continuous Z_eff/rho_e colormaps.
    """
    labels = classify_material_volume(zeff_vol, rhoe_vol)
    return colorize_material_labels(labels)


def colorize_material_labels(label_vol: np.ndarray) -> np.ndarray:
    """Colorize an exact (Z, H, W) material-ID label volume -- e.g. the
    ground-truth labels `Voxelizer.voxelize_stackup(return_material_ids=True)`
    produces from known stackup materials, as opposed to `classify_material_volume`'s
    nearest-match guess for noisy real/AM-recovered data. Same MATERIAL_NAMES
    index convention as `classify_material_volume`, so labels from either
    source colorize identically.
    """
    z_depth, h, w = label_vol.shape
    rgba_zhw = np.zeros((z_depth, h, w, 4), dtype=np.uint8)
    for idx, name in enumerate(MATERIAL_NAMES):
        rgba_zhw[label_vol == idx] = MATERIAL_COLORS[name]
    return np.moveaxis(rgba_zhw, 0, 2)  # -> (H, W, Z, 4)

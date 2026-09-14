
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

WATER_ELECTRON_DENSITY_E23_CM3 = 3.343


@dataclass(frozen=True)
class MaterialProperties:
    """Reference properties used by the synthetic spectral-CT pipeline.

    ``relative_electron_density`` is electron density divided by the NIST
    liquid-water reference (approximately 3.343e23 electrons/cm3). It is not
    mass density. The old project model used g/cm3 as an electron-density
    proxy; ``mass_density_g_cm3`` preserves that distinct input explicitly.
    Values for compounds are engineering approximations because FR4, solder,
    and polyimide composition varies by formulation.
    """

    name: str
    z_eff: float  # Effective atomic number
    mass_density_g_cm3: float
    relative_electron_density: float | None = None

    def __post_init__(self) -> None:
        if self.relative_electron_density is None:
            # Backward-compatible construction for callers that still pass
            # only the historical (name, Z_eff, mass-density) triplet. The
            # fallback retains the former proxy semantics for that ad-hoc
            # object; canonical database entries always provide calibrated
            # relative electron density explicitly.
            object.__setattr__(
                self, "relative_electron_density", self.mass_density_g_cm3
            )

    @property
    def electron_density(self) -> float:
        """Compatibility name for relative electron density to water."""
        return float(self.relative_electron_density)

    @property
    def electron_density_e23_cm3(self) -> float:
        """Absolute electron density in units of 1e23 electrons/cm3."""
        return float(self.relative_electron_density) * WATER_ELECTRON_DENSITY_E23_CM3


# Mass densities follow the reference values already used by this project.
# Relative electron densities are derived from density and electron-per-mass
# composition, normalized to NIST liquid water. Compound entries are explicit
# approximations rather than falsely precise universal material constants.
# Reference water composition/density: https://physics.nist.gov/cgi-bin/Star/compos.pl?matno=276

MATERIALS_DB: Dict[str, MaterialProperties] = {
    "AIR": MaterialProperties("Air", 7.2, 0.0012, 0.0010),
    "COPPER": MaterialProperties("Copper", 29.0, 8.96, 7.38),
    "FR4": MaterialProperties("FR4", 6.5, 1.85, 1.70),
    "SOLDER": MaterialProperties("Solder", 60.0, 8.50, 6.20),
    "GOLD": MaterialProperties("Gold", 79.0, 19.30, 13.95),
    "SILVER": MaterialProperties("Silver", 47.0, 10.49, 8.24),
    "NICKEL": MaterialProperties("Nickel", 28.0, 8.90, 7.66),
    "TIN": MaterialProperties("Tin", 50.0, 7.31, 5.56),
    "LEAD": MaterialProperties("Lead", 82.0, 11.34, 8.10),
    "ALUMINUM": MaterialProperties("Aluminum", 13.0, 2.70, 2.35),
    "TITANIUM": MaterialProperties("Titanium", 22.0, 4.51, 3.74),
    "TUNGSTEN": MaterialProperties("Tungsten", 74.0, 19.25, 13.98),
    "ALUMINA": MaterialProperties("Alumina", 10.2, 3.95, 3.49),
    "POLYIMIDE": MaterialProperties("Polyimide", 6.5, 1.42, 1.35),
}


def get_material(name: str) -> MaterialProperties:
    return MATERIALS_DB.get(name.upper(), MATERIALS_DB["AIR"])


# Canonical display color per material, shared by the stackup editor's color
# pickers and the Alvarez-Macovski material legend/lookup so both draw from
# the same reference (RGBA, alpha carries default 3D-view opacity).
MATERIAL_COLORS: Dict[str, Tuple[int, int, int, int]] = {
    "AIR": (0, 0, 200, 5),
    "COPPER": (255, 140, 0, 200),
    "FR4": (80, 80, 80, 60),
    "SOLDER": (200, 200, 0, 200),
    "GOLD": (255, 215, 0, 200),
    "SILVER": (192, 192, 192, 200),
    "NICKEL": (140, 140, 150, 200),
    "TIN": (210, 210, 220, 180),
    "LEAD": (90, 90, 110, 200),
    "ALUMINUM": (180, 190, 200, 140),
    "TITANIUM": (120, 130, 140, 200),
    "TUNGSTEN": (70, 60, 90, 220),
    "ALUMINA": (230, 225, 210, 70),
    "POLYIMIDE": (200, 160, 60, 50),
}


# Canonical ground-truth material label ordering -- single source of truth
# shared by voxelizer.py (building label volumes) and alvarez_macovski.py
# (colorizing them), so a label volume's integer indices always mean the
# same thing regardless of which module produced or consumes them.
MATERIAL_NAMES: List[str] = sorted(MATERIALS_DB.keys())


def material_labels_to_zeff_rhoe(label_volume: np.ndarray) -> np.ndarray:
    """Map canonical IDs to ``[..., (Z_eff, rho_e / rho_e,water)]``.

    ``label_volume`` may use any spatial axis order. Its integer values must
    index ``MATERIAL_NAMES``, matching the voxelizer, showcase, AM classifier,
    and material colorizer. The returned float32 array appends a two-channel
    property axis and is directly consumable by ``simulate_am_volume`` once
    the spatial axes are in ``(Z, H, W)`` order.
    """
    labels = np.asarray(label_volume)
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError("material labels must use an integer dtype")
    if labels.size:
        minimum = int(labels.min())
        maximum = int(labels.max())
        if minimum < 0 or maximum >= len(MATERIAL_NAMES):
            raise ValueError(
                f"material labels must be in [0, {len(MATERIAL_NAMES) - 1}]"
            )

    properties = np.empty(labels.shape + (2,), dtype=np.float32)
    for material_id, name in enumerate(MATERIAL_NAMES):
        material = MATERIALS_DB[name]
        mask = labels == material_id
        properties[..., 0][mask] = material.z_eff
        properties[..., 1][mask] = material.electron_density
    return properties

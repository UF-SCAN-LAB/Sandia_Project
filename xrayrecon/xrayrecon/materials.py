
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class MaterialProperties:
    name: str
    z_eff: float  # Effective atomic number
    electron_density: float  # e-/A^3 (approx or relative)
             # Real physical units or relative scan units can be used. 
             # For simulation, physical units like 10^23 e-/cm^3 might be best, but normalized 0-1 is easier for visualization.

# Reference values (approximate)
# Z_eff values: Cu=29, FR4~6-7, Solder (Sn/Pb)~50-82 range, Air~7
# Density (g/cm^3) -> convert to electron density if needed, or just use mass density proxy.
# We will use mass density (g/cm^3) for simplicity in this intermediate format unless strictly e- density is requested.
# Prompt asked for "Effective nuclear charge and electron density".

MATERIALS_DB: Dict[str, MaterialProperties] = {
    "AIR": MaterialProperties("Air", 7.2, 0.0012),
    "COPPER": MaterialProperties("Copper", 29.0, 8.96),
    "FR4": MaterialProperties("FR4", 6.5, 1.85),
    "SOLDER": MaterialProperties("Solder", 60.0, 8.5), # Averaged for SnPb
    "GOLD": MaterialProperties("Gold", 79.0, 19.3),
    "SILVER": MaterialProperties("Silver", 47.0, 10.49)
}

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
}


def get_material(name: str) -> MaterialProperties:
    return MATERIALS_DB.get(name.upper(), MATERIALS_DB["AIR"])

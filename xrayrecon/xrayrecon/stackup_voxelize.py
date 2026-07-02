from __future__ import annotations

from pathlib import Path

import numpy as np

from .gerber_utils import GerberLayer
from .materials import MATERIALS_DB, get_material
from .pygerber_integration import PyGerberRasterizer
from .stackup import StackupConfig
from .voxelizer import Voxelizer


def material_or_error(name: str):
    if name.upper() not in MATERIALS_DB:
        available = ", ".join(sorted(MATERIALS_DB.keys()))
        raise ValueError(f"Unknown material '{name}'. Available materials: {available}")
    return get_material(name)


def voxelize_stackup_config(
    config: StackupConfig,
    stackup_path: Path,
    return_material_ids: bool = False,
):
    """Rasterize + voxelize every layer in *config*.

    Relative layer file paths are resolved against *stackup_path*'s parent
    directory (or config.gerber_dir); absolute layer file paths are used
    as-is regardless of *stackup_path*. Returns (vol, layer_names,
    layer_slices, visual_entries, spacing_mm) -- the same shape
    `MainWin._apply_gerber_volume` expects, plus the mm/voxel (x, y, z)
    spacing used for this voxelization (discarded internally by Voxelizer
    otherwise). Pure function, no Qt/GUI dependency, so it's usable directly
    from tests and MainWin alike.
    """
    if config.resolution_xy <= 0 or config.resolution_z <= 0:
        raise ValueError("resolution_xy and resolution_z must both be > 0")

    rasterizer = PyGerberRasterizer(dpi=config.dpi)
    substrate_mat = material_or_error(config.substrate_material)
    air_mat = material_or_error("AIR")

    stackup = []
    visual_entries = []
    layer_names = []
    layer_slices = []
    z_cursor = 0

    for layer_cfg in config.layers:
        layer_path = config.resolve_layer_path(stackup_path, layer_cfg.file)
        if not layer_path.exists():
            raise FileNotFoundError(
                f"Gerber file for layer '{layer_cfg.name}' not found: {layer_path}"
            )

        with open(layer_path, "r") as f:
            content = f.read()
        mask = rasterizer.rasterize(content)
        if np.max(mask) == 0:
            raise ValueError(f"Layer '{layer_cfg.name}' produced an empty raster mask.")

        layer = GerberLayer(layer_cfg.name)
        layer.raster_mask = mask

        h_px, w_px = mask.shape
        width_mm = w_px / (config.dpi / 25.4)
        height_mm = h_px / (config.dpi / 25.4)
        layer.update_bounds(0, 0)
        layer.update_bounds(width_mm, height_mm)

        feature_mat = material_or_error(layer_cfg.feature_material)
        bg_mat = material_or_error(layer_cfg.background_material)
        stackup.append((layer, layer_cfg.thickness, feature_mat, bg_mat))

        layer_num_slices = max(1, int(np.round(layer_cfg.thickness / config.resolution_z)))
        layer_names.append(layer_cfg.name)
        layer_slices.append(np.arange(z_cursor, z_cursor + layer_num_slices))
        z_cursor += layer_num_slices
        visual_entries.append(
            {
                "name": layer_cfg.name,
                "is_substrate": False,
                "num_slices": layer_num_slices,
                "color": layer_cfg.color,
            }
        )

        if layer_cfg.gap > 0:
            substrate_name = f"{layer_cfg.name}_substrate"
            substrate_layer = GerberLayer(substrate_name)
            substrate_layer.update_bounds(0, 0)
            substrate_layer.update_bounds(width_mm, height_mm)
            stackup.append((substrate_layer, layer_cfg.gap, substrate_mat, substrate_mat))

            gap_slices = max(1, int(np.round(layer_cfg.gap / config.resolution_z)))
            layer_names.append(substrate_name)
            layer_slices.append(np.arange(z_cursor, z_cursor + gap_slices))
            z_cursor += gap_slices
            visual_entries.append(
                {
                    "name": substrate_name,
                    "is_substrate": True,
                    "num_slices": gap_slices,
                    "color": (80, 80, 80, 10),
                }
            )

    vox = Voxelizer(resolution_xy=config.resolution_xy, resolution_z=config.resolution_z)
    result = vox.voxelize_stackup(
        stackup, air_mat, return_material_ids=return_material_ids
    )
    if return_material_ids:
        vol, material_ids = result
    else:
        vol, material_ids = result, None

    spacing_mm = (config.resolution_xy, config.resolution_xy, config.resolution_z)
    return vol, layer_names, layer_slices, visual_entries, spacing_mm, material_ids

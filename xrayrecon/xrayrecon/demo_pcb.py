"""A self-contained, portable, in-memory PCB stackup demonstrating multi-layer
fabrication finishes without touching external Gerber files.

Showcases:
- ENIG (Electroless Nickel Immersion Gold): Ni barrier + Au flash over copper traces
- HASL (Hot Air Solder Leveling): Sn + Pb on separate pads

Built directly from hand-crafted numpy raster masks, feeding directly into
Voxelizer.voxelize_stackup() for CT simulation and material ID volume generation.
"""

from __future__ import annotations

import numpy as np

from xrayrecon.gerber_utils import GerberLayer
from xrayrecon.materials import get_material
from xrayrecon.voxelizer import Voxelizer


def build_sophisticated_pcb_stackup(size: int = 96, resolution_z: float = 0.02):
    """Build a multi-layer PCB stackup with ENIG and HASL finishes.

    Returns (stackup, air_material, layer_names, layer_slices):
    - stackup: list of (GerberLayer, thickness_mm, feature_material,
      background_material) tuples, in the same bottom-to-top Z order
      Voxelizer.voxelize_stackup consumes it in.
    - layer_names / layer_slices: per-layer display name and the
      (0-indexed) Z-slice range it occupies in the voxelized volume,
      computed with the exact same thickness/resolution_z -> slice-count
      rounding Voxelizer itself uses internally (max(1, round(thickness /
      resolution_z))) -- so callers can pass these straight to
      LayerModel.setupTblRows() instead of relying on the generic
      auto-layer-detection heuristic, which is tuned for real CT scans
      with hundreds of slices and collapses to a single degenerate layer
      on a volume this small.

    Physical layout (X-Y plane, size x size raster):
    - Copper trace bar: horizontal bar from row size//3 to 2*size//3,
      columns margin to size-margin (margin = size//6)
    - Tin/Lead pad: smaller square in upper-left quadrant, away from trace
      and edges
    """
    margin = size // 6
    half_size = size // 2

    # Copper trace bar: horizontal rectangle
    trace_mask = np.zeros((size, size), dtype=np.uint8)
    trace_mask[size // 3 : 2 * size // 3, margin : -margin] = 255

    # Tin/Lead pad: smaller square in upper-left quadrant
    pad_size = size // 4
    pad_start = size // 6
    pad_end = pad_start + pad_size
    pad_mask = np.zeros((size, size), dtype=np.uint8)
    pad_mask[pad_start:pad_end, pad_start:pad_end] = 255

    # Full coverage substrate layer
    full_mask = np.ones((size, size), dtype=np.uint8) * 255

    # Material references
    copper = get_material("COPPER")
    nickel = get_material("NICKEL")
    gold = get_material("GOLD")
    tin = get_material("TIN")
    lead = get_material("LEAD")
    fr4 = get_material("FR4")
    air = get_material("AIR")

    # Reusable GerberLayer objects (one per physical feature region)
    trace_layer = GerberLayer("Copper Trace")
    trace_layer.raster_mask = trace_mask
    trace_layer.update_bounds(0, 0)
    trace_layer.update_bounds(size, size)

    pad_layer = GerberLayer("Solder Pad")
    pad_layer.raster_mask = pad_mask
    pad_layer.update_bounds(0, 0)
    pad_layer.update_bounds(size, size)

    substrate_layer = GerberLayer("Substrate")
    substrate_layer.raster_mask = full_mask
    substrate_layer.update_bounds(0, 0)
    substrate_layer.update_bounds(size, size)

    # Stackup (bottom to top):
    # 1. Bottom FR4 substrate (keeps features away from Z boundaries)
    # 2. Copper trace
    # 3. Nickel ENIG barrier
    # 4. Gold ENIG flash (completes ENIG finish over copper)
    # 5. Tin pad (HASL bottom layer)
    # 6. Lead on tin (HASL top layer)
    # 7. Top FR4 substrate (keeps features away from Z boundaries)
    stackup = [
        ("Bottom FR4", substrate_layer, 0.4, fr4, fr4),
        ("Copper Trace", trace_layer, 0.05, copper, air),
        ("ENIG Nickel Barrier", trace_layer, 0.01, nickel, air),
        ("ENIG Gold Flash", trace_layer, 0.001, gold, air),
        ("HASL Tin Pad", pad_layer, 0.03, tin, air),
        ("HASL Lead", pad_layer, 0.01, lead, air),
        ("Top FR4", substrate_layer, 0.4, fr4, fr4),
    ]

    layer_names = []
    layer_slices = []
    z_cursor = 0
    for name, _layer, thickness, _feat, _bg in stackup:
        num_slices = max(1, int(np.round(thickness / resolution_z)))
        layer_names.append(name)
        layer_slices.append(np.arange(z_cursor, z_cursor + num_slices))
        z_cursor += num_slices

    stackup = [(layer, thickness, feat, bg) for _name, layer, thickness, feat, bg in stackup]

    return stackup, air, layer_names, layer_slices


if __name__ == "__main__":
    stackup, air, layer_names, layer_slices = build_sophisticated_pcb_stackup()
    vox = Voxelizer(resolution_xy=0.05, resolution_z=0.02)
    vol, material_ids = vox.voxelize_stackup(stackup, air, return_material_ids=True)
    print(f"Sophisticated PCB stackup voxelized: {vol.shape}")
    print(f"Layers: {list(zip(layer_names, [s[0] for s in layer_slices], [s[-1] for s in layer_slices]))}")
    assert vol.shape[0] == layer_slices[-1][-1] + 1, "layer_slices don't cover the full voxelized depth"

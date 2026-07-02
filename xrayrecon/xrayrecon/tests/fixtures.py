from __future__ import annotations

import numpy as np

from xrayrecon.ct_simulation import simulate_ct_volume
from xrayrecon.gerber_utils import GerberLayer
from xrayrecon.materials import get_material
from xrayrecon.registration import RigidScaleTransform, resample_volume
from xrayrecon.voxelizer import Voxelizer

# Scaled to this fixture's physical extent (~10mm x 10mm x 1.1mm) -- e.g. a
# 0.5mm Z shift would be almost the entire depth of this thin PCB-like
# volume and push everything out of frame, unlike a plausible real
# misalignment.
DEFAULT_KNOWN_TRANSFORM = RigidScaleTransform(tx=0.2, ty=-0.15, tz=0.02, rot_z_deg=3.0, scale=1.02)


def _demo_stackup(size: int = 48):
    """A tiny 2-layer copper-trace-on-FR4 stackup, built directly from a
    hand-crafted raster mask -- avoids any dependency on real Gerber files on
    disk, so this fixture is portable across environments. (Raster-only
    GerberLayers have no `objects`, so Voxelizer falls back to its default
    10mm x 10mm grid regardless of `size` -- a pre-existing quirk, not
    something this fixture needs to work around.)
    """
    # Keep a comfortable margin around the trace (not near the raster edges)
    # so a modest registration misalignment doesn't push it out of frame --
    # information that leaves the frame during resampling is lost for good,
    # which would defeat a round-trip recovery test regardless of whether
    # the resampling math itself is correct.
    margin = size // 6
    trace_mask = np.zeros((size, size), dtype=np.uint8)
    trace_mask[size // 3 : 2 * size // 3, margin:-margin] = 255  # a copper bar

    cu_layer = GerberLayer("Top Cu")
    cu_layer.raster_mask = trace_mask

    copper = get_material("COPPER")
    fr4 = get_material("FR4")
    air = get_material("AIR")

    stackup = [
        # FR4 slabs both above and below keep the trace comfortably in the
        # Z-interior: `scipy.ndimage.affine_transform`'s default 'constant'
        # boundary mode zeroes an entire interpolated sample outright the
        # instant its source coordinate goes even slightly out of range (no
        # partial blending) -- content placed at a volume edge would get
        # wiped out by any transform with so much as a fractional-voxel shift
        # in that direction, regardless of how small the misalignment is.
        (cu_layer, 0.4, fr4, fr4),  # bottom FR4 slab (mask irrelevant, feat==bg)
        (cu_layer, 0.1, copper, air),  # copper trace layer
        (cu_layer, 0.4, fr4, fr4),  # top FR4 slab
    ]
    return stackup, air


def make_synthetic_pair(transform: RigidScaleTransform | None = None) -> dict:
    """Build a known-ground-truth registration test case from the already-
    verified Gerber-voxelize -> simulate_ct_volume bridge (Phase 1): a Gerber
    ground-truth material-ID volume, its "real CT" FBP-simulated proxy
    (perfectly aligned by construction, since both derive from the same
    voxel grid), and a second copy of that CT proxy with a KNOWN synthetic
    misalignment applied -- so registration recovery has a known-correct
    answer to check against.

    Since both start on the identical grid, `gerber_spacing == fixed_spacing`
    here; the fixture's `known_transform.scale` stands in for "unknown
    relative voxel size" between Gerber and real CT, which real data would
    otherwise require estimating.
    """
    stackup, air_mat = _demo_stackup()
    spacing = (0.1, 0.1, 0.1)
    vox = Voxelizer(resolution_xy=spacing[0], resolution_z=spacing[2])
    vol, material_ids_zhw = vox.voxelize_stackup(stackup, air_mat, return_material_ids=True)

    sim_zhw = simulate_ct_volume(vol)
    aligned_ct_hwz = np.moveaxis(sim_zhw, 0, -1)
    gerber_material_ids_hwz = np.moveaxis(material_ids_zhw, 0, -1)

    known_transform = transform or DEFAULT_KNOWN_TRANSFORM
    misaligned_ct_hwz = resample_volume(
        aligned_ct_hwz, spacing, aligned_ct_hwz.shape, spacing, known_transform, order=1
    )

    return dict(
        gerber_material_ids_hwz=gerber_material_ids_hwz,
        gerber_spacing=spacing,
        aligned_ct_hwz=aligned_ct_hwz,
        misaligned_ct_hwz=misaligned_ct_hwz,
        fixed_spacing=spacing,
        known_transform=known_transform,
    )

import numpy as np

from xrayrecon.alvarez_macovski import MATERIAL_NAMES as AM_MATERIAL_NAMES
from xrayrecon.alvarez_macovski import colorize_material_labels
from xrayrecon.gerber_utils import GerberLayer
from xrayrecon.materials import MATERIAL_COLORS, MaterialProperties, get_material
from xrayrecon.voxelizer import MATERIAL_NAMES as VOX_MATERIAL_NAMES
from xrayrecon.voxelizer import Voxelizer


def _checkerboard_layer(size=16):
    gl = GerberLayer("layer")
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[: size // 2, : size // 2] = 255
    gl.raster_mask = mask
    gl.update_bounds(0, 0)
    gl.update_bounds(2, 2)
    return gl


def test_material_name_ordering_matches_across_modules():
    assert AM_MATERIAL_NAMES == VOX_MATERIAL_NAMES


def test_voxelize_stackup_return_material_ids_matches_feature_background():
    gl = _checkerboard_layer()
    copper = get_material("COPPER")
    air = get_material("AIR")
    vox = Voxelizer(resolution_xy=0.125, resolution_z=0.05)

    vol, ids = vox.voxelize_stackup([(gl, 0.1, copper, air)], air, return_material_ids=True)

    assert vol.shape[:3] == ids.shape
    assert ids.dtype == np.int32

    cu_idx = VOX_MATERIAL_NAMES.index("COPPER")
    air_idx = VOX_MATERIAL_NAMES.index("AIR")
    assert set(np.unique(ids).tolist()) <= {cu_idx, air_idx}
    assert ids.shape == (2, 32, 32)
    # The raster occupies its 2x2 mm physical bounds inside the 1 mm margin.
    assert np.all(ids[:, 8:16, 8:16] == cu_idx)
    assert np.all(ids[:, :8, :] == air_idx)
    assert np.all(ids[:, 16:24, 16:24] == air_idx)


def test_voxelize_stackup_uses_raster_bounds_for_grid_size():
    gl = _checkerboard_layer()
    copper = get_material("COPPER")
    air = get_material("AIR")
    vox = Voxelizer(resolution_xy=0.25, resolution_z=0.05)

    vol = vox.voxelize_stackup([(gl, 0.1, copper, air)], air)

    assert vol.shape[:3] == (2, 16, 16)


def test_voxelize_stackup_material_id_matches_by_name_not_identity():
    # A freshly-constructed MaterialProperties with matching field values
    # (not the literal MATERIALS_DB singleton object) must still resolve to
    # the correct label index -- lookup is by name, not Python object identity.
    gl = _checkerboard_layer()
    copper_copy = MaterialProperties("Copper", 29.0, 8.96)
    air_copy = MaterialProperties("Air", 7.2, 0.0012)
    vox = Voxelizer(resolution_xy=0.125, resolution_z=0.05)

    vol, ids = vox.voxelize_stackup(
        [(gl, 0.1, copper_copy, air_copy)], air_copy, return_material_ids=True
    )

    cu_idx = VOX_MATERIAL_NAMES.index("COPPER")
    air_idx = VOX_MATERIAL_NAMES.index("AIR")
    assert set(np.unique(ids).tolist()) <= {cu_idx, air_idx}


def test_voxelize_stackup_without_flag_returns_bare_array():
    gl = _checkerboard_layer()
    copper = get_material("COPPER")
    air = get_material("AIR")
    vox = Voxelizer(resolution_xy=0.125, resolution_z=0.05)

    result = vox.voxelize_stackup([(gl, 0.1, copper, air)], air)

    assert isinstance(result, np.ndarray)


def test_colorize_material_labels_matches_material_colors():
    z, h, w = 2, 4, 4
    cu_idx = AM_MATERIAL_NAMES.index("COPPER")
    air_idx = AM_MATERIAL_NAMES.index("AIR")
    labels = np.full((z, h, w), air_idx, dtype=np.int32)
    labels[:, :2, :2] = cu_idx

    rgba = colorize_material_labels(labels)

    assert rgba.shape == (h, w, z, 4)
    assert rgba.dtype == np.uint8
    assert tuple(rgba[0, 0, 0]) == MATERIAL_COLORS["COPPER"]
    assert tuple(rgba[2, 2, 0]) == MATERIAL_COLORS["AIR"]

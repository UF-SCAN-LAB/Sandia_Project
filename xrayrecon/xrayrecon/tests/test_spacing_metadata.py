import numpy as np

from xrayrecon.gui.layermodel import LayerModel
from xrayrecon.pygerber_integration import PyGerberRasterizer
from xrayrecon.stackup import StackupConfig, StackupLayer
from xrayrecon.stackup_voxelize import voxelize_stackup_config


def test_layermodel_spacing_defaults_to_unitless(qapp):
    m = LayerModel(dataCube=np.zeros((8, 8, 4)))
    assert m.spacing_xyz == (1.0, 1.0, 1.0)


def test_layermodel_set_spacing_updates_and_emits(qapp):
    m = LayerModel(dataCube=np.zeros((8, 8, 4)))
    received = []
    m.sigSpacingChanged.connect(lambda: received.append(m.spacing_xyz))

    m.setSpacing(0.2, 0.2, 0.05)

    assert m.spacing_xyz == (0.2, 0.2, 0.05)
    assert received == [(0.2, 0.2, 0.05)]


def test_layermodel_spacing_confirmed_flag_lifecycle(qapp):
    m = LayerModel(dataCube=np.zeros((8, 8, 4)))
    assert m._spacing_confirmed is False

    m.setSpacing(0.2, 0.2, 0.05)
    assert m._spacing_confirmed is True

    # Loading a new, potentially differently-scaled cube must un-confirm the
    # spacing -- otherwise a stale spacing value silently survives onto data
    # it was never actually measured for.
    m.setCube(np.zeros((8, 8, 6)))
    assert m._spacing_confirmed is False
    # The stale numeric value itself is untouched (still whatever it was) --
    # only the "has this been confirmed for the CURRENT cube" flag resets.
    assert m.spacing_xyz == (0.2, 0.2, 0.05)


def test_layermodel_set_rgba_cube_also_unconfirms_spacing(qapp):
    m = LayerModel(dataCube=np.zeros((8, 8, 4)))
    m.setSpacing(0.1, 0.1, 0.1)
    assert m._spacing_confirmed is True

    m.setRGBACube(np.zeros((8, 8, 4)), np.zeros((8, 8, 4, 4), dtype=np.uint8))
    assert m._spacing_confirmed is False


def test_change_ax_permutes_spacing_and_preserves_confirmed(qapp):
    m = LayerModel(dataCube=np.zeros((5, 8, 4)))
    m.setSpacing(0.1, 0.2, 0.3)  # anisotropic, so a permutation bug is visible
    assert m.spacing_xyz == (0.1, 0.2, 0.3)

    # x-z swap: axis order becomes [z, y, x] -> spacing should follow
    m.changeAx(type("W", (), {"name": "Z-Y"})())

    assert m._spacing_confirmed is True  # same physical data, just relabeled
    assert set(m.spacing_xyz) == {0.1, 0.2, 0.3}
    assert m.spacing_xyz != (0.1, 0.2, 0.3)  # order actually changed


def test_voxelize_stackup_config_spacing_matches_config(tmp_path, monkeypatch):
    # Stub out the actual Gerber rasterization (PyGerber parsing real file
    # content) so this test exercises voxelize_stackup_config's own
    # spacing-tuple assembly without depending on real Gerber files on disk.
    def _fake_rasterize(self, content):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[4:6, 4:6] = 255
        return mask

    monkeypatch.setattr(PyGerberRasterizer, "rasterize", _fake_rasterize)
    gerber_file = tmp_path / "layer.gbr"
    gerber_file.write_text("dummy")

    config = StackupConfig(
        layers=[StackupLayer(file="layer.gbr", name="L1", thickness=0.1)],
        resolution_xy=0.15,
        resolution_z=0.07,
    )

    _vol, _names, _slices, _visual, spacing_mm, _ids = voxelize_stackup_config(
        config, tmp_path / "stackup.json"
    )

    assert spacing_mm == (0.15, 0.15, 0.07)


def test_voxelize_stackup_config_rejects_nonpositive_resolution(tmp_path):
    bad_config = StackupConfig(layers=[], resolution_xy=0.0, resolution_z=0.05)
    try:
        voxelize_stackup_config(bad_config, tmp_path / "stackup.json")
        assert False, "expected ValueError for non-positive resolution_xy"
    except ValueError:
        pass

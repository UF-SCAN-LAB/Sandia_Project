"""Compare adapter output with the unmodified installed PyGerber renderer."""

import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

pytest.importorskip("pygerber")

from pygerber.backend import Rasterized2DBackend
from pygerber.backend.rasterized_2d.backend_cls import Rasterized2DBackendOptions
from pygerber.gerberx3.api import ColorScheme, Layer, LayerParams

from xrayrecon.pygerber_integration import PyGerberRasterizer
from xrayrecon.stackup import StackupConfig, StackupLayer
from xrayrecon.stackup_voxelize import voxelize_stackup_config


def gerber(aperture="1.0", *, units="%MOMM*%", decimals=5):
    coordinate = "020000" if decimals == 4 else "0200000"
    return (
        f"%FSLAX2{decimals}Y2{decimals}*%\n{units}\n"
        f"%ADD10C,{aperture}*%\nD10*\n"
        f"X{coordinate}Y{coordinate}D03*\nM02*\n"
    )


LEGACY = gerber("1.0X0", units="%MOMM*%\nG71*\nG71*", decimals=4)
MIXED = (
    "%FSLAX25Y25*%\n%MOMM*%\n"
    "%ADD10C,1.0X0*%\n%ADD11C,0.1*%\n%ADD12R,1.0X0.5*%\n"
    "D10*\nX0200000Y0200000D03*\nX0400000Y0200000D01*\n"
    "D11*\nX0500000Y0300000D03*\n"
    "D12*\nX0400000Y0400000D03*\nM02*\n"
)
REGION = (
    "%FSLAX25Y25*%\n%MOMM*%\n%ADD10C,0.2*%\nD10*\n"
    "G36*\nX0100000Y0100000D02*\nX0300000Y0100000D01*\n"
    "X0300000Y0300000D01*\nX0100000Y0300000D01*\n"
    "X0100000Y0100000D01*\nG37*\nM02*\n"
)


def reference_mask(content, dpi):
    """Use the pre-change backend and parser with precisely the same options."""
    class ReferenceLayer(Layer):
        def _create_backend(self):
            return Rasterized2DBackend(options=Rasterized2DBackendOptions(
                dpi=dpi, color_scheme=ColorScheme.COPPER_ALPHA,
            ))

    result = ReferenceLayer(LayerParams(source_code=content)).render()
    # The existing adapter uses the raw handle.result. get_image() flips it
    # vertically, which would change the downstream coordinate convention.
    image = result._result_handle.result
    return np.array(image.getchannel("A") if image.mode == "RGBA" else image.convert("L"))


def findings(rasterizer):
    return {finding.category: finding for finding in rasterizer.diagnostics}


@pytest.mark.parametrize("dpi", [300, 508, 1000])
@pytest.mark.parametrize("content", [
    LEGACY, MIXED, REGION, gerber("1.0X0.3"), gerber("0.1"),
    gerber(units="%MOMM*%\nG70*\nG71*"),
    gerber(units="%MOIN*%\nG70*"),
], ids=["legacy-zero-hole", "mixed-traces", "region", "positive-hole",
        "tiny-circle", "conflicting-units", "imperial-units"])
def test_masks_are_bit_exact_with_original_renderer(content, dpi):
    expected = reference_mask(content, dpi)
    rasterizer = PyGerberRasterizer(dpi=dpi, source_name="synthetic.gbr")

    actual = rasterizer.rasterize(content)

    assert actual.dtype == expected.dtype == np.uint8
    np.testing.assert_array_equal(actual, expected)
    assert rasterizer.dpi == dpi


def test_legacy_noise_is_classified_and_precision_warned_once(caplog):
    rasterizer = PyGerberRasterizer(dpi=300, source_name="top_copper.gbr")
    with caplog.at_level(logging.INFO):
        mask = rasterizer.rasterize(LEGACY)

    assert mask.max() == 255
    result = findings(rasterizer)
    assert result["coordinate_precision"].count == 2  # X and Y
    assert result["legacy_units"].count == 2
    assert result["redundant_units"].count == 2
    assert result["empty_circle"].count == 1
    assert "conflicting_units" not in result
    assert "lost_circle" not in result
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warning_records) == 1
    assert "top_copper.gbr (300 DPI)" in warning_records[0].message
    assert "Coordinate precision" in warning_records[0].message
    assert "Overriding coordinate units is illegal" not in caplog.text


@pytest.mark.parametrize("units, category", [
    ("%MOMM*%\n%MOMM*%", "redundant_units"),
    ("%MOMM*%\n%MOIN*%", "conflicting_units"),
    ("%MOMM*%\nG70*\nG71*", "conflicting_units"),
    ("%MOIN*%\nG70*", "redundant_units"),
])
def test_unit_classification_uses_actual_parser_state(caplog, units, category):
    rasterizer = PyGerberRasterizer(dpi=300, source_name="units.gbr")
    # Two unit conflicts end back in metric units in the G70/G71 case.
    rasterizer.rasterize(gerber(units=units))

    result = findings(rasterizer)
    assert category in result
    if category == "conflicting_units":
        assert result[category].level == logging.WARNING
        assert "units.gbr (300 DPI): Conflicting unit declarations" in caplog.text
        assert "MM -> IN" in caplog.text
    else:
        assert result[category].level == logging.INFO
        assert "conflicting_units" not in result


def test_gerber_comments_do_not_change_unit_classification():
    content = gerber(units="%MOMM*%\nG04 Old exporter mentioned G70 and G71*")
    rasterizer = PyGerberRasterizer()
    rasterizer.rasterize(content)

    assert rasterizer.diagnostics == ()


def test_coordinate_format_override_is_not_treated_as_redundant_units(caplog):
    content = gerber().replace("%MOMM*%", "%FSLAX25Y25*%\n%MOMM*%")
    rasterizer = PyGerberRasterizer(dpi=300)
    rasterizer.rasterize(content)

    assert "Overriding coordinate format is illegal" in caplog.text
    assert "redundant_units" not in findings(rasterizer)


def test_zero_holes_have_no_pixels_and_no_dpi_warning(caplog):
    rasterizer = PyGerberRasterizer(dpi=300, source_name="empty_hole.gbr")
    actual = rasterizer.rasterize(gerber("1.0X0"))

    assert findings(rasterizer)["empty_circle"].level == logging.INFO
    assert "lost_circle" not in findings(rasterizer)
    assert not caplog.records
    np.testing.assert_array_equal(actual, reference_mask(gerber("1.0"), 300))


def test_nonzero_lost_circles_warn_once_with_source_and_dpi(caplog):
    rasterizer = PyGerberRasterizer(dpi=300, source_name="tiny_features.gbr")
    rasterizer.rasterize(gerber("0.1"))

    assert findings(rasterizer)["lost_circle"].count == 1
    assert "tiny_features.gbr (300 DPI)" in caplog.text
    assert "Nonzero-diameter" in caplog.text
    assert rasterizer.dpi == 300


def test_diagnostics_reset_between_layers(caplog):
    rasterizer = PyGerberRasterizer(dpi=300, source_name="first.gbr")
    rasterizer.rasterize(LEGACY)
    assert rasterizer.diagnostics
    caplog.clear()
    rasterizer.source_name = "second.gbr"
    rasterizer.rasterize(gerber())

    assert rasterizer.diagnostics == ()
    assert not caplog.records


def test_logging_scope_preserves_unrelated_warnings_errors_and_threads(monkeypatch, caplog):
    root = logging.getLogger()
    original_filters = list(root.filters)
    original_level = root.level
    render = Layer.render
    path = "C:/site-packages/pygerber/backend/rasterized_2d/draw_commands/draw_circle.py"
    known_message = "Drawing zero surface circle. DPI may be too low. (1, 1, 1, 1)"

    def emit(message, *, level=logging.WARNING, filename=path):
        root.handle(logging.LogRecord("root", level, filename, 1, message, (), None))

    def render_with_unrelated_logs(layer):
        emit("unrecognized PyGerber warning")
        emit(known_message, level=logging.ERROR)
        emit(known_message, filename="C:/application/other.py")
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(emit, known_message).result()
        return render(layer)

    monkeypatch.setattr(Layer, "render", render_with_unrelated_logs)
    rasterizer = PyGerberRasterizer(dpi=300)
    rasterizer.rasterize(gerber())

    assert rasterizer.diagnostics == ()
    assert len(caplog.records) == 4
    assert "unrecognized PyGerber warning" in caplog.text
    assert sum(r.levelno == logging.ERROR for r in caplog.records) == 1
    assert root.filters == original_filters
    assert root.level == original_level


def test_filter_cleanup_and_failure_reporting(monkeypatch, capsys):
    root = logging.getLogger()
    original_filters = list(root.filters)

    def fail(layer):
        raise ValueError("synthetic rendering failure")

    monkeypatch.setattr(Layer, "render", fail)
    rasterizer = PyGerberRasterizer()
    rasterizer.rasterize(gerber())

    assert "PyGerber Rasterization Failed: synthetic rendering failure" in capsys.readouterr().out
    assert root.filters == original_filters


def test_nested_renders_keep_their_own_diagnostics(monkeypatch, caplog):
    render = Layer.render
    inner = PyGerberRasterizer(dpi=300, source_name="inner.gbr")
    nested = False

    def render_with_inner(layer):
        nonlocal nested
        if not nested:
            nested = True
            inner.rasterize(gerber("0.1"))
        return render(layer)

    monkeypatch.setattr(Layer, "render", render_with_inner)
    outer = PyGerberRasterizer(dpi=300, source_name="outer.gbr")
    outer.rasterize(LEGACY)

    assert "lost_circle" in findings(inner)
    assert "lost_circle" not in findings(outer)
    assert findings(outer)["coordinate_precision"].count == 2
    assert "inner.gbr (300 DPI)" in caplog.text
    assert "outer.gbr (300 DPI)" in caplog.text


def test_stackup_voxels_labels_and_spacing_equal_original_renderer(tmp_path, monkeypatch, caplog):
    first = tmp_path / "top.gbr"
    second = tmp_path / "bottom.gbr"
    first.write_text(LEGACY)
    second.write_text(MIXED)
    source_bytes = (first.read_bytes(), second.read_bytes())
    config = StackupConfig(
        layers=[
            StackupLayer("top.gbr", "Top", thickness=0.1, gap=0.1),
            StackupLayer("bottom.gbr", "Bottom", thickness=0.05, background_material="FR4"),
        ],
        dpi=300, resolution_xy=0.05, resolution_z=0.05,
    )
    stackup_path = tmp_path / "stackup.json"
    actual = voxelize_stackup_config(config, stackup_path, return_material_ids=True)
    assert f"{first} (508 DPI)" in caplog.text
    assert f"{second} (508 DPI)" in caplog.text

    def original_rasterize(self, content):
        return reference_mask(content, self.dpi)

    with monkeypatch.context() as context:
        context.setattr(PyGerberRasterizer, "rasterize", original_rasterize)
        expected = voxelize_stackup_config(config, stackup_path, return_material_ids=True)

    np.testing.assert_array_equal(actual[0], expected[0])  # physical/material volume
    np.testing.assert_array_equal(actual[5], expected[5])  # explicit material IDs
    assert actual[1] == expected[1]  # names
    for actual_slices, expected_slices in zip(actual[2], expected[2]):
        np.testing.assert_array_equal(actual_slices, expected_slices)
    assert actual[3] == expected[3]  # visual metadata
    assert actual[4] == expected[4] == (0.05, 0.05, 0.05)
    assert (first.read_bytes(), second.read_bytes()) == source_bytes

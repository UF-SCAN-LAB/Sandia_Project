"""Render unchanged Gerber sources with diagnostics scoped to one render."""

from collections import Counter
from contextvars import ContextVar
from dataclasses import dataclass
import logging
from threading import get_ident

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)
_active_diagnostics = ContextVar("gerber_diagnostics", default=None)


@dataclass(frozen=True)
class GerberDiagnostic:
    """One consolidated finding from the most recent rasterization."""

    category: str
    count: int
    level: int
    message: str


class _RenderDiagnostics(logging.Filter):
    """Collect only known PyGerber warnings on the active rendering thread.

    PyGerber 2.x logs these messages through the root logger. Other warnings,
    errors, threads and nested renders keep their normal logging behavior.
    """

    def __init__(self):
        super().__init__()
        self.thread_id = get_ident()
        self.counts = Counter()
        self.conflicts = []

    def filter(self, record):
        if (
            _active_diagnostics.get() is not self
            or record.thread != self.thread_id
            or record.levelno != logging.WARNING
        ):
            return True
        path = record.pathname.replace("\\", "/").lower()
        if "/pygerber/" not in path:
            return True
        filename = path.rsplit("/", 1)[-1]
        message = record.getMessage()
        category = None
        if filename == "fs_coordinate_format.py" and message.startswith(
            "It is recommended to use at least 5 decimal places for coordinate data"
        ):
            category = "coordinate_precision"
        elif filename == "warnings.py" and message.startswith(
            ("Detected deprecated code: G70.", "Detected deprecated code: G71.")
        ):
            category = "legacy_units"
        elif (
            filename in {"g70_set_unit_inch.py", "g71_set_unit_mm.py", "mo_unit_mode.py"}
            and message.startswith((
                "Overriding coordinate units is illegal.",
                "Overriding coordinate format is illegal.",
            ))
        ):
            # The parser hook records actual previous/new units. Do not infer
            # a conflict from PyGerber's warning text alone.
            category = "unit_override"
        elif filename == "draw_circle.py" and message.startswith(
            "Drawing zero surface circle. DPI may be too low."
        ):
            category = "lost_circle"
        if category is None:
            return True
        self.counts[category] += 1
        return False

    def findings(self):
        findings = []
        if self.counts["coordinate_precision"]:
            findings.append(GerberDiagnostic(
                "coordinate_precision", self.counts["coordinate_precision"], logging.WARNING,
                "Coordinate precision is below the recommendation (5 decimal places "
                "for metric, 6 for imperial). Source coordinate formats are preserved.",
            ))
        if self.counts["legacy_units"]:
            findings.append(GerberDiagnostic(
                "legacy_units", self.counts["legacy_units"], logging.INFO,
                "Legacy G70/G71 unit declarations were parsed without rewriting the source.",
            ))
        if self.counts["redundant_units"]:
            findings.append(GerberDiagnostic(
                "redundant_units", self.counts["redundant_units"], logging.INFO,
                "Repeated unit declarations specify the same units.",
            ))
        if self.conflicts:
            findings.append(GerberDiagnostic(
                "conflicting_units", len(self.conflicts), logging.WARNING,
                f"Conflicting unit declarations: {'; '.join(self.conflicts[:3])}. "
                "Review the source; the existing parser interpretation is preserved.",
            ))
        classified_overrides = self.counts["redundant_units"] + len(self.conflicts)
        if self.counts["unit_override"] > classified_overrides:
            findings.append(GerberDiagnostic(
                "unit_override", self.counts["unit_override"] - classified_overrides,
                logging.WARNING, "Unit override reported by PyGerber; review the source.",
            ))
        if self.counts["empty_circle"]:
            findings.append(GerberDiagnostic(
                "empty_circle", self.counts["empty_circle"], logging.INFO,
                "Zero-diameter circle components have no pixels to draw.",
            ))
        if self.counts["lost_circle"]:
            findings.append(GerberDiagnostic(
                "lost_circle", self.counts["lost_circle"], logging.WARNING,
                "Nonzero-diameter circle components could not be drawn at this raster "
                "DPI. Review feature sizes and DPI; the raster DPI is preserved.",
            ))
        return tuple(findings)


class PyGerberParser:
    def __init__(self):
        pass


class PyGerberRasterizer:
    """Return the existing alpha mask and expose per-render diagnostic summaries.

    ``source_name`` is logging context only. ``diagnostics`` is replaced after
    each render, including a failed render. No source normalization is applied.
    """

    def __init__(self, dpi=1000, source_name=None):
        self.dpi = dpi
        self.source_name = source_name
        self.diagnostics = ()

    def rasterize(self, content: str) -> np.ndarray:
        diagnostics = _RenderDiagnostics()
        root = logging.getLogger()
        context_token = _active_diagnostics.set(diagnostics)
        root.addFilter(diagnostics)
        try:
            from pygerber.gerberx3.api import Layer, ColorScheme, LayerParams
            from pygerber.backend import Rasterized2DBackend
            from pygerber.backend.rasterized_2d.backend_cls import (
                Rasterized2DBackendOptions,
            )
            from pygerber.backend.rasterized_2d.draw_commands.draw_circle import (
                Rasterized2DApertureDrawCircle,
            )
            from pygerber.backend.rasterized_2d.drawing_target import (
                Rasterized2DDrawingTarget,
            )
            from pygerber.gerberx3.parser.parser import Parser, ParserOptions
            from pygerber.gerberx3.state_enums import Unit
            from pygerber.gerberx3.tokenizer.tokens.g70_set_unit_inch import SetUnitInch
            from pygerber.gerberx3.tokenizer.tokens.g71_set_unit_mm import SetUnitMillimeters
            from pygerber.gerberx3.tokenizer.tokens.mo_unit_mode import UnitMode

            class DiagnosticParser(Parser):
                def _update_drawing_state(self, token):
                    new_unit = None
                    if isinstance(token, UnitMode):
                        new_unit = token.unit
                    elif isinstance(token, SetUnitMillimeters):
                        new_unit = Unit.Millimeters
                    elif isinstance(token, SetUnitInch):
                        new_unit = Unit.Inches
                    previous_unit = self.state.draw_units
                    if new_unit is not None and previous_unit is not None:
                        if previous_unit == new_unit:
                            diagnostics.counts["redundant_units"] += 1
                        else:
                            diagnostics.conflicts.append(
                                f"{previous_unit.value} -> {new_unit.value} at "
                                f"{token.get_token_position()}"
                            )
                    super()._update_drawing_state(token)

            class DiagnosticCircle(Rasterized2DApertureDrawCircle):
                def draw(self, target):
                    if (
                        isinstance(target, Rasterized2DDrawingTarget)
                        and self.diameter.as_millimeters() == 0
                    ):
                        # The original renderer skips this empty component too.
                        # Keep its command and bounding box, avoiding only its
                        # misleading DPI warning (e.g. an explicit 0-size hole).
                        diagnostics.counts["empty_circle"] += 1
                        return
                    super().draw(target)

            class DiagnosticBackend(Rasterized2DBackend):
                def get_draw_circle_cls(self):
                    return DiagnosticCircle

            class CustomLayer(Layer):
                def __init__(self, options: LayerParams, dpi: int):
                    self.custom_dpi = dpi
                    super().__init__(options)

                def _create_backend(self):
                    return DiagnosticBackend(
                        options=Rasterized2DBackendOptions(
                            dpi=self.custom_dpi,
                            color_scheme=ColorScheme.COPPER_ALPHA, 
                        )
                    )

                def _create_parser(self):
                    return DiagnosticParser(ParserOptions(
                        backend=self.backend,
                        on_update_drawing_state_error=self.options.parser_error,
                    ))
            
            layer = CustomLayer(options=LayerParams(source_code=content), dpi=self.dpi)
            result = layer.render()
            
            # Access internal handle
            if hasattr(result, '_result_handle'):
                handle = result._result_handle
                # print(f"DEBUG Internal Handle Dir: {dir(handle)}")
                
                # Check for image access
                if hasattr(handle, 'result'):
                    # Sometimes handle.result is the image
                    img = handle.result
                elif hasattr(handle, 'get_image'):
                    img = handle.get_image()
                else:
                    # Check backend specific handle
                    # Step 209 said Rasterized2DBackend has get_result_handle
                    # The handle is Rasterized2DResultHandle
                    # I'll print to find out
                    print(f"DEBUG Handle Dir: {dir(handle)}")
                    raise AttributeError("Cannot find image in internal handle")
            else:
                 raise AttributeError("Cannot find _result_handle")

            # Verify img is PIL Image
            if not isinstance(img, Image.Image):
                 print(f"DEBUG Retrieved object type: {type(img)}")
                 # If it's not image, maybe it's something else
                 raise AttributeError("Retrieved object is not PIL Image")

            if img.mode == 'RGBA':
                r, g, b, a = img.split()
                mask = np.array(a)
            else:
                mask = np.array(img.convert("L")) 
            return mask
            
        except ImportError as e:
            print(f"PyGerber Import Error: {e}")
            return np.zeros((10,10), dtype=np.uint8)
        except Exception as e:
            print(f"PyGerber Rasterization Failed: {e}")
            return np.zeros((10,10), dtype=np.uint8)
        finally:
            root.removeFilter(diagnostics)
            _active_diagnostics.reset(context_token)
            self.diagnostics = diagnostics.findings()
            for finding in self.diagnostics:
                logger.log(
                    finding.level, "Gerber %s (%s DPI): %s (count=%d)",
                    self.source_name or "<memory>", self.dpi,
                    finding.message, finding.count,
                )

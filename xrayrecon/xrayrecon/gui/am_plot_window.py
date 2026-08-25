from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from pyqtgraph.Qt import QtCore, QtWidgets

from ..materials import MATERIALS_DB, MATERIAL_COLORS


class AMAnalysisPanel(QtWidgets.QWidget):
    """Embedded quantitative Alvarez-Macovski review workspace.

    The panel deliberately keeps direct inversion and material-regularized
    interpretation selectable, shows data-consistency/confidence maps beside
    them, and labels electron-density units explicitly. Synthetic ground truth
    is shown when available; production scans may omit those arrays.
    """

    sigCloseRequested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.result: dict | None = None

        self.backBtn = QtWidgets.QPushButton("Back to volume workspace")
        self.backBtn.clicked.connect(self.sigCloseRequested)
        self.modeCombo = QtWidgets.QComboBox()
        self.modeCombo.addItem("Material-regularized interpretation", "regularized")
        self.modeCombo.addItem("Raw inversion (audit)", "raw")
        self.modeCombo.currentIndexChanged.connect(self._draw)

        self.sliceSlider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.sliceSlider.setRange(0, 0)
        self.sliceSlider.valueChanged.connect(self._draw)
        self.sliceLabel = QtWidgets.QLabel("Slice 0")
        self.metricSummary = QtWidgets.QLabel("Run a decomposition to inspect it here.")
        self.metricSummary.setWordWrap(True)
        self.metricSummary.setStyleSheet("color: #555;")

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(self.backBtn)
        controls.addSpacing(12)
        controls.addWidget(QtWidgets.QLabel("Display:"))
        controls.addWidget(self.modeCombo)
        controls.addSpacing(12)
        controls.addWidget(self.sliceLabel)
        controls.addWidget(self.sliceSlider, 1)

        self.figure = Figure(figsize=(10, 7))
        self.canvas = FigureCanvasQTAgg(self.figure)

        layout = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel("Quantitative Alvarez–Macovski Analysis")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)
        layout.addLayout(controls)
        layout.addWidget(self.metricSummary)
        layout.addWidget(self.canvas, 1)

    def setResult(self, result: dict) -> None:
        required = {
            "zeff_recovered",
            "rhoe_recovered",
            "projection_residual",
            "confidence",
        }
        missing = sorted(required - set(result))
        if missing:
            raise ValueError(
                "AM result is missing quantitative arrays: " + ", ".join(missing)
            )
        shape = np.asarray(result["zeff_recovered"]).shape
        if len(shape) != 3 or any(np.asarray(result[key]).shape != shape for key in required):
            raise ValueError("AM quantitative arrays must share a (Z, H, W) shape")

        self.result = result
        representative = AMPlotWindow._pick_representative_slice(
            result.get("zeff_ground_truth", result["zeff_recovered"])
        )
        self.sliceSlider.blockSignals(True)
        self.sliceSlider.setRange(0, shape[0] - 1)
        self.sliceSlider.setValue(representative)
        self.sliceSlider.blockSignals(False)
        self._draw()

    def _draw(self, *_args) -> None:
        if self.result is None:
            return
        result = self.result
        index = self.sliceSlider.value()
        self.sliceLabel.setText(f"Slice {index + 1} of {self.sliceSlider.maximum() + 1}")
        regularized = self.modeCombo.currentData() == "regularized"
        z_key = "zeff_regularized" if regularized and "zeff_regularized" in result else "zeff_recovered"
        r_key = "rhoe_regularized" if regularized and "rhoe_regularized" in result else "rhoe_recovered"
        mode_name = "Material-regularized" if regularized else "Raw inversion"

        zeff = result[z_key][index]
        rhoe = result[r_key][index]
        zeff_gt = result.get("zeff_ground_truth")
        rhoe_gt = result.get("rhoe_ground_truth")
        confidence = result["confidence"][index]
        residual = result["projection_residual"][index]

        self.figure.clear()
        axes = self.figure.subplots(2, 3)
        zmax = max(
            float(np.max(zeff)),
            float(np.max(zeff_gt[index])) if zeff_gt is not None else 0.0,
            1e-6,
        )
        rmax = max(
            float(np.max(rhoe)),
            float(np.max(rhoe_gt[index])) if rhoe_gt is not None else 0.0,
            1e-6,
        )
        panels = [
            (
                axes[0, 0],
                zeff_gt[index] if zeff_gt is not None else zeff,
                "Ground-truth Z-effective" if zeff_gt is not None else "Z-effective reference unavailable",
                "plasma",
                0.0,
                zmax,
                "Z-effective",
            ),
            (axes[0, 1], zeff, f"{mode_name} Z-effective", "plasma", 0.0, zmax, "Z-effective"),
            (axes[0, 2], confidence, "Reconstruction confidence", "viridis", 0.0, 1.0, "0–1 quality indicator"),
            (
                axes[1, 0],
                rhoe_gt[index] if rhoe_gt is not None else rhoe,
                "Ground-truth relative electron density" if rhoe_gt is not None else "Electron-density reference unavailable",
                "cividis",
                0.0,
                rmax,
                "rho_e / rho_e,water",
            ),
            (axes[1, 1], rhoe, f"{mode_name} electron density", "cividis", 0.0, rmax, "rho_e / rho_e,water"),
            (axes[1, 2], residual, "Projection residual", "magma", 0.0, 1.0, "normalized mismatch"),
        ]
        for axis, image, title, cmap, vmin, vmax, label in panels:
            artist = axis.imshow(image, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
            axis.set_title(title, fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
            self.figure.colorbar(artist, ax=axis, label=label, fraction=0.046, pad=0.04)

        self.figure.suptitle(
            "Raw results remain available for audit; confidence is not a probability.",
            fontsize=10,
        )
        self.figure.tight_layout()
        self.canvas.draw_idle()
        self._update_metrics(index, z_key, r_key)

    def _update_metrics(self, index: int, z_key: str, r_key: str) -> None:
        result = self.result
        zeff_gt = result.get("zeff_ground_truth")
        rhoe_gt = result.get("rhoe_ground_truth")
        confidence = result["confidence"][index]
        residual = result["projection_residual"][index]
        summary = (
            f"Median confidence {np.median(confidence):.2f} · "
            f"median normalized projection residual {np.median(residual):.3f}. "
        )
        if zeff_gt is not None and rhoe_gt is not None:
            mask = rhoe_gt[index] > 0.01
            if np.any(mask):
                z_error = np.median(
                    np.abs(result[z_key][index][mask] - zeff_gt[index][mask])
                    / np.maximum(zeff_gt[index][mask], 1e-6)
                )
                r_error = np.median(
                    np.abs(result[r_key][index][mask] - rhoe_gt[index][mask])
                    / np.maximum(rhoe_gt[index][mask], 1e-6)
                )
                summary += (
                    f"Synthetic-truth median errors: Z-effective {z_error:.1%}; "
                    f"relative electron density {r_error:.1%}."
                )
        self.metricSummary.setText(summary)


class AMPlotWindow(QtWidgets.QDialog):
    """Legacy non-modal comparison retained for API compatibility.

    ``MainWin`` uses :class:`AMAnalysisPanel` so the supported application
    workflow stays embedded. This dialog still shows Alvarez-Macovski-recovered
    Z_eff / electron-density maps for a representative slice.

    The visual comparison uses explicitly named material-regularized outputs
    when present; raw decomposition arrays remain in the result for audit.
    Also prints a per-material accuracy table to the console (mean +/- std
    displayed value grouped by ground-truth Z_eff), mirroring the
    quantitative comparison in alvarez_macovski_demo.ipynb Section 12.
    """

    def __init__(self, result: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Alvarez-Macovski: Z_eff / Electron Density")
        self.resize(900, 800)

        slice_idx = self._pick_representative_slice(result["zeff_ground_truth"])

        fig = Figure(figsize=(9, 8))
        canvas = FigureCanvasQTAgg(fig)

        zeff_gt = result["zeff_ground_truth"][slice_idx]
        zeff_rec = result.get("zeff_regularized", result["zeff_recovered"])[slice_idx]
        rhoe_gt = result["rhoe_ground_truth"][slice_idx]
        rhoe_rec = result.get("rhoe_regularized", result["rhoe_recovered"])[slice_idx]

        vmax_z = max(float(zeff_gt.max()), float(zeff_rec.max()), 1e-6)
        vmax_r = max(float(rhoe_gt.max()), float(rhoe_rec.max()), 1e-6)

        axes = fig.subplots(2, 2)
        panels = [
            (axes[0, 0], zeff_gt, "Ground-truth $Z_{eff}$", "plasma", vmax_z, "$Z_{eff}$"),
            (axes[0, 1], zeff_rec, "Material-regularized $Z_{eff}$", "plasma", vmax_z, "$Z_{eff}$"),
            (axes[1, 0], rhoe_gt, "Ground-truth $\\rho_e$", "cividis", vmax_r, "$\\rho_e$"),
            (axes[1, 1], rhoe_rec, "Material-regularized $\\rho_e$", "cividis", vmax_r, "$\\rho_e$"),
        ]
        for ax, img, title, cmap, vmax, label in panels:
            im = ax.imshow(img, cmap=cmap, origin="lower", vmin=0, vmax=vmax)
            ax.set_title(title)
            fig.colorbar(im, ax=ax, label=label, fraction=0.046, pad=0.04)

        fig.suptitle(f"Alvarez-Macovski Decomposition — slice {slice_idx}", fontweight="bold")
        fig.tight_layout()

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(canvas)

        self._print_accuracy_table(result)

    @staticmethod
    def _pick_representative_slice(zeff_ground_truth: np.ndarray) -> int:
        """Pick the Z-slice containing the highest-Z material (e.g. copper),
        so the plot shows a feature-containing slice rather than a uniform
        substrate/gap slice."""
        max_zeff_per_slice = zeff_ground_truth.max(axis=(1, 2))
        return int(np.argmax(max_zeff_per_slice))

    @staticmethod
    def _print_accuracy_table(result: dict):
        zeff_gt = result["zeff_ground_truth"]
        zeff_rec = result.get("zeff_regularized", result["zeff_recovered"])
        rhoe_gt = result["rhoe_ground_truth"]
        rhoe_rec = result.get("rhoe_regularized", result["rhoe_recovered"])

        print("\nAlvarez-Macovski decomposition accuracy (mean +/- std, grouped by material):")
        print(
            f"{'True Zeff':>10s} {'Recovered Zeff':>16s} {'True rho_e':>12s} "
            f"{'Recovered rho_e':>18s} {'voxels':>8s}"
        )

        nonzero = zeff_gt > 0
        if not nonzero.any():
            print("  (no non-air voxels found)")
            return

        rounded = np.round(zeff_gt, 1)
        for z_val in sorted(set(rounded[nonzero].tolist())):
            mask = nonzero & (rounded == z_val)
            n = int(mask.sum())
            if n == 0:
                continue
            rz = zeff_rec[mask]
            rr = rhoe_rec[mask]
            true_rho = float(rhoe_gt[mask].mean())
            print(
                f"{z_val:10.2f} {rz.mean():9.2f}+/-{rz.std():<6.2f} {true_rho:12.3f} "
                f"{rr.mean():11.3f}+/-{rr.std():<6.3f} {n:8d}"
            )


class MaterialLegendWidget(QtWidgets.QWidget):
    """Embeddable legend for the material lookup table used by the
    'Material' AM color mode: color swatch, name, and reference
    Z_eff/electron-density for every material in materials.MATERIALS_DB,
    laid out as a compact horizontal strip for docking at the bottom of the
    main window. Reference values are standard textbook figures consistent
    with NIST's tabulated element/compound attenuation data
    (https://www.nist.gov/pml/x-ray-mass-attenuation-coefficients).

    Each voxel of the 'Material' color mode is assigned to whichever of
    these reference materials its recovered (Z_eff, rho_e) is nearest to --
    an approximation, not a certainty, especially where Z_eff recovery is
    noisy (see the accuracy table printed by AMPlotWindow).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        outer_layout = QtWidgets.QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        scroll.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        scroll.setFixedHeight(48)
        content = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(content)
        layout.setContentsMargins(6, 2, 6, 2)

        title = QtWidgets.QLabel("Material Legend:")
        title.setStyleSheet("font-weight: bold;")
        layout.addWidget(title)

        for name in sorted(MATERIALS_DB.keys()):
            props = MATERIALS_DB[name]
            r, g, b, _a = MATERIAL_COLORS[name]

            # Full opacity in the legend swatch regardless of the material's
            # 3D-view alpha (e.g. AIR's alpha=5 would be nearly invisible
            # here) -- this legend is about identifying colors, not replaying
            # the 3D view's transparency.
            swatch = QtWidgets.QLabel()
            swatch.setFixedSize(14, 14)
            swatch.setStyleSheet(f"background-color: rgb({r},{g},{b}); border: 1px solid #444;")
            layout.addWidget(swatch)

            text = QtWidgets.QLabel(
                f"{props.name} (Z={props.z_eff:.1f}, "
                f"rho_e/water={props.relative_electron_density:.2f})"
            )
            layout.addWidget(text)

        layout.addStretch()

        note = QtWidgets.QLabel(
            "Approximate -- density-led material regularization with raw AM outputs retained for audit."
        )
        note.setStyleSheet("color: #888; font-style: italic;")
        layout.addWidget(note)
        scroll.setWidget(content)
        outer_layout.addWidget(scroll)

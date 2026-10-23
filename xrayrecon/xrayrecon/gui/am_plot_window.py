from __future__ import annotations

import textwrap
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyqtgraph.Qt import QtCore, QtWidgets

from ..materials import MATERIALS_DB, MATERIAL_COLORS
from ..reconstruction_metrics import reconstruction_fidelity


class AMAnalysisPanel(QtWidgets.QWidget):
    """Embedded quantitative Alvarez-Macovski review workspace.

    The panel deliberately keeps direct inversion and material-regularized
    interpretation selectable, shows data-consistency/confidence maps beside
    them, and labels electron-density units explicitly. Synthetic ground truth
    is shown when available; production scans may omit those arrays.
    """

    sigCloseRequested = QtCore.Signal()
    sigInspectRequested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.result: dict | None = None

        self.backBtn = QtWidgets.QPushButton("Return to Viewer")
        self.backBtn.clicked.connect(self.sigCloseRequested)
        self.modeCombo = QtWidgets.QComboBox()
        self.modeCombo.addItem("Calibrated quantitative", "calibrated")
        self.modeCombo.addItem("Material-regularized interpretation", "regularized")
        self.modeCombo.addItem("Raw inversion (audit)", "raw")
        self.modeCombo.currentIndexChanged.connect(self._draw)
        self.qualityCombo = QtWidgets.QComboBox()
        self.qualityCombo.addItem("Z-effective 90% uncertainty", "zeff_uncertainty")
        self.qualityCombo.addItem("Electron-density 90% uncertainty", "rhoe_uncertainty")
        self.qualityCombo.addItem("Projection residual", "residual")
        self.qualityCombo.currentIndexChanged.connect(self._draw)

        self.sliceSlider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.sliceSlider.setRange(0, 0)
        self.sliceSlider.valueChanged.connect(self._draw)
        self.sliceLabel = QtWidgets.QLabel("Slice 0")
        self.metricSummary = QtWidgets.QLabel("Run a decomposition to inspect it here.")
        self.metricSummary.setWordWrap(True)
        self.metricSummary.setStyleSheet("color: #555;")
        self.acquisitionSummary = QtWidgets.QLabel("Acquisition metadata unavailable")
        self.acquisitionSummary.setWordWrap(True)
        self.calibrationStatus = QtWidgets.QLabel("Uncalibrated")
        self.calibrationStatus.setWordWrap(True)
        for combo in (self.modeCombo, self.qualityCombo):
            combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(12)
            combo.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        for label in (self.metricSummary, self.acquisitionSummary, self.calibrationStatus):
            label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
            label.setMinimumWidth(0)
        # Slice-dependent text must not move the canvas as lines wrap differently.
        self.metricSummary.setFixedHeight(self.fontMetrics().lineSpacing() * 3 + 4)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(self.backBtn)
        self.inspectBtn = QtWidgets.QPushButton("Error inspection…")
        self.inspectBtn.setToolTip("Inspect reference errors with linked maps, selections, profiles, and numerical export")
        self.inspectBtn.clicked.connect(self.sigInspectRequested)
        controls.addWidget(self.inspectBtn)
        controls.addSpacing(12)
        controls.addWidget(QtWidgets.QLabel("Display:"))
        controls.addWidget(self.modeCombo)
        controls.addWidget(QtWidgets.QLabel("Quality map:"))
        controls.addWidget(self.qualityCombo)
        navigation = QtWidgets.QHBoxLayout()
        self.sliceLabel.setMinimumWidth(self.sliceLabel.fontMetrics().horizontalAdvance("Slice 9999 of 9999"))
        navigation.addWidget(self.sliceLabel)
        navigation.addWidget(self.sliceSlider, 1)

        self.figure = Figure(figsize=(10, 7))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.canvas.setMinimumSize(0, 0)
        self._createPlots()

        layout = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel("Quantitative Alvarez–Macovski Analysis")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)
        layout.addLayout(controls)
        layout.addLayout(navigation)
        layout.addWidget(self.acquisitionSummary)
        layout.addWidget(self.calibrationStatus)
        layout.addWidget(self.metricSummary)
        layout.addWidget(self.canvas, 1)

    def _createPlots(self):
        """Allocate axes/colorbars once; data updates cannot renegotiate margins.

        A divider fits each image and its colorbar inside a fixed grid cell on
        resize. Unlike tight_layout on every slice, tick magnitudes and titles
        cannot shift the grid, or progressively shrink it on repeated redraws.
        """
        self._axes = self.figure.subplots(2, 3)
        self.figure.subplots_adjust(left=.025, right=.94, bottom=.05, top=.88,
                                    wspace=.42, hspace=.30)
        self._images, self._colorbars = [], []
        for axis in self._axes.flat:
            image = axis.imshow(np.zeros((1, 1)), origin="lower", vmin=0, vmax=1)
            axis.set_xticks([])
            axis.set_yticks([])
            divider = make_axes_locatable(axis)
            color_axis = divider.append_axes("right", size="4%", pad=.08)
            formatter = ScalarFormatter(useOffset=False, useMathText=True)
            formatter.set_powerlimits((-2, 3))
            colorbar = self.figure.colorbar(image, cax=color_axis, format=formatter)
            color_axis.tick_params(labelsize=8, pad=2)
            color_axis.yaxis.get_offset_text().set_fontsize(8)
            self._images.append(image)
            self._colorbars.append(colorbar)
        self.figure.suptitle(
            "Raw results remain available for audit; confidence is not a probability.",
            fontsize=9, y=.99,
        )

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

        # Changing modes while replacing the result must not draw using the old
        # slider index (which can be outside the new volume).
        blockers = [QtCore.QSignalBlocker(widget) for widget in (self.modeCombo, self.qualityCombo)]
        self.result = result
        self._reference_ranges = {
            key: float(result[key].max()) - float(result[key].min())
            for key in ("zeff_ground_truth", "rhoe_ground_truth") if key in result
        }
        config = getattr(result, "acquisition_config", None)
        if config is None:
            self.acquisitionSummary.setText("Acquisition metadata unavailable")
        else:
            noise = (
                f"{config.photon_count:,} photons/view (seed {config.random_seed})"
                if config.photon_count is not None
                else "noise-free forward model"
            )
            self.acquisitionSummary.setText(
                f"{config.source_kind.title()} acquisition · "
                f"{config.low_spectrum.label} / {config.high_spectrum.label} · "
                f"{config.num_angles} views · {config.pixel_size_cm * 10:.3f} mm pixels · {noise}"
            )
        calibration = getattr(result, "calibration_model", None)
        if calibration is None:
            self.calibrationStatus.setText(
                "Calibration: none. Raw and material-regularized outputs are available; "
                "no quantitative uncertainty interval is claimed."
            )
            self.calibrationStatus.setStyleSheet("color: #9a6700;")
            self.modeCombo.setCurrentIndex(1)
            self.qualityCombo.setCurrentIndex(2)
        else:
            compatibility = getattr(
                result, "calibration_compatibility", "not recorded"
            )
            compatibility_note = (
                "acquisition physics verified"
                if compatibility == "verified"
                else "acquisition compatibility not recorded"
            )
            source_note = (
                "synthetic reference calibration; measured scanner data not supplied"
                if getattr(calibration, "calibration_source", "synthetic_reference")
                == "synthetic_reference"
                else "measured reference calibration"
            )
            self.calibrationStatus.setText(
                f"Calibration: {calibration.calibration_id} · nominal "
                f"{calibration.nominal_coverage:.0%} intervals · "
                f"{compatibility_note} · {source_note}."
            )
            self.calibrationStatus.setStyleSheet(
                "color: #176b2c;"
                if compatibility == "verified"
                else "color: #9a6700;"
            )
            self.modeCombo.setCurrentIndex(0)
            self.qualityCombo.setCurrentIndex(0)
        representative = AMPlotWindow._pick_representative_slice(
            result.get("zeff_ground_truth", result["zeff_recovered"])
        )
        self.sliceSlider.blockSignals(True)
        self.sliceSlider.setRange(0, shape[0] - 1)
        self.sliceSlider.setValue(representative)
        self.sliceSlider.blockSignals(False)
        del blockers
        self._draw()

    def _draw(self, *_args) -> None:
        if self.result is None:
            return
        result = self.result
        index = self.sliceSlider.value()
        self.sliceLabel.setText(f"Slice {index + 1} of {self.sliceSlider.maximum() + 1}")
        mode = self.modeCombo.currentData()
        if mode == "calibrated" and "zeff_calibrated" in result:
            z_key, r_key, mode_name = (
                "zeff_calibrated",
                "rhoe_calibrated",
                "Calibrated quantitative",
            )
        elif mode == "regularized":
            z_key, r_key, mode_name = (
                "zeff_regularized",
                "rhoe_regularized",
                "Material-regularized",
            )
        else:
            z_key, r_key, mode_name = (
                "zeff_recovered",
                "rhoe_recovered",
                "Raw inversion",
            )

        zeff = result[z_key][index]
        rhoe = result[r_key][index]
        zeff_gt = result.get("zeff_ground_truth")
        rhoe_gt = result.get("rhoe_ground_truth")
        confidence = result.get("confidence_calibrated", result["confidence"])[index]
        residual = result["projection_residual"][index]
        quality_mode = self.qualityCombo.currentData()
        if quality_mode == "zeff_uncertainty" and "zeff_interval_upper" in result:
            quality_image = 0.5 * (
                result["zeff_interval_upper"][index]
                - result["zeff_interval_lower"][index]
            )
            quality_title = "Z-effective 90% half-width"
            quality_label = "Z-effective"
        elif quality_mode == "rhoe_uncertainty" and "rhoe_interval_upper" in result:
            quality_image = 0.5 * (
                result["rhoe_interval_upper"][index]
                - result["rhoe_interval_lower"][index]
            )
            quality_title = "Electron-density 90% half-width"
            quality_label = "rho_e / rho_e,water"
        else:
            quality_image = residual
            quality_title = "Backprojected projection inconsistency"
            quality_label = "normalized mismatch"
        quality_max = max(float(np.percentile(quality_image, 99)), 1e-6)

        axes = self._axes
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
            (axes[0, 2], confidence, "Reconstruction quality indicator", "viridis", 0.0, 1.0, "0–1 quality indicator"),
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
            (axes[1, 2], quality_image, quality_title, "magma", 0.0, quality_max, quality_label),
        ]
        for (axis, image, title, cmap, vmin, vmax, label), artist, colorbar in zip(panels, self._images, self._colorbars):
            artist.set_data(image)
            artist.set_extent((-.5, image.shape[1] - .5, -.5, image.shape[0] - .5))
            artist.set_cmap(cmap)
            artist.set_clim(vmin, vmax)
            axis.set_title(textwrap.fill(title, width=32), fontsize=9)
            colorbar.set_label(label, fontsize=8, labelpad=3)
        self.canvas.draw_idle()
        self._update_metrics(index, z_key, r_key)

    def _update_metrics(self, index: int, z_key: str, r_key: str) -> None:
        result = self.result
        zeff_gt = result.get("zeff_ground_truth")
        rhoe_gt = result.get("rhoe_ground_truth")
        confidence = result.get("confidence_calibrated", result["confidence"])[index]
        residual = result["projection_residual"][index]
        summary = (
            f"Median confidence {np.median(confidence):.2f} · "
            f"median normalized projection residual {np.median(residual):.3f}. "
        )
        if zeff_gt is not None and rhoe_gt is not None:
            fidelity_text = []
            for name, reference, key in (("Z-effective", zeff_gt, z_key), ("Electron density", rhoe_gt, r_key)):
                metrics = reconstruction_fidelity(
                    reference[index], result[key][index],
                    data_range=self._reference_ranges["zeff_ground_truth" if key == z_key else "rhoe_ground_truth"],
                )
                snr = "unavailable" if metrics["snr_db"] is None else f"{metrics['snr_db']:.2f} dB"
                ssi = "unavailable" if metrics["ssi"] is None else f"{metrics['ssi']:.4f}"
                fidelity_text.append(f"{name}: SNR {snr}, SSI (SSIM) {ssi}")
            summary += "Full-slice fidelity: " + "; ".join(fidelity_text) + ". "
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
                    f"Synthetic-truth slice errors (including boundaries): "
                    f"Z-effective {z_error:.1%}; "
                    f"relative electron density {r_error:.1%}."
                )
                if "zeff_interval_lower" in result:
                    z_covered = (
                        result["zeff_interval_lower"][index][mask]
                        <= zeff_gt[index][mask]
                    ) & (
                        zeff_gt[index][mask]
                        <= result["zeff_interval_upper"][index][mask]
                    )
                    r_covered = (
                        result["rhoe_interval_lower"][index][mask]
                        <= rhoe_gt[index][mask]
                    ) & (
                        rhoe_gt[index][mask]
                        <= result["rhoe_interval_upper"][index][mask]
                    )
                    summary += (
                        f" Empirical interval coverage: Z-effective {np.mean(z_covered):.1%}; "
                        f"electron density {np.mean(r_covered):.1%}."
                    )
        else:
            summary += "SNR and SSI unavailable: no ground-truth reference."
        self.metricSummary.setText(summary)
        self.metricSummary.setToolTip(summary)


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


class _HorizontalMaterialLegendWidget(QtWidgets.QWidget):
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


class MaterialLegendWidget(QtWidgets.QWidget):
    """Compact material legend used by the AM material color mode."""

    MAX_HEIGHT = 96
    MIN_ENTRY_WIDTH = 165

    def __init__(self, parent=None):
        super().__init__(parent)
        try:
            fixedPolicy = QtWidgets.QSizePolicy.Policy.Fixed
            preferredPolicy = QtWidgets.QSizePolicy.Policy.Preferred
            expandingPolicy = QtWidgets.QSizePolicy.Policy.Expanding
            noFrame = QtWidgets.QFrame.Shape.NoFrame
            alignTop = QtCore.Qt.AlignmentFlag.AlignTop
            scrollAlwaysOff = QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            scrollAsNeeded = QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        except AttributeError:
            fixedPolicy = QtWidgets.QSizePolicy.Fixed
            preferredPolicy = QtWidgets.QSizePolicy.Preferred
            expandingPolicy = QtWidgets.QSizePolicy.Expanding
            noFrame = QtWidgets.QFrame.NoFrame
            alignTop = QtCore.Qt.AlignTop
            scrollAlwaysOff = QtCore.Qt.ScrollBarAlwaysOff
            scrollAsNeeded = QtCore.Qt.ScrollBarAsNeeded

        self.setMaximumHeight(self.MAX_HEIGHT)
        self.setSizePolicy(expandingPolicy, fixedPolicy)
        self.setToolTip(
            "Approximate nearest-match to recovered Z_eff/rho_e; noisier for low-Z materials like FR4."
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 2, 6, 2)
        layout.setSpacing(1)

        title = QtWidgets.QLabel("Material Legend:")
        title.setStyleSheet("font-weight: bold;")
        layout.addWidget(title)

        self._scroll = QtWidgets.QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(noFrame)
        self._scroll.setHorizontalScrollBarPolicy(scrollAlwaysOff)
        self._scroll.setVerticalScrollBarPolicy(scrollAsNeeded)
        self._scroll.setSizePolicy(expandingPolicy, preferredPolicy)
        layout.addWidget(self._scroll)

        content = QtWidgets.QWidget()
        self._grid = QtWidgets.QGridLayout(content)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(10)
        self._grid.setVerticalSpacing(1)
        self._entries = []
        self._columnCount = 0

        for idx, name in enumerate(sorted(MATERIALS_DB.keys())):
            props = MATERIALS_DB[name]
            r, g, b, _a = MATERIAL_COLORS[name]

            entry = QtWidgets.QWidget()
            entry.setSizePolicy(expandingPolicy, fixedPolicy)
            entryLayout = QtWidgets.QHBoxLayout(entry)
            entryLayout.setContentsMargins(0, 0, 0, 0)
            entryLayout.setSpacing(4)

            swatch = QtWidgets.QLabel()
            swatch.setFixedSize(10, 10)
            swatch.setStyleSheet(
                f"background-color: rgb({r},{g},{b}); border: 1px solid #444;"
            )
            entryLayout.addWidget(swatch)

            text = QtWidgets.QLabel(
                f"{props.name} Z={props.z_eff:.1f} rho={props.electron_density:.2f}"
            )
            text.setWordWrap(False)
            text.setSizePolicy(expandingPolicy, fixedPolicy)
            entryLayout.addWidget(text)
            entryLayout.addStretch()
            self._entries.append(entry)

        self._alignTop = alignTop
        self._scroll.setWidget(content)
        self._reflowLegend()

    def _reflowLegend(self):
        viewportWidth = max(1, self._scroll.viewport().width())
        columns = max(1, viewportWidth // self.MIN_ENTRY_WIDTH)
        if columns == self._columnCount and self._grid.count() == len(self._entries):
            return

        while self._grid.count():
            self._grid.takeAt(0)

        self._columnCount = columns
        for idx, entry in enumerate(self._entries):
            row = idx // columns
            col = idx % columns
            self._grid.addWidget(entry, row, col, alignment=self._alignTop)
        for col in range(columns):
            self._grid.setColumnStretch(col, 1)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reflowLegend()

    def sizeHint(self):
        hint = super().sizeHint()
        return QtCore.QSize(
            max(hint.width(), self.MIN_ENTRY_WIDTH * 2),
            min(hint.height(), self.MAX_HEIGHT),
        )

    def minimumSizeHint(self):
        hint = super().minimumSizeHint()
        return QtCore.QSize(
            min(hint.width(), self.MIN_ENTRY_WIDTH),
            min(hint.height(), self.MAX_HEIGHT),
        )

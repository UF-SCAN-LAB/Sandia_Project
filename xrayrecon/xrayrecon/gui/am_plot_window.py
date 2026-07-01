from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from pyqtgraph.Qt import QtWidgets

from ..materials import MATERIALS_DB, MATERIAL_COLORS


class AMPlotWindow(QtWidgets.QDialog):
    """Non-modal window showing ground-truth vs. Alvarez-Macovski-recovered
    Z_eff / electron-density maps for a representative slice.

    Also prints a per-material accuracy table to the console (mean +/- std
    recovered value grouped by ground-truth Z_eff), mirroring the
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
        zeff_rec = result["zeff_recovered"][slice_idx]
        rhoe_gt = result["rhoe_ground_truth"][slice_idx]
        rhoe_rec = result["rhoe_recovered"][slice_idx]

        vmax_z = max(float(zeff_gt.max()), float(zeff_rec.max()), 1e-6)
        vmax_r = max(float(rhoe_gt.max()), float(rhoe_rec.max()), 1e-6)

        axes = fig.subplots(2, 2)
        panels = [
            (axes[0, 0], zeff_gt, "Ground-truth $Z_{eff}$", "plasma", vmax_z, "$Z_{eff}$"),
            (axes[0, 1], zeff_rec, "Recovered $Z_{eff}$", "plasma", vmax_z, "$Z_{eff}$"),
            (axes[1, 0], rhoe_gt, "Ground-truth $\\rho_e$", "cividis", vmax_r, "$\\rho_e$"),
            (axes[1, 1], rhoe_rec, "Recovered $\\rho_e$", "cividis", vmax_r, "$\\rho_e$"),
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
        zeff_rec = result["zeff_recovered"]
        rhoe_gt = result["rhoe_ground_truth"]
        rhoe_rec = result["rhoe_recovered"]

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
        layout = QtWidgets.QHBoxLayout(self)
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

            text = QtWidgets.QLabel(f"{props.name} (Z={props.z_eff:.1f}, ρ={props.electron_density:.2f})")
            layout.addWidget(text)

        layout.addStretch()

        note = QtWidgets.QLabel(
            "Approximate -- nearest-match to recovered (Z_eff, ρ_e); noisier for low-Z materials like FR4."
        )
        note.setStyleSheet("color: #888; font-style: italic;")
        layout.addWidget(note)

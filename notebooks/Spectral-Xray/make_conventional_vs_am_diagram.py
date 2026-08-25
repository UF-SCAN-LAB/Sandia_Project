"""
Generate a poster-ready comparison: Conventional (single-energy) CT reconstruction
vs. Alvarez-Macovski (dual-energy) CT reconstruction.

Reuses the exact phantom + pipeline building blocks from make_am_process_diagram.py
so every panel is a real computed result, not a schematic mock-up. The conventional
row FBPs the single high-energy sinogram directly (what a standard scanner does);
the AM row decomposes into basis sinograms *before* reconstructing (what
alvarez_macovski.py does) -- that ordering is the actual point of difference, and
it's called out explicitly in the arrow labels and row callouts.

Output: conventional_vs_am_diagram.png
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.interpolate import griddata
from skimage.transform import radon, iradon

from make_am_process_diagram import build_phantom, make_spectrum, forward_arr, to_Zeff, to_rho_e, MATERIALS

OUT_PATH = os.path.join(os.path.dirname(__file__), "conventional_vs_am_diagram.png")

# Demo-only display colors for the four phantom materials (distinct from the
# physical MATERIAL_COLORS in xrayrecon/materials.py, which use real Z-based
# tinting -- this demo phantom's (a1, a2) values are illustrative, see
# make_am_process_diagram.py's MATERIALS comment).
DISPLAY_COLORS = {
    "air": (0.05, 0.05, 0.08),
    "fr4": (0.13, 0.55, 0.55),
    "copper": (0.87, 0.65, 0.13),
    "solder": (0.80, 0.15, 0.15),
}
_NAMES = list(MATERIALS.keys())
_PTS = np.array([MATERIALS[n] for n in _NAMES])
_SCALE = np.where(_PTS.max(axis=0) > 0, _PTS.max(axis=0), 1.0)


def classify_rgb(a1_map, a2_map):
    """Nearest-material classification in normalized (a1, a2) space -> RGB image."""
    flat = np.stack([a1_map.ravel(), a2_map.ravel()], axis=1) / _SCALE
    ref = _PTS / _SCALE
    d2 = ((flat[:, None, :] - ref[None, :, :]) ** 2).sum(axis=2)
    idx = d2.argmin(axis=1)
    rgb = np.array([DISPLAY_COLORS[n] for n in _NAMES])[idx]
    return rgb.reshape(*a1_map.shape, 3)


def run_both(N=128, n_angles=90, verbose=True):
    def say(*a):
        if verbose:
            print(*a)

    x, a1_map, a2_map = build_phantom(N)
    dx = x[1] - x[0]

    E_lo, phi_lo = make_spectrum(20, 80)
    E_hi, phi_hi = make_spectrum(20, 140)
    theta = np.linspace(0, 180, n_angles, endpoint=False)

    say("Forward projecting phantom...")
    sino_A1 = radon(a1_map, theta=theta, circle=True) * dx
    sino_A2 = radon(a2_map, theta=theta, circle=True) * dx

    sino_Llo = forward_arr(sino_A1, sino_A2, E_lo, phi_lo).reshape(sino_A1.shape)
    sino_Lhi = forward_arr(sino_A1, sino_A2, E_hi, phi_hi).reshape(sino_A1.shape)

    say("Conventional: FBP the single-energy sinogram directly...")
    recon_conventional = iradon(sino_Lhi, theta=theta, filter_name="ramp", circle=True) / dx

    say("AM: decomposing in sinogram space, then FBP each basis...")
    A1g = np.linspace(0, 0.09, 45)
    A2g = np.linspace(0, 40, 55)
    AA1, AA2 = np.meshgrid(A1g, A2g, indexing="ij")
    Llo_grid = forward_arr(AA1, AA2, E_lo, phi_lo)
    Lhi_grid = forward_arr(AA1, AA2, E_hi, phi_hi)
    pts = np.column_stack([Llo_grid, Lhi_grid])
    query = np.column_stack([sino_Llo.ravel(), sino_Lhi.ravel()])

    def invert(values):
        lin = griddata(pts, values, query, method="linear")
        nan = ~np.isfinite(lin)
        if nan.any():
            lin[nan] = griddata(pts, values, query[nan], method="nearest")
        return lin

    sino_A1_rec = np.clip(invert(AA1.ravel()), 0, None).reshape(sino_A1.shape)
    sino_A2_rec = np.clip(invert(AA2.ravel()), 0, None).reshape(sino_A1.shape)

    recon_a1 = np.clip(iradon(sino_A1_rec, theta=theta, filter_name="ramp", circle=True) / dx, 0, None)
    recon_a2 = np.clip(iradon(sino_A2_rec, theta=theta, filter_name="ramp", circle=True) / dx, 0, None)

    say("Done.")
    return dict(
        object_rgb=classify_rgb(a1_map, a2_map),
        sino_Lhi=sino_Lhi, sino_Llo=sino_Llo,
        recon_conventional=recon_conventional,
        material_rgb=classify_rgb(recon_a1, recon_a2),
        Zeff=to_Zeff(recon_a1, recon_a2), rhoe=to_rho_e(recon_a2),
    )


# ============================================================================
# Figure
# ============================================================================

def panel(fig, x, y, w, h, img, cmap, edge, label):
    ax = fig.add_axes([x, y, w, h])
    ax.imshow(img, cmap=cmap, origin="lower")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(edge); sp.set_linewidth(2.2)
    ax.text(0.5, -0.06, label, transform=ax.transAxes, ha="center", va="top",
             fontsize=9.5, color="#222")
    return ax


def harrow(ax, x0, x1, y, color, label=None, fontsize=9.0):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                 mutation_scale=22, linewidth=2.4, color=color, zorder=1))
    if label:
        ax.text((x0 + x1) / 2, y + 0.014, label, ha="center", va="bottom",
                 fontsize=fontsize, style="italic", color="#333", wrap=True)


def callout(ax, x, y, w, text, face, edge, textcolor):
    box = FancyBboxPatch((x, y), w, 0.075, boxstyle="round,pad=0.008,rounding_size=0.01",
                          fc=face, ec=edge, linewidth=1.6, zorder=2)
    ax.add_patch(box)
    ax.text(x + w / 2, y + 0.0375, text, ha="center", va="center", fontsize=9.3,
             fontweight="bold", color=textcolor, wrap=True)


def main():
    R = run_both(N=128, n_angles=90)

    fig = plt.figure(figsize=(18, 10.5))
    bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, 1); bg.set_ylim(0, 1)
    bg.axis("off")

    bg.text(0.5, 0.978, "Conventional CT vs. Alvarez-Macovski Dual-Energy CT",
             ha="center", va="top", fontsize=21, fontweight="bold", color="#111")
    bg.text(0.5, 0.938,
             "Same phantom, same detector geometry -- the only change is a second energy "
             "channel and one extra step: decomposing into physical basis maps\n"
             r"$\mu(x,y,E)=a_1(x,y)f_1(E)+a_2(x,y)f_2(E)$ before reconstruction, instead of after.",
             ha="center", va="top", fontsize=11, color="#333")

    A_CONV = "#a8322f"   # conventional row accent (warm red)
    A_AM = "#2f9e64"     # AM row accent (green)
    OBJ_EDGE = "#444444"

    col_x = [0.05, 0.32, 0.70]
    col_w = [0.13, 0.30, 0.22]
    H = 0.20
    y_top = 0.585
    y_bot = 0.20

    # ---------------- Row A: Conventional ----------------
    bg.text(0.02, y_top + H + 0.05, "A", fontsize=15, fontweight="bold", color="white",
             ha="center", va="center",
             bbox=dict(boxstyle="circle,pad=0.35", fc=A_CONV, ec="none"))
    bg.text(0.05, y_top + H + 0.05, "Conventional (single-energy) reconstruction",
             fontsize=13.5, fontweight="bold", color=A_CONV, ha="left", va="center")

    panel(fig, col_x[0], y_top, col_w[0], H, R["object_rgb"], None, OBJ_EDGE,
          "Object\n(Cu / solder / FR4 PCB)")
    panel(fig, col_x[1], y_top, col_w[1], H, R["sino_Lhi"], "Reds", A_CONV,
          r"Single-energy sinogram $L(\theta,t)$ -- 140 kVp only")
    panel(fig, col_x[2], y_top, col_w[2], H, R["recon_conventional"], "gray", A_CONV,
          "Reconstructed attenuation image (FBP)")

    ymid_t = y_top + H / 2
    harrow(bg, col_x[0] + col_w[0] + 0.006, col_x[1] - 0.006, ymid_t, A_CONV, "acquire\n(1 energy)")
    harrow(bg, col_x[1] + col_w[1] + 0.006, col_x[2] - 0.006, ymid_t, A_CONV, "FBP directly on $L$")

    callout(bg, col_x[2], y_top - 0.095, col_w[2],
            "Cu and solder both read as \"bright\" -- one grayscale\n"
            "value can't separate them, plus beam-hardening cupping.",
            face="#fbe9e7", edge=A_CONV, textcolor="#7a1f1c")

    # ---------------- Row B: Alvarez-Macovski ----------------
    bg.text(0.02, y_bot + H + 0.05, "B", fontsize=15, fontweight="bold", color="white",
             ha="center", va="center",
             bbox=dict(boxstyle="circle,pad=0.35", fc=A_AM, ec="none"))
    bg.text(0.05, y_bot + H + 0.05, "Alvarez-Macovski (dual-energy) reconstruction",
             fontsize=13.5, fontweight="bold", color=A_AM, ha="left", va="center")

    panel(fig, col_x[0], y_bot, col_w[0], H, R["object_rgb"], None, OBJ_EDGE,
          "Object\n(same PCB phantom)")

    gap = 0.012
    pw = (col_w[1] - gap) / 2
    panel(fig, col_x[1], y_bot, pw, H, R["sino_Llo"], "Blues", A_AM, r"$L_{lo}$ 80 kVp")
    panel(fig, col_x[1] + pw + gap, y_bot, pw, H, R["sino_Lhi"], "Reds", A_AM, r"$L_{hi}$ 140 kVp")

    panel(fig, col_x[2], y_bot, col_w[2], H, R["material_rgb"], None, A_AM,
          r"Material-ID map from recovered $\hat{Z}_{eff}$, $\hat{\rho}_e$")

    ymid_b = y_bot + H / 2
    harrow(bg, col_x[0] + col_w[0] + 0.006, col_x[1] - 0.006, ymid_b, A_AM, "acquire\n(2 energies)")
    harrow(bg, col_x[1] + col_w[1] + 0.006, col_x[2] - 0.006, ymid_b, A_AM,
           "decompose\n→ FBP ×2", fontsize=9.0)

    callout(bg, col_x[2], y_bot - 0.095, col_w[2],
            "Decomposition happens before reconstruction, so beam\n"
            "hardening cancels out and materials separate automatically.",
            face="#e6f6ec", edge=A_AM, textcolor="#1c6b3f")

    bg.text(0.5, 0.025,
            "Both pipelines start from the same object and the same acquisition geometry -- Alvarez-Macovski adds one more "
            "energy channel and moves basis decomposition ahead of reconstruction, trading a single qualitative image for two "
            "quantitative, artifact-reduced material maps.",
            ha="center", va="bottom", fontsize=9.3, color="#555", style="italic")

    fig.savefig(OUT_PATH, dpi=150, facecolor="white")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()

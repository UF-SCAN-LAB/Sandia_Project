"""
Diagram explaining K-edge subtraction in spectral CT.

Shows four panels:
  A. Physics of K-edge absorption (electron shell diagram)
  B. Attenuation coefficient μ(E) — jump at K-edge energy
  C. K-edge subtraction formula: ΔL = L_post − L_pre → contrast agent map
  D. Why standard AM fails: L_hi > L_lo when K-edge sits between windows

Output: kedge_diagram.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch
from matplotlib.gridspec import GridSpec

OUT_PATH = os.path.join(os.path.dirname(__file__), "kedge_diagram.png")

# ── Attenuation model ────────────────────────────────────────────────────────
def klein_nishina(E):
    alpha = E / 511.0
    t1 = (1 + alpha) / alpha**3
    t2 = (2 * alpha * (1 + alpha)) / (1 + 2 * alpha) - np.log(1 + 2 * alpha)
    t3 = np.log(1 + 2 * alpha) / (2 * alpha)
    t4 = (1 + 3 * alpha) / (1 + 2 * alpha)**2
    return t1 * t2 + t3 - t4

_KN50 = klein_nishina(50.0)


def mu_tissue(E):
    """Soft tissue: Compton + photoelectric, no K-edge."""
    return 0.6 * E**(-3) / 20.0**(-3) + 0.4 * klein_nishina(E) / _KN50


def mu_kedge(E, E_K, jump=1.8, scale=0.9):
    """Material with a K-edge: smooth tissue-like baseline + step at E_K."""
    base = scale * mu_tissue(E)
    edge = jump * np.where(E >= E_K, np.exp(-(E - E_K) / 12.0), 0.0)
    return base + edge


# ── Colour palette ────────────────────────────────────────────────────────────
C_IODINE = "#2e7cbe"
C_GD     = "#c44b8a"
C_TISSUE = "#6aaa5c"
C_PRE    = "#f5a623"
C_POST   = "#d0021b"
C_DIFF   = "#7b5ea7"
C_BG     = "#f7f7f7"
C_PANEL  = "white"


def label_box(ax, x, y, text, color, fs=9.5, ha="center", va="center"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color="white",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", fc=color, ec="none", zorder=5))


# ════════════════════════════════════════════════════════════════════════════════
def main():
    fig = plt.figure(figsize=(17, 11), facecolor=C_BG)
    fig.text(0.5, 0.975, "K-Edge Subtraction in Spectral CT",
             ha="center", va="top", fontsize=20, fontweight="bold", color="#111")
    fig.text(
        0.5, 0.945,
        "K-edges create discontinuous jumps in μ(E). Subtracting just-below from just-above "
        "isolates the K-edge material, extending the Alvarez-Macovski framework to 3+ bases.",
        ha="center", va="top", fontsize=11, color="#444", style="italic")

    gs = GridSpec(2, 2, figure=fig, left=0.06, right=0.97, top=0.90, bottom=0.06,
                  hspace=0.45, wspace=0.35)

    # ── Panel A: Physics ──────────────────────────────────────────────────────
    axA = fig.add_subplot(gs[0, 0])
    axA.set_xlim(0, 10); axA.set_ylim(0, 10)
    axA.axis("off")
    axA.set_facecolor(C_PANEL)
    axA.set_title("A   Physics of K-Edge Absorption",
                  fontsize=12, fontweight="bold", loc="left", pad=8)

    # Atom nucleus
    nucleus = Circle((5, 5), 0.55, color="#e8893b", zorder=6)
    axA.add_patch(nucleus)
    axA.text(5, 5, "nucleus", ha="center", va="center", fontsize=7, color="white",
             fontweight="bold", zorder=7)

    # Electron shells (K, L, M)
    shell_r = [1.6, 2.8, 4.2]
    shell_names = ["K-shell\n(n=1)", "L-shell\n(n=2)", "M-shell\n(n=3)"]
    shell_colors = ["#3b7fc4", "#7a5bb0", "#3ba36b"]
    for r, name, col in zip(shell_r, shell_names, shell_colors):
        circ = plt.Circle((5, 5), r, fill=False, edgecolor=col,
                          linewidth=1.8, linestyle="--", zorder=3)
        axA.add_patch(circ)
        axA.text(5 + r * 0.72, 5 + r * 0.72, name, ha="center", va="center",
                 fontsize=7.5, color=col, fontweight="bold")

    # Electrons on K-shell
    for angle_deg in [0, 180]:
        rad = np.radians(angle_deg)
        ex = 5 + shell_r[0] * np.cos(rad)
        ey = 5 + shell_r[0] * np.sin(rad)
        axA.add_patch(Circle((ex, ey), 0.22, color="#3b7fc4", zorder=7))
        axA.text(ex, ey, "e⁻", ha="center", va="center", fontsize=7,
                 color="white", fontweight="bold", zorder=8)

    # Incoming photon arrow (from lower-left)
    axA.annotate("", xy=(5 - shell_r[0] * 0.95, 5),
                 xytext=(0.6, 5),
                 arrowprops=dict(arrowstyle="-|>", color=C_PRE, lw=2.5))
    axA.text(1.8, 5.3, r"$h\nu \geq E_K$", ha="center", fontsize=10,
             color=C_PRE, fontweight="bold")

    # Ejected electron
    axA.annotate("", xy=(1.0, 8.2),
                 xytext=(5 - shell_r[0] * 0.9, 5.1),
                 arrowprops=dict(arrowstyle="-|>", color=C_POST, lw=2.0,
                                 linestyle="dashed"))
    axA.text(1.5, 8.6, "e⁻ ejected\n(photoionisation)", ha="center",
             fontsize=8.5, color=C_POST, fontweight="bold")

    # Fluorescence photon
    axA.annotate("", xy=(8.8, 7.8),
                 xytext=(5 + shell_r[1] * 0.65, 5 + shell_r[1] * 0.65),
                 arrowprops=dict(arrowstyle="-|>", color="#aaa", lw=1.8,
                                 linestyle="dotted"))
    axA.text(8.8, 7.5, "fluorescence\nphoton", ha="center", fontsize=7.5,
             color="#888", style="italic")

    # Key fact box
    axA.text(5, 0.65,
             r"K-edge energy $E_K \propto Z^{4/3}$ — higher Z → higher $E_K$",
             ha="center", va="center", fontsize=8.8, color="#222",
             bbox=dict(boxstyle="round,pad=0.4", fc="#eef3fb", ec="#3b7fc4",
                       linewidth=1.5))

    # ── Panel B: μ(E) with K-edge jumps ──────────────────────────────────────
    axB = fig.add_subplot(gs[0, 1])
    axB.set_facecolor(C_PANEL)
    axB.set_title("B   Attenuation Coefficient μ(E) with K-Edges",
                  fontsize=12, fontweight="bold", loc="left", pad=8)

    E = np.linspace(10, 60, 2000)
    E_I  = 33.2   # Iodine K-edge
    E_Gd = 50.2   # Gadolinium K-edge

    mu_t   = mu_tissue(E)
    mu_I   = mu_kedge(E, E_I,  jump=2.2, scale=1.1)
    mu_Gd  = mu_kedge(E, E_Gd, jump=2.5, scale=0.85)

    axB.semilogy(E, mu_t,  color=C_TISSUE, lw=2.0, label="Soft tissue (no K-edge)", zorder=4)
    axB.semilogy(E, mu_I,  color=C_IODINE, lw=2.5, label="Iodine (I,  Z=53)", zorder=5)
    axB.semilogy(E, mu_Gd, color=C_GD,     lw=2.5, label="Gadolinium (Gd, Z=64)", zorder=5)

    # Vertical K-edge lines
    for E_K, col, name, ek_label in [
        (E_I,  C_IODINE, "I",  "I K-edge\n33.2 keV"),
        (E_Gd, C_GD,     "Gd", "Gd K-edge\n50.2 keV"),
    ]:
        axB.axvline(E_K, color=col, ls="--", lw=1.8, alpha=0.85, zorder=3)
        axB.text(E_K + 0.5, axB.get_ylim()[1] if axB.get_ylim()[1] > 1 else 3.5,
                 ek_label, fontsize=8.5, color=col, fontweight="bold", va="top")

    # Shade pre/post windows for Iodine as example
    pre_lo, pre_hi   = 25.0, 32.5
    post_lo, post_hi = 33.5, 43.0
    axB.axvspan(pre_lo,  pre_hi,  alpha=0.15, color=C_PRE,  label="Pre-K window")
    axB.axvspan(post_lo, post_hi, alpha=0.15, color=C_POST, label="Post-K window")
    axB.text((pre_lo + pre_hi) / 2,   1e-3, "pre-K",  ha="center", fontsize=8,
             color=C_PRE,  fontweight="bold", va="bottom")
    axB.text((post_lo + post_hi) / 2, 1e-3, "post-K", ha="center", fontsize=8,
             color=C_POST, fontweight="bold", va="bottom")

    # Jump annotation arrow for Iodine
    mu_just_below = float(mu_I[np.searchsorted(E, E_I) - 5])
    mu_just_above = float(mu_I[np.searchsorted(E, E_I) + 5])
    axB.annotate("", xy=(E_I + 1.5, mu_just_above),
                 xytext=(E_I + 1.5, mu_just_below),
                 arrowprops=dict(arrowstyle="<->", color=C_IODINE, lw=2.0))
    axB.text(E_I + 3.0, (mu_just_below * mu_just_above)**0.5,
             r"$\Delta\mu_K$", fontsize=10, color=C_IODINE, fontweight="bold",
             va="center")

    axB.set_xlabel("Photon energy  E (keV)", fontsize=10)
    axB.set_ylabel(r"$\mu(E)$  (a.u., log scale)", fontsize=10)
    axB.legend(fontsize=8, loc="upper right", framealpha=0.9)
    axB.grid(alpha=0.25, which="both")
    axB.set_xlim(10, 60)

    # ── Panel C: K-edge subtraction formula ──────────────────────────────────
    axC = fig.add_subplot(gs[1, 0])
    axC.set_xlim(0, 10); axC.set_ylim(0, 10)
    axC.axis("off")
    axC.set_facecolor(C_PANEL)
    axC.set_title("C   K-Edge Subtraction — Isolating the Contrast Agent",
                  fontsize=12, fontweight="bold", loc="left", pad=8)

    def formula_box(ax, x, y, w, h, lines, facecolor, border):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                   boxstyle="round,pad=0.1",
                                   fc=facecolor, ec=border, lw=2.0, zorder=4))
        for i, (txt, fs, col, bold) in enumerate(lines):
            ax.text(x + w / 2, y + h - 0.35 - i * 0.72, txt,
                    ha="center", va="top", fontsize=fs, color=col,
                    fontweight="bold" if bold else "normal",
                    math_fontfamily="cm", zorder=5)

    # Step 1 — measure pre-K
    formula_box(axC, 0.2, 7.5, 4.2, 2.0, [
        (r"$L_{\rm pre} = -\ln\!\!\int_{\rm pre}\! \Phi(E)\,e^{-A_1 f_1(E)-A_2 f_2(E)}\,dE$",
         9.0, C_PRE, True),
        ("Measure BELOW K-edge: tissue + bone only", 8.5, "#444", False),
        (r"No K-edge jump — standard Beer-Lambert", 8.0, "#777", False),
    ], "#fff8f0", C_PRE)

    # Step 2 — measure post-K
    formula_box(axC, 5.6, 7.5, 4.2, 2.0, [
        (r"$L_{\rm post} = -\ln\!\!\int_{\rm post}\!\Phi(E)\,e^{-A_1 f_1-A_2 f_2-A_3 f_3}\,dE$",
         9.0, C_POST, True),
        ("Measure ABOVE K-edge: tissue + bone + agent", 8.5, "#444", False),
        (r"$f_3(E)$ = K-edge basis function (step jump)", 8.0, "#777", False),
    ], "#fff0f0", C_POST)

    # Arrow down to subtraction
    for x_mid in [2.3, 7.7]:
        axC.annotate("", xy=(x_mid, 6.95), xytext=(x_mid, 7.45),
                     arrowprops=dict(arrowstyle="-|>", color="#999", lw=2.0))

    # Subtraction box
    formula_box(axC, 1.5, 5.1, 7.0, 1.75, [
        (r"$\Delta L = L_{\rm post} - L_{\rm pre}$", 13, C_DIFF, True),
        ("Tissue/bone terms cancel  →  contrast agent only", 9.5, "#333", False),
    ], "#f3eeff", C_DIFF)

    # Arrow down to result
    axC.annotate("", xy=(5, 4.55), xytext=(5, 5.05),
                 arrowprops=dict(arrowstyle="-|>", color=C_DIFF, lw=2.5))

    # Result box
    formula_box(axC, 1.5, 2.8, 7.0, 1.65, [
        (r"Reconstruct  $\hat{\mu}_{\rm agent}(x,y) = \mathrm{FBP}(\Delta L)$",
         10.5, "#111", True),
        (r"Direct map of K-edge material (iodine / gadolinium) concentration",
         9.0, "#444", False),
    ], "#efffef", C_TISSUE)

    # Comparison note
    axC.text(5, 1.7,
             "Same idea as Alvarez-Macovski but with 3 bases: {photoelectric, Compton, K-edge step}",
             ha="center", va="center", fontsize=8.8, color="#555", style="italic",
             bbox=dict(boxstyle="round,pad=0.3", fc="#f0f0f0", ec="#ccc", lw=1.2))

    # ── Panel D: Why standard AM fails ───────────────────────────────────────
    axD = fig.add_subplot(gs[1, 1])
    axD.set_facecolor(C_PANEL)
    axD.set_title("D   Why Standard AM Fails for K-Edge Data",
                  fontsize=12, fontweight="bold", loc="left", pad=8)

    # Simulate L_lo and L_hi along a 1-D detector for two cases
    det = np.linspace(0, 1, 200)
    # Case 1: no contrast agent — L_lo > L_hi (normal: more attenuation at low E)
    L_lo_normal = 0.4 + 0.3 * np.sin(np.pi * det)**2
    L_hi_normal = 0.25 + 0.18 * np.sin(np.pi * det)**2
    # Case 2: iodine in the object — L_hi window crosses K-edge → L_hi > L_lo
    iodine_profile = 0.35 * np.exp(-((det - 0.5)**2) / 0.02)
    L_lo_iodine = L_lo_normal.copy()
    L_hi_iodine = L_hi_normal + iodine_profile * 1.6

    axD.plot(det, L_lo_normal, color=C_PRE,    lw=2.0, ls="--", label=r"$L_{lo}$ (no agent)",  zorder=4)
    axD.plot(det, L_hi_normal, color=C_POST,   lw=2.0, ls="--", label=r"$L_{hi}$ (no agent)",  zorder=4)
    axD.plot(det, L_lo_iodine, color=C_PRE,    lw=2.5,          label=r"$L_{lo}$ (with iodine)", zorder=5)
    axD.plot(det, L_hi_iodine, color=C_POST,   lw=2.5,          label=r"$L_{hi}$ (with iodine)", zorder=5)
    axD.fill_between(det, L_lo_iodine, L_hi_iodine,
                     where=(L_hi_iodine > L_lo_iodine),
                     alpha=0.18, color=C_DIFF, label=r"Inversion region $L_{hi}>L_{lo}$")

    axD.axhline(0, color="#ccc", lw=0.8)
    axD.set_xlabel("Detector position", fontsize=10)
    axD.set_ylabel("Log-attenuation  L", fontsize=10)
    axD.legend(fontsize=8, loc="upper right", framealpha=0.92)
    axD.grid(alpha=0.25)
    axD.set_xlim(0, 1)

    # Annotation arrows and text
    peak_x = 0.5
    axD.annotate("", xy=(peak_x, float(L_hi_iodine[100]) + 0.03),
                 xytext=(peak_x, float(L_lo_iodine[100]) - 0.02),
                 arrowprops=dict(arrowstyle="<->", color=C_DIFF, lw=2.2))
    axD.text(peak_x + 0.04, (float(L_hi_iodine[100]) + float(L_lo_iodine[100])) / 2,
             r"$L_{hi} > L_{lo}$" + "\n(inverted!)", fontsize=9, color=C_DIFF,
             fontweight="bold", va="center")

    # Explanation box
    fail_text = (
        "AM assumes  $L_{lo} > L_{hi}$  (low-E always attenuates more).\n"
        "Iodine K-edge (33.2 keV) sits inside the high-energy window →\n"
        r"$L_{hi}$ spikes; AM returns unphysical $A_1 < 0$ to compensate."
    )
    axD.text(0.5, -0.2, fail_text,
             ha="center", va="top", fontsize=8.8, color="#333",
             transform=axD.transAxes, style="italic",
             bbox=dict(boxstyle="round,pad=0.4", fc="#fff0e0", ec=C_POST, lw=1.5))

    # ── Save ─────────────────────────────────────────────────────────────────
    fig.savefig(OUT_PATH, dpi=130, facecolor=C_BG, bbox_inches="tight")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()

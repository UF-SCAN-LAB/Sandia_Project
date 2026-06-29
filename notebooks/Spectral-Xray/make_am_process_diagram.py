"""
Generate an overall-process *storyboard* for the Alvarez-Macovski dual-energy CT demo.

This runs the exact phantom and pipeline from alvarez_macovski_demo.ipynb end-to-end
and embeds the real generated image(s) at every step, so the figure shows what the
data actually looks like as it flows through:

  1 Object (phantom)  a1, a2 maps
  2 Forward project   Radon -> A1, A2 sinograms
  3 Polychromatic     Beer-Lambert x 2 spectra -> measured L_lo, L_hi sinograms
  4 AM decomposition  invert per ray (2-D LUT) -> recovered A1_hat, A2_hat sinograms
  5 FBP reconstruction-> a1_hat, a2_hat images
  6 Physical maps     -> Z_eff, rho_e

Output: am_process_diagram.png
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.interpolate import griddata
from skimage.transform import radon, iradon

OUT_PATH = os.path.join(os.path.dirname(__file__), "am_process_diagram.png")


# ============================================================================
# Pipeline (mirrors alvarez_macovski_demo.ipynb)
# ============================================================================

def f1(E):
    return E ** (-3)


def klein_nishina(E):
    alpha = E / 511.0
    term1 = (1 + alpha) / alpha ** 3
    term2 = (2 * alpha * (1 + alpha)) / (1 + 2 * alpha) - np.log(1 + 2 * alpha)
    term3 = np.log(1 + 2 * alpha) / (2 * alpha)
    term4 = (1 + 3 * alpha) / (1 + 2 * alpha) ** 2
    return term1 * term2 + term3 - term4


_KN_REF = klein_nishina(50.0)


def f2(E):
    return klein_nishina(E) / _KN_REF


def make_spectrum(E_min, E_max, n_bins=200):
    E = np.linspace(E_min, E_max, n_bins)
    phi = np.maximum(0, (E_max - E) / (E_max - E_min))
    phi /= np.trapz(phi, E)
    return E, phi


def forward_arr(A1_arr, A2_arr, E, phi):
    """Vectorised polychromatic Beer-Lambert over arrays of (A1, A2)."""
    A1_arr = np.asarray(A1_arr, float).ravel()
    A2_arr = np.asarray(A2_arr, float).ravel()
    exponent = -np.outer(A1_arr, f1(E)) - np.outer(A2_arr, f2(E))
    transmitted = np.trapz(phi * np.exp(exponent), E, axis=1)
    return -np.log(np.clip(transmitted, 1e-12, None))


# Printed-circuit-board materials (a1 photoelectric, a2 Compton). Units are
# illustrative; chosen so substrate / copper / solder separate cleanly in
# (a1, a2) space while keeping line integrals within the inversion-LUT range.
# Z_eff values below are illustrative (compressed), not physical -- real copper
# is Z~29 and Sn-based solder Z~50; here they only need to order fr4 < Cu < solder.
MATERIALS = {
    "air":    (0.0,    0.0),    # board outline, drilled holes
    "fr4":    (0.0025, 9.0),    # glass-epoxy substrate   (illustrative Z_eff ~ 12)
    "copper": (0.045,  22.0),   # traces, pads, ground rail (illustrative Z_eff ~ 20)
    "solder": (0.075,  20.0),   # solder joints / bumps    (illustrative Z_eff ~ 24)
}


def build_phantom(N=128):
    """Top-down view of a simple printed circuit board."""
    x = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, x)

    def circle(cx, cy, r):
        return (X - cx) ** 2 + (Y - cy) ** 2 < r ** 2

    def rect(x0, x1, y0, y1):
        return (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)

    board = rect(-0.80, 0.80, -0.55, 0.55)

    copper = np.zeros((N, N), bool)
    copper |= rect(-0.72, 0.72, -0.50, -0.40)            # bottom ground rail
    finger_x = np.linspace(-0.60, 0.60, 8)
    for fx in finger_x:                                   # edge-connector fingers
        copper |= rect(fx - 0.035, fx + 0.035, 0.40, 0.53)
    for p in np.linspace(-0.24, 0.24, 7):                 # central IC pad ring
        copper |= rect(p - 0.018, p + 0.018, 0.30, 0.36)
        copper |= rect(p - 0.018, p + 0.018, -0.36, -0.30)
        copper |= rect(-0.36, -0.30, p - 0.018, p + 0.018)
        copper |= rect(0.30, 0.36, p - 0.018, p + 0.018)
    copper |= circle(-0.62, -0.05, 0.10)                  # power pads
    copper |= circle(0.62, -0.05, 0.10)
    for fx in finger_x[::2]:                              # routing traces
        copper |= rect(fx - 0.012, fx + 0.012, 0.12, 0.40)
    copper |= rect(-0.62, 0.62, 0.10, 0.135)             # horizontal bus
    via_centres = [(vx, 0.0) for vx in np.linspace(-0.45, 0.45, 7)]
    for vx, vy in via_centres:
        copper |= circle(vx, vy, 0.022)

    solder = np.zeros((N, N), bool)
    for fx in finger_x:
        solder |= circle(fx, 0.46, 0.030)
    solder |= circle(-0.62, -0.05, 0.05)
    solder |= circle(0.62, -0.05, 0.05)

    holes = np.zeros((N, N), bool)
    for hx, hy in [(-0.70, 0.46), (0.70, 0.46), (-0.70, -0.47), (0.70, -0.47)]:
        holes |= circle(hx, hy, 0.055)                    # mounting holes
    for vx, vy in via_centres:
        holes |= circle(vx, vy, 0.009)                    # drilled via barrels

    a1 = np.zeros((N, N))
    a2 = np.zeros((N, N))
    for mask, mat in [(board, "fr4"), (copper, "copper"),
                      (solder, "solder"), (holes, "air")]:
        a1[mask], a2[mask] = MATERIALS[mat]
    return x, a1, a2


# --- physical conversions (anchored to the FR4 substrate) ---
PE_EXPONENT = 3.8
_a1w, _a2w = MATERIALS["fr4"]
_K = 12.0 / (_a1w / _a2w) ** (1 / PE_EXPONENT)


def to_Zeff(a1, a2, eps=1e-9):
    ratio = np.where(a2 > eps, a1 / np.where(a2 > eps, a2, 1.0), 0.0)
    return np.where(a2 > eps, _K * np.clip(ratio, 0, None) ** (1 / PE_EXPONENT), 0.0)


def to_rho_e(a2):
    return a2 / _a2w


def run_pipeline(N=128, n_angles=90, verbose=True):
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
    n_det = sino_A1.shape[0]

    say("Polychromatic Beer-Lambert...")
    sino_Llo = forward_arr(sino_A1, sino_A2, E_lo, phi_lo).reshape(sino_A1.shape)
    sino_Lhi = forward_arr(sino_A1, sino_A2, E_hi, phi_hi).reshape(sino_A1.shape)

    say("Building AM inverse LUT and decomposing...")
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

    say("Filtered back-projection...")
    recon_a1 = np.clip(iradon(sino_A1_rec, theta=theta, filter_name="ramp",
                              circle=True) / dx, 0, None)
    recon_a2 = np.clip(iradon(sino_A2_rec, theta=theta, filter_name="ramp",
                              circle=True) / dx, 0, None)

    Zeff = to_Zeff(recon_a1, recon_a2)
    rhoe = to_rho_e(recon_a2)
    say("Done.")

    return dict(
        a1_map=a1_map, a2_map=a2_map,
        sino_A1=sino_A1, sino_A2=sino_A2,
        sino_Llo=sino_Llo, sino_Lhi=sino_Lhi,
        sino_A1_rec=sino_A1_rec, sino_A2_rec=sino_A2_rec,
        recon_a1=recon_a1, recon_a2=recon_a2,
        Zeff=Zeff, rhoe=rhoe,
        E_lo=E_lo, phi_lo=phi_lo, E_hi=E_hi, phi_hi=phi_hi,
    )


# ============================================================================
# Storyboard figure
# ============================================================================

def add_step(fig, ax_bg, cx, y0, w, h, step_no, title, panels, accent):
    """panels = list of (image2d, subtitle, cmap). Lays them side by side."""
    n = len(panels)
    gap = 0.012
    pw = (w - gap * (n - 1)) / n
    x_left = cx - w / 2
    # step title bar
    ax_bg.text(cx, y0 + h + 0.052, f"{step_no}", ha="center", va="center",
               fontsize=12, fontweight="bold", color="white", zorder=5,
               bbox=dict(boxstyle="circle,pad=0.32", fc=accent, ec="none"))
    ax_bg.text(cx, y0 + h + 0.012, title, ha="center", va="bottom",
               fontsize=10.5, fontweight="bold", color="#1a1a1a", zorder=4)
    for i, (img, subtitle, cmap) in enumerate(panels):
        xi = x_left + i * (pw + gap)
        iax = fig.add_axes([xi, y0, pw, h])
        iax.imshow(img, cmap=cmap, aspect="auto", origin="lower")
        iax.set_xticks([]); iax.set_yticks([])
        for sp in iax.spines.values():
            sp.set_edgecolor(accent); sp.set_linewidth(1.8)
        iax.text(0.5, -0.055, subtitle, transform=iax.transAxes, ha="center",
                 va="top", fontsize=8.0, color="#333333")


def harrow(ax, x0, x1, y, color="#555", label=None):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                 mutation_scale=20, linewidth=2.0, color=color, zorder=1))
    if label:
        ax.text((x0 + x1) / 2, y + 0.012, label, ha="center", va="bottom",
                fontsize=7.6, style="italic", color="#555")


def varrow(ax, x, y0, y1, color="#555", label=None):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
                 mutation_scale=20, linewidth=2.0, color=color, zorder=1))
    if label:
        ax.text(x + 0.012, (y0 + y1) / 2, label, ha="left", va="center",
                fontsize=7.6, style="italic", color="#555")


def main():
    R = run_pipeline(N=128, n_angles=90)

    fig = plt.figure(figsize=(18, 11))
    bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, 1); bg.set_ylim(0, 1)
    bg.axis("off")

    bg.text(0.5, 0.975, "Alvarez-Macovski Dual-Energy CT - Overall Process",
            ha="center", va="top", fontsize=20, fontweight="bold", color="#111")
    bg.text(0.5, 0.94,
            r"$\mu(x,y,E)=a_1(x,y)\,f_1(E)+a_2(x,y)\,f_2(E)$    "
            r"($f_1=E^{-3}$ photoelectric,  $f_2=$ Klein-Nishina Compton);  "
            r"two beams: 80 & 140 kVp",
            ha="center", va="top", fontsize=12, color="#333")

    # accents
    A_OBJ, A_FWD, A_MEAS = "#e8893b", "#3b7fc4", "#c44b3b"
    A_DEC, A_REC, A_PHYS = "#7a5bb0", "#3ba36b", "#b89a2e"

    col = [0.165, 0.5, 0.835]      # column centres
    W = 0.235                       # step block width
    H = 0.205                       # image height (≈ panel width in inches → near-square)
    y_top = 0.56
    y_bot = 0.10

    # ---- Top row (forward), left -> right ----
    add_step(fig, bg, col[0], y_top, W, H, "1", "Object (phantom)", [
        (R["a1_map"], r"$a_1$  photoelectric", "inferno"),
        (R["a2_map"], r"$a_2$  Compton", "viridis")], A_OBJ)
    add_step(fig, bg, col[1], y_top, W, H, "2", "Forward project (Radon)", [
        (R["sino_A1"], r"$A_1(\theta,t)$", "inferno"),
        (R["sino_A2"], r"$A_2(\theta,t)$", "viridis")], A_FWD)
    add_step(fig, bg, col[2], y_top, W, H, "3", "Polychromatic Beer-Lambert", [
        (R["sino_Llo"], r"$L_{lo}$  80 kVp", "Blues"),
        (R["sino_Lhi"], r"$L_{hi}$  140 kVp", "Reds")], A_MEAS)

    # ---- Bottom row (inverse), right -> left ----
    add_step(fig, bg, col[2], y_bot, W, H, "4", "AM decomposition (per ray)", [
        (R["sino_A1_rec"], r"$\hat{A}_1(\theta,t)$", "inferno"),
        (R["sino_A2_rec"], r"$\hat{A}_2(\theta,t)$", "viridis")], A_DEC)
    add_step(fig, bg, col[1], y_bot, W, H, "5", "FBP reconstruction", [
        (R["recon_a1"], r"$\hat{a}_1(x,y)$", "inferno"),
        (R["recon_a2"], r"$\hat{a}_2(x,y)$", "viridis")], A_REC)
    add_step(fig, bg, col[0], y_bot, W, H, "6", "Physical maps", [
        (R["Zeff"], r"$Z_{eff}$", "plasma"),
        (R["rhoe"], r"$\rho_e/\rho_{fr4}$", "cividis")], A_PHYS)

    # ---- Arrows ----
    ymid_t = y_top + H / 2
    ymid_b = y_bot + H / 2
    harrow(bg, col[0] + W / 2 + 0.004, col[1] - W / 2 - 0.004, ymid_t, A_FWD)
    harrow(bg, col[1] + W / 2 + 0.004, col[2] - W / 2 - 0.004, ymid_t, A_MEAS,
           "× 2 spectra")
    varrow(bg, col[2], y_top - 0.004, y_bot + H + 0.058, A_DEC,
           "solve 2×2 per ray")
    harrow(bg, col[2] - W / 2 - 0.004, col[1] + W / 2 + 0.004, ymid_b, A_REC)
    harrow(bg, col[1] - W / 2 - 0.004, col[0] + W / 2 + 0.004, ymid_b, A_PHYS,
           r"$\rho_e\propto a_2,\ Z_{eff}=K(a_1/a_2)^{1/3.8}$")

    bg.text(0.5, 0.022,
            "Decomposition happens in sinogram space (steps 3-4, before reconstruction), "
            "so beam hardening is removed mathematically rather than corrected empirically. "
            "Every panel is generated by running the notebook phantom through the pipeline.",
            ha="center", va="bottom", fontsize=9, color="#666", style="italic")

    fig.savefig(OUT_PATH, dpi=130, facecolor="white")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()

"""
Gerber Design vs Simulated X-Ray comparison demo.

Loads copper Gerber layers from a PCB project, then shows side-by-side:
  Left  – Gerber copper overlay (design intent, colored by layer)
  Right – Simulated Beer-Lambert X-ray projection

Run from the xrayrecon/ directory:
    python demo_gerber_xray.py
"""
import numpy as np
import cv2 as cv
import pyqtgraph as pg
from pathlib import Path

from xrayrecon.gerber_utils import GerberLayer, SimpleGerberParser
from xrayrecon.pygerber_integration import PyGerberRasterizer
from xrayrecon.materials import get_material

# ── PCB to use ───────────────────────────────────────────────────────────────
GERBER_DIR = Path(
    r"E:\git\kicad\demos\royalblue54L_feather\nfc_antenna"
    r"\RoyalBlue54L-NFC-Antenna\jlcpcb\gerber"
)

# (filename, display name, RGBA color for overlay)
LAYERS = [
    ("RoyalBlue54L-NFC-Antenna-CuTop.gbr",    "Top Cu",    (255, 140,  0, 220)),
    ("RoyalBlue54L-NFC-Antenna-CuBottom.gbr", "Bottom Cu", ( 50, 160, 255, 220)),
]

DPI           = 300
RES_XY_MM     = 25.4 / DPI   # mm per pixel at this DPI
THICKNESS_MM  = 0.035         # 1 oz copper layer thickness
FR4_THICK_MM  = 1.6           # typical 2-layer PCB substrate

# Beer-Lambert attenuation scale (tune so contrast is visible)
MU_SCALE = 0.018


# ── Helpers ──────────────────────────────────────────────────────────────────

def rasterize_gerber(gbr_path: Path) -> np.ndarray:
    """Return (H, W) uint8 alpha mask for a Gerber file."""
    content = gbr_path.read_text()

    # Try PyGerber first
    mask = PyGerberRasterizer(dpi=DPI).rasterize(content)
    if mask.max() > 0:
        return mask

    # Fallback: simple vector parser → cv2 rasterization
    print(f"  PyGerber empty for {gbr_path.name}, using SimpleGerberParser")
    parser = SimpleGerberParser()
    gl = parser.parse(content)
    if not gl.objects or gl.bounds[0] >= gl.bounds[2]:
        return np.zeros((10, 10), dtype=np.uint8)

    w_px = int((gl.bounds[2] - gl.bounds[0]) / RES_XY_MM) + 8
    h_px = int((gl.bounds[3] - gl.bounds[1]) / RES_XY_MM) + 8
    mask_img = np.zeros((h_px, w_px), dtype=np.uint8)
    for obj in gl.objects:
        def to_px(pt):
            return (int((pt[0] - gl.bounds[0]) / RES_XY_MM + 4),
                    int((pt[1] - gl.bounds[1]) / RES_XY_MM + 4))
        p1, p2 = to_px(obj.start), to_px(obj.end)
        t_px = max(1, int(obj.aperture.params[0] / RES_XY_MM))
        if obj.type == "LINE":
            cv.line(mask_img, p1, p2, 255, t_px)
            cv.circle(mask_img, p1, t_px // 2, 255, -1)
            cv.circle(mask_img, p2, t_px // 2, 255, -1)
        elif obj.type == "FLASH" and obj.aperture.type == "C":
            cv.circle(mask_img, p1, t_px // 2, 255, -1)
    return mask_img


def build_gerber_overlay(masks, colors, shape) -> np.ndarray:
    """Composite all copper masks as a colored RGBA image (H, W, 4)."""
    H, W = shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    # Dark gray PCB substrate background
    rgba[:, :] = (25, 30, 25, 255)
    for mask, color in zip(masks, colors):
        m = cv.resize(mask, (W, H), interpolation=cv.INTER_NEAREST)
        rgba[m > 128] = color
    return rgba


def build_xray_projection(masks, shape) -> np.ndarray:
    """
    Beer-Lambert X-ray transmission image (H, W) float32 in [0, 1].
    Bright = high transmission (air/FR4), dark = copper absorbs X-rays.
    """
    H, W = shape
    mat_cu  = get_material("COPPER")
    mat_fr4 = get_material("FR4")

    cu_att  = mat_cu.z_eff  * mat_cu.electron_density  * THICKNESS_MM
    fr4_att = mat_fr4.z_eff * mat_fr4.electron_density * FR4_THICK_MM

    # Start with uniform FR4 substrate
    total_att = np.full((H, W), fr4_att, dtype=np.float32)

    # Add copper from each layer
    for mask in masks:
        m = cv.resize(mask, (W, H), interpolation=cv.INTER_NEAREST)
        total_att += (m > 128).astype(np.float32) * cu_att

    xray = np.exp(-MU_SCALE * total_att)
    lo, hi = xray.min(), xray.max()
    if hi > lo:
        xray = (xray - lo) / (hi - lo)
    return xray


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Rasterizing Gerber layers...")
    masks, colors, names = [], [], []
    for fname, name, color in LAYERS:
        path = GERBER_DIR / fname
        if not path.exists():
            print(f"  SKIP (not found): {fname}")
            continue
        mask = rasterize_gerber(path)
        print(f"  {name}: {mask.shape}  max={mask.max()}")
        masks.append(mask)
        colors.append(color)
        names.append(name)

    if not masks:
        print("No Gerber layers loaded — check GERBER_DIR.")
        return

    # Target display size (cap at 1024px on longest axis)
    max_h = max(m.shape[0] for m in masks)
    max_w = max(m.shape[1] for m in masks)
    scale = min(1.0, 1024 / max(max_h, max_w))
    H, W = int(max_h * scale), int(max_w * scale)
    print(f"Display: {W}×{H} px")

    rgba_gerber = build_gerber_overlay(masks, colors, (H, W))
    xray_float  = build_xray_projection(masks, (H, W))
    xray_uint8  = (xray_float * 255).astype(np.uint8)

    # ── Save images to disk ──────────────────────────────────────────────
    out_dir = Path("demo_output")
    out_dir.mkdir(exist_ok=True)

    # Gerber overlay (RGBA → save as-is)
    cv.imwrite(str(out_dir / "gerber_overlay.png"),
               cv.cvtColor(rgba_gerber, cv.COLOR_RGBA2BGRA))

    # X-ray: save grayscale
    cv.imwrite(str(out_dir / "xray_simulated.png"), xray_uint8)

    # Side-by-side composite (pad to same height, concatenate)
    gH, gW = rgba_gerber.shape[:2]
    xH, xW = xray_uint8.shape
    target_H = max(gH, xH)
    pad_g = np.zeros((target_H - gH, gW, 4), dtype=np.uint8)
    pad_x = np.zeros((target_H - xH, xW),    dtype=np.uint8)
    g_padded = np.concatenate([rgba_gerber, pad_g], axis=0)
    x_padded = np.concatenate([xray_uint8,  pad_x], axis=0)
    # Convert xray to RGBA for concatenation
    x_rgba = cv.cvtColor(x_padded, cv.COLOR_GRAY2RGBA)
    # Add 8px white separator
    sep = np.full((target_H, 8, 4), 200, dtype=np.uint8)
    composite = np.concatenate([g_padded, sep, x_rgba], axis=1)
    cv.imwrite(str(out_dir / "gerber_vs_xray.png"),
               cv.cvtColor(composite, cv.COLOR_RGBA2BGRA))

    print(f"Images saved to {out_dir.resolve()}/")
    print("  gerber_overlay.png  — Gerber copper layers colored by layer")
    print("  xray_simulated.png  — Beer-Lambert X-ray projection")
    print("  gerber_vs_xray.png  — Side-by-side composite")

    # ── Also launch PyQtGraph window ─────────────────────────────────────
    app = pg.mkQApp()
    win = pg.GraphicsLayoutWidget(title="Gerber Design vs Simulated X-Ray")
    win.resize(1400, 700)
    win.setBackground("k")

    layer_label = "  |  ".join(names)

    p1 = win.addPlot(title=f"Gerber Copper Layers  ({layer_label})")
    img1 = pg.ImageItem(rgba_gerber)
    p1.addItem(img1)
    p1.getViewBox().invertY(True)
    p1.setAspectLocked(True)
    p1.hideAxis("bottom")
    p1.hideAxis("left")

    p2 = win.addPlot(title="Simulated X-Ray Projection  (Beer–Lambert)")
    img2 = pg.ImageItem(xray_uint8, levels=(0, 255))
    p2.addItem(img2)
    p2.getViewBox().invertY(True)
    p2.setAspectLocked(True)
    p2.hideAxis("bottom")
    p2.hideAxis("left")

    p2.setXLink(p1)
    p2.setYLink(p1)

    win.show()
    print("Window open — scroll/drag to zoom and pan both panels together.")
    app.exec_()


if __name__ == "__main__":
    main()

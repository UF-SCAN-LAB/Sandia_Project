"""
3D visualization of the lizard-head spectral CT scan, segmented into material
(density) regions. Reads lizard_recon.npz (produced by recon_lizard.py).

The raw reconstruction is dominated by (a) a cylindrical specimen holder and
(b) concentric CT ring artifacts -- both rotationally symmetric about the scan
axis. Real anatomy is asymmetric, so we subtract the per-slice azimuthal (radial)
mean to remove the holder, the embedding-medium gradient and the ring artifacts,
leaving the head structures. The ring-corrected signal is then split into:

  soft tissue        : low  positive residual  (translucent)
  bone / dense parts : high positive residual  (opaque)

Output: lizard_materials_3d.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter, median_filter, label
from skimage.measure import marching_cubes

OUT = r"E:\git\Spectral-Xray\lizard_materials_3d.png"

TISSUE_LO = 0.0035    # ring-corrected residual level for soft tissue
BONE_LO = 0.0130      # residual level for bone / dense structures
HEAD_R = 28.0         # radius enclosing the head, inside the holder (px)

C_TISSUE = "#d98c6a"  # soft tissue (translucent)
C_BONE = "#f2ead6"    # bone / dense (opaque ivory)


def load_anatomy():
    """Return ring-corrected anatomy volume (asymmetric signal) and voxel size."""
    d = np.load(os.path.join(os.path.dirname(OUT), "lizard_recon.npz"))
    vol = median_filter(d["vol_br"].astype(np.float64), size=3)
    vol = gaussian_filter(vol, sigma=0.8)
    voxel = float(d["voxel_mm"])

    nz, ny, nx = vol.shape
    cy, cx = ny / 2 - 0.5, nx / 2 - 0.5
    Y, X = np.mgrid[0:ny, 0:nx]
    R = np.hypot(X - cx, Y - cy)
    Ri = np.round(R).astype(int)
    nb = Ri.max() + 1
    counts = np.maximum(np.bincount(Ri.ravel(), None, nb), 1)

    # subtract the rotationally symmetric background (holder + medium + rings)
    anat = np.empty_like(vol)
    for z in range(nz):
        rmean = np.bincount(Ri.ravel(), vol[z].ravel(), nb) / counts
        anat[z] = vol[z] - rmean[Ri]
    anat = np.clip(anat, 0, None)
    anat[:, R > HEAD_R] = 0.0
    anat = gaussian_filter(anat, sigma=0.8)
    hi = np.percentile(anat[anat > 0], 99.5)      # clip line-artifact spikes
    anat = np.clip(anat, 0, hi)

    # keep the largest connected component above the tissue level -> the head
    lab, n = label(anat > TISSUE_LO)
    if n:
        sizes = np.bincount(lab.ravel()); sizes[0] = 0
        anat = np.where(lab == sizes.argmax(), anat, 0.0)
    return anat, voxel


def add_surface(ax, vol, level, color, alpha, voxel, step=1):
    if vol.max() <= level:
        return
    verts, faces, _, _ = marching_cubes(vol, level=level, step_size=step)
    verts = verts * voxel                                  # mm, (z, y, x)
    mesh = Poly3DCollection(verts[faces], alpha=alpha, linewidths=0)
    mesh.set_facecolor(color); mesh.set_edgecolor("none")
    ax.add_collection3d(mesh)


def style_axis(ax, vol, voxel, elev, azim, title):
    nz, ny, nx = vol.shape
    ax.set_xlim(0, nx * voxel); ax.set_ylim(0, ny * voxel); ax.set_zlim(0, nz * voxel)
    ax.set_box_aspect((nx, ny, nz))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)"); ax.set_zlabel("z (mm)")
    ax.set_title(title, fontsize=11, fontweight="bold")


def material_rgba(sl):
    rgba = np.zeros(sl.shape + (4,))
    tissue = (sl >= TISSUE_LO) & (sl < BONE_LO)
    bone = sl >= BONE_LO
    rgba[tissue] = (0.85, 0.55, 0.41, 0.80)
    rgba[bone] = (0.95, 0.92, 0.84, 1.0)
    return rgba


def main():
    vol, voxel = load_anatomy()
    print(f"anatomy {vol.shape}, voxel {voxel:.3f} mm, max {vol.max():.4f}")

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("Lizard-head spectral CT — material regions in 3D "
                 "(holder & ring artifacts removed)", fontsize=15, fontweight="bold")

    for i, (elev, azim, name) in enumerate(
            [(18, -65, "view A"), (18, 35, "view B"), (78, -90, "top-down")]):
        ax = fig.add_subplot(2, 3, i + 1, projection="3d")
        add_surface(ax, vol, TISSUE_LO, C_TISSUE, 0.20, voxel)
        add_surface(ax, vol, BONE_LO, C_BONE, 0.95, voxel)
        style_axis(ax, vol, voxel, elev, azim, name)

    nz, ny, nx = vol.shape
    cz, cy, cx = nz // 2, ny // 2, nx // 2
    vmax = np.percentile(vol[vol > 0], 99)
    for j, (sl, title) in enumerate([
            (vol[cz], f"axial  z={cz}"),
            (vol[:, cy], f"coronal  y={cy}"),
            (vol[:, :, cx], f"sagittal  x={cx}")]):
        ax = fig.add_subplot(2, 3, j + 4)
        ax.imshow(sl, cmap="bone", origin="lower", vmax=vmax)
        ax.imshow(material_rgba(sl), origin="lower")
        ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])

    fig.legend(handles=[
        Patch(facecolor=C_TISSUE, label="soft tissue (low residual density)"),
        Patch(facecolor=C_BONE, edgecolor="#999", label="bone / dense structures")],
        loc="lower center", ncol=2, fontsize=10, frameon=False,
        bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(OUT, dpi=130, facecolor="white")
    print("Saved:", OUT)


if __name__ == "__main__":
    main()

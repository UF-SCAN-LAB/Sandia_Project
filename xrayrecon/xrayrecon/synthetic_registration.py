"""Deterministic synthetic datasets for registration demos and regression tests.

The generators live in the package rather than under ``tests`` so the desktop
application never imports test code. Each case carries its known physical-space
transform and identifiability expectations, making it useful as an executable
registration contract as well as a demo source.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter

from .materials import MATERIAL_NAMES, MATERIALS_DB
from .registration import RigidScaleTransform, resample_labels, resample_volume


PCB_SHOWCASE_FEATURES = (
    "rounded board with keyed notch",
    "four mounting holes",
    "QFP controller with visible pins",
    "secondary IC package",
    "shielded connector shell",
    "gold edge fingers",
    "routed top and bottom copper traces",
    "plated through-vias",
    "gold fiducials",
    "surface-mount passives",
)

PCB_SHOWCASE_MATERIALS = (
    "AIR",
    "FR4",
    "COPPER",
    "GOLD",
    "TIN",
    "ALUMINUM",
    "ALUMINA",
    "TUNGSTEN",
)


@dataclass(frozen=True)
class SyntheticRegistrationSpec:
    name: str
    transform: RigidScaleTransform = field(default_factory=RigidScaleTransform.identity)
    moving_shape: tuple[int, int, int] = (40, 44, 12)
    fixed_shape: tuple[int, int, int] | None = None
    moving_spacing: tuple[float, float, float] = (0.15, 0.15, 0.10)
    fixed_spacing: tuple[float, float, float] = (0.15, 0.15, 0.10)
    noise_std: float = 0.0
    intensity_gamma: float = 1.0
    crop_fraction: float = 0.0
    fixed_blur_sigma: float = 0.0
    geometry: str = "asymmetric"
    identifiable: bool = True
    minimum_score_gain: float = 0.03
    seed: int = 1729


@dataclass(frozen=True)
class SyntheticRegistrationCase:
    spec: SyntheticRegistrationSpec
    moving_hwz: np.ndarray
    fixed_hwz: np.ndarray
    moving_labels_hwz: np.ndarray
    fixed_labels_hwz: np.ndarray
    feature_names: tuple[str, ...] = ()
    material_names: tuple[str, ...] = ()
    layer_names: tuple[str, ...] = ()
    layer_slices: tuple[tuple[int, ...], ...] = ()

    @property
    def known_transform(self) -> RigidScaleTransform:
        return self.spec.transform

    @property
    def moving_spacing(self) -> tuple[float, float, float]:
        return self.spec.moving_spacing

    @property
    def fixed_spacing(self) -> tuple[float, float, float]:
        return self.spec.fixed_spacing


def _asymmetric_phantom(shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    axes = np.meshgrid(
        *[np.linspace(-1.0, 1.0, size, dtype=np.float32) for size in shape],
        indexing="ij",
    )
    x, y, z = axes
    volume = np.zeros(shape, dtype=np.float32)
    blobs = (
        (0.95, (-0.42, -0.30, -0.25), (0.18, 0.13, 0.22)),
        (0.72, (0.36, -0.12, 0.28), (0.12, 0.22, 0.16)),
        (0.55, (-0.08, 0.46, 0.05), (0.25, 0.10, 0.28)),
    )
    for amplitude, center, widths in blobs:
        exponent = sum(
            ((axis - axis_center) / width) ** 2
            for axis, axis_center, width in zip(axes, center, widths)
        )
        volume += amplitude * np.exp(-0.5 * exponent)

    # Asymmetric trace and via features break rotational/translation symmetry.
    trace = (x > -0.75) & (x < 0.15) & (y > 0.05) & (y < 0.16) & (abs(z) < 0.45)
    via = ((x - 0.57) ** 2 + (y - 0.42) ** 2 < 0.065**2) & (z > -0.65)
    corner = (x < -0.55) & (y > 0.45) & (z < -0.05)
    volume[trace] += 0.70
    volume[via] += 0.90
    volume[corner] += 0.45
    volume = np.clip(volume, 0.0, 1.0)

    labels = np.zeros(shape, dtype=np.int16)
    labels[volume > 0.12] = 1
    labels[volume > 0.42] = 2
    labels[volume > 0.72] = 3
    return volume, labels


def _symmetric_phantom(shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    axes = np.meshgrid(
        *[np.linspace(-1.0, 1.0, size, dtype=np.float32) for size in shape],
        indexing="ij",
    )
    radius_xy = np.sqrt(axes[0] ** 2 + axes[1] ** 2)
    volume = np.exp(-((radius_xy - 0.48) ** 2) / 0.012) * np.exp(
        -(axes[2] ** 2) / 0.35
    )
    volume = volume.astype(np.float32)
    labels = (volume > 0.25).astype(np.int16)
    return volume, labels


def _material_id(name: str) -> np.int16:
    """Return the canonical label used by the voxelizer/color pipeline."""
    return np.int16(MATERIAL_NAMES.index(name))


def _disk_mask(shape: tuple[int, int], center, radius: float) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= radius**2


def _polyline_mask(
    shape: tuple[int, int], points: tuple[tuple[int, int], ...], width: int
) -> np.ndarray:
    """Rasterize a readable Manhattan/diagonal PCB trace without GUI deps."""
    mask = np.zeros(shape, dtype=bool)
    radius = max(0, int(width) // 2)
    for start, end in zip(points, points[1:]):
        distance = max(abs(end[0] - start[0]), abs(end[1] - start[1]), 1)
        rows = np.rint(np.linspace(start[0], end[0], distance + 1)).astype(int)
        cols = np.rint(np.linspace(start[1], end[1], distance + 1)).astype(int)
        for row, col in zip(rows, cols):
            row0, row1 = max(0, row - radius), min(shape[0], row + radius + 1)
            col0, col1 = max(0, col - radius), min(shape[1], col + radius + 1)
            mask[row0:row1, col0:col1] = True
    return mask


def _pcb_showcase_phantom(
    shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Build a human-readable multi-material PCB in ``(H, W, Z)`` order."""
    h, w, depth = shape
    if h < 80 or w < 104 or depth < 20:
        raise ValueError(
            "pcb_showcase geometry requires at least (80, 104, 20) voxels"
        )

    yy, xx = np.ogrid[:h, :w]
    center_y, center_x = (h - 1) / 2.0, (w - 1) / 2.0
    half_h, half_w = h * 0.43, w * 0.45
    board = (
        (np.abs((yy - center_y) / half_h) ** 8)
        + (np.abs((xx - center_x) / half_w) ** 8)
        <= 1.0
    )
    # A keyed bite in the upper-right edge makes orientation unambiguous.
    board[(yy < h * 0.18) & (xx > w * 0.80)] = False

    labels = np.full(shape, _material_id("AIR"), dtype=np.int16)
    substrate_z = slice(3, depth - 4)
    labels[:, :, substrate_z][board] = _material_id("FR4")

    copper = _material_id("COPPER")
    gold = _material_id("GOLD")
    tin = _material_id("TIN")
    alumina = _material_id("ALUMINA")
    aluminum = _material_id("ALUMINUM")
    tungsten = _material_id("TUNGSTEN")

    top_z = depth - 7
    bottom_z = 4
    top_traces = np.zeros((h, w), dtype=bool)
    bottom_traces = np.zeros((h, w), dtype=bool)

    def point(y_frac: float, x_frac: float) -> tuple[int, int]:
        return int(round(y_frac * (h - 1))), int(round(x_frac * (w - 1)))

    top_routes = (
        (point(0.40, 0.50), point(0.26, 0.50), point(0.26, 0.82)),
        (point(0.45, 0.56), point(0.35, 0.68), point(0.35, 0.88)),
        (point(0.52, 0.57), point(0.52, 0.76), point(0.42, 0.86)),
        (point(0.60, 0.53), point(0.72, 0.63), point(0.72, 0.88)),
        (point(0.60, 0.45), point(0.72, 0.39), point(0.83, 0.39)),
        (point(0.50, 0.42), point(0.50, 0.28), point(0.34, 0.28)),
        (point(0.43, 0.43), point(0.31, 0.35), point(0.25, 0.22)),
    )
    bottom_routes = (
        (point(0.31, 0.24), point(0.68, 0.24), point(0.76, 0.36)),
        (point(0.24, 0.37), point(0.18, 0.62), point(0.28, 0.78)),
        (point(0.67, 0.47), point(0.82, 0.55), point(0.78, 0.82)),
        (point(0.36, 0.67), point(0.63, 0.68), point(0.71, 0.84)),
    )
    for route in top_routes:
        top_traces |= _polyline_mask((h, w), route, width=3)
    for route in bottom_routes:
        bottom_traces |= _polyline_mask((h, w), route, width=2)
    top_traces &= board
    bottom_traces &= board
    labels[:, :, top_z : top_z + 2][top_traces] = copper
    labels[:, :, bottom_z : bottom_z + 2][bottom_traces] = copper

    # Central QFP controller body, die, and four banks of visible pins.
    qfp_y0, qfp_y1 = int(h * 0.36), int(h * 0.64)
    qfp_x0, qfp_x1 = int(w * 0.39), int(w * 0.61)
    component_z = slice(depth - 6, depth - 1)
    labels[qfp_y0:qfp_y1, qfp_x0:qfp_x1, component_z] = alumina
    labels[
        int(h * 0.44) : int(h * 0.56),
        int(w * 0.45) : int(w * 0.55),
        depth - 4 : depth - 1,
    ] = tungsten
    for row in range(qfp_y0 + 2, qfp_y1 - 1, 4):
        labels[row : row + 2, qfp_x0 - 5 : qfp_x0, depth - 7 : depth - 3] = tin
        labels[row : row + 2, qfp_x1 : qfp_x1 + 5, depth - 7 : depth - 3] = tin
    for col in range(qfp_x0 + 2, qfp_x1 - 1, 5):
        labels[qfp_y0 - 5 : qfp_y0, col : col + 2, depth - 7 : depth - 3] = tin
        labels[qfp_y1 : qfp_y1 + 5, col : col + 2, depth - 7 : depth - 3] = tin

    # Secondary IC makes the component population deliberately asymmetric.
    labels[
        int(h * 0.19) : int(h * 0.33),
        int(w * 0.16) : int(w * 0.31),
        depth - 6 : depth - 2,
    ] = alumina
    for row in range(int(h * 0.20), int(h * 0.32), 4):
        labels[row : row + 2, int(w * 0.13) : int(w * 0.16), depth - 7 : depth - 3] = tin
        labels[row : row + 2, int(w * 0.31) : int(w * 0.34), depth - 7 : depth - 3] = tin

    # Shielded board-edge connector on the left.
    labels[
        int(h * 0.40) : int(h * 0.65),
        int(w * 0.045) : int(w * 0.14),
        depth - 6 : depth,
    ] = aluminum
    labels[
        int(h * 0.46) : int(h * 0.59),
        int(w * 0.045) : int(w * 0.11),
        depth - 5 : depth,
    ] = _material_id("AIR")

    # Eight gold edge fingers along the opposite edge.
    for row0 in np.linspace(int(h * 0.22), int(h * 0.72), 8, dtype=int):
        labels[row0 : row0 + 4, int(w * 0.82) : int(w * 0.94), top_z : top_z + 3] = gold

    # Through-vias and two gold fiducials are visually distinct landmarks.
    via_centers = (
        point(0.27, 0.38),
        point(0.31, 0.70),
        point(0.48, 0.30),
        point(0.68, 0.32),
        point(0.70, 0.69),
        point(0.80, 0.55),
    )
    for center in via_centers:
        outer = _disk_mask((h, w), center, 3.2)
        inner = _disk_mask((h, w), center, 1.3)
        ring = outer & ~inner
        labels[:, :, 3 : depth - 3][ring] = copper
        labels[:, :, :][inner] = _material_id("AIR")
    for center in (point(0.20, 0.71), point(0.76, 0.18)):
        labels[:, :, top_z : top_z + 3][_disk_mask((h, w), center, 3.0)] = gold

    # Surface-mount passives: ceramic bodies with tin end caps.
    for row0, col0, horizontal in (
        (int(h * 0.69), int(w * 0.43), True),
        (int(h * 0.73), int(w * 0.52), True),
        (int(h * 0.28), int(w * 0.57), False),
        (int(h * 0.25), int(w * 0.63), False),
        (int(h * 0.63), int(w * 0.76), False),
    ):
        if horizontal:
            labels[row0 : row0 + 4, col0 : col0 + 10, depth - 5 : depth - 2] = alumina
            labels[row0 : row0 + 4, col0 : col0 + 2, depth - 6 : depth - 2] = tin
            labels[row0 : row0 + 4, col0 + 8 : col0 + 10, depth - 6 : depth - 2] = tin
        else:
            labels[row0 : row0 + 10, col0 : col0 + 4, depth - 5 : depth - 2] = alumina
            labels[row0 : row0 + 2, col0 : col0 + 4, depth - 6 : depth - 2] = tin
            labels[row0 + 8 : row0 + 10, col0 : col0 + 4, depth - 6 : depth - 2] = tin

    # Four large mounting holes remain air and are surrounded by copper pads.
    mounting_centers = (
        point(0.14, 0.12),
        point(0.14, 0.72),
        point(0.86, 0.12),
        point(0.86, 0.88),
    )
    for center in mounting_centers:
        outer = _disk_mask((h, w), center, 5.2)
        inner = _disk_mask((h, w), center, 3.2)
        labels[:, :, top_z : top_z + 2][outer & ~inner] = copper
        labels[:, :, :][inner] = _material_id("AIR")

    # Map physical material response to a compact CT-like intensity range.
    volume = np.zeros(shape, dtype=np.float32)
    responses = {
        name: np.log1p(MATERIALS_DB[name].z_eff * MATERIALS_DB[name].electron_density)
        for name in PCB_SHOWCASE_MATERIALS
        if name != "AIR"
    }
    response_max = max(responses.values())
    for name, response in responses.items():
        volume[labels == _material_id(name)] = response / response_max
    return volume, labels


def _pcb_showcase_layer_metadata(
    depth: int,
) -> tuple[tuple[str, ...], tuple[tuple[int, ...], ...]]:
    names = (
        "Bottom copper routing",
        "FR4 core and plated vias",
        "Top copper and gold finish",
        "Mounted components",
    )
    slices = (
        tuple(range(4, 6)),
        tuple(range(6, depth - 7)),
        tuple(range(depth - 7, depth - 5)),
        tuple(range(depth - 5, depth)),
    )
    return names, slices


def make_synthetic_registration_case(
    spec: SyntheticRegistrationSpec,
) -> SyntheticRegistrationCase:
    """Build one reproducible moving/fixed pair from ``spec``."""
    if spec.geometry == "asymmetric":
        moving, moving_labels = _asymmetric_phantom(spec.moving_shape)
    elif spec.geometry == "symmetric":
        moving, moving_labels = _symmetric_phantom(spec.moving_shape)
    elif spec.geometry == "constant":
        moving = np.ones(spec.moving_shape, dtype=np.float32)
        moving_labels = np.ones(spec.moving_shape, dtype=np.int16)
    elif spec.geometry == "pcb_showcase":
        moving, moving_labels = _pcb_showcase_phantom(spec.moving_shape)
    else:
        raise ValueError(f"unknown synthetic geometry: {spec.geometry}")

    fixed_shape = spec.fixed_shape or spec.moving_shape
    fixed = resample_volume(
        moving,
        spec.moving_spacing,
        fixed_shape,
        spec.fixed_spacing,
        spec.transform,
        order=1,
    ).astype(np.float32)
    fixed_labels = resample_labels(
        moving_labels,
        spec.moving_spacing,
        fixed_shape,
        spec.fixed_spacing,
        spec.transform,
    )

    if spec.fixed_blur_sigma > 0.0:
        fixed = gaussian_filter(
            fixed, sigma=float(spec.fixed_blur_sigma), mode="constant"
        ).astype(np.float32)

    if spec.intensity_gamma != 1.0:
        fixed = np.power(np.clip(fixed, 0.0, 1.0), spec.intensity_gamma).astype(
            np.float32
        )
    if spec.noise_std > 0.0:
        rng = np.random.default_rng(spec.seed)
        fixed = np.clip(
            fixed + rng.normal(0.0, spec.noise_std, size=fixed.shape), 0.0, 1.0
        ).astype(np.float32)
    if spec.crop_fraction > 0.0:
        crop_slices = int(round(fixed.shape[0] * spec.crop_fraction))
        if crop_slices:
            fixed[:crop_slices] = 0.0
            fixed_labels[:crop_slices] = 0

    layer_names: tuple[str, ...] = ()
    layer_slices: tuple[tuple[int, ...], ...] = ()
    feature_names: tuple[str, ...] = ()
    material_names: tuple[str, ...] = ()
    if spec.geometry == "pcb_showcase":
        feature_names = PCB_SHOWCASE_FEATURES
        material_names = PCB_SHOWCASE_MATERIALS
        layer_names, layer_slices = _pcb_showcase_layer_metadata(spec.moving_shape[2])

    return SyntheticRegistrationCase(
        spec=spec,
        moving_hwz=moving,
        fixed_hwz=fixed,
        moving_labels_hwz=moving_labels,
        fixed_labels_hwz=fixed_labels,
        feature_names=feature_names,
        material_names=material_names,
        layer_names=layer_names,
        layer_slices=layer_slices,
    )


def pcb_registration_showcase_spec() -> SyntheticRegistrationSpec:
    """Return the visually rich, known-ground-truth application showcase."""
    return SyntheticRegistrationSpec(
        name="interpretable_pcb_showcase",
        transform=RigidScaleTransform(
            tx=0.55,
            ty=-0.45,
            tz=0.12,
            rot_z_deg=4.0,
            scale=1.025,
        ),
        moving_shape=(96, 128, 24),
        moving_spacing=(0.10, 0.10, 0.08),
        fixed_spacing=(0.10, 0.10, 0.08),
        noise_std=0.012,
        intensity_gamma=1.12,
        fixed_blur_sigma=0.55,
        geometry="pcb_showcase",
        minimum_score_gain=0.08,
        seed=20260825,
    )


def make_pcb_registration_showcase() -> SyntheticRegistrationCase:
    """Build the canonical rich PCB demo used by the application and tests."""
    return make_synthetic_registration_case(pcb_registration_showcase_spec())


def default_synthetic_registration_specs() -> tuple[SyntheticRegistrationSpec, ...]:
    """Return the canonical deterministic regression matrix."""
    transform = RigidScaleTransform
    return (
        SyntheticRegistrationSpec("identity", minimum_score_gain=0.0),
        SyntheticRegistrationSpec("translate_x", transform(tx=0.45)),
        SyntheticRegistrationSpec("translate_y", transform(ty=-0.45)),
        SyntheticRegistrationSpec("translate_z", transform(tz=0.20)),
        SyntheticRegistrationSpec("rotate_z", transform(rot_z_deg=6.0)),
        SyntheticRegistrationSpec("uniform_scale", transform(scale=1.06)),
        SyntheticRegistrationSpec(
            "combined", transform(tx=0.30, ty=-0.30, tz=0.10, rot_z_deg=5.0, scale=1.04)
        ),
        SyntheticRegistrationSpec(
            "different_spacing",
            transform(tx=0.30, ty=-0.15, rot_z_deg=4.0, scale=1.03),
            fixed_shape=(36, 48, 15),
            fixed_spacing=(0.17, 0.13, 0.08),
        ),
        SyntheticRegistrationSpec(
            "noise_and_intensity_remap",
            transform(tx=0.30, ty=-0.30, rot_z_deg=4.0),
            noise_std=0.025,
            intensity_gamma=1.6,
            minimum_score_gain=0.015,
        ),
        SyntheticRegistrationSpec(
            "partial_crop",
            transform(tx=0.30, ty=-0.30, rot_z_deg=4.0),
            crop_fraction=0.12,
            minimum_score_gain=0.015,
        ),
        SyntheticRegistrationSpec(
            "constant_low_information",
            geometry="constant",
            identifiable=False,
            minimum_score_gain=0.0,
        ),
        SyntheticRegistrationSpec(
            "symmetric_ambiguous",
            transform=transform(rot_z_deg=8.0),
            geometry="symmetric",
            identifiable=False,
            minimum_score_gain=0.0,
        ),
    )


def make_default_synthetic_registration_cases() -> tuple[SyntheticRegistrationCase, ...]:
    return tuple(
        make_synthetic_registration_case(spec)
        for spec in default_synthetic_registration_specs()
    )

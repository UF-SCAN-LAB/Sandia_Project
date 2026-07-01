from __future__ import annotations

import re
from pathlib import Path

import numpy as np

try:
    import tifffile as _tifffile
    _HAS_TIFFFILE = True
except ImportError:
    _HAS_TIFFFILE = False


def _sort_key(p: Path) -> list[int]:
    return [int(n) for n in re.findall(r"\d+", p.stem)]


def _read_tif(path: Path) -> np.ndarray:
    if _HAS_TIFFFILE:
        return _tifffile.imread(str(path)).astype(np.float32)
    from PIL import Image
    return np.array(Image.open(str(path)), dtype=np.float32)


def load_channel_volume(
    chan_dir: str | Path,
    pattern: str = "rec_negLog*.tif",
) -> np.ndarray:
    """Load all matching TIFFs in *chan_dir* as a single (Z, H, W) float32 volume.

    Tries *pattern* first, then lower-case variant, then all *.tif files.
    Slices are sorted by the numeric indices embedded in their filenames.
    """
    chan_dir = Path(chan_dir)
    files = sorted(chan_dir.glob(pattern), key=_sort_key)
    if not files:
        files = sorted(chan_dir.glob(pattern.lower()), key=_sort_key)
    if not files:
        files = sorted(chan_dir.glob("*.tif"), key=_sort_key)
    if not files:
        raise FileNotFoundError(f"No TIFF files found in {chan_dir}")

    slices = [_read_tif(f) for f in files]
    vol = np.stack(slices, axis=0)   # (Z, H, W)
    print(f"Loaded {len(slices)} slices from {chan_dir.name}  shape={vol.shape}")
    return vol


def load_spectral_volume(chan_dirs: list[str | Path]) -> np.ndarray:
    """Stack per-channel volumes into a (C, Z, H, W) float32 array."""
    channels = [load_channel_volume(d) for d in chan_dirs]
    # Pad to the same Z depth if channels differ (e.g. chan0003 may be shorter)
    max_z = max(c.shape[0] for c in channels)
    padded = []
    for c in channels:
        if c.shape[0] < max_z:
            pad = np.zeros((max_z - c.shape[0], *c.shape[1:]), dtype=np.float32)
            c = np.concatenate([c, pad], axis=0)
        padded.append(c)
    return np.stack(padded, axis=0)   # (C, Z, H, W)


def load_ed_data(root_dir: str | Path) -> np.ndarray:
    """Auto-detect and load the ED Data directory structure.

    Supported layouts (tried in order):
    1. <root>/part1/part1/reconRecs/*.tif            → (1, Z, H, W)
    2. <root>/reconRecs/*.tif                        → (1, Z, H, W)
    3. <root>/part2/part2/sinogramData/rawSino/chan????/  → (C, Z, H, W)
    4. <root>/sinogramData/rawSino/chan????/          → (C, Z, H, W)
    5. <root>/chan????/                               → (C, Z, H, W)

    Returns (C, Z, H, W) float32.
    """
    root = Path(root_dir)

    # --- Single-channel FBP candidates ---
    for candidate in [
        root / "part1" / "part1" / "reconRecs",
        root / "reconRecs",
    ]:
        if candidate.is_dir():
            vol = load_channel_volume(candidate)
            return vol[np.newaxis]   # (1, Z, H, W)

    # --- Multi-channel candidates ---
    for rawSino_candidate in [
        root / "part2" / "part2" / "sinogramData" / "rawSino",
        root / "sinogramData" / "rawSino",
        root,
    ]:
        chan_dirs = sorted(rawSino_candidate.glob("chan????"), key=_sort_key)
        if chan_dirs:
            return load_spectral_volume(chan_dirs)

    raise FileNotFoundError(
        f"No recognised ED Data layout found under {root_dir}.\n"
        "Expected 'reconRecs/' or 'chan????/' subdirectories."
    )

from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MaterialEntry:
    name: str
    color: tuple[int, int, int, int]    # RGBA uint8
    threshold: tuple[float, float]      # normalized [0.0, 1.0] inclusive range


@dataclass
class MultiMaterialConfig:
    name: str
    channels: list[str]
    materials: list[MaterialEntry]
    data_dir: str | None = None         # abs path or relative to the config file

    @classmethod
    def from_json(cls, path: str | Path) -> "MultiMaterialConfig":
        cfg_path = Path(path)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {cfg_path}")

        data = json.loads(cfg_path.read_text())

        if "materials" not in data or not data["materials"]:
            raise ValueError("Config must define a non-empty 'materials' list.")

        materials: list[MaterialEntry] = []
        for idx, raw in enumerate(data["materials"]):
            for key in ("name", "color", "threshold"):
                if key not in raw:
                    raise ValueError(f"Material #{idx + 1} is missing required key '{key}'.")

            color_vals = raw["color"]
            if len(color_vals) != 4:
                raise ValueError(
                    f"Material '{raw['name']}' color must be RGBA with 4 values."
                )
            rgba = tuple(int(max(0, min(255, c))) for c in color_vals)

            thresh = raw["threshold"]
            if len(thresh) != 2 or thresh[0] > thresh[1]:
                raise ValueError(
                    f"Material '{raw['name']}' threshold must be [lo, hi] with lo <= hi."
                )

            materials.append(
                MaterialEntry(
                    name=str(raw["name"]),
                    color=rgba,
                    threshold=(float(thresh[0]), float(thresh[1])),
                )
            )

        return cls(
            name=str(data.get("name", cfg_path.stem)),
            channels=list(data.get("channels", ["intensity"])),
            materials=materials,
            data_dir=data.get("data_dir"),
        )

    def to_json(self, path: str | Path) -> None:
        out = {
            "name": self.name,
            "channels": self.channels,
            "materials": [
                {
                    "name": m.name,
                    "color": list(m.color),
                    "threshold": list(m.threshold),
                }
                for m in self.materials
            ],
        }
        if self.data_dir is not None:
            out["data_dir"] = self.data_dir
        Path(path).write_text(json.dumps(out, indent=2))

    def resolve_data_dir(self, config_path: str | Path) -> Path:
        """Return the data directory, resolving relative paths against config_path's parent."""
        if self.data_dir is None:
            return Path(config_path).parent
        data = Path(self.data_dir)
        if data.is_absolute():
            return data
        return (Path(config_path).parent / data).resolve()


def save_label_map(
    label_map: np.ndarray,
    config: MultiMaterialConfig,
    output_stem: str | Path,
) -> None:
    """Write label_map.npy and label_map_meta.json as the outtake artefacts.

    label_map: uint8 array of shape (Z, H, W); 0 = unassigned.
    output_stem: path without extension — e.g. 'output/part1'.
    """
    stem = Path(output_stem)
    np.save(str(stem) + "_label_map.npy", label_map)

    meta: dict = {
        "config_name": config.name,
        "shape": list(label_map.shape),
        "total_voxels": int(label_map.size),
        "materials": [],
    }
    for idx, mat in enumerate(config.materials, start=1):
        count = int(np.sum(label_map == idx))
        meta["materials"].append(
            {
                "id": idx,
                "name": mat.name,
                "color": list(mat.color),
                "threshold": list(mat.threshold),
                "voxel_count": count,
                "volume_fraction": round(count / label_map.size, 6),
            }
        )

    with open(str(stem) + "_label_map_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

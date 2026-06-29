from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class StackupLayer:
    file: str
    name: str
    thickness: float
    gap: float = 0.0
    feature_material: str = "COPPER"
    background_material: str = "AIR"
    color: tuple[int, int, int, int] = (255, 140, 0, 200)


@dataclass
class StackupConfig:
    layers: list[StackupLayer] = field(default_factory=list)
    gerber_dir: str | None = None
    dpi: int = 300
    resolution_xy: float = 0.1
    resolution_z: float = 0.01
    substrate_material: str = "FR4"
    default_gap_slices: int = 20

    @classmethod
    def from_json(cls, path: str | Path) -> "StackupConfig":
        cfg_path = Path(path)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Stackup config does not exist: {cfg_path}")

        data = json.loads(cfg_path.read_text())
        layers_data = data.get("layers", [])
        if not layers_data:
            raise ValueError("Stackup config must define a non-empty 'layers' list.")

        layers: list[StackupLayer] = []
        for idx, raw in enumerate(layers_data):
            if "file" not in raw:
                raise ValueError(f"Layer #{idx + 1} is missing required key 'file'.")

            thickness = float(raw.get("thickness", 0.035))
            if thickness <= 0:
                raise ValueError(
                    f"Layer '{raw.get('name', raw['file'])}' must have thickness > 0."
                )

            gap = float(raw.get("gap", 0.0))
            if gap < 0:
                raise ValueError(
                    f"Layer '{raw.get('name', raw['file'])}' cannot have negative gap."
                )

            color_vals = raw.get("color", [255, 140, 0, 200])
            if len(color_vals) != 4:
                raise ValueError(
                    f"Layer '{raw.get('name', raw['file'])}' color must be RGBA with 4 values."
                )
            rgba = tuple(int(max(0, min(255, c))) for c in color_vals)

            layers.append(
                StackupLayer(
                    file=str(raw["file"]),
                    name=str(raw.get("name", Path(raw["file"]).stem)),
                    thickness=thickness,
                    gap=gap,
                    feature_material=str(raw.get("feature_material", "COPPER")),
                    background_material=str(raw.get("background_material", "AIR")),
                    color=rgba,
                )
            )

        return cls(
            layers=layers,
            gerber_dir=data.get("gerber_dir"),
            dpi=int(data.get("dpi", 300)),
            resolution_xy=float(data.get("resolution_xy", 0.1)),
            resolution_z=float(data.get("resolution_z", 0.01)),
            substrate_material=str(data.get("substrate_material", "FR4")),
            default_gap_slices=int(data.get("default_gap_slices", 20)),
        )

    def resolve_layer_path(self, stackup_file: str | Path, layer_file: str) -> Path:
        stackup_path = Path(stackup_file)
        if self.gerber_dir:
            base_dir = Path(self.gerber_dir)
            if not base_dir.is_absolute():
                base_dir = (stackup_path.parent / base_dir).resolve()
        else:
            base_dir = stackup_path.parent.resolve()
        return (base_dir / layer_file).resolve()
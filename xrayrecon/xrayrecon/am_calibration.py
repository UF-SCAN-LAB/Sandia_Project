"""Cross-validated material calibration and empirical AM uncertainty."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.stats import spearmanr

from .alvarez_macovski import AMResult
from .materials import (
    MATERIAL_NAMES,
    MATERIALS_DB,
    WATER_ELECTRON_DENSITY_E23_CM3,
)


def _empirical_confidence(result: AMResult) -> np.ndarray:
    """Ground-truth-free confidence calibrated to observed error predictors."""
    density_disagreement = np.abs(
        result["rhoe_recovered"] - result["rhoe_regularized"]
    ) / np.maximum(result["rhoe_regularized"], 0.05)
    risk = result["projection_residual"] + 2.0 * np.clip(
        density_disagreement, 0.0, 1.0
    )
    return np.clip(np.exp(-risk), 0.0, 1.0).astype(np.float32)


@dataclass(frozen=True)
class AMCalibrationSample:
    material_name: str
    true_zeff: float
    true_rhoe: float
    recovered_zeff: float
    recovered_rhoe: float
    confidence: float


@dataclass(frozen=True)
class AMCalibrationModel:
    """Material-response calibration plus empirical 90% interval scales."""

    calibration_id: str
    training_materials: tuple[str, ...]
    acquisition_fingerprint: str
    zeff_log_slope: float
    zeff_log_intercept: float
    rhoe_slope: float
    rhoe_intercept: float
    zeff_uncertainty_scale: float = 0.15
    rhoe_uncertainty_scale: float = 0.12
    nominal_coverage: float = 0.90
    zeff_input_mode: str = "material_regularized"
    rhoe_input_mode: str = "raw"
    calibration_source: str = "synthetic_reference"
    schema_version: int = 1

    def __post_init__(self) -> None:
        calibration_id = str(self.calibration_id).strip()
        training_materials = tuple(
            str(name).strip().upper() for name in self.training_materials
        )
        prefix, separator, digest = str(self.acquisition_fingerprint).partition(":")
        if not calibration_id:
            raise ValueError("calibration_id must not be empty")
        if len(training_materials) < 3 or any(not name for name in training_materials):
            raise ValueError("calibration requires at least three named training materials")
        if len(set(training_materials)) != len(training_materials):
            raise ValueError("calibration training materials must be unique")
        if (
            prefix != "am-physics-v1"
            or separator != ":"
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("invalid AM acquisition physics fingerprint")
        numeric_values = (
            self.zeff_log_slope,
            self.zeff_log_intercept,
            self.rhoe_slope,
            self.rhoe_intercept,
            self.zeff_uncertainty_scale,
            self.rhoe_uncertainty_scale,
            self.nominal_coverage,
        )
        if not np.isfinite(numeric_values).all():
            raise ValueError("calibration coefficients must be finite")
        if self.zeff_log_slope <= 0 or self.rhoe_slope <= 0:
            raise ValueError("calibration response slopes must be positive")
        if self.zeff_uncertainty_scale <= 0 or self.rhoe_uncertainty_scale <= 0:
            raise ValueError("calibration uncertainty scales must be positive")
        if not 0 < self.nominal_coverage < 1:
            raise ValueError("nominal_coverage must be between zero and one")
        valid_modes = {"raw", "material_regularized"}
        if self.zeff_input_mode not in valid_modes or self.rhoe_input_mode not in valid_modes:
            raise ValueError("calibration input mode must be raw or material_regularized")
        if self.calibration_source not in {"synthetic_reference", "measured_reference"}:
            raise ValueError("unsupported calibration_source")
        if self.schema_version != 1:
            raise ValueError("unsupported calibration schema_version")
        object.__setattr__(self, "calibration_id", calibration_id)
        object.__setattr__(self, "training_materials", training_materials)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["artifact_kind"] = "xrayrecon.am_calibration"
        payload["training_materials"] = list(self.training_materials)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AMCalibrationModel":
        data = dict(payload)
        if data.pop("artifact_kind", None) != "xrayrecon.am_calibration":
            raise ValueError("invalid AM calibration artifact kind")
        try:
            data["training_materials"] = tuple(data["training_materials"])
            return cls(**data)
        except (KeyError, TypeError) as error:
            raise ValueError(f"invalid AM calibration bundle: {error}") from error

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, allow_nan=False)

    @classmethod
    def from_json(cls, payload: str) -> "AMCalibrationModel":
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid AM calibration JSON: {error}") from error
        if not isinstance(decoded, dict):
            raise ValueError("AM calibration JSON must contain an object")
        return cls.from_dict(decoded)

    def calibrate_zeff(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        calibrated = np.exp(
            self.zeff_log_intercept
            + self.zeff_log_slope * np.log(np.maximum(array, 1e-3))
        )
        return np.where(array > 0, calibrated, 0.0).astype(np.float32)

    def calibrate_rhoe(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        return np.clip(
            self.rhoe_intercept + self.rhoe_slope * array, 0.0, None
        ).astype(np.float32)

    def apply(self, result: AMResult) -> AMResult:
        zeff_source = (
            result["zeff_regularized"]
            if self.zeff_input_mode == "material_regularized"
            else result["zeff_recovered"]
        )
        rhoe_source = (
            result["rhoe_regularized"]
            if self.rhoe_input_mode == "material_regularized"
            else result["rhoe_recovered"]
        )
        zeff = self.calibrate_zeff(zeff_source)
        rhoe = self.calibrate_rhoe(rhoe_source)
        confidence = _empirical_confidence(result)
        factor = 1.25 - confidence
        zeff_halfwidth = (
            self.zeff_uncertainty_scale * factor * np.maximum(zeff, 1.0)
        ).astype(np.float32)
        rhoe_halfwidth = (
            self.rhoe_uncertainty_scale * factor * np.maximum(rhoe, 0.05)
        ).astype(np.float32)

        calibrated = AMResult(
            result,
            acquisition_config=result.acquisition_config,
            has_ground_truth=result.has_ground_truth,
        )
        calibrated.update(
            zeff_calibrated=zeff,
            rhoe_calibrated=rhoe,
            zeff_interval_lower=np.clip(zeff - zeff_halfwidth, 0.0, None),
            zeff_interval_upper=zeff + zeff_halfwidth,
            rhoe_interval_lower=np.clip(rhoe - rhoe_halfwidth, 0.0, None),
            rhoe_interval_upper=rhoe + rhoe_halfwidth,
            rhoe_e23_cm3_calibrated=(
                rhoe * WATER_ELECTRON_DENSITY_E23_CM3
            ).astype(np.float32),
            confidence_calibrated=confidence,
        )
        calibrated.calibration_model = self
        calibrated.calibration_id = self.calibration_id
        calibrated.nominal_coverage = self.nominal_coverage
        return calibrated


def make_am_calibration_phantom(
    material_names: Iterable[str], *, size: int = 72
) -> np.ndarray:
    """Create separated inserts for fitting and held-out material validation."""
    names = tuple(name.upper() for name in material_names)
    if not names or any(name not in MATERIALS_DB for name in names):
        raise ValueError("calibration phantom materials must exist in MATERIALS_DB")
    if len(set(names)) != len(names):
        raise ValueError("calibration phantom materials must be unique")
    if size < 36:
        raise ValueError("calibration phantom size must be at least 36")

    labels = np.full(
        (1, size, size), MATERIAL_NAMES.index("AIR"), dtype=np.int16
    )
    columns = int(np.ceil(np.sqrt(len(names))))
    rows = int(np.ceil(len(names) / columns))
    cell_h = size // rows
    cell_w = size // columns
    margin = max(4, min(cell_h, cell_w) // 5)
    for index, name in enumerate(names):
        row, column = divmod(index, columns)
        y0, y1 = row * cell_h + margin, min((row + 1) * cell_h - margin, size)
        x0, x1 = column * cell_w + margin, min((column + 1) * cell_w - margin, size)
        labels[:, y0:y1, x0:x1] = MATERIAL_NAMES.index(name)
    return labels


def _material_interior_mask(
    labels: np.ndarray, material_name: str, *, erosion: int = 2
) -> np.ndarray:
    material_id = MATERIAL_NAMES.index(material_name.upper())
    mask = np.asarray(labels) == material_id
    if mask.ndim == 3:
        return np.stack(
            [binary_erosion(slice_mask, iterations=erosion) for slice_mask in mask]
        )
    return binary_erosion(mask, iterations=erosion)


def samples_from_result(
    result: AMResult,
    labels: np.ndarray,
    material_names: Iterable[str],
) -> tuple[AMCalibrationSample, ...]:
    if not result.has_ground_truth:
        raise ValueError("calibration samples require known material ground truth")
    samples = []
    for material_name in material_names:
        name = material_name.upper()
        mask = _material_interior_mask(labels, name)
        if not np.any(mask):
            raise ValueError(f"calibration phantom has no interior voxels for {name}")
        props = MATERIALS_DB[name]
        samples.append(
            AMCalibrationSample(
                material_name=name,
                true_zeff=props.z_eff,
                true_rhoe=props.relative_electron_density,
                recovered_zeff=float(np.median(result["zeff_regularized"][mask])),
                recovered_rhoe=float(np.median(result["rhoe_recovered"][mask])),
                confidence=float(np.median(result["confidence"][mask])),
            )
        )
    return tuple(samples)


def fit_material_calibration(
    samples: Iterable[AMCalibrationSample], *, calibration_id: str,
    acquisition_fingerprint: str,
) -> AMCalibrationModel:
    sample_list = tuple(samples)
    if len(sample_list) < 3:
        raise ValueError("at least three material samples are required for calibration")
    recovered_z = np.asarray([sample.recovered_zeff for sample in sample_list])
    recovered_r = np.asarray([sample.recovered_rhoe for sample in sample_list])
    true_z = np.asarray([sample.true_zeff for sample in sample_list])
    true_r = np.asarray([sample.true_rhoe for sample in sample_list])
    if np.any(recovered_z <= 0) or np.ptp(recovered_z) <= 1e-6:
        raise ValueError("calibration requires positive, varying recovered Z-effective values")
    if np.ptp(recovered_r) <= 1e-6:
        raise ValueError("calibration requires varying recovered electron density")
    z_slope, z_intercept = np.polyfit(np.log(recovered_z), np.log(true_z), 1)
    r_slope, r_intercept = np.polyfit(recovered_r, true_r, 1)
    model = AMCalibrationModel(
        calibration_id=str(calibration_id),
        training_materials=tuple(sample.material_name for sample in sample_list),
        acquisition_fingerprint=acquisition_fingerprint,
        zeff_log_slope=float(z_slope),
        zeff_log_intercept=float(z_intercept),
        rhoe_slope=float(r_slope),
        rhoe_intercept=float(r_intercept),
    )
    # Leave-one-material-out errors quantify response-model extrapolation that
    # voxel residuals alone cannot see. Preserve the 90th-percentile normalized
    # miss as the minimum uncertainty scale used by later voxel calibration.
    z_ratios = []
    r_ratios = []
    for index, held_out in enumerate(sample_list):
        keep = np.ones(len(sample_list), dtype=bool)
        keep[index] = False
        loo_z_slope, loo_z_intercept = np.polyfit(
            np.log(recovered_z[keep]), np.log(true_z[keep]), 1
        )
        loo_r_slope, loo_r_intercept = np.polyfit(
            recovered_r[keep], true_r[keep], 1
        )
        predicted_z = np.exp(
            loo_z_intercept + loo_z_slope * np.log(held_out.recovered_zeff)
        )
        predicted_r = loo_r_intercept + loo_r_slope * held_out.recovered_rhoe
        factor = max(1.25 - held_out.confidence, 0.25)
        z_ratios.append(
            abs(predicted_z - held_out.true_zeff)
            / (factor * max(predicted_z, 1.0))
        )
        r_ratios.append(
            abs(predicted_r - held_out.true_rhoe)
            / (factor * max(predicted_r, 0.05))
        )
    return replace(
        model,
        zeff_uncertainty_scale=max(
            model.zeff_uncertainty_scale,
            float(np.quantile(z_ratios, model.nominal_coverage)),
        ),
        rhoe_uncertainty_scale=max(
            model.rhoe_uncertainty_scale,
            float(np.quantile(r_ratios, model.nominal_coverage)),
        ),
    )


def fit_uncertainty_calibration(
    model: AMCalibrationModel,
    result: AMResult,
    mask: np.ndarray,
) -> AMCalibrationModel:
    """Fit interval scales to the empirical nominal-coverage quantile."""
    if not result.has_ground_truth:
        raise ValueError("uncertainty calibration requires ground truth")
    selected = np.asarray(mask, dtype=bool)
    if selected.shape != result["zeff_recovered"].shape or not np.any(selected):
        raise ValueError("uncertainty calibration mask must select result voxels")
    zeff_source = (
        result["zeff_regularized"]
        if model.zeff_input_mode == "material_regularized"
        else result["zeff_recovered"]
    )
    rhoe_source = (
        result["rhoe_regularized"]
        if model.rhoe_input_mode == "material_regularized"
        else result["rhoe_recovered"]
    )
    zeff = model.calibrate_zeff(zeff_source)
    rhoe = model.calibrate_rhoe(rhoe_source)
    confidence = _empirical_confidence(result)
    factor = np.maximum(1.25 - confidence[selected], 0.25)
    z_ratio = np.abs(zeff[selected] - result["zeff_ground_truth"][selected]) / (
        factor * np.maximum(zeff[selected], 1.0)
    )
    r_ratio = np.abs(rhoe[selected] - result["rhoe_ground_truth"][selected]) / (
        factor * np.maximum(rhoe[selected], 0.05)
    )
    return replace(
        model,
        zeff_uncertainty_scale=max(
            model.zeff_uncertainty_scale,
            float(np.quantile(z_ratio, model.nominal_coverage)),
            1e-4,
        ),
        rhoe_uncertainty_scale=max(
            model.rhoe_uncertainty_scale,
            float(np.quantile(r_ratio, model.nominal_coverage)),
            1e-4,
        ),
    )


def fit_calibration_from_result(
    result: AMResult,
    labels: np.ndarray,
    training_materials: Iterable[str],
    *,
    calibration_id: str,
) -> AMCalibrationModel:
    if result.acquisition_config is None:
        raise ValueError("calibration fitting requires acquisition metadata")
    names = tuple(name.upper() for name in training_materials)
    model = fit_material_calibration(
        samples_from_result(result, labels, names),
        calibration_id=calibration_id,
        acquisition_fingerprint=result.acquisition_config.calibration_fingerprint,
    )
    mask = np.zeros(np.asarray(labels).shape, dtype=bool)
    for name in names:
        mask |= _material_interior_mask(labels, name)
    return fit_uncertainty_calibration(model, result, mask)


def evaluate_calibrated_result(
    result: AMResult, mask: np.ndarray
) -> dict[str, float]:
    if not hasattr(result, "calibration_model") or not result.has_ground_truth:
        raise ValueError("evaluation requires calibrated synthetic ground truth")
    selected = np.asarray(mask, dtype=bool)
    zeff_gt = result["zeff_ground_truth"][selected]
    rhoe_gt = result["rhoe_ground_truth"][selected]
    zeff = result["zeff_calibrated"][selected]
    rhoe = result["rhoe_calibrated"][selected]
    z_error = np.abs(zeff - zeff_gt) / np.maximum(zeff_gt, 1e-6)
    r_error = np.abs(rhoe - rhoe_gt) / np.maximum(rhoe_gt, 1e-6)
    z_covered = (
        result["zeff_interval_lower"][selected] <= zeff_gt
    ) & (zeff_gt <= result["zeff_interval_upper"][selected])
    r_covered = (
        result["rhoe_interval_lower"][selected] <= rhoe_gt
    ) & (rhoe_gt <= result["rhoe_interval_upper"][selected])
    combined_error = 0.5 * z_error + 0.5 * r_error
    correlation = spearmanr(
        1.0 - result["confidence_calibrated"][selected], combined_error
    ).statistic
    return {
        "zeff_median_relative_error": float(np.median(z_error)),
        "rhoe_median_relative_error": float(np.median(r_error)),
        "zeff_coverage": float(np.mean(z_covered)),
        "rhoe_coverage": float(np.mean(r_covered)),
        "joint_coverage": float(np.mean(z_covered & r_covered)),
        "confidence_error_rank_correlation": float(correlation),
    }


def material_validation_mask(
    labels: np.ndarray, material_names: Iterable[str]
) -> np.ndarray:
    mask = np.zeros(np.asarray(labels).shape, dtype=bool)
    for name in material_names:
        mask |= _material_interior_mask(labels, name.upper())
    return mask


def save_calibration_model(
    model: AMCalibrationModel,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a calibration bundle without overwriting by default."""
    destination = Path(path)
    if destination.suffix.lower() != ".json":
        raise ValueError("AM calibration bundle path must end in .json")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"calibration bundle already exists: {destination}")
    try:
        destination.write_text(model.to_json() + "\n", encoding="utf-8")
    except OSError as error:
        raise OSError(f"could not save AM calibration bundle: {error}") from error
    return destination


def load_calibration_model(path: str | Path) -> AMCalibrationModel:
    """Load and fully validate a versioned calibration bundle."""
    source = Path(path)
    if source.suffix.lower() != ".json":
        raise ValueError("AM calibration bundle path must end in .json")
    try:
        payload = source.read_text(encoding="utf-8")
    except OSError as error:
        raise OSError(f"could not load AM calibration bundle: {error}") from error
    return AMCalibrationModel.from_json(payload)

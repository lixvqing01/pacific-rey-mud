
import os
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import xarray as xr
from core.utils.netcdf_io import open_dataset

from core.prediction.predictor import GridPredictor
from core.utils.config_models import (
    CascadeEnsembleArtifacts,
    CascadeTrainingArtifacts,
    PredictionConfig,
    PredictionGridSettings,
    TrainingArtifacts,
)
from core.utils.constants import RESULTS_GUI_DIR
from core.utils.logger_config import logger
from core.utils.results_io import RESULTS_SCHEMA_VERSION, build_paths_manifest, snapshot_environment, write_json
from core.utils.run_manager import save_run_summary
from core.utils.run_report import make_prediction_run_report_md
from core.visualization.map_plots import plot_categorical_map_2d, plot_rey_map_2d
from core.visualization.summary_2d_diagnostics import save_summary_2d_json

try:
    from core.visualization.prediction_paper_figures import generate_multiclass_paper_figures
except ModuleNotFoundError:
    generate_multiclass_paper_figures = None


class PredictionPipelineError(Exception):
    pass


class ConfigurationError(PredictionPipelineError):
    pass


class FileNotFoundError(PredictionPipelineError):
    pass


class PredictionFailedError(PredictionPipelineError):
    pass


def _to_native_attr_value(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _to_native_attr_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_native_attr_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_native_attr_value(v) for v in value)
    return value


def _update_attrs_native(target, attrs: dict) -> None:
    target.attrs.update({k: _to_native_attr_value(v) for k, v in attrs.items()})


def _infer_source_run_dir(training_artifacts_path: str) -> Path | None:
    try:
        p = Path(str(training_artifacts_path))
    except Exception:
        return None

    if p.is_dir():
        return p

    if p.name.lower() == "training_artifacts.json" and p.parent.name.lower() == "models":
        return p.parent.parent

    return p.parent if p.parent.exists() else None


def _infer_env_grid_from_run_dir(run_dir: Path) -> str | None:
    try:
        cfg_path = run_dir / "pipeline_config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            paths = (cfg.get("paths", {}) or {})
            p = paths.get("processed_env_grid")
            if isinstance(p, str) and p and os.path.exists(p):
                return p

        summary_path = run_dir / "run_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            paths = ((summary.get("config", {}) or {}).get("paths", {}) or {})
            p = paths.get("processed_env_grid")
            if isinstance(p, str) and p and os.path.exists(p):
                return p
    except Exception:
        return None

    return None


def _resolve_model_path_from_artifacts(artifacts_path: str, model_path: str | None) -> str | None:
    if model_path and os.path.exists(model_path):
        return model_path

    try:
        ap = Path(str(artifacts_path))
        if ap.is_dir():
            candidate = ap / "models" / "final_trained_model.pth"
        else:
            candidate = ap.parent / "final_trained_model.pth" if ap.parent.name.lower() == "models" else ap.parent / "final_trained_model.pth"
        if candidate.exists():
            return str(candidate)
    except Exception:
        return None

    return None


def _high_grade_prospectivity_score_from_stack(
    prob_stack: np.ndarray,
    class_names: list[str],
    formula: str | None = None,
) -> tuple[np.ndarray, str]:
    formula_norm = str(formula or "prob_class2_high").strip() or "prob_class2_high"
    n_classes = int(prob_stack.shape[0])
    is_exploration_4class = class_names[:4] == [
        "carbonate_background",
        "class1_enriched",
        "class2_low_grade",
        "class2_high_grade",
    ]
    is_grade_geochem = class_names[:5] == [
        "carbonate_background",
        "class1_low_grade",
        "class1_high_grade",
        "class2_low_grade",
        "class2_high_grade",
    ]
    if is_exploration_4class and n_classes >= 4:
        p_class2 = prob_stack[2] + prob_stack[3]
        p_class2_high = prob_stack[3]
    elif is_grade_geochem and n_classes >= 5:
        p_class2 = prob_stack[3] + prob_stack[4]
        p_class2_high = prob_stack[4]
    else:
        high_idx = n_classes - 1
        p_class2 = prob_stack[high_idx]
        p_class2_high = prob_stack[high_idx]
    p_high_given_class2 = p_class2_high / np.clip(p_class2, 1e-12, None)

    if formula_norm in {"prob_class2_high", "score_1", "p_class2_high"}:
        score = p_class2_high
    elif formula_norm in {"class2_high_given_class2", "p_high_given_class2", "score_2"}:
        score = p_high_given_class2
    elif formula_norm in {"class2_times_sqrt_high_given_class2", "score_3"}:
        score = p_class2 * np.sqrt(np.clip(p_high_given_class2, 0.0, 1.0))
    elif formula_norm in {"blend_0p70_class2_0p30_high_given_class2", "score_4"}:
        score = (0.70 * p_class2) + (0.30 * p_high_given_class2)
    elif formula_norm.startswith("validation_blend_alpha_"):
        try:
            alpha = float(formula_norm.rsplit("_", maxsplit=1)[-1])
        except ValueError:
            alpha = 0.7
        alpha = float(np.clip(alpha, 0.0, 1.0))
        score = (alpha * p_class2) + ((1.0 - alpha) * p_high_given_class2)
    else:
        logger.warning("Unknown high-grade prospectivity score formula '%s'; using prob_class2_high.", formula_norm)
        formula_norm = "prob_class2_high"
        score = p_class2_high
    return np.clip(score.astype(np.float32), 0.0, 1.0), formula_norm


def _load_training_or_cascade_artifacts(artifacts_path: str) -> tuple[str, TrainingArtifacts | CascadeTrainingArtifacts | CascadeEnsembleArtifacts]:
    with open(artifacts_path, "r", encoding="utf-8") as f:
        raw = f.read()
    try:
        obj = json.loads(raw)
        artifact_type = str(obj.get("artifact_type", "")).lower() if isinstance(obj, dict) else ""
    except Exception:
        artifact_type = ""
    if artifact_type in {"cascade_grade_geochem_ensemble", "cascade_exploration_4class_ensemble"}:
        ensemble = CascadeEnsembleArtifacts.model_validate_json(raw)
        return "ensemble", ensemble
    try:
        cascade = CascadeTrainingArtifacts.model_validate_json(raw)
        return "cascade", cascade
    except Exception:
        single = TrainingArtifacts.model_validate_json(raw)
        return "single", single


def _validate_env_grid_has_features(env_gridded_nc_path: str, feature_names: list[str]) -> None:
    try:
        with open_dataset(env_gridded_nc_path) as ds:
            present = set(ds.data_vars.keys())
            dims = set(ds.sizes.keys())
            coords = set(ds.coords.keys())
    except Exception as e:
        raise FileNotFoundError(f"Failed to open env grid NetCDF: {env_gridded_nc_path}. Error: {e}") from e

    if "latitude" not in dims and "latitude" not in coords:
        raise ConfigurationError(f"Environment grid is missing required coordinate/dimension 'latitude': {env_gridded_nc_path}")
    if "longitude" not in dims and "longitude" not in coords:
        raise ConfigurationError(f"Environment grid is missing required coordinate/dimension 'longitude': {env_gridded_nc_path}")

    missing = [name for name in feature_names if name not in present]
    if missing:
        missing_preview = ", ".join(missing[:20])
        more = "" if len(missing) <= 20 else f" ... (+{len(missing) - 20} more)"
        raise ConfigurationError(
            "Environment grid does not contain all features required by the trained model.\n"
            f"- env_gridded.nc: {env_gridded_nc_path}\n"
            f"- missing ({len(missing)}/{len(feature_names)}): {missing_preview}{more}\n"
            "Typical cause: using an env_gridded.nc produced by a different preprocessing configuration/run.\n"
            "Fix: re-run preprocessing with matching settings, or point prediction to the env_gridded.nc used during training."
        )


def run_prediction_pipeline_multiclass(
    pred_config_obj: PredictionConfig,
    progress_callback=None,
) -> Tuple[bool, dict]:
    logger.info("Prediction Pipeline (multiclass): Starting...")
    start_time = time.time()

    pred_run_id = f"prediction_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_base_dir = Path(RESULTS_GUI_DIR) / pred_run_id

    pred_config_obj = pred_config_obj.model_copy(deep=True)
    pred_config_obj.output_mode = "summaries_2d"
    pred_config_obj.enable_2d_age_summaries = True

    output_dirs = {
        "base": str(run_base_dir),
        "predictions": str(run_base_dir / "predictions_nc"),
        "summary_2d": str(run_base_dir / "plots" / "summary_2d"),
        "summary_2d_maps": str(run_base_dir / "plots" / "summary_2d" / "maps"),
        "reports": str(run_base_dir / "reports"),
        "tables": str(run_base_dir / "tables"),
    }

    try:
        for dir_path in output_dirs.values():
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Failed to create output directories: {e}") from e

    pred_run_summary = {
        "run_id": pred_run_id,
        "run_type": "prediction",
        "timestamp_start": datetime.now().isoformat(),
        "status": "started",
        "results_schema_version": RESULTS_SCHEMA_VERSION,
        "inputs": {"prediction_config": pred_config_obj.model_dump()},
        "results": {},
        "artifacts": {"paths": output_dirs},
    }

    try:
        artifacts_path = pred_config_obj.training_artifacts_path
        if not artifacts_path or not os.path.exists(artifacts_path):
            raise FileNotFoundError(f"Training artifacts file not found at: {artifacts_path}")

        artifact_kind, loaded_artifacts = _load_training_or_cascade_artifacts(artifacts_path)

        pred_run_summary["inputs"]["source_training_artifacts_path"] = artifacts_path

        if artifact_kind == "single":
            training_artifacts = loaded_artifacts
            task_type = str(getattr(training_artifacts.norm_params.rey, "task_type", None) or "").lower()
            if task_type != "multiclass_classification":
                raise ConfigurationError(
                    "This prediction pipeline supports multiclass classification only. "
                    f"Got task_type='{task_type or None}'."
                )
            pred_run_summary["results"]["target_task_type"] = task_type

            model_path_from_artifacts = getattr(training_artifacts, "model_path", None)
            resolved_model_path = _resolve_model_path_from_artifacts(artifacts_path, model_path_from_artifacts)
            if not resolved_model_path or not os.path.exists(resolved_model_path):
                raise FileNotFoundError(
                    "Model file path is missing or file not found.\n"
                    f"- from artifacts: {model_path_from_artifacts}\n"
                    f"- tried resolved: {resolved_model_path}\n"
                    f"- artifacts: {artifacts_path}"
                )
            if resolved_model_path != model_path_from_artifacts:
                logger.warning(f"Prediction Pipeline: Using resolved model_path={resolved_model_path}")
            training_artifacts = training_artifacts.model_copy(update={"model_path": resolved_model_path})
            pred_run_summary["inputs"]["resolved_model_path"] = resolved_model_path
            feature_names_for_env = list(training_artifacts.model_init_config.feature_names)
            num_classes = int(getattr(training_artifacts.norm_params.rey, "num_classes", 0) or 0)
            class_names = list(getattr(training_artifacts.norm_params.rey, "class_names", None) or [])
            high_grade_score_formula = "prob_class2_high"
            high_grade_score_source = "default"
            cascade_artifacts = None
            ensemble_artifacts = None
        else:
            ensemble_artifacts = loaded_artifacts if artifact_kind == "ensemble" else None
            cascade_artifacts = loaded_artifacts if artifact_kind == "cascade" else None
            pred_run_summary["results"]["target_task_type"] = "multiclass_classification"
            feature_names_for_env = []
            artifact_for_schema = cascade_artifacts
            if ensemble_artifacts is not None:
                if not ensemble_artifacts.members:
                    raise ConfigurationError("Cascade ensemble artifact has no members.")
                first_member_path = ensemble_artifacts.members[0].cascade_artifacts_path
                with open(first_member_path, "r", encoding="utf-8") as f:
                    artifact_for_schema = CascadeTrainingArtifacts.model_validate_json(f.read())
            num_classes = int(getattr(loaded_artifacts, "final_num_classes", 0) or 0)
            class_names = list(getattr(loaded_artifacts, "final_class_names", None) or [])
            high_grade_score_formula = str(getattr(loaded_artifacts, "high_grade_score_formula", None) or "prob_class2_high")
            high_grade_score_source = str(getattr(loaded_artifacts, "high_grade_score_source", None) or "artifact_or_default")
            for stage in [
                artifact_for_schema.stage_main,
                artifact_for_schema.stage_class1_grade,
                artifact_for_schema.stage_class2_grade,
            ]:
                if stage is None:
                    continue
                with open(stage.training_artifacts_path, "r", encoding="utf-8") as f:
                    stage_artifacts = TrainingArtifacts.model_validate_json(f.read())
                feature_names_for_env = list(stage_artifacts.model_init_config.feature_names)
                break

        env_gridded_nc_path = getattr(pred_config_obj, "env_gridded_nc_path", None)
        if not env_gridded_nc_path or not os.path.exists(env_gridded_nc_path):
            run_dir = _infer_source_run_dir(artifacts_path)
            inferred = _infer_env_grid_from_run_dir(run_dir) if run_dir else None
            if inferred:
                logger.info(f"Prediction Pipeline: using training-run env grid: {inferred}")
                env_gridded_nc_path = inferred
                pred_config_obj.env_gridded_nc_path = env_gridded_nc_path
                pred_run_summary["inputs"]["prediction_config"] = pred_config_obj.model_dump()

        if not env_gridded_nc_path or not os.path.exists(env_gridded_nc_path):
            raise FileNotFoundError(f"Environment grid data path missing or file not found: {env_gridded_nc_path}")

        _validate_env_grid_has_features(env_gridded_nc_path, feature_names_for_env)

        if progress_callback:
            progress_callback(10, "Initializing predictor...")

        predictor = None
        cascade_predictors = None
        ensemble_predictors = None
        if artifact_kind == "single":
            predictor = GridPredictor(
                training_artifacts=training_artifacts,
                env_gridded_nc_path=env_gridded_nc_path,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                batch_size=pred_config_obj.batch_size,
                progress_callback=progress_callback,
                pred_config=pred_config_obj,
            )
        elif artifact_kind == "cascade":
            cascade_predictors = {}
            for key, stage in [
                ("main", cascade_artifacts.stage_main),
                ("class1", cascade_artifacts.stage_class1_grade),
                ("class2", cascade_artifacts.stage_class2_grade),
            ]:
                if stage is None:
                    continue
                with open(stage.training_artifacts_path, "r", encoding="utf-8") as f:
                    stage_artifacts = TrainingArtifacts.model_validate_json(f.read())
                resolved_stage_model = _resolve_model_path_from_artifacts(stage.training_artifacts_path, stage_artifacts.model_path)
                if not resolved_stage_model or not os.path.exists(resolved_stage_model):
                    raise FileNotFoundError(f"Missing cascade stage model for {key}: {resolved_stage_model}")
                stage_artifacts = stage_artifacts.model_copy(update={"model_path": resolved_stage_model})
                cascade_predictors[key] = GridPredictor(
                    training_artifacts=stage_artifacts,
                    env_gridded_nc_path=env_gridded_nc_path,
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                    batch_size=pred_config_obj.batch_size,
                    progress_callback=None,
                    pred_config=pred_config_obj,
                )
        else:
            ensemble_predictors = []
            for member in ensemble_artifacts.members:
                with open(member.cascade_artifacts_path, "r", encoding="utf-8") as f:
                    member_artifacts = CascadeTrainingArtifacts.model_validate_json(f.read())
                member_predictors = {}
                for key, stage in [
                    ("main", member_artifacts.stage_main),
                    ("class1", member_artifacts.stage_class1_grade),
                    ("class2", member_artifacts.stage_class2_grade),
                ]:
                    if stage is None:
                        continue
                    with open(stage.training_artifacts_path, "r", encoding="utf-8") as f:
                        stage_artifacts = TrainingArtifacts.model_validate_json(f.read())
                    resolved_stage_model = _resolve_model_path_from_artifacts(stage.training_artifacts_path, stage_artifacts.model_path)
                    if not resolved_stage_model or not os.path.exists(resolved_stage_model):
                        raise FileNotFoundError(f"Missing cascade ensemble stage model for {key}: {resolved_stage_model}")
                    stage_artifacts = stage_artifacts.model_copy(update={"model_path": resolved_stage_model})
                    member_predictors[key] = GridPredictor(
                        training_artifacts=stage_artifacts,
                        env_gridded_nc_path=env_gridded_nc_path,
                        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        batch_size=pred_config_obj.batch_size,
                        progress_callback=None,
                        pred_config=pred_config_obj,
                    )
                ensemble_predictors.append({"weight": float(member.weight or 1.0), "predictors": member_predictors})

        if predictor is not None:
            predictor_for_mode = predictor
        elif cascade_predictors is not None:
            predictor_for_mode = cascade_predictors["main"]
        else:
            predictor_for_mode = ensemble_predictors[0]["predictors"]["main"]
        if predictor_for_mode.depth_feature_type != "age_ka_proxy":
            logger.warning(
                "Prediction Pipeline: 2D age-window summaries requested, but training artifacts do not "
                "declare depth_feature_type='age_ka_proxy'. Summaries may be less interpretable."
            )

        bounds = pred_config_obj.prediction_grid_settings.geospatial_bounds
        user_lat_min = bounds.min_lat if bounds else None
        user_lat_max = bounds.max_lat if bounds else None
        user_lon_min = bounds.min_lon if bounds else None
        user_lon_max = bounds.max_lon if bounds else None

        def _sample_ages(end_ka: float, n: int, scheme: str) -> list[float]:
            end_ka = float(end_ka)
            n = max(2, int(n))
            if end_ka <= 0:
                return [0.0]
            x = np.linspace(0.0, 1.0, n)
            if scheme == "quadratic":
                x = x * x
            ages = (x * end_ka).tolist()
            ages[0] = 0.0
            ages[-1] = end_ka
            return ages

        scheme = str(pred_config_obj.summary_sampling_scheme or "linear").lower()
        age_samples = set()
        windows: list[tuple[str, float, float]] = []
        if pred_config_obj.enable_holocene_summary:
            windows.append(("holocene", 0.0, float(pred_config_obj.holocene_end_ka)))
            age_samples.update(_sample_ages(pred_config_obj.holocene_end_ka, pred_config_obj.summary_n_samples_holocene, scheme))
        if pred_config_obj.enable_lgc_summary:
            windows.append(("lgc", 0.0, float(pred_config_obj.lgc_end_ka)))
            age_samples.update(_sample_ages(pred_config_obj.lgc_end_ka, pred_config_obj.summary_n_samples_lgc, scheme))
        if pred_config_obj.enable_quaternary_summary:
            end_ka = float(pred_config_obj.training_age_domain_ka or 2580.0)
            windows.append(("quaternary", 0.0, end_ka))
            age_samples.update(_sample_ages(end_ka, pred_config_obj.summary_n_samples_quaternary, scheme))
        if not windows:
            windows = [("quaternary", 0.0, float(pred_config_obj.training_age_domain_ka or 2580.0))]

        age_levels = sorted([a for a in age_samples if np.isfinite(a) and a >= 0.0])
        for _, _, b in windows:
            age_levels.append(float(b))
        age_levels = sorted(set([float(a) for a in age_levels]))

        tmp_grid = pred_config_obj.prediction_grid_settings.model_copy(deep=True)
        tmp_grid.vertical_axis_mode = "age_ka"
        tmp_grid.age_levels_ka = age_levels
        tmp_grid.depth_levels = []

        if progress_callback:
            progress_callback(20, f"Predicting age-grid (n_age_levels={len(age_levels)})...")

        def _compose_cascade_probability_stack(pred_main, pred_c1, pred_c2) -> tuple[xr.Dataset, np.ndarray, dict[str, np.ndarray]]:
            p0 = np.asarray(pred_main["predicted_prob_class0"].data, dtype=np.float32)
            p_main1 = np.asarray(pred_main["predicted_prob_class1"].data, dtype=np.float32)
            p_main2 = np.asarray(pred_main["predicted_prob_class2"].data, dtype=np.float32)
            p_c2_low = np.asarray(pred_c2["predicted_prob_class0"].data, dtype=np.float32)
            p_c2_high = np.asarray(pred_c2["predicted_prob_class1"].data, dtype=np.float32)

            if pred_c1 is None:
                prob_stack_local = np.stack(
                    [
                        p0,
                        p_main1,
                        p_main2 * p_c2_low,
                        p_main2 * p_c2_high,
                    ],
                    axis=0,
                )
                intermediates = {
                    "prob_main_carbonate": p0,
                    "prob_main_class1": p_main1,
                    "prob_main_class2": p_main2,
                    "prob_class2_low_given_class2": p_c2_low,
                    "prob_class2_high_given_class2": p_c2_high,
                }
            else:
                p_c1_low = np.asarray(pred_c1["predicted_prob_class0"].data, dtype=np.float32)
                p_c1_high = np.asarray(pred_c1["predicted_prob_class1"].data, dtype=np.float32)
                prob_stack_local = np.stack(
                    [
                        p0,
                        p_main1 * p_c1_low,
                        p_main1 * p_c1_high,
                        p_main2 * p_c2_low,
                        p_main2 * p_c2_high,
                    ],
                    axis=0,
                )
                intermediates = {
                    "prob_main_carbonate": p0,
                    "prob_main_class1": p_main1,
                    "prob_main_class2": p_main2,
                    "prob_class1_low_given_class1": p_c1_low,
                    "prob_class1_high_given_class1": p_c1_high,
                    "prob_class2_low_given_class2": p_c2_low,
                    "prob_class2_high_given_class2": p_c2_high,
                }
            prob_sum_local = np.clip(np.sum(prob_stack_local, axis=0, keepdims=True), 1e-12, None)
            prob_stack_local = prob_stack_local / prob_sum_local
            return pred_main, prob_stack_local.astype(np.float32), intermediates

        def _dataset_from_prob_stack(
            template_ds: xr.Dataset,
            prob_stack: np.ndarray,
            *,
            intermediates: dict[str, np.ndarray] | None = None,
        ) -> xr.Dataset:
            pred_class = np.argmax(prob_stack, axis=0).astype(np.int16)
            pred_entropy = (-np.sum(np.clip(prob_stack, 1e-12, 1.0) * np.log(np.clip(prob_stack, 1e-12, 1.0)), axis=0)).astype(np.float32)
            pred_conf = np.max(prob_stack, axis=0).astype(np.float32)

            out_ds = template_ds.copy(deep=True)
            for var_name in list(out_ds.data_vars):
                if var_name.startswith("predicted_prob_class") or var_name in {
                    "predicted_class_argmax",
                    "pred_grade_geochem_class_id",
                    "predicted_class_entropy",
                    "prediction_entropy",
                    "prediction_confidence",
                    "prob_carbonate_background",
                    "prob_class1_low_grade",
                    "prob_class1_high_grade",
                    "prob_class2_low_grade",
                    "prob_class2_high_grade",
                    "prob_main_carbonate",
                    "prob_main_class1",
                    "prob_main_class2",
                    "prob_class1_low_given_class1",
                    "prob_class1_high_given_class1",
                    "prob_class2_low_given_class2",
                    "prob_class2_high_given_class2",
                    "high_grade_prospectivity_score",
                    "pred_exploration_class_id",
                }:
                    out_ds = out_ds.drop_vars(var_name)

            vertical_dim = "age_ka" if "age_ka" in out_ds.coords else "depth_mbsf"
            if prob_stack.shape[0] == 4:
                grade_aliases_local = {
                    0: "prob_carbonate_background",
                    1: "prob_class1_enriched",
                    2: "prob_class2_low_grade",
                    3: "prob_class2_high_grade",
                }
            else:
                grade_aliases_local = {
                    0: "prob_carbonate_background",
                    1: "prob_class1_low_grade",
                    2: "prob_class1_high_grade",
                    3: "prob_class2_low_grade",
                    4: "prob_class2_high_grade",
                }
            for class_id in range(prob_stack.shape[0]):
                out_ds[f"predicted_prob_class{class_id}"] = (
                    (vertical_dim, "latitude", "longitude"),
                    prob_stack[class_id].astype(np.float32),
                )
                out_ds[grade_aliases_local[class_id]] = (
                    (vertical_dim, "latitude", "longitude"),
                    prob_stack[class_id].astype(np.float32),
                )
            out_ds["predicted_class_argmax"] = ((vertical_dim, "latitude", "longitude"), pred_class)
            out_ds["pred_grade_geochem_class_id"] = ((vertical_dim, "latitude", "longitude"), pred_class)
            if prob_stack.shape[0] == 4:
                out_ds["pred_exploration_class_id"] = ((vertical_dim, "latitude", "longitude"), pred_class)
                out_ds["pred_exploration_class_id"].attrs.update(
                    {
                        "units": "class_id",
                        "long_name": "Predicted exploration class id",
                    }
                )
            score, score_formula = _high_grade_prospectivity_score_from_stack(prob_stack, class_names, high_grade_score_formula)
            out_ds["high_grade_prospectivity_score"] = ((vertical_dim, "latitude", "longitude"), score)
            out_ds["high_grade_prospectivity_score"].attrs.update(
                {
                    "units": "1",
                    "long_name": "High-grade prospectivity score",
                    "formula": score_formula,
                    "source": high_grade_score_source,
                    "description": "Validation-selectable ranking score for class2 high-grade prospectivity, separate from class argmax.",
                }
            )
            out_ds.attrs["high_grade_score_formula"] = score_formula
            out_ds.attrs["high_grade_score_source"] = high_grade_score_source
            out_ds["prediction_entropy"] = ((vertical_dim, "latitude", "longitude"), pred_entropy)
            out_ds["predicted_class_entropy"] = ((vertical_dim, "latitude", "longitude"), pred_entropy)
            out_ds["prediction_confidence"] = ((vertical_dim, "latitude", "longitude"), pred_conf)
            for key, arr in (intermediates or {}).items():
                out_ds[key] = ((vertical_dim, "latitude", "longitude"), arr.astype(np.float32))
            return out_ds

        if artifact_kind == "single":
            pred_age_ds = predictor.predict_3d_grid(pred_grid_settings=tmp_grid)
        elif artifact_kind == "cascade":
            pred_main = cascade_predictors["main"].predict_3d_grid(pred_grid_settings=tmp_grid)
            pred_c1 = cascade_predictors["class1"].predict_3d_grid(pred_grid_settings=tmp_grid) if "class1" in cascade_predictors else None
            pred_c2 = cascade_predictors["class2"].predict_3d_grid(pred_grid_settings=tmp_grid)
            if pred_main is None or pred_c2 is None:
                raise PredictionFailedError("Cascade predictors returned None.")
            template_ds, prob_stack, intermediates = _compose_cascade_probability_stack(pred_main, pred_c1, pred_c2)
            pred_age_ds = _dataset_from_prob_stack(template_ds, prob_stack, intermediates=intermediates)
            pred_age_ds.attrs["class_names"] = ",".join(cascade_artifacts.final_class_names)
            pred_age_ds.attrs["cascade_artifact_type"] = cascade_artifacts.artifact_type
        else:
            weighted_probs = []
            weights = []
            template_ds = None
            for member in ensemble_predictors:
                predictors = member["predictors"]
                pred_main = predictors["main"].predict_3d_grid(pred_grid_settings=tmp_grid)
                pred_c1 = predictors["class1"].predict_3d_grid(pred_grid_settings=tmp_grid) if "class1" in predictors else None
                pred_c2 = predictors["class2"].predict_3d_grid(pred_grid_settings=tmp_grid)
                if pred_main is None or pred_c2 is None:
                    raise PredictionFailedError("Cascade ensemble member returned None.")
                member_template, member_prob, _member_intermediates = _compose_cascade_probability_stack(pred_main, pred_c1, pred_c2)
                if template_ds is None:
                    template_ds = member_template
                weighted_probs.append(member_prob * float(member["weight"]))
                weights.append(float(member["weight"]))
            if not weighted_probs or template_ds is None:
                raise PredictionFailedError("Cascade ensemble produced no probabilities.")
            prob_stack = np.sum(np.stack(weighted_probs, axis=0), axis=0) / max(float(np.sum(weights)), 1e-12)
            prob_stack = prob_stack / np.clip(np.sum(prob_stack, axis=0, keepdims=True), 1e-12, None)
            pred_age_ds = _dataset_from_prob_stack(template_ds, prob_stack.astype(np.float32), intermediates=None)
            pred_age_ds.attrs["class_names"] = ",".join(ensemble_artifacts.final_class_names)
            pred_age_ds.attrs["cascade_artifact_type"] = ensemble_artifacts.artifact_type
            pred_age_ds.attrs["cascade_ensemble_n_members"] = int(len(ensemble_predictors))
            if ensemble_artifacts.recommended_high_grade_threshold is not None:
                pred_age_ds.attrs["recommended_high_grade_threshold"] = float(ensemble_artifacts.recommended_high_grade_threshold)
        if pred_age_ds is None:
            raise PredictionFailedError("Age-axis prediction returned None; cannot build 2D summaries.")
        if "age_ka" not in pred_age_ds.coords:
            raise PredictionFailedError("Expected 'age_ka' coordinate for summary aggregation.")

        quantile = float(pred_config_obj.summary_quantile or 0.95)
        quantile = min(0.999, max(0.5, quantile))

        num_classes = max(num_classes, len(class_names))
        if not class_names and num_classes > 0:
            class_names = [f"class{i}" for i in range(num_classes)]

        def _label(c: int) -> str:
            return str(class_names[c]) if (0 <= c < len(class_names)) else f"class{c}"

        summary_vars: dict[str, xr.DataArray] = {}
        is_grade_geochem = class_names[:5] == [
            "carbonate_background",
            "class1_low_grade",
            "class1_high_grade",
            "class2_low_grade",
            "class2_high_grade",
        ]
        grade_aliases = {
            0: "prob_carbonate_background",
            1: "prob_class1_low_grade",
            2: "prob_class1_high_grade",
            3: "prob_class2_low_grade",
            4: "prob_class2_high_grade",
        }


        try:
            dominant_thr = float(getattr(pred_config_obj, "dominant_confidence_threshold", 0.75) or 0.75)
        except Exception:
            dominant_thr = 0.75
        dominant_thr = float(np.clip(dominant_thr, 0.0, 1.0))
        dominant_thr_tag = f"{int(round(dominant_thr * 100)):03d}"

        for tag, a0, a1 in windows:
            sel = pred_age_ds.sel(age_ka=slice(a0, a1))
            if sel.sizes.get("age_ka", 0) <= 0:
                continue

            mean_maps = []
            for c in range(num_classes):
                prob_name = f"predicted_prob_class{c}"
                if prob_name not in sel:
                    continue
                prob = (
                    sel[prob_name].where(sel["is_in_training_age_domain"] > 0)
                    if pred_config_obj.mask_out_of_domain_in_maps and "is_in_training_age_domain" in sel
                    else sel[prob_name]
                )
                mean2d = prob.mean(dim="age_ka", skipna=True)
                q2d = prob.quantile(quantile, dim="age_ka", skipna=True)
                mean2d.attrs.update(
                    {
                        "units": "1",
                        "long_name": f"Mean probability of {_label(c)} over {a0:g}-{a1:g} ka ({tag})",
                        "window_start_ka": float(a0),
                        "window_end_ka": float(a1),
                        "class_index": int(c),
                        "class_label": _label(c),
                    }
                )
                q2d.attrs.update(
                    {
                        "units": "1",
                        "long_name": f"High-quantile probability (p{int(quantile*100)}) of {_label(c)} over {a0:g}-{a1:g} ka ({tag})",
                        "window_start_ka": float(a0),
                        "window_end_ka": float(a1),
                        "class_index": int(c),
                        "class_label": _label(c),
                    }
                )
                summary_vars[f"predicted_prob_class{c}_mean_0_{tag}"] = mean2d
                summary_vars[f"predicted_prob_class{c}_p{int(quantile*100)}_0_{tag}"] = q2d
                if is_grade_geochem and c in grade_aliases:
                    alias = grade_aliases[c]
                    summary_vars[f"{alias}_mean_0_{tag}"] = mean2d
                    summary_vars[f"{alias}_p{int(quantile*100)}_0_{tag}"] = q2d
                if class_names[:4] == [
                    "carbonate_background",
                    "class1_enriched",
                    "class2_low_grade",
                    "class2_high_grade",
                ]:
                    exploration_aliases = {
                        0: "prob_carbonate_background",
                        1: "prob_class1_enriched",
                        2: "prob_class2_low_grade",
                        3: "prob_class2_high_grade",
                    }
                    alias = exploration_aliases.get(c)
                    if alias:
                        summary_vars[f"{alias}_mean_0_{tag}"] = mean2d
                        summary_vars[f"{alias}_p{int(quantile*100)}_0_{tag}"] = q2d
                mean_maps.append(mean2d)

                std_name = f"{prob_name}_std"
                if pred_config_obj.enable_mc_dropout and std_name in sel:
                    unc = sel[std_name].mean(dim="age_ka", skipna=True)
                    unc.attrs.update(
                        {
                            "units": "1",
                            "long_name": f"Mean uncertainty (std, MC Dropout) of {_label(c)} over {a0:g}-{a1:g} ka ({tag})",
                            "window_start_ka": float(a0),
                            "window_end_ka": float(a1),
                            "class_index": int(c),
                            "class_label": _label(c),
                        }
                    )
                    summary_vars[f"predicted_prob_class{c}_std_0_{tag}"] = unc


            if mean_maps:
                try:
                    mean_stack = xr.concat(mean_maps, dim="class")
                    mean_stack = mean_stack.assign_coords(
                        {"class": np.arange(int(mean_stack.sizes.get("class", len(mean_maps))))}
                    )
                    arr = np.asarray(mean_stack.data, dtype=float)
                    arr2 = np.where(np.isfinite(arr), arr, -np.inf)
                    valid_any = np.any(np.isfinite(arr), axis=0)

                    mode = np.argmax(arr2, axis=0).astype(np.int16)
                    top1 = np.max(arr2, axis=0)
                    if arr2.shape[0] >= 2:
                        part = np.partition(arr2, kth=arr2.shape[0] - 2, axis=0)
                        top2 = part[-2, :, :]
                    else:
                        top2 = np.full_like(top1, -np.inf)

                    top1 = np.where(np.isfinite(top1) & (top1 > -np.inf / 2), top1, np.nan).astype(np.float32)
                    top2 = np.where(np.isfinite(top2) & (top2 > -np.inf / 2), top2, np.nan).astype(np.float32)
                    margin = (top1 - top2).astype(np.float32)

                    mode = np.where(valid_any, mode, -1).astype(np.int16)
                    top1 = np.where(valid_any, top1, np.nan).astype(np.float32)
                    margin = np.where(valid_any, margin, np.nan).astype(np.float32)

                    mode_da = xr.DataArray(mode.astype(np.int16), coords=mean_maps[0].coords, dims=mean_maps[0].dims)
                    mode_da.attrs.update(
                        {
                            "units": "class_id",
                            "long_name": f"Dominant class by window-mean probability over {a0:g}-{a1:g} ka ({tag})",
                            "window_start_ka": float(a0),
                            "window_end_ka": float(a1),
                            "definition": "argmax over window-mean per-class probabilities (age proxy). -1 indicates all-NaN pixel.",
                        }
                    )
                    summary_vars[f"predicted_class_mode_mean_0_{tag}"] = mode_da
                    if is_grade_geochem:
                        summary_vars[f"dominant_class_0_{tag}"] = mode_da
                    if class_names[:4] == [
                        "carbonate_background",
                        "class1_enriched",
                        "class2_low_grade",
                        "class2_high_grade",
                    ]:
                        summary_vars[f"pred_exploration_class_id_mode_mean_0_{tag}"] = mode_da

                    mp_da = xr.DataArray(top1, coords=mean_maps[0].coords, dims=mean_maps[0].dims)
                    mp_da.attrs.update(
                        {
                            "units": "1",
                            "long_name": f"Max mean class probability over {a0:g}-{a1:g} ka ({tag})",
                            "window_start_ka": float(a0),
                            "window_end_ka": float(a1),
                            "definition": "max over window-mean per-class probabilities (age proxy).",
                        }
                    )
                    summary_vars[f"predicted_class_maxprob_mean_0_{tag}"] = mp_da
                    if is_grade_geochem:
                        summary_vars[f"prediction_confidence_mean_0_{tag}"] = mp_da

                    mg_da = xr.DataArray(margin, coords=mean_maps[0].coords, dims=mean_maps[0].dims)
                    mg_da.attrs.update(
                        {
                            "units": "1",
                            "long_name": f"Top-1 minus top-2 mean probability over {a0:g}-{a1:g} ka ({tag})",
                            "window_start_ka": float(a0),
                            "window_end_ka": float(a1),
                            "definition": "Confidence proxy from window-mean probabilities: (p_max - p_2nd).",
                        }
                    )
                    summary_vars[f"predicted_class_margin_mean_0_{tag}"] = mg_da

                    confmask = np.where(np.isfinite(top1) & (top1 >= dominant_thr), mode, -1).astype(np.int16)
                    confmask = np.where(valid_any, confmask, -1).astype(np.int16)
                    conf_da = xr.DataArray(confmask, coords=mean_maps[0].coords, dims=mean_maps[0].dims)
                    conf_da.attrs.update(
                        {
                            "units": "class_id",
                            "long_name": f"Dominant class (pmax >= {dominant_thr:.2f}) over {a0:g}-{a1:g} ka ({tag})",
                            "window_start_ka": float(a0),
                            "window_end_ka": float(a1),
                            "pmax_threshold": float(dominant_thr),
                            "definition": "Mask dominant class map by pmax threshold. -1 indicates masked or all-NaN pixel.",
                        }
                    )
                    summary_vars[f"predicted_class_mode_mean_confmask_pmax{dominant_thr_tag}_0_{tag}"] = conf_da
                    is_exploration_4class = class_names[:4] == [
                        "carbonate_background",
                        "class1_enriched",
                        "class2_low_grade",
                        "class2_high_grade",
                    ]
                    if is_grade_geochem or is_exploration_4class:
                        high_indices = [1, 3] if is_exploration_4class else [2, 4]
                        high_stack = xr.concat([mean_maps[i] for i in high_indices], dim="high_class")
                        high_stack = high_stack.assign_coords({"high_class": high_indices})
                        any_high = high_stack.max(dim="high_class", skipna=True)
                        any_high.attrs.update(
                            {
                                "units": "1",
                                "long_name": f"Maximum high-grade probability over {a0:g}-{a1:g} ka ({tag})",
                                "window_start_ka": float(a0),
                                "window_end_ka": float(a1),
                            }
                        )
                        summary_vars[f"max_prob_any_high_grade_0_{tag}"] = any_high
                        if is_exploration_4class:
                            summary_vars[f"max_prob_class1_enriched_0_{tag}"] = mean_maps[1]
                            summary_vars[f"max_prob_class2_high_0_{tag}"] = mean_maps[3]
                        else:
                            summary_vars[f"max_prob_class1_high_0_{tag}"] = mean_maps[2]
                            summary_vars[f"max_prob_class2_high_0_{tag}"] = mean_maps[4]

                        high_arr = np.stack(
                            [np.asarray(mean_maps[i].data, dtype=float) for i in high_indices],
                            axis=0,
                        )
                        high_arg = np.argmax(np.where(np.isfinite(high_arr), high_arr, -np.inf), axis=0)
                        high_valid = np.any(np.isfinite(high_arr), axis=0)
                        high_dom = np.where(high_valid, np.where(high_arg == 0, high_indices[0], high_indices[1]), -1).astype(np.int16)
                        high_da = xr.DataArray(high_dom, coords=mean_maps[0].coords, dims=mean_maps[0].dims)
                        high_da.attrs.update(
                            {
                                "units": "class_id",
                                "long_name": f"Dominant high-grade class over {a0:g}-{a1:g} ka ({tag})",
                                "definition": (
                                    "1=class1_enriched, 3=class2_high_grade, -1=missing."
                                    if is_exploration_4class
                                    else "2=class1_high_grade, 4=class2_high_grade, -1=missing."
                                ),
                            }
                        )
                        summary_vars[f"dominant_high_grade_class_0_{tag}"] = high_da
                except Exception:
                    logger.debug("Prediction Pipeline: Failed to compute dominant-class summaries.", exc_info=True)


            if bool(getattr(pred_config_obj, "enable_multiclass_target_threshold_maps", True)) and num_classes > 0:
                target_idx = int(getattr(pred_config_obj, "multiclass_target_class_index", 1) or 0)
                if 0 <= target_idx < num_classes:
                    thr_source = str(getattr(pred_config_obj, "multiclass_target_threshold_source", "manual") or "manual")
                    target_thr = float(getattr(pred_config_obj, "multiclass_target_threshold", 0.5) or 0.5)
                    target_thr_note = "manual"
                    if thr_source != "manual":
                        try:
                            run_dir = _infer_source_run_dir(artifacts_path)

                            metric_map = {
                                "from_val_recommendation_f0_5": "F0.5",
                                "from_val_recommendation_f1": "F1",
                                "from_val_recommendation_f2": "F2",
                            }
                            metric_key = metric_map.get(thr_source)
                            if metric_key:

                                for split_name in ("val", "test", "train"):
                                    p = (
                                        run_dir
                                        / "tables"
                                        / split_name
                                        / f"target_class{int(target_idx)}_threshold_recommendations.json"
                                    )
                                    if not p.exists():
                                        continue
                                    rec = json.loads(p.read_text(encoding="utf-8"))
                                    best = (rec.get("best", {}) or {}) if isinstance(rec, dict) else {}
                                    entry = best.get(metric_key)
                                    if isinstance(entry, dict) and ("threshold" in entry):
                                        target_thr = float(entry["threshold"])
                                        target_thr_note = f"{thr_source}:{split_name}:{metric_key}"
                                        break
                        except Exception:
                            pass

                    target_thr = float(np.clip(target_thr, 0.0, 1.0))
                    pred_run_summary["results"]["multiclass_target_threshold"] = {
                        "multiclass_target_class_index": int(target_idx),
                        "multiclass_target_threshold": float(target_thr),
                        "multiclass_target_threshold_source": str(target_thr_note),
                    }

                    prob_k = (
                        sel[f"predicted_prob_class{target_idx}"].where(sel["is_in_training_age_domain"] > 0)
                        if pred_config_obj.mask_out_of_domain_in_maps and "is_in_training_age_domain" in sel
                        else sel[f"predicted_prob_class{target_idx}"]
                    )
                    indicator = (prob_k >= target_thr).astype(np.float32)
                    thr_tag_local = str(int(round(target_thr * 1000)))

                    if bool(getattr(pred_config_obj, "multiclass_target_include_freq", True)):
                        freq = indicator.mean(dim="age_ka", skipna=True)
                        freq.attrs.update(
                            {
                                "units": "1",
                                "long_name": f"Hotspot frequency of {_label(target_idx)} (P≥{target_thr:.3f}) over {a0:g}-{a1:g} ka ({tag})",
                                "window_start_ka": float(a0),
                                "window_end_ka": float(a1),
                                "class_index": int(target_idx),
                                "class_label": _label(target_idx),
                                "prob_threshold": float(target_thr),
                                "threshold_source": target_thr_note,
                            }
                        )
                        summary_vars[f"predicted_target_class{target_idx}_hotspot_freq_thr{thr_tag_local}_0_{tag}"] = freq

                    if bool(getattr(pred_config_obj, "multiclass_target_include_any", True)):
                        any_ex = indicator.max(dim="age_ka", skipna=True)
                        any_ex.attrs.update(
                            {
                                "units": "1",
                                "long_name": f"Any-hotspot of {_label(target_idx)} (P≥{target_thr:.3f}) over {a0:g}-{a1:g} ka ({tag})",
                                "window_start_ka": float(a0),
                                "window_end_ka": float(a1),
                                "class_index": int(target_idx),
                                "class_label": _label(target_idx),
                                "prob_threshold": float(target_thr),
                                "threshold_source": target_thr_note,
                            }
                        )
                        summary_vars[f"predicted_target_class{target_idx}_hotspot_any_thr{thr_tag_local}_0_{tag}"] = any_ex


            if "predicted_class_entropy" in sel:
                ent_sel = (
                    sel["predicted_class_entropy"].where(sel["is_in_training_age_domain"] > 0)
                    if pred_config_obj.mask_out_of_domain_in_maps and "is_in_training_age_domain" in sel
                    else sel["predicted_class_entropy"]
                )
                ent_mean = ent_sel.mean(dim="age_ka", skipna=True)
                ent_q = ent_sel.quantile(quantile, dim="age_ka", skipna=True)
                summary_vars[f"predicted_class_entropy_mean_0_{tag}"] = ent_mean
                summary_vars[f"predicted_class_entropy_p{int(quantile*100)}_0_{tag}"] = ent_q
                if is_grade_geochem:
                    summary_vars[f"uncertainty_entropy_mean_0_{tag}"] = ent_mean
                    summary_vars[f"uncertainty_entropy_p{int(quantile*100)}_0_{tag}"] = ent_q

            if "high_grade_prospectivity_score" in sel:
                hg_score = (
                    sel["high_grade_prospectivity_score"].where(sel["is_in_training_age_domain"] > 0)
                    if pred_config_obj.mask_out_of_domain_in_maps and "is_in_training_age_domain" in sel
                    else sel["high_grade_prospectivity_score"]
                )
                hg_score_mean = hg_score.mean(dim="age_ka", skipna=True)
                hg_score_q = hg_score.quantile(quantile, dim="age_ka", skipna=True)
                for da, suffix, long_name in [
                    (hg_score_mean, "mean", "Mean high-grade prospectivity score"),
                    (hg_score_q, f"p{int(quantile*100)}", f"High-quantile high-grade prospectivity score (p{int(quantile*100)})"),
                ]:
                    da.attrs.update(
                        {
                            "units": "1",
                            "long_name": f"{long_name} over {a0:g}-{a1:g} ka ({tag})",
                            "window_start_ka": float(a0),
                            "window_end_ka": float(a1),
                            "formula": pred_age_ds.attrs.get("high_grade_score_formula", high_grade_score_formula),
                        }
                )
                summary_vars[f"high_grade_prospectivity_score_mean_0_{tag}"] = hg_score_mean
                summary_vars[f"high_grade_prospectivity_score_p{int(quantile*100)}_0_{tag}"] = hg_score_q


            if "predicted_class_argmax" in sel and num_classes > 0:
                arg = (
                    sel["predicted_class_argmax"].where(sel["is_in_training_age_domain"] > 0)
                    if pred_config_obj.mask_out_of_domain_in_maps and "is_in_training_age_domain" in sel
                    else sel["predicted_class_argmax"]
                )
                for c in range(num_classes):
                    freq = (arg == c).mean(dim="age_ka", skipna=True)
                    summary_vars[f"predicted_class_argmax_freq_class{c}_0_{tag}"] = freq


            if "is_in_training_age_domain" in sel:
                frac = (sel["is_in_training_age_domain"] > 0).mean(dim="age_ka")
                summary_vars[f"in_domain_fraction_0_{tag}"] = frac


        if bool(getattr(pred_config_obj, "summary_include_env_diagnostics_maps", True)):
            for diag_var in ["env_center_valid_fraction", "env_center_any_missing", "sedimentation_rate", "sedimentation_rate_was_filled"]:
                if diag_var in pred_age_ds and diag_var not in summary_vars:
                    summary_vars[diag_var] = pred_age_ds[diag_var]

        summary_2d_ds = xr.Dataset(
            summary_vars,
            coords={"latitude": pred_age_ds["latitude"], "longitude": pred_age_ds["longitude"]},
        )
        _update_attrs_native(
            summary_2d_ds,
            {
                "description": "2D summary products aggregated over geologically defined age windows (age proxy).",
                "target_task_type": "multiclass_classification",
                "num_classes": int(num_classes),
                "class_names": ",".join(class_names),
                "training_age_domain_ka": float(pred_config_obj.training_age_domain_ka or 2580.0),
                "holocene_end_ka": float(pred_config_obj.holocene_end_ka),
                "lgc_end_ka": float(pred_config_obj.lgc_end_ka),
                "summary_quantile": float(quantile),
                "summary_sampling_scheme": scheme,
                "age_samples_ka_used": ",".join([f"{a:g}" for a in age_levels]),
                "high_grade_score_formula": pred_age_ds.attrs.get("high_grade_score_formula", high_grade_score_formula),
                "high_grade_score_source": pred_age_ds.attrs.get("high_grade_score_source", high_grade_score_source),
            }
        )

        summary_nc_path = os.path.join(output_dirs["predictions"], "predicted_mechanism_2d_summaries.nc")
        summary_2d_ds.to_netcdf(summary_nc_path)
        pred_run_summary["artifacts"]["paths"]["predicted_summary_2d_nc"] = summary_nc_path
        pred_run_summary["results"]["predicted_summary_2d_nc_path"] = summary_nc_path


        try:
            window_tags = [t for t, _, _ in windows]
            n_cls = int(num_classes)
            lat = summary_2d_ds["latitude"].to_numpy(dtype=float)
            lon = summary_2d_ds["longitude"].to_numpy(dtype=float)

            def _empty_2d_float() -> xr.DataArray:
                return xr.DataArray(
                    np.full((lat.size, lon.size), np.nan, dtype=np.float32),
                    coords={"latitude": summary_2d_ds["latitude"], "longitude": summary_2d_ds["longitude"]},
                    dims=("latitude", "longitude"),
                )

            def _empty_2d_class() -> xr.DataArray:
                return xr.DataArray(
                    np.full((lat.size, lon.size), -1, dtype=np.int16),
                    coords={"latitude": summary_2d_ds["latitude"], "longitude": summary_2d_ds["longitude"]},
                    dims=("latitude", "longitude"),
                )

            def _stack_prob(suffix: str) -> xr.DataArray:
                win_blocks = []
                for t in window_tags:
                    cls_blocks = []
                    for c in range(n_cls):
                        key = f"predicted_prob_class{c}_{suffix}_0_{t}"
                        cls_blocks.append(summary_2d_ds[key] if key in summary_2d_ds else _empty_2d_float())
                    cls_da = xr.concat(cls_blocks, dim="class").assign_coords({"class": np.arange(n_cls)})
                    win_blocks.append(cls_da)
                return xr.concat(win_blocks, dim="window").assign_coords({"window": window_tags})

            prob_mean = _stack_prob("mean")
            prob_p95 = _stack_prob(f"p{int(quantile*100)}")

            dominant = xr.concat(
                [summary_2d_ds[f"predicted_class_mode_mean_0_{t}"] if f"predicted_class_mode_mean_0_{t}" in summary_2d_ds else _empty_2d_class() for t in window_tags],
                dim="window",
            ).assign_coords({"window": window_tags})
            pmax = xr.concat(
                [summary_2d_ds[f"predicted_class_maxprob_mean_0_{t}"] if f"predicted_class_maxprob_mean_0_{t}" in summary_2d_ds else _empty_2d_float() for t in window_tags],
                dim="window",
            ).assign_coords({"window": window_tags})
            margin = xr.concat(
                [summary_2d_ds[f"predicted_class_margin_mean_0_{t}"] if f"predicted_class_margin_mean_0_{t}" in summary_2d_ds else _empty_2d_float() for t in window_tags],
                dim="window",
            ).assign_coords({"window": window_tags})

            conf_key_tpl = f"predicted_class_mode_mean_confmask_pmax{dominant_thr_tag}_0_{{t}}"
            dominant_conf = xr.concat(
                [summary_2d_ds[conf_key_tpl.format(t=t)] if conf_key_tpl.format(t=t) in summary_2d_ds else _empty_2d_class() for t in window_tags],
                dim="window",
            ).assign_coords({"window": window_tags})

            entropy_mean = xr.concat(
                [summary_2d_ds[f"predicted_class_entropy_mean_0_{t}"] if f"predicted_class_entropy_mean_0_{t}" in summary_2d_ds else _empty_2d_float() for t in window_tags],
                dim="window",
            ).assign_coords({"window": window_tags})
            entropy_p95 = xr.concat(
                [summary_2d_ds[f"predicted_class_entropy_p{int(quantile*100)}_0_{t}"] if f"predicted_class_entropy_p{int(quantile*100)}_0_{t}" in summary_2d_ds else _empty_2d_float() for t in window_tags],
                dim="window",
            ).assign_coords({"window": window_tags})

            in_domain_fraction = xr.concat(
                [summary_2d_ds[f"in_domain_fraction_0_{t}"] if f"in_domain_fraction_0_{t}" in summary_2d_ds else _empty_2d_float() for t in window_tags],
                dim="window",
            ).assign_coords({"window": window_tags})

            pub = xr.Dataset(
                {
                    "prob_mean": prob_mean,
                    "prob_p95": prob_p95,
                    "dominant_class": dominant,
                    "dominant_pmax": pmax,
                    "dominant_margin": margin,
                    "dominant_class_confmask": dominant_conf,
                    "entropy_mean": entropy_mean,
                    "entropy_p95": entropy_p95,
                    "in_domain_fraction": in_domain_fraction,
                },
                coords={
                    "window": window_tags,
                    "class": np.arange(n_cls),
                    "latitude": summary_2d_ds["latitude"],
                    "longitude": summary_2d_ds["longitude"],
                },
            )
            pub.coords["window_start_ka"] = ("window", np.array([float(a0) for _, a0, _ in windows], dtype=np.float32))
            pub.coords["window_end_ka"] = ("window", np.array([float(a1) for _, _, a1 in windows], dtype=np.float32))
            pub.coords["class_name"] = ("class", np.array([str(x) for x in class_names], dtype=str))

            thr_info = pred_run_summary.get("results", {}).get("multiclass_target_threshold") or {}
            target_thr = thr_info.get("multiclass_target_threshold")
            target_idx = thr_info.get("multiclass_target_class_index")
            if target_thr is not None and target_idx is not None:
                thr_tag_local = str(int(round(float(target_thr) * 1000)))
                freq_blocks = []
                any_blocks = []
                for t in window_tags:
                    kf = f"predicted_target_class{int(target_idx)}_hotspot_freq_thr{thr_tag_local}_0_{t}"
                    ka = f"predicted_target_class{int(target_idx)}_hotspot_any_thr{thr_tag_local}_0_{t}"
                    freq_blocks.append(summary_2d_ds[kf] if kf in summary_2d_ds else _empty_2d_float())
                    any_blocks.append(summary_2d_ds[ka] if ka in summary_2d_ds else _empty_2d_float())
                pub["hotspot_freq"] = xr.concat(freq_blocks, dim="window").assign_coords({"window": window_tags})
                pub["hotspot_any"] = xr.concat(any_blocks, dim="window").assign_coords({"window": window_tags})
                pub["hotspot_freq"].attrs.update(
                    {
                        "target_class_index": int(target_idx),
                        "prob_threshold": float(target_thr),
                        "threshold_source": str(thr_info.get("multiclass_target_threshold_source") or "unknown"),
                    }
                )

            for pub_var, source_tpl in [
                ("prob_class2_high_grade", "prob_class2_high_grade_mean_0_{t}"),
            ]:
                if any(source_tpl.format(t=t) in summary_2d_ds for t in window_tags):
                    pub[pub_var] = xr.concat(
                        [summary_2d_ds[source_tpl.format(t=t)] if source_tpl.format(t=t) in summary_2d_ds else _empty_2d_float() for t in window_tags],
                        dim="window",
                    ).assign_coords({"window": window_tags})

            for diag_var in [
                "env_center_valid_fraction",
                "env_center_any_missing",
                "sedimentation_rate",
                "sedimentation_rate_was_filled",
            ]:
                if diag_var in summary_2d_ds:
                    pub[diag_var] = summary_2d_ds[diag_var]

            _update_attrs_native(
                pub,
                {
                    "description": "Publication-ready 2D summary products for multiclass mechanism prediction.",
                    "target_task_type": "multiclass_classification",
                    "num_classes": int(num_classes),
                    "class_names": ",".join([str(x) for x in class_names]),
                    "dominant_confidence_threshold": float(dominant_thr),
                    "summary_quantile": float(quantile),
                    "summary_sampling_scheme": scheme,
                    "mask_out_of_domain_in_maps": bool(pred_config_obj.mask_out_of_domain_in_maps),
                }
            )

            publication_nc_path = os.path.join(output_dirs["predictions"], "predicted_mechanism_2d_publication.nc")
            pub.to_netcdf(publication_nc_path)
            pred_run_summary["artifacts"]["paths"]["predicted_publication_2d_nc"] = publication_nc_path
            pred_run_summary["results"]["predicted_publication_2d_nc_path"] = publication_nc_path
        except Exception as e_pub:
            logger.warning(f"Prediction Pipeline: Failed to build publication NetCDF: {e_pub}")
            pred_run_summary["results"]["predicted_publication_2d_nc_path"] = None

        meta = {
            "windows": [{"tag": t, "start_ka": a0, "end_ka": a1} for t, a0, a1 in windows],
            "age_samples_ka": age_levels,
            "quantile": quantile,
            "mask_out_of_domain_in_maps": bool(pred_config_obj.mask_out_of_domain_in_maps),
        }
        diag_json = save_summary_2d_json(summary_2d_ds, output_dir=output_dirs["summary_2d"], meta=meta)
        pred_run_summary["artifacts"]["paths"]["summary_2d_diagnostics_json"] = diag_json


        entropy_vmax = float(np.log(float(max(num_classes, 2))))
        summary_map_paths: list[str] = []
        for var_name, da in summary_2d_ds.data_vars.items():
            cmap = None
            vmin = None
            vmax = None
            center_zero = False
            colorbar_label = var_name
            title = var_name

            if var_name.startswith("predicted_prob_class") and ("_mean_" in var_name or "_p" in var_name):
                cmap = "viridis"
                vmin, vmax = 0.0, 1.0
                colorbar_label = "Probability"
            elif var_name.startswith("predicted_prob_class") and "_std_" in var_name:
                cmap = "magma"
                vmin = 0.0
                a = np.asarray(da.values)
                m = np.isfinite(a)
                if np.any(m):
                    vmax = float(np.percentile(a[m], 99))
                colorbar_label = "Uncertainty (std)"
            elif var_name.startswith("predicted_target_class") and ("_hotspot_freq_" in var_name or "_hotspot_any_" in var_name):
                cmap = "magma" if "_freq_" in var_name else "gray_r"
                vmin, vmax = 0.0, 1.0
                colorbar_label = "Hotspot frequency" if "_freq_" in var_name else "Any hotspot (0/1)"
            elif var_name.startswith("predicted_class_entropy"):
                cmap = "magma"
                vmin, vmax = 0.0, entropy_vmax
                colorbar_label = "Entropy (nats)"
            elif var_name.startswith("predicted_class_maxprob_mean_"):
                cmap = "viridis"
                vmin, vmax = 0.0, 1.0
                colorbar_label = "pmax (mean)"
            elif var_name.startswith("predicted_class_margin_mean_"):
                cmap = "viridis"
                vmin, vmax = 0.0, 1.0
                colorbar_label = "margin (p1-p2)"
            elif var_name.startswith("predicted_class_argmax_freq_class") or var_name.startswith("in_domain_fraction"):
                cmap = "viridis"
                vmin, vmax = 0.0, 1.0
            elif var_name.endswith("_any_missing") or var_name.endswith("_was_filled"):
                cmap = "gray_r"
                vmin, vmax = 0.0, 1.0
            elif var_name.startswith("delta_"):
                cmap = "RdBu_r"
                center_zero = True

            if var_name.startswith("predicted_class_mode_mean_"):
                p = plot_categorical_map_2d(
                    categories_2d=da,
                    categories=class_names,
                    output_dir=output_dirs["summary_2d_maps"],
                    plot_config_params=pred_config_obj.plotting.model_dump(),
                    user_lat_min=user_lat_min,
                    user_lat_max=user_lat_max,
                    user_lon_min=user_lon_min,
                    user_lon_max=user_lon_max,
                    filename_prefix=f"summary_{var_name}",
                    title=title,
                )
            else:
                p = plot_rey_map_2d(
                    rey_data_2d=da,
                    output_dir=output_dirs["summary_2d_maps"],
                    plot_config_params=pred_config_obj.plotting.model_dump(),
                    user_lat_min=user_lat_min,
                    user_lat_max=user_lat_max,
                    user_lon_min=user_lon_min,
                    user_lon_max=user_lon_max,
                    filename_prefix=f"summary_{var_name}",
                    colorbar_label=colorbar_label,
                    title=title,
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                    center_zero=center_zero,
                )
            if p:
                summary_map_paths.append(p)

        pred_run_summary["artifacts"]["paths"]["summary_2d_maps"] = summary_map_paths
        pred_run_summary["results"]["summary_2d_maps_dir_path"] = output_dirs["summary_2d_maps"]


        try:
            if generate_multiclass_paper_figures is None:
                raise RuntimeError("core.visualization.prediction_paper_figures is not available")
            paper_dir = Path(output_dirs["summary_2d"]) / "paper_figures"
            thr_info = pred_run_summary.get("results", {}).get("multiclass_target_threshold") or {}
            target_thr = thr_info.get("multiclass_target_threshold")
            target_idx = thr_info.get(
                "multiclass_target_class_index",
                int(getattr(pred_config_obj, "multiclass_target_class_index", 1) or 1),
            )
            paper_pngs = generate_multiclass_paper_figures(
                summary_2d_ds=summary_2d_ds,
                windows=windows,
                class_names=class_names,
                output_dir=paper_dir,
                plot_config=pred_config_obj.plotting.model_dump(),
                target_class_index=int(target_idx),
                target_threshold=float(target_thr) if target_thr is not None else None,
                dominant_confidence_threshold=float(dominant_thr),
            )
            pred_run_summary["artifacts"]["paths"]["paper_figures_dir"] = str(paper_dir)
            pred_run_summary["artifacts"]["paths"]["paper_figures"] = paper_pngs
        except Exception as e_paper:
            logger.warning(f"Prediction Pipeline: Failed to generate paper figures: {e_paper}")
            pred_run_summary["artifacts"]["paths"]["paper_figures"] = []

        pred_run_summary["status"] = "completed"
        pred_run_summary["results"]["predicted_grid_nc_path"] = None
        pred_run_summary["results"]["prediction_maps_dir_path"] = None
        pred_run_summary["duration_seconds"] = time.time() - start_time
        logger.info("Prediction Pipeline (multiclass): Completed successfully.")
        return True, pred_run_summary

    except PredictionPipelineError as e:
        error_message = f"Prediction pipeline failed: {e}"
        logger.error(error_message, exc_info=True)
        pred_run_summary["status"] = "error"
        pred_run_summary["error_message"] = error_message
        return False, pred_run_summary
    except Exception as e:
        error_message = f"An unexpected error occurred in the prediction pipeline: {e}"
        logger.error(error_message, exc_info=True)
        pred_run_summary["status"] = "error"
        pred_run_summary["error_message"] = error_message
        return False, pred_run_summary
    finally:
        pred_run_summary["timestamp_end"] = datetime.now().isoformat()
        if "timestamp_start" in pred_run_summary:
            try:
                pred_run_summary["duration_seconds"] = (
                    datetime.fromisoformat(pred_run_summary["timestamp_end"]) - datetime.fromisoformat(pred_run_summary["timestamp_start"])
                ).total_seconds()
            except Exception:
                pass

        try:
            run_dir_for_outputs = run_base_dir
            pipeline_config_path = run_dir_for_outputs / "pipeline_config.json"
            write_json(pipeline_config_path, pred_run_summary.get("inputs", {}))
            pred_run_summary["artifacts"]["paths"]["pipeline_config"] = str(pipeline_config_path)

            env_path = run_dir_for_outputs / "environment_snapshot.json"
            write_json(env_path, snapshot_environment(extra={"run_id": pred_run_id, "run_type": "prediction"}))
            pred_run_summary["artifacts"]["paths"]["environment_snapshot"] = str(env_path)

            report_path = run_dir_for_outputs / "RUN_REPORT.md"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(make_prediction_run_report_md(pred_run_summary))
            pred_run_summary["artifacts"]["paths"]["run_report"] = str(report_path)

            manifest_path = run_dir_for_outputs / "results_manifest.json"
            write_json(manifest_path, build_paths_manifest(run_dir_for_outputs, pred_run_summary))
            pred_run_summary["artifacts"]["paths"]["results_manifest"] = str(manifest_path)
        except Exception as e_aux:
            logger.warning(f"Prediction Pipeline: Failed to write journal-ready outputs: {e_aux}")

        try:
            save_run_summary(pred_run_id, pred_run_summary)
        except Exception as e:
            logger.critical(
                f"CRITICAL: Failed to save final prediction run summary for run {pred_run_id}: {e}",
                exc_info=True,
            )

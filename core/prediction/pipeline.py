
import os
import time
import json
import numpy as np
import torch
import xarray as xr
from core.utils.netcdf_io import open_dataset
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional

from core.utils.logger_config import logger
from core.utils.run_manager import save_run_summary
from core.utils.constants import RESULTS_GUI_DIR
from core.visualization.map_plots import plot_rey_map_by_depth, plot_rey_map_by_level, plot_rey_map_2d, plot_categorical_map_2d
from .predictor import GridPredictor
from core.utils.config_models import PredictionConfig, TrainingArtifacts
from core.utils.results_io import write_json, snapshot_environment, build_paths_manifest, RESULTS_SCHEMA_VERSION
from core.utils.run_report import make_prediction_run_report_md
from core.visualization.prediction_diagnostics import (
    save_prediction_level_diagnostics,
    plot_domain_fraction_by_level,
    plot_vertical_transform_summary,
)
from core.visualization.summary_2d_diagnostics import save_summary_2d_json

class PredictionPipelineError(Exception):
    """Base class for prediction pipeline errors"""
    pass

class ConfigurationError(PredictionPipelineError):
    """Raised when there's an error in the configuration"""
    pass

class FileNotFoundError(PredictionPipelineError):
    """Raised when a required file is not found"""
    pass

class PredictionFailedError(PredictionPipelineError):
    """Raised when the prediction process fails"""
    pass


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
    """
    Best-effort: resolve the env_gridded.nc path used during training so prediction
    can run even when the current GUI session has not re-run preprocessing.
    """
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
    """
    Robustly resolve model path in the presence of renamed run directories.

    If `model_path` is missing, try to find `final_trained_model.pth` next to the artifacts file.
    """
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

def _validate_env_grid_has_features(env_gridded_nc_path: str, feature_names: list[str]) -> None:
    """
    Fail-fast with a helpful message when the env grid does not match the model's expected channels.
    """
    try:
        with open_dataset(env_gridded_nc_path) as ds:
            present = set(ds.data_vars.keys())
            dims = set(ds.sizes.keys())
            coords = set(ds.coords.keys())
    except Exception as e:
        raise FileNotFoundError(
            f"Failed to open env grid NetCDF: {env_gridded_nc_path}. Error: {e}"
        ) from e


    if "latitude" not in dims and "latitude" not in coords:
        raise ConfigurationError(
            "Environment grid is missing required coordinate/dimension 'latitude': "
            f"{env_gridded_nc_path}"
        )
    if "longitude" not in dims and "longitude" not in coords:
        raise ConfigurationError(
            "Environment grid is missing required coordinate/dimension 'longitude': "
            f"{env_gridded_nc_path}"
        )

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

def run_prediction_pipeline(
    pred_config_obj: PredictionConfig,
    progress_callback=None
) -> Tuple[bool, dict]:
    """
    Orchestrates 3D REY prediction and map plotting using a completed training run.
    This version uses structured Pydantic models for configuration.

    Args:
        pred_config_obj (PredictionConfig): A Pydantic object containing all prediction settings.
        progress_callback (Optional[callable]): A callback function for progress updates.

    Returns:
        Tuple[bool, dict]: (success status, prediction run summary)
    """
    logger.info("Prediction Pipeline: Starting...")
    start_time = time.time()

    pred_run_id = f"prediction_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_base_dir = Path(RESULTS_GUI_DIR) / pred_run_id


    pred_config_obj = pred_config_obj.model_copy(deep=True)

    output_dirs = {
        "base": str(run_base_dir),
        "tables": str(run_base_dir / "tables"),
        "predictions": str(run_base_dir / "predictions_nc"),
        "prediction_maps_by_depth": str(run_base_dir / "prediction_maps_by_depth"),
        "prediction_maps_by_age": str(run_base_dir / "prediction_maps_by_age"),
        "plots": str(run_base_dir / "plots"),
        "diagnostics": str(run_base_dir / "plots" / "diagnostics"),
        "summary_2d": str(run_base_dir / "plots" / "summary_2d"),
        "summary_2d_maps": str(run_base_dir / "plots" / "summary_2d" / "maps"),
        "reports": str(run_base_dir / "reports"),
    }


    try:
        for dir_path in output_dirs.values():
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create output directories: {e}")
        raise RuntimeError(f"Failed to create output directories: {e}") from e


    pred_run_summary = {
        "run_id": pred_run_id,
        "run_type": "prediction",
        "timestamp_start": datetime.now().isoformat(),
        "status": "started",
        "results_schema_version": RESULTS_SCHEMA_VERSION,
        "inputs": {
            "prediction_config": pred_config_obj.model_dump()
        },
        "results": {},
        "artifacts": {"paths": output_dirs}
    }

    try:
        logger.debug("Prediction Pipeline: Loading training artifacts...")


        artifacts_path = pred_config_obj.training_artifacts_path
        if not artifacts_path or not os.path.exists(artifacts_path):
            raise FileNotFoundError(f"Training artifacts file not found at: {artifacts_path}")

        with open(artifacts_path, 'r') as f:
            training_artifacts = TrainingArtifacts.model_validate_json(f.read())

        pred_run_summary["inputs"]["source_training_artifacts_path"] = artifacts_path


        target_task_type = str(getattr(training_artifacts.norm_params.rey, "task_type", None) or "regression").lower()
        if target_task_type not in ("regression", "binary_classification", "multiclass_classification"):
            logger.warning(f"Prediction Pipeline: Unknown target task_type='{target_task_type}', falling back to 'regression'.")
            target_task_type = "regression"


        if target_task_type == "multiclass_classification":
            if pred_config_obj.output_mode != "summaries_2d":
                logger.info(
                    "Prediction Pipeline: Forcing output_mode='summaries_2d' for multiclass classification "
                    "(mechanism probability maps)."
                )
            pred_config_obj.output_mode = "summaries_2d"
            pred_config_obj.enable_2d_age_summaries = True

            if not (pred_config_obj.enable_holocene_summary or pred_config_obj.enable_lgc_summary or pred_config_obj.enable_quaternary_summary):
                pred_config_obj.enable_quaternary_summary = True

            try:
                pred_run_summary["inputs"]["prediction_config"] = pred_config_obj.model_dump()
            except Exception:
                pass
        thr_ppm_art = getattr(training_artifacts.norm_params.rey, "threshold_ppm", None)
        if target_task_type == "multiclass_classification":
            thr_ppm = None
            thr_tag = None

            pred_var_name = "predicted_prob_class0"
            pred_std_name = "predicted_prob_class0_std"
        elif target_task_type == "binary_classification":
            thr_ppm = float(thr_ppm_art or 1000.0)
            thr_tag = str(int(round(thr_ppm))) if abs(thr_ppm - round(thr_ppm)) < 1e-6 else str(thr_ppm).replace(".", "p")
            pred_var_name = f"predicted_prob_ge_{thr_tag}ppm"
            pred_std_name = f"{pred_var_name}_std"
        else:
            thr_ppm = None
            thr_tag = None
            pred_var_name = "predicted_rey"
            pred_std_name = "predicted_rey_std"
        pred_run_summary["results"]["target_task_type"] = target_task_type
        pred_run_summary["results"]["target_threshold_ppm"] = thr_ppm


        logger.debug("Prediction Pipeline: Extracting configuration from objects...")

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
            logger.warning(
                "Prediction Pipeline: training_artifacts.model_path does not exist; "
                f"using resolved path instead: {resolved_model_path}"
            )
        training_artifacts = training_artifacts.model_copy(update={"model_path": resolved_model_path})
        pred_run_summary["inputs"]["resolved_model_path"] = resolved_model_path


        env_gridded_nc_path = getattr(pred_config_obj, "env_gridded_nc_path", None)
        if not env_gridded_nc_path or not os.path.exists(env_gridded_nc_path):
            run_dir = _infer_source_run_dir(artifacts_path)
            inferred = _infer_env_grid_from_run_dir(run_dir) if run_dir else None
            if inferred:
                logger.info(
                    "Prediction Pipeline: env_gridded.nc path missing/invalid; "
                    f"using training-run env grid: {inferred}"
                )
                env_gridded_nc_path = inferred
                try:
                    pred_config_obj.env_gridded_nc_path = env_gridded_nc_path
                    pred_run_summary["inputs"]["prediction_config"] = pred_config_obj.model_dump()
                except Exception:
                    pass

        if not env_gridded_nc_path or not os.path.exists(env_gridded_nc_path):
            raise FileNotFoundError(
                "Environment grid data path is missing or file not found.\n"
                f"- provided: {getattr(pred_config_obj, 'env_gridded_nc_path', None)}\n"
                f"- inferred: {env_gridded_nc_path}\n"
                f"- artifacts: {artifacts_path}"
            )


        _validate_env_grid_has_features(env_gridded_nc_path, list(training_artifacts.model_init_config.feature_names))


        pred_maps_output_dir_depth = output_dirs["prediction_maps_by_depth"]
        pred_maps_output_dir_age = output_dirs["prediction_maps_by_age"]
        pred_nc_output_path = os.path.join(output_dirs["predictions"], "predicted_rey_3d_grid.nc")

        logger.debug("Prediction Pipeline: Configuration extracted successfully.")


        logger.debug("Prediction Pipeline: Initializing GridPredictor...")
        if progress_callback: 
            progress_callback(10, "Initializing predictor...")

        try:
            predictor = GridPredictor(
                training_artifacts=training_artifacts,
                env_gridded_nc_path=env_gridded_nc_path,
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                batch_size=pred_config_obj.batch_size,
                progress_callback=progress_callback,
                pred_config=pred_config_obj
            )
        except Exception as e:
            logger.error(f"Failed to initialize GridPredictor: {e}")
            raise PredictionFailedError(f"Failed to initialize GridPredictor: {e}") from e


        logger.debug("Prediction Pipeline: Running 3D grid prediction...")
        if progress_callback: 
            progress_callback(20, "Generating 3D prediction grid...")

        predicted_ds = None
        if pred_config_obj.output_mode in ("slices_3d", "both"):
            try:
                predicted_ds = predictor.predict_3d_grid(
                    pred_grid_settings=pred_config_obj.prediction_grid_settings
                )
            except Exception as e:
                logger.error(f"3D grid prediction failed: {e}")
                raise PredictionFailedError(f"3D grid prediction failed: {e}") from e


        summary_2d_ds = None
        pred_age_ds_for_summary = None
        if pred_config_obj.output_mode in ("summaries_2d", "both") and pred_config_obj.enable_2d_age_summaries:
            try:

                if predictor.depth_feature_type != "age_ka_proxy":
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

                windows = []
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

                age_levels = sorted([a for a in age_samples if np.isfinite(a) and a >= 0.0])

                for _, _, b in windows:
                    age_levels.append(float(b))
                age_levels = sorted(set([float(a) for a in age_levels]))


                tmp_grid = pred_config_obj.prediction_grid_settings.model_copy(deep=True)
                tmp_grid.vertical_axis_mode = "age_ka"
                tmp_grid.age_levels_ka = age_levels
                tmp_grid.depth_levels = []

                logger.info(
                    "Prediction Pipeline: Generating compact age-axis grid for 2D summaries "
                    f"(n_age_levels={len(age_levels)})..."
                )
                pred_age_ds_for_summary = predictor.predict_3d_grid(pred_grid_settings=tmp_grid)
                if pred_age_ds_for_summary is None:
                    raise ValueError("Age-axis prediction returned None; cannot build 2D summaries.")

                if "age_ka" not in pred_age_ds_for_summary.coords:
                    raise ValueError("Expected 'age_ka' coordinate for summary aggregation.")

                quantile = float(pred_config_obj.summary_quantile or 0.95)
                quantile = min(0.999, max(0.5, quantile))
                low_quantile = 1.0 - quantile


                summary_vars = {}
                is_cls = target_task_type == "binary_classification"
                is_multiclass = target_task_type == "multiclass_classification"
                enable_thr = bool(getattr(pred_config_obj, "enable_threshold_exceedance_maps", False)) and (target_task_type == "regression")

                thr_ppm_local = float((thr_ppm if is_cls else getattr(pred_config_obj, "threshold_exceedance_ppm", 1000.0)) or 1000.0)
                thr_ppm_local = max(0.0, thr_ppm_local)
                thr_tag = str(int(round(thr_ppm_local))) if abs(thr_ppm_local - round(thr_ppm_local)) < 1e-6 else str(thr_ppm_local).replace(".", "p")
                for tag, a0, a1 in windows:
                    sel = pred_age_ds_for_summary.sel(age_ka=slice(a0, a1))
                    if sel.sizes.get("age_ka", 0) <= 0:
                        continue
                    if is_multiclass:
                        num_classes = int(getattr(training_artifacts.norm_params.rey, "num_classes", 0) or 0)
                        class_names = list(getattr(training_artifacts.norm_params.rey, "class_names", None) or [])
                        num_classes = max(num_classes, len(class_names))

                        def _label(c: int) -> str:
                            return class_names[c] if c < len(class_names) else f"class{c}"


                        enable_target_maps = bool(getattr(pred_config_obj, "enable_multiclass_target_threshold_maps", True))
                        target_idx = int(getattr(pred_config_obj, "multiclass_target_class_index", 1) or 0)
                        target_idx_valid = 0 <= target_idx < num_classes

                        def _load_target_threshold_from_run_dir(metric_key: str) -> Optional[float]:
                            try:
                                model_path = getattr(training_artifacts, "model_path", None)
                                if not model_path:
                                    return None
                                run_dir = Path(str(model_path)).resolve().parent.parent
                                for split_name in ["val", "test", "train"]:
                                    p = run_dir / "tables" / split_name / f"target_class{target_idx}_threshold_recommendations.json"
                                    if not p.exists():
                                        continue
                                    data = json.loads(p.read_text(encoding="utf-8"))
                                    best = data.get("best", {})
                                    entry = best.get(metric_key)
                                    if isinstance(entry, dict) and "threshold" in entry:
                                        return float(entry["threshold"])
                            except Exception as e:
                                logger.debug(f"Prediction Pipeline: Failed to load target threshold from run dir: {e}")
                            return None

                        thr_source = str(getattr(pred_config_obj, "multiclass_target_threshold_source", "manual") or "manual")
                        target_thr = float(getattr(pred_config_obj, "multiclass_target_threshold", 0.5) or 0.5)
                        target_thr = float(np.clip(target_thr, 0.0, 1.0))
                        target_thr_note = "manual"
                        if enable_target_maps and target_idx_valid and thr_source != "manual":
                            metric_map = {
                                "from_val_recommendation_f0_5": "F0.5",
                                "from_val_recommendation_f1": "F1",
                                "from_val_recommendation_f2": "F2",
                            }
                            metric_key = metric_map.get(thr_source)
                            if metric_key:
                                loaded = _load_target_threshold_from_run_dir(metric_key)
                                if loaded is not None:
                                    target_thr = float(np.clip(loaded, 0.0, 1.0))
                                    target_thr_note = f"{thr_source}:{metric_key}"


                        try:
                            pred_run_summary.setdefault("results", {})
                            if "multiclass_target_threshold" not in pred_run_summary["results"]:
                                pred_run_summary["results"].update(
                                    {
                                        "multiclass_target_class_index": int(target_idx),
                                        "multiclass_target_threshold": float(target_thr),
                                        "multiclass_target_threshold_source": str(target_thr_note),
                                    }
                                )
                        except Exception:
                            pass

                        mean_maps = [None] * num_classes

                        for c in range(num_classes):
                            var = f"predicted_prob_class{c}"
                            if var not in sel:
                                continue
                            if pred_config_obj.mask_out_of_domain_in_maps and "is_in_training_age_domain" in sel:
                                sel_pred = sel[var].where(sel["is_in_training_age_domain"] > 0)
                            else:
                                sel_pred = sel[var]

                            mean2d = sel_pred.mean(dim="age_ka", skipna=True)
                            q2d = sel_pred.quantile(quantile, dim="age_ka", skipna=True)
                            mean2d.attrs.update(
                                {
                                    "units": "1",
                                    "long_name": f"Mean predicted probability of {_label(c)} over {a0:g}-{a1:g} ka ({tag})",
                                    "window_start_ka": float(a0),
                                    "window_end_ka": float(a1),
                                    "class_index": int(c),
                                    "class_label": _label(c),
                                    "definition": "Mean predicted class probability over sampled ages within window (age proxy).",
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
                                    "definition": "Quantile of predicted class probability over sampled ages within window (age proxy).",
                                }
                            )
                            summary_vars[f"predicted_prob_class{c}_mean_0_{tag}"] = mean2d
                            summary_vars[f"predicted_prob_class{c}_p{int(quantile*100)}_0_{tag}"] = q2d
                            mean_maps[c] = mean2d

                            std_var = f"{var}_std"
                            if std_var in sel:
                                unc = sel[std_var].mean(dim="age_ka", skipna=True)
                                unc.attrs.update(
                                    {
                                        "units": "1",
                                        "long_name": f"Mean MC-dropout std of P({_label(c)}) over {a0:g}-{a1:g} ka ({tag})",
                                        "window_start_ka": float(a0),
                                        "window_end_ka": float(a1),
                                        "class_index": int(c),
                                        "class_label": _label(c),
                                        "definition": "Mean predicted-probability uncertainty (MC Dropout) over sampled ages within window.",
                                    }
                                )
                                summary_vars[f"predicted_prob_class{c}_std_0_{tag}"] = unc


                        if enable_target_maps and target_idx_valid and f"predicted_prob_class{target_idx}" in sel:
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
                                        "definition": "Mean of I(P(target_class) >= threshold) over sampled ages within window.",
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
                                        "definition": "Max of I(P(target_class) >= threshold) over sampled ages within window (0/1).",
                                    }
                                )
                                summary_vars[f"predicted_target_class{target_idx}_hotspot_any_thr{thr_tag_local}_0_{tag}"] = any_ex


                        try:
                            if all(m is not None for m in mean_maps):
                                prob_stack = np.stack(
                                    [np.asarray(m.values, dtype=np.float64) for m in mean_maps], axis=0
                                )
                                valid_any = np.any(np.isfinite(prob_stack), axis=0)


                                safe = np.where(np.isfinite(prob_stack), prob_stack, -np.inf)
                                mode = np.argmax(safe, axis=0).astype(np.int16)
                                maxprob = np.max(safe, axis=0).astype(np.float32)


                                sorted_probs = np.sort(safe, axis=0)
                                margin = (sorted_probs[-1] - sorted_probs[-2]).astype(np.float32)

                                margin = np.where(np.isfinite(margin), margin, np.nan).astype(np.float32)


                                mode = np.where(valid_any, mode, -1).astype(np.int16)
                                maxprob = np.where(valid_any, maxprob, np.nan).astype(np.float32)
                                margin = np.where(valid_any, margin, np.nan).astype(np.float32)

                                mode_da = xr.DataArray(mode.astype(np.float32), coords=mean_maps[0].coords, dims=mean_maps[0].dims)
                                mode_da.attrs.update(
                                    {
                                        "units": "class_id",
                                        "long_name": f"Most likely class by mean probability over {a0:g}-{a1:g} ka ({tag})",
                                        "window_start_ka": float(a0),
                                        "window_end_ka": float(a1),
                                        "definition": "argmax over window-mean per-class probabilities (age proxy). Value -1 indicates all-NaN (masked) pixel.",
                                    }
                                )
                                summary_vars[f"predicted_class_mode_mean_0_{tag}"] = mode_da

                                mp_da = xr.DataArray(maxprob, coords=mean_maps[0].coords, dims=mean_maps[0].dims)
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
                        except Exception:
                            pass

                        if "predicted_class_entropy" in sel:
                            ent_sel = (
                                sel["predicted_class_entropy"].where(sel["is_in_training_age_domain"] > 0)
                                if pred_config_obj.mask_out_of_domain_in_maps and "is_in_training_age_domain" in sel
                                else sel["predicted_class_entropy"]
                            )
                            ent_mean = ent_sel.mean(dim="age_ka", skipna=True)
                            ent_q = ent_sel.quantile(quantile, dim="age_ka", skipna=True)
                            ent_mean.attrs.update(
                                {
                                    "units": "nats",
                                    "long_name": f"Mean predictive entropy over {a0:g}-{a1:g} ka ({tag})",
                                    "window_start_ka": float(a0),
                                    "window_end_ka": float(a1),
                                    "definition": "Mean entropy over sampled ages within window (age proxy).",
                                }
                            )
                            ent_q.attrs.update(
                                {
                                    "units": "nats",
                                    "long_name": f"High-quantile entropy (p{int(quantile*100)}) over {a0:g}-{a1:g} ka ({tag})",
                                    "window_start_ka": float(a0),
                                    "window_end_ka": float(a1),
                                    "definition": "Quantile of entropy over sampled ages within window (age proxy).",
                                }
                            )
                            summary_vars[f"predicted_class_entropy_mean_0_{tag}"] = ent_mean
                            summary_vars[f"predicted_class_entropy_p{int(quantile*100)}_0_{tag}"] = ent_q

                        if "predicted_class_argmax" in sel and num_classes > 0:
                            arg = (
                                sel["predicted_class_argmax"].where(sel["is_in_training_age_domain"] > 0)
                                if pred_config_obj.mask_out_of_domain_in_maps and "is_in_training_age_domain" in sel
                                else sel["predicted_class_argmax"]
                            )
                            for c in range(num_classes):
                                freq = (arg == c).mean(dim="age_ka", skipna=True)
                                freq.attrs.update(
                                    {
                                        "units": "1",
                                        "long_name": f"Fraction of ages where argmax is {_label(c)} over {a0:g}-{a1:g} ka ({tag})",
                                        "window_start_ka": float(a0),
                                        "window_end_ka": float(a1),
                                        "class_index": int(c),
                                        "class_label": _label(c),
                                        "definition": "Mean of I(argmax==class) over sampled ages within window.",
                                    }
                                )
                                summary_vars[f"predicted_class_argmax_freq_class{c}_0_{tag}"] = freq

                        continue

                    if pred_config_obj.mask_out_of_domain_in_maps and "is_in_training_age_domain" in sel:
                        sel_pred = sel[pred_var_name].where(sel["is_in_training_age_domain"] > 0)
                    else:
                        sel_pred = sel[pred_var_name]

                    mean2d = sel_pred.mean(dim="age_ka", skipna=True)
                    q2d = sel_pred.quantile(quantile, dim="age_ka", skipna=True)

                    if is_cls:
                        mean2d.attrs.update(
                            {
                                "units": "1",
                                "long_name": f"Mean probability of ΣREY ≥ {thr_ppm_local:g} ppm over {a0:g}-{a1:g} ka ({tag})",
                                "window_start_ka": float(a0),
                                "window_end_ka": float(a1),
                                "threshold_ppm": float(thr_ppm_local),
                                "definition": "Mean predicted probability over sampled ages within window (age proxy).",
                            }
                        )
                        q2d.attrs.update(
                            {
                                "units": "1",
                                "long_name": f"High-quantile probability (p{int(quantile*100)}) of ΣREY ≥ {thr_ppm_local:g} ppm over {a0:g}-{a1:g} ka ({tag})",
                                "window_start_ka": float(a0),
                                "window_end_ka": float(a1),
                                "threshold_ppm": float(thr_ppm_local),
                                "definition": "Quantile of predicted probability over sampled ages within window (age proxy).",
                            }
                        )

                        summary_vars[f"predicted_rey_prob_ge_{thr_tag}ppm_0_{tag}"] = mean2d
                        summary_vars[f"predicted_rey_prob_ge_{thr_tag}ppm_p{int(quantile*100)}_0_{tag}"] = q2d
                    else:
                        mean2d.attrs.update(
                            {
                                "units": "ppm",
                                "long_name": f"Time-averaged REY over {a0:g}-{a1:g} ka ({tag})",
                                "window_start_ka": float(a0),
                                "window_end_ka": float(a1),
                                "definition": "Mean over sampled ages within window (age proxy).",
                            }
                        )
                        q2d.attrs.update(
                            {
                                "units": "ppm",
                                "long_name": f"High-quantile REY (p{int(quantile*100)}) over {a0:g}-{a1:g} ka ({tag})",
                                "window_start_ka": float(a0),
                                "window_end_ka": float(a1),
                                "definition": "Quantile over sampled ages within window (age proxy).",
                            }
                        )
                        summary_vars[f"predicted_rey_age_mean_0_{tag}"] = mean2d
                        summary_vars[f"predicted_rey_age_p{int(quantile*100)}_0_{tag}"] = q2d


                    if enable_thr:
                        try:
                            exceed = (sel_pred >= thr_ppm_local).astype("float32")
                            if bool(getattr(pred_config_obj, "threshold_exceedance_include_freq", True)):
                                freq2d = exceed.mean(dim="age_ka", skipna=True)
                                freq2d.attrs.update(
                                    {
                                        "units": "1",
                                        "long_name": f"Exceedance frequency of predicted REY >= {thr_ppm_local:g} ppm over {a0:g}-{a1:g} ka ({tag})",
                                        "window_start_ka": float(a0),
                                        "window_end_ka": float(a1),
                                        "threshold_ppm": float(thr_ppm_local),
                                        "definition": "Fraction of sampled ages within window with pred>=threshold (proxy probability).",
                                    }
                                )
                                summary_vars[f"predicted_rey_prob_ge_{thr_tag}ppm_0_{tag}"] = freq2d

                            if bool(getattr(pred_config_obj, "threshold_exceedance_include_any", True)):
                                any2d = (exceed.max(dim="age_ka", skipna=True) > 0).astype("int8")
                                any2d.attrs.update(
                                    {
                                        "units": "1",
                                        "long_name": f"Any exceedance of predicted REY >= {thr_ppm_local:g} ppm within {a0:g}-{a1:g} ka ({tag})",
                                        "window_start_ka": float(a0),
                                        "window_end_ka": float(a1),
                                        "threshold_ppm": float(thr_ppm_local),
                                        "definition": "1 if any sampled age within window exceeds threshold; else 0.",
                                    }
                                )
                                summary_vars[f"predicted_rey_any_ge_{thr_tag}ppm_0_{tag}"] = any2d

                            if bool(getattr(pred_config_obj, "threshold_exceedance_include_p95", False)):
                                p2d = exceed.quantile(quantile, dim="age_ka", skipna=True)
                                p2d.attrs.update(
                                    {
                                        "units": "1",
                                        "long_name": f"High-quantile exceedance indicator (p{int(quantile*100)}) for pred REY >= {thr_ppm_local:g} ppm over {a0:g}-{a1:g} ka ({tag})",
                                        "window_start_ka": float(a0),
                                        "window_end_ka": float(a1),
                                        "threshold_ppm": float(thr_ppm_local),
                                        "definition": "Quantile of the exceedance indicator across sampled ages within window.",
                                    }
                                )
                                summary_vars[f"predicted_rey_exceed_p{int(quantile*100)}_ge_{thr_tag}ppm_0_{tag}"] = p2d
                        except Exception:
                            pass

                    if bool(getattr(pred_config_obj, "summary_generate_age_sensitivity_maps", True)):
                        low2d = sel_pred.quantile(low_quantile, dim="age_ka", skipna=True)
                        std2d = sel_pred.std(dim="age_ka", skipna=True)
                        spread2d = q2d - low2d

                        if is_cls:
                            std2d.attrs.update(
                                {
                                    "units": "1",
                                    "long_name": f"Std of predicted probability (ΣREY ≥ {thr_ppm_local:g} ppm) over {a0:g}-{a1:g} ka ({tag})",
                                    "window_start_ka": float(a0),
                                    "window_end_ka": float(a1),
                                    "threshold_ppm": float(thr_ppm_local),
                                    "definition": "Std over sampled ages within window (age sensitivity proxy).",
                                }
                            )
                            spread2d.attrs.update(
                                {
                                    "units": "1",
                                    "long_name": f"Spread of predicted probability (p{int(quantile*100)}-p{int(low_quantile*100)}) over {a0:g}-{a1:g} ka ({tag})",
                                    "window_start_ka": float(a0),
                                    "window_end_ka": float(a1),
                                    "threshold_ppm": float(thr_ppm_local),
                                    "definition": "Quantile spread over sampled ages within window (age sensitivity proxy).",
                                }
                            )
                            summary_vars[f"predicted_rey_prob_ge_{thr_tag}ppm_std_0_{tag}"] = std2d
                            summary_vars[f"predicted_rey_prob_ge_{thr_tag}ppm_spread_p{int(quantile*100)}_p{int(low_quantile*100)}_0_{tag}"] = spread2d
                        else:
                            std2d.attrs.update(
                                {
                                    "units": "ppm",
                                    "long_name": f"Std of predicted REY over {a0:g}-{a1:g} ka ({tag})",
                                    "window_start_ka": float(a0),
                                    "window_end_ka": float(a1),
                                    "definition": "Standard deviation over sampled ages within window (age sensitivity proxy).",
                                }
                            )
                            spread2d.attrs.update(
                                {
                                    "units": "ppm",
                                    "long_name": f"Spread of predicted REY (p{int(quantile*100)}-p{int(low_quantile*100)}) over {a0:g}-{a1:g} ka ({tag})",
                                    "window_start_ka": float(a0),
                                    "window_end_ka": float(a1),
                                    "definition": "Quantile spread over sampled ages within window (age sensitivity proxy).",
                                }
                            )
                            summary_vars[f"predicted_rey_age_std_0_{tag}"] = std2d
                            summary_vars[f"predicted_rey_age_spread_p{int(quantile*100)}_p{int(low_quantile*100)}_0_{tag}"] = spread2d

                    if "is_in_training_age_domain" in sel:
                        frac = (sel["is_in_training_age_domain"] > 0).mean(dim="age_ka")
                        frac.attrs.update(
                            {
                                "units": "1",
                                "long_name": f"Fraction of sampled ages within training age domain over {a0:g}-{a1:g} ka ({tag})",
                                "window_start_ka": float(a0),
                                "window_end_ka": float(a1),
                            }
                        )
                        summary_vars[f"in_domain_fraction_0_{tag}"] = frac

                    if pred_std_name in sel:
                        unc = sel[pred_std_name].mean(dim="age_ka", skipna=True)
                        unc.attrs.update(
                            {
                                "units": "1" if is_cls else "ppm",
                                "long_name": f"Mean MC Dropout uncertainty over {a0:g}-{a1:g} ka ({tag})",
                                "window_start_ka": float(a0),
                                "window_end_ka": float(a1),
                            }
                        )
                        summary_vars[f"uncertainty_mean_0_{tag}"] = unc


                if bool(getattr(pred_config_obj, "summary_generate_delta_maps", True)):
                    try:
                        if is_cls:

                            key_h = f"predicted_rey_prob_ge_{thr_tag}ppm_0_holocene"
                            key_q = f"predicted_rey_prob_ge_{thr_tag}ppm_0_quaternary"
                            if key_h in summary_vars and key_q in summary_vars:
                                delta = summary_vars[key_h] - summary_vars[key_q]
                                delta.attrs.update(
                                    {
                                        "units": "1",
                                        "long_name": f"Holocene mean probability minus Quaternary mean probability (≥ {thr_ppm_local:g} ppm)",
                                        "threshold_ppm": float(thr_ppm_local),
                                        "definition": "Difference between two probability summary products (same model, different age windows).",
                                    }
                                )
                                summary_vars[f"delta_predicted_rey_prob_ge_{thr_tag}ppm_holocene_minus_quaternary"] = delta
                        else:
                            if (
                                "predicted_rey_age_mean_0_holocene" in summary_vars
                                and "predicted_rey_age_mean_0_quaternary" in summary_vars
                            ):
                                delta = summary_vars["predicted_rey_age_mean_0_holocene"] - summary_vars["predicted_rey_age_mean_0_quaternary"]
                                delta.attrs.update(
                                    {
                                        "units": "ppm",
                                        "long_name": "Holocene mean minus Quaternary mean (age-window summary)",
                                        "definition": "Difference between two summary products (same model, different age windows).",
                                    }
                                )
                                summary_vars["delta_predicted_rey_age_mean_holocene_minus_quaternary"] = delta

                            if (
                                f"predicted_rey_age_p{int(quantile*100)}_0_holocene" in summary_vars
                                and f"predicted_rey_age_p{int(quantile*100)}_0_quaternary" in summary_vars
                            ):
                                delta = (
                                    summary_vars[f"predicted_rey_age_p{int(quantile*100)}_0_holocene"]
                                    - summary_vars[f"predicted_rey_age_p{int(quantile*100)}_0_quaternary"]
                                )
                                delta.attrs.update(
                                    {
                                        "units": "ppm",
                                        "long_name": f"Holocene p{int(quantile*100)} minus Quaternary p{int(quantile*100)} (age-window summary)",
                                        "definition": "Difference between two summary products (same model, different age windows).",
                                    }
                                )
                                summary_vars[f"delta_predicted_rey_age_p{int(quantile*100)}_holocene_minus_quaternary"] = delta


                            if enable_thr and bool(getattr(pred_config_obj, "enable_holocene_summary", True)) and bool(getattr(pred_config_obj, "enable_quaternary_summary", True)):
                                key_h = f"predicted_rey_prob_ge_{thr_tag}ppm_0_holocene"
                                key_q = f"predicted_rey_prob_ge_{thr_tag}ppm_0_quaternary"
                                if key_h in summary_vars and key_q in summary_vars:
                                    delta = summary_vars[key_h] - summary_vars[key_q]
                                    delta.attrs.update(
                                        {
                                            "units": "1",
                                            "long_name": f"Holocene exceedance frequency minus Quaternary exceedance frequency (>= {thr_ppm_local:g} ppm)",
                                            "threshold_ppm": float(thr_ppm_local),
                                            "definition": "Difference between two exceedance-frequency summary products (same model, different age windows).",
                                        }
                                    )
                                    summary_vars[f"delta_predicted_rey_prob_ge_{thr_tag}ppm_holocene_minus_quaternary"] = delta
                    except Exception:
                        pass


                if bool(getattr(pred_config_obj, "summary_include_env_diagnostics_maps", True)):
                    for diag_var in [
                        "env_center_valid_fraction",
                        "env_center_any_missing",
                        "sedimentation_rate",
                        "sedimentation_rate_was_filled",
                    ]:
                        try:
                            if diag_var in pred_age_ds_for_summary and diag_var not in summary_vars:
                                summary_vars[diag_var] = pred_age_ds_for_summary[diag_var]
                        except Exception:
                            pass


                summary_2d_ds = xr.Dataset(
                    summary_vars,
                    coords={"latitude": pred_age_ds_for_summary["latitude"], "longitude": pred_age_ds_for_summary["longitude"]},
                )
                summary_attrs = {
                    "description": "2D summary products aggregated over geologically defined age windows (age proxy).",
                    "target_task_type": str(target_task_type),
                    "target_threshold_ppm": float(thr_ppm_local) if is_cls else None,
                    "num_classes": int(getattr(training_artifacts.norm_params.rey, "num_classes", 0) or 0)
                    if target_task_type == "multiclass_classification"
                    else None,
                    "class_names": ",".join(list(getattr(training_artifacts.norm_params.rey, "class_names", None) or []))
                    if target_task_type == "multiclass_classification"
                    else None,
                    "training_age_domain_ka": float(pred_config_obj.training_age_domain_ka or 2580.0),
                    "holocene_end_ka": float(pred_config_obj.holocene_end_ka),
                    "lgc_end_ka": float(pred_config_obj.lgc_end_ka),
                    "summary_quantile": quantile,
                    "summary_sampling_scheme": scheme,
                    "age_samples_ka_used": ",".join([f"{a:g}" for a in age_levels]),
                    "depth_definition": "sub-bottom depth (mbsf), not water depth",
                }

                summary_2d_ds.attrs.update({k: v for k, v in summary_attrs.items() if v is not None})


                summary_nc_filename = (
                    "predicted_mechanism_2d_summaries.nc"
                    if target_task_type == "multiclass_classification"
                    else "predicted_rey_2d_summaries.nc"
                )
                summary_nc_path = os.path.join(output_dirs["predictions"], summary_nc_filename)
                Path(summary_nc_path).parent.mkdir(parents=True, exist_ok=True)
                summary_2d_ds.to_netcdf(summary_nc_path)
                pred_run_summary["artifacts"]["paths"]["predicted_summary_2d_nc"] = summary_nc_path
                pred_run_summary["results"]["predicted_summary_2d_nc_path"] = summary_nc_path


                summary_diag_dir = str(Path(output_dirs["summary_2d"]))
                meta = {
                    "windows": [{"tag": t, "start_ka": a0, "end_ka": a1} for t, a0, a1 in windows],
                    "age_samples_ka": age_levels,
                    "quantile": quantile,
                    "mask_out_of_domain_in_maps": bool(pred_config_obj.mask_out_of_domain_in_maps),
                }
                diag_json = save_summary_2d_json(summary_2d_ds, output_dir=summary_diag_dir, meta=meta)
                pred_run_summary["artifacts"]["paths"]["summary_2d_diagnostics_json"] = diag_json


                summary_maps_dir = output_dirs["summary_2d_maps"]
                Path(summary_maps_dir).mkdir(parents=True, exist_ok=True)
                summary_map_paths = []


                def _robust_scale(var_names: list[str]):
                    vals = []
                    for vn in var_names:
                        if vn not in summary_2d_ds:
                            continue
                        a = np.asarray(summary_2d_ds[vn].values)
                        m = np.isfinite(a)
                        if np.any(m):
                            vals.append(a[m].ravel())
                    if not vals:
                        return None, None
                    x = np.concatenate(vals, axis=0)
                    qmin = float(pred_config_obj.plotting.pred_map_qmin)
                    qmax = float(pred_config_obj.plotting.pred_map_qmax)
                    return float(np.percentile(x, qmin)), float(np.percentile(x, qmax))

                mean_vars = [n for n in summary_2d_ds.data_vars if n.startswith("predicted_rey_age_mean_0_")]
                p_vars = [n for n in summary_2d_ds.data_vars if n.startswith("predicted_rey_age_p")]
                if bool(getattr(pred_config_obj, "summary_plot_shared_scale", True)):
                    vmin_mean, vmax_mean = _robust_scale(mean_vars)
                    vmin_p, vmax_p = _robust_scale(p_vars)
                else:
                    vmin_mean = vmax_mean = vmin_p = vmax_p = None
                entropy_vmax = None
                if target_task_type == "multiclass_classification":
                    n_cls = int(getattr(training_artifacts.norm_params.rey, "num_classes", 0) or 0)
                    n_cls = max(n_cls, 2)
                    entropy_vmax = float(np.log(float(n_cls)))

                for var_name, da in summary_2d_ds.data_vars.items():
                    cmap = None
                    vmin = None
                    vmax = None
                    center_zero = False
                    colorbar_label = var_name
                    title = var_name

                    if var_name.startswith("in_domain_fraction") or var_name.endswith("_fraction"):
                        cmap = "viridis"
                        vmin, vmax = 0.0, 1.0
                    elif var_name.startswith("predicted_rey_prob_ge_") or var_name.startswith("predicted_rey_exceed_"):
                        cmap = "magma"
                        vmin, vmax = 0.0, 1.0
                        colorbar_label = "Exceedance frequency (proxy probability)"
                        title = f"P(pred ΣREY ≥ threshold) ({var_name.split('_0_')[-1]})"
                    elif var_name.startswith("predicted_rey_any_ge_"):
                        cmap = "gray_r"
                        vmin, vmax = 0.0, 1.0
                        colorbar_label = "Any exceedance (0/1)"
                        title = f"Any pred ΣREY ≥ threshold ({var_name.split('_0_')[-1]})"
                    elif var_name.startswith("predicted_prob_class") and "_std_" in var_name:
                        cmap = "magma"
                        colorbar_label = "Uncertainty (std)"
                        a = np.asarray(da.values)
                        m = np.isfinite(a)
                        if np.any(m):
                            vmax = float(np.percentile(a[m], 99))
                            vmin = 0.0
                    elif var_name.startswith("predicted_prob_class") and ("_mean_" in var_name or "_p" in var_name):
                        cmap = "viridis"
                        vmin = 0.0
                        a = np.asarray(da.values, dtype=np.float64)
                        m = np.isfinite(a)
                        if np.any(m):
                            vmax = float(np.percentile(a[m], float(pred_config_obj.plotting.pred_map_qmax)))
                            vmax = min(1.0, max(0.05, vmax))
                        else:
                            vmax = 1.0
                        colorbar_label = "Probability"
                    elif var_name.startswith("predicted_target_class") and "_hotspot_freq_thr" in var_name:
                        cmap = "magma"
                        vmin, vmax = 0.0, 1.0
                        colorbar_label = "Hotspot frequency"
                        cls_label = da.attrs.get("class_label", "target")
                        thr = da.attrs.get("prob_threshold", None)
                        title = f"{cls_label} hotspot frequency (P≥{thr:.3f})" if isinstance(thr, (int, float)) else f"{cls_label} hotspot frequency"
                    elif var_name.startswith("predicted_target_class") and "_hotspot_any_thr" in var_name:
                        cmap = "gray_r"
                        vmin, vmax = 0.0, 1.0
                        colorbar_label = "Any hotspot (0/1)"
                        cls_label = da.attrs.get("class_label", "target")
                        thr = da.attrs.get("prob_threshold", None)
                        title = f"{cls_label} any-hotspot (P≥{thr:.3f})" if isinstance(thr, (int, float)) else f"{cls_label} any-hotspot"
                    elif var_name.startswith("predicted_class_argmax_freq_class"):
                        cmap = "viridis"
                        vmin = 0.0
                        a = np.asarray(da.values, dtype=np.float64)
                        m = np.isfinite(a)
                        if np.any(m):
                            vmax = float(np.percentile(a[m], float(pred_config_obj.plotting.pred_map_qmax)))
                            vmax = min(1.0, max(0.1, vmax))
                        else:
                            vmax = 1.0
                        colorbar_label = "Argmax frequency"
                    elif var_name.startswith("predicted_class_maxprob_mean_"):
                        cmap = "viridis"
                        vmin, vmax = 0.0, 1.0
                        colorbar_label = "Max mean probability"
                    elif var_name.startswith("predicted_class_margin_mean_"):
                        cmap = "magma"
                        vmin = 0.0
                        a = np.asarray(da.values, dtype=np.float64)
                        m = np.isfinite(a)
                        if np.any(m):
                            vmax = float(np.percentile(a[m], 98))
                        colorbar_label = "Top1-Top2 margin"
                    elif var_name.startswith("predicted_class_entropy"):
                        cmap = "magma"
                        vmin = 0.0
                        vmax = entropy_vmax
                        colorbar_label = "Entropy (nats)"
                    elif var_name.endswith("_any_missing") or var_name.endswith("_was_filled"):
                        cmap = "gray_r"
                        vmin, vmax = 0.0, 1.0
                    elif var_name.startswith("delta_"):
                        cmap = "RdBu_r"
                        center_zero = True
                        a = np.asarray(da.values)
                        m = np.isfinite(a)
                        if np.any(m):
                            lim = float(np.percentile(np.abs(a[m]), 98))
                            if lim <= 0:
                                lim = float(np.nanmax(np.abs(a[m]))) if np.any(m) else 1.0
                            vmin, vmax = -lim, lim
                    elif var_name.startswith("predicted_rey_age_mean_0_") and vmin_mean is not None and vmax_mean is not None:
                        vmin, vmax = vmin_mean, vmax_mean
                        colorbar_label = "Predicted ΣREY (ppm)"
                        title = f"Predicted ΣREY mean ({var_name.split('_0_')[-1]})"
                    elif var_name.startswith("predicted_rey_age_p") and vmin_p is not None and vmax_p is not None:
                        vmin, vmax = vmin_p, vmax_p
                        colorbar_label = "Predicted ΣREY (ppm)"
                        title = f"Predicted ΣREY p{int(quantile*100)} ({var_name.split('_0_')[-1]})"
                    elif var_name.startswith("predicted_rey_age_std_0_") or var_name.startswith("predicted_rey_age_spread_"):
                        cmap = "magma"
                        colorbar_label = "Age sensitivity (ppm)"

                    if var_name.startswith("predicted_class_mode_mean_"):
                        class_names = list(getattr(training_artifacts.norm_params.rey, "class_names", None) or [])
                        if not class_names:
                            ncls = int(getattr(training_artifacts.norm_params.rey, "num_classes", 0) or 0)
                            class_names = [f"class{i}" for i in range(ncls)]
                        p = plot_categorical_map_2d(
                            categories_2d=da,
                            categories=class_names,
                            output_dir=summary_maps_dir,
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
                            output_dir=summary_maps_dir,
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
                pred_run_summary["results"]["summary_2d_maps_dir_path"] = summary_maps_dir
            except Exception as e_sum:
                logger.warning(f"Prediction Pipeline: Failed to generate 2D summaries: {e_sum}")


        if predicted_ds is not None:
            logger.debug(f"Prediction Pipeline: Saving prediction to NetCDF: {pred_nc_output_path}")
            if progress_callback:
                progress_callback(80, "Saving prediction to NetCDF...")

            try:
                Path(pred_nc_output_path).parent.mkdir(parents=True, exist_ok=True)
                predicted_ds.to_netcdf(pred_nc_output_path)
                logger.info(f"Prediction Pipeline: 3D grid saved to {pred_nc_output_path}")


                pred_run_summary["artifacts"]["paths"]["predicted_grid_nc"] = pred_nc_output_path
            except Exception as e:
                logger.error(f"Failed to save prediction to NetCDF: {e}")
                raise PredictionFailedError(f"Failed to save prediction to NetCDF: {e}") from e


        diag_source_ds = pred_age_ds_for_summary if pred_config_obj.output_mode == "summaries_2d" else predicted_ds
        try:
            diag_dir = output_dirs.get("diagnostics")
            if target_task_type != "multiclass_classification" and diag_dir and diag_source_ds is not None:
                Path(diag_dir).mkdir(parents=True, exist_ok=True)

                rey_var_name = pred_var_name
                vertical_dim = None
                for cand in ["depth_mbsf", "age_ka", "depth"]:
                    if cand in diag_source_ds[rey_var_name].coords:
                        vertical_dim = cand
                        break

                if vertical_dim:
                    diag_json = save_prediction_level_diagnostics(
                        diag_source_ds,
                        vertical_dim=vertical_dim,
                        output_dir=diag_dir,
                        pred_var_name=pred_var_name,
                        pred_std_var_name=pred_std_name,
                    )
                    pred_run_summary["artifacts"]["paths"]["prediction_level_diagnostics_json"] = diag_json

                    dom_plot = plot_domain_fraction_by_level(
                        diag_source_ds,
                        vertical_dim=vertical_dim,
                        output_dir=diag_dir,
                        pred_var_name=pred_var_name,
                    )
                    if dom_plot:
                        pred_run_summary["artifacts"]["paths"]["prediction_domain_fraction_plot"] = dom_plot

                    mapping_plot = plot_vertical_transform_summary(diag_source_ds, vertical_dim=vertical_dim, output_dir=diag_dir)
                    if mapping_plot:
                        pred_run_summary["artifacts"]["paths"]["prediction_vertical_mapping_plot"] = mapping_plot
        except Exception as e_diag:
            logger.warning(f"Prediction Pipeline: Diagnostics generation failed: {e_diag}")


        if pred_config_obj.output_mode == "summaries_2d":
            pred_run_summary["status"] = "completed"
            pred_run_summary["results"]["predicted_grid_nc_path"] = None
            pred_run_summary["results"]["prediction_maps_dir_path"] = None
            logger.info("Prediction Pipeline: Completed successfully (summaries_2d mode)")
            return True, pred_run_summary


        logger.debug("Prediction Pipeline: Generating maps...")
        if progress_callback:
            progress_callback(85, "Generating prediction maps...")

        try:
            logger.debug("Prediction Pipeline: Entering map generation try block.")

            plot_config = pred_config_obj.plotting
            bounds = pred_config_obj.prediction_grid_settings.geospatial_bounds
            user_lat_min = bounds.min_lat if bounds else None
            user_lat_max = bounds.max_lat if bounds else None
            user_lon_min = bounds.min_lon if bounds else None
            user_lon_max = bounds.max_lon if bounds else None

            rey_var_name = pred_var_name
            vertical_dim = None
            for cand in ["depth_mbsf", "age_ka", "depth"]:
                if cand in predicted_ds[rey_var_name].coords:
                    vertical_dim = cand
                    break
            if not vertical_dim:
                raise ConfigurationError("Could not find a vertical coordinate in predicted dataset (expected one of: depth_mbsf, age_ka, depth).")

            level_values = predicted_ds[vertical_dim].values
            logger.debug(f"Prediction Pipeline: Found vertical axis '{vertical_dim}' with levels: {level_values}")

            if vertical_dim in ("depth_mbsf", "depth"):
                level_name = "Depth (mbsf)"
                level_unit = "m"
                pred_maps_output_dir = pred_maps_output_dir_depth
                filename_prefix = "rey_map_depth_mbsf"
            else:
                level_name = "Age"
                level_unit = "ka"
                pred_maps_output_dir = pred_maps_output_dir_age
                filename_prefix = "rey_map_age_ka"

            Path(pred_maps_output_dir).mkdir(parents=True, exist_ok=True)
            pred_run_summary["results"]["prediction_maps_axis"] = vertical_dim
            pred_run_summary["results"]["prediction_maps_dir_path"] = pred_maps_output_dir

            map_paths = []
            if target_task_type == "multiclass_classification":
                num_classes = int(getattr(training_artifacts.norm_params.rey, "num_classes", 0) or 0)
                class_names = list(getattr(training_artifacts.norm_params.rey, "class_names", None) or [])
                num_classes = max(num_classes, len(class_names))

                def _label(c: int) -> str:
                    return class_names[c] if c < len(class_names) else f"class{c}"

                for c in range(num_classes):
                    prob_var = f"predicted_prob_class{c}"
                    if prob_var not in predicted_ds:
                        continue
                    out_dir = os.path.join(pred_maps_output_dir, f"class_{c}")
                    Path(out_dir).mkdir(parents=True, exist_ok=True)
                    for idx, lvl_val in enumerate(level_values):
                        rey_at_level = predicted_ds[prob_var].sel({vertical_dim: lvl_val})
                        if bool(getattr(pred_config_obj, "mask_out_of_domain_in_maps", True)) and "is_in_training_age_domain" in predicted_ds:
                            try:
                                dom = predicted_ds["is_in_training_age_domain"].sel({vertical_dim: lvl_val})
                                rey_at_level = rey_at_level.where(dom > 0)
                            except Exception:
                                pass
                        map_path = plot_rey_map_by_level(
                            rey_data_2d=rey_at_level,
                            level_value=float(lvl_val),
                            level_name=level_name,
                            level_unit=level_unit,
                            output_dir=out_dir,
                            plot_config_params=plot_config.model_dump(),
                            user_lat_min=user_lat_min,
                            user_lat_max=user_lat_max,
                            user_lon_min=user_lon_min,
                            user_lon_max=user_lon_max,
                            filename_prefix=f"{filename_prefix}_class{c}",
                            colorbar_label=f"P({_label(c)})",
                            title_prefix=f"Predicted P({_label(c)})",
                            vmin=0.0,
                            vmax=1.0,
                            cmap="viridis",
                        )
                        if map_path:
                            map_paths.append(map_path)
                        if progress_callback:
                            progress_callback(
                                85 + (15 * (idx + 1) / len(level_values)),
                                f"Plotted maps for {vertical_dim}={lvl_val}",
                            )
            else:
                if target_task_type == "binary_classification":
                    cb_label = f"P(ΣREY ≥ {float(thr_ppm or 1000.0):g} ppm)"
                    title_prefix = "Predicted probability"
                    cmap = "viridis"
                    vmin = 0.0
                    vmax = 1.0
                else:
                    cb_label = "Predicted REY (ppm)"
                    title_prefix = "Predicted REY"
                    cmap = None
                    vmin = None
                    vmax = None

                for idx, lvl_val in enumerate(level_values):
                    logger.debug(
                        f"Prediction Pipeline: Processing level {idx+1}/{len(level_values)}: {lvl_val} ({vertical_dim})"
                    )
                    rey_at_level = predicted_ds[rey_var_name].sel({vertical_dim: lvl_val})

                    if bool(getattr(pred_config_obj, "mask_out_of_domain_in_maps", True)) and "is_in_training_age_domain" in predicted_ds:
                        try:
                            dom = predicted_ds["is_in_training_age_domain"].sel({vertical_dim: lvl_val})
                            rey_at_level = rey_at_level.where(dom > 0)
                        except Exception:
                            pass

                    map_path = plot_rey_map_by_level(
                        rey_data_2d=rey_at_level,
                        level_value=float(lvl_val),
                        level_name=level_name,
                        level_unit=level_unit,
                        output_dir=pred_maps_output_dir,
                        plot_config_params=plot_config.model_dump(),
                        user_lat_min=user_lat_min,
                        user_lat_max=user_lat_max,
                        user_lon_min=user_lon_min,
                        user_lon_max=user_lon_max,
                        filename_prefix=filename_prefix,
                        colorbar_label=cb_label,
                        title_prefix=title_prefix,
                        vmin=vmin,
                        vmax=vmax,
                        cmap=cmap,
                    )
                    logger.debug(f"Prediction Pipeline: plot_rey_map_by_level returned. Map path: {map_path}")
                    if map_path:
                        map_paths.append(map_path)
                    if progress_callback:
                        progress_callback(
                            85 + (15 * (idx + 1) / len(level_values)),
                            f"Plotted map for {vertical_dim}={lvl_val}",
                        )

            pred_run_summary["artifacts"]["paths"]["prediction_maps"] = map_paths


            aux_maps = {}
            try:
                aux_outs = {
                    "in_domain": (pred_maps_output_dir + os.sep + "aux_in_domain_mask", "is_in_training_age_domain", "In-domain (age proxy)", "0/1"),
                    "extrap_factor": (pred_maps_output_dir + os.sep + "aux_extrapolation_factor", "depth_extrapolation_factor", "Extrapolation attenuation", ""),
                }
                if target_task_type == "multiclass_classification":
                    aux_outs["entropy"] = (
                        pred_maps_output_dir + os.sep + "aux_entropy",
                        "predicted_class_entropy",
                        "Predictive entropy",
                        "nats",
                    )
                if pred_std_name in predicted_ds and pred_config_obj.enable_mc_dropout:
                    aux_outs["uncertainty"] = (
                        pred_maps_output_dir + os.sep + "aux_uncertainty_per_level",
                        pred_std_name,
                        "Uncertainty (MC Dropout)",
                        "1" if target_task_type in ("binary_classification", "multiclass_classification") else "ppm",
                    )

                for key, (out_dir, var_name, title, unit) in aux_outs.items():
                    if var_name not in predicted_ds:
                        continue
                    Path(out_dir).mkdir(parents=True, exist_ok=True)
                    paths = []
                    for lvl_val in level_values:
                        da = predicted_ds[var_name].sel({vertical_dim: lvl_val})
                        if key == "in_domain":
                            da = da.astype(float)
                        p = plot_rey_map_by_level(
                            rey_data_2d=da,
                            level_value=float(lvl_val),
                            level_name=level_name,
                            level_unit=level_unit,
                            output_dir=out_dir,
                            plot_config_params=plot_config.model_dump(),
                            user_lat_min=user_lat_min,
                            user_lat_max=user_lat_max,
                            user_lon_min=user_lon_min,
                            user_lon_max=user_lon_max,
                            colorbar_label=(title if not unit else f"{title} ({unit})"),
                            title_prefix=title,
                            filename_prefix=f"{key}_{vertical_dim}",
                        )
                        if p:
                            paths.append(p)
                    if paths:
                        aux_maps[key] = paths

                if aux_maps:
                    pred_run_summary["artifacts"]["paths"]["prediction_aux_maps"] = aux_maps
            except Exception as e_aux_maps:
                logger.warning(f"Prediction Pipeline: Failed to generate auxiliary maps: {e_aux_maps}")


            if pred_std_name in predicted_ds and pred_config_obj.enable_mc_dropout:
                uncertainty_maps = []
                if vertical_dim in ("depth_mbsf", "depth"):
                    bins = pred_config_obj.depth_bins_m or [0.0, 3.0, 6.0, 15.0]
                    unit_tag = "m"
                    title_unit = "m"
                else:
                    bins = getattr(pred_config_obj, "age_bins_ka", None) or [0.0, 1000.0, 2580.0, 10000.0]
                    unit_tag = "ka"
                    title_unit = "ka"

                for i in range(len(bins) - 1):
                    d_min = bins[i]
                    d_max = bins[i + 1]
                    layer_name = f"uncertainty_std_{int(d_min)}_{int(d_max)}{unit_tag}"
                    if layer_name not in predicted_ds:
                        continue
                    unc_layer = predicted_ds[layer_name]
                    map_path_unc = plot_rey_map_by_level(
                        rey_data_2d=unc_layer,
                        level_value=float(d_min),
                        level_name="Uncertainty layer",
                        level_unit=unit_tag,
                        level_label=f"{d_min}-{d_max} {title_unit}",
                        output_dir=pred_maps_output_dir,
                        plot_config_params=plot_config.model_dump(),
                        user_lat_min=user_lat_min,
                        user_lat_max=user_lat_max,
                        user_lon_min=user_lon_min,
                        user_lon_max=user_lon_max,
                        colorbar_label=("Uncertainty (std)" if target_task_type != "regression" else "Uncertainty (std, ppm)"),
                        title_prefix="Uncertainty (MC Dropout)",
                        filename_prefix=f"uncertainty_std_{unit_tag}",
                    )
                    if map_path_unc:
                        uncertainty_maps.append(map_path_unc)
                pred_run_summary["artifacts"]["paths"]["uncertainty_maps"] = uncertainty_maps

            logger.debug("Prediction Pipeline: Exiting map generation try block successfully.")
        except Exception as e:
            logger.warning(f"Map generation failed: {e}")
            logger.exception("Full traceback for map generation failure:")

            pred_run_summary["artifacts"]["paths"]["prediction_maps"] = []
            logger.debug("Continuing with prediction pipeline despite map generation failure")


        pred_run_summary["status"] = "completed"
        pred_run_summary["results"]["predicted_grid_nc_path"] = pred_nc_output_path

        logger.info("Prediction Pipeline: Completed successfully")
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
            pred_run_summary["duration_seconds"] = (
                datetime.fromisoformat(pred_run_summary["timestamp_end"]) - 
                datetime.fromisoformat(pred_run_summary["timestamp_start"])
            ).total_seconds()


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

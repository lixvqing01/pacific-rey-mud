
import os
import time
import json
import random
import numpy as np
import torch
import xarray as xr
from core.utils.netcdf_io import open_dataset
import pandas as pd
import matplotlib


matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from typing import Any
from torch.utils.data import DataLoader, SubsetRandomSampler
from datetime import datetime
from pathlib import Path

from core.training.samplers import BlockBalancedSamplingConfig, WeightedSubsetRandomSampler, make_block_balanced_sampler

from core.utils.logger_config import logger
from core.utils.run_manager import save_run_summary
from core.utils.constants import RESULTS_GUI_DIR
from core.datasets.rey_dataset import REYPointDataset
from core.utils.exceptions import StopExecutionError
from core.utils.serialization_utils import NumpyJSONEncoder
from core.utils.config_models import TrainingArtifacts, ModelInitConfig, NormalizationParams
from core.utils.config_models import CascadeStageArtifacts, CascadeTrainingArtifacts

from .trainer import ModelTrainer
from core.explainability.shap_analyzer import analyze_shap_values
from core.explainability.post_hoc_analyzer import run_post_hoc_analysis
from core.explainability.mop_analyzer import analyze_mop_depth_profiles
from core.explainability.peak_profile_analyzer import analyze_peak_profiles
from core.explainability.env_profile_analyzer import analyze_env_profile_relationships
from core.explainability.profile_cluster_analyzer import analyze_profile_clusters
from core.utils.results_io import write_json, snapshot_environment, build_paths_manifest, RESULTS_SCHEMA_VERSION
from core.utils.run_report import make_training_run_report_md
from core.visualization.classification_diagnostics import (
    compute_high_grade_metrics,
    infer_high_grade_classes,
    save_classification_tables,
    save_b2_separability_diagnostics,
    save_env_patch_missingness_diagnostics,
    save_grade_class_diagnostics,
    save_hierarchical_classification_diagnostics,
    save_split_group_diagnostics,
)


CASCADE_GRADE_TASK_TYPE = "cascade_grade_geochem_classification"
CASCADE_EXPLORATION_4CLASS_TASK_TYPE = "cascade_exploration_4class"
CASCADE_FINAL_CLASS_NAMES = [
    "carbonate_background",
    "class1_low_grade",
    "class1_high_grade",
    "class2_low_grade",
    "class2_high_grade",
]
CASCADE_EXPLORATION_4CLASS_NAMES = [
    "carbonate_background",
    "class1_enriched",
    "class2_low_grade",
    "class2_high_grade",
]


def _copy_output_dir_tree(output_dirs: dict[str, str], new_base_dir: Path) -> dict[str, str]:
    src_base = Path(output_dirs["base"])
    new_dirs: dict[str, str] = {}
    for key, value in output_dirs.items():
        value_path = Path(value)
        try:
            rel = value_path.relative_to(src_base)
            new_dirs[key] = str(new_base_dir / rel)
        except Exception:
            new_dirs[key] = str(new_base_dir / value_path.name)
    return new_dirs


def _prepare_cascade_labels(rey_points_path: str, output_dir: str) -> str:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(rey_points_path).copy()

    if "mech_class_id" not in df.columns:
        raise ValueError("Cascade training requires 'mech_class_id' in the point table.")

    df["main_class_id"] = pd.to_numeric(df["mech_class_id"], errors="coerce").astype("Int64")

    grade_ids = None
    if "grade_geochem_class_id" in df.columns:
        grade_ids = pd.to_numeric(df["grade_geochem_class_id"], errors="coerce")

    if "class1_grade_subclass_id" in df.columns:
        c1 = pd.to_numeric(df["class1_grade_subclass_id"], errors="coerce")
    else:
        c1 = pd.Series(np.where(grade_ids.isin([1, 2]), grade_ids - 1, np.nan), index=df.index, dtype=float)

    if "class2_grade_subclass_id" in df.columns:
        c2 = pd.to_numeric(df["class2_grade_subclass_id"], errors="coerce")
    else:
        c2 = pd.Series(np.where(grade_ids.isin([3, 4]), grade_ids - 3, np.nan), index=df.index, dtype=float)

    df["class1_grade_label"] = c1.astype("Int64")
    df["class2_grade_label"] = c2.astype("Int64")
    if "exploration_class_id" not in df.columns:
        exploration = np.zeros(len(df), dtype=np.int16)
        mech = pd.to_numeric(df["mech_class_id"], errors="coerce").to_numpy(dtype=float)
        c2_np = pd.to_numeric(df["class2_grade_label"], errors="coerce").to_numpy(dtype=float)
        exploration[mech == 1] = 1
        class2_mask = mech == 2
        exploration[class2_mask] = 2 + np.nan_to_num(c2_np[class2_mask], nan=0.0).astype(np.int16)
        df["exploration_class_id"] = exploration.astype(int)
    if "exploration_class_name" not in df.columns:
        df["exploration_class_name"] = [
            CASCADE_EXPLORATION_4CLASS_NAMES[int(i)]
            if 0 <= int(i) < len(CASCADE_EXPLORATION_4CLASS_NAMES)
            else f"class_{int(i)}"
            for i in pd.to_numeric(df["exploration_class_id"], errors="coerce").fillna(0).astype(int).tolist()
        ]
    df["cascade_global_index"] = np.arange(len(df), dtype=int)
    df["cascade_source_index"] = np.asarray(df.index, dtype=int)

    output_path = out_dir / "rey_points_cascade_labels.parquet"
    df.to_parquet(output_path, index=False)
    return str(output_path)


def _write_cascade_stage_points(
    *,
    cascade_df: pd.DataFrame,
    mask: pd.Series,
    output_path: Path,
) -> tuple[str, dict[int, int]]:
    subset = cascade_df.loc[mask].copy().reset_index(drop=True)
    if subset.empty:
        raise ValueError(f"Cascade stage subset is empty: {output_path}")
    mapping = {
        int(global_idx): int(local_idx)
        for local_idx, global_idx in enumerate(subset["cascade_global_index"].astype(int).tolist())
    }
    subset.to_parquet(output_path, index=False)
    return str(output_path), mapping


def _add_class2_soft_label_columns(
    df: pd.DataFrame,
    *,
    output_path: Path,
    mode: str = "geochem_probs",
) -> str:
    """Write a B2 stage table with optional soft low/high labels."""

    mode_norm = str(mode or "geochem_probs").strip().lower()
    out = df.copy()
    if mode_norm not in {"geochem_probs", "exploration_probs"}:
        raise ValueError(f"Unsupported B2 soft-label mode: {mode}")

    if mode_norm == "exploration_probs":
        cols = ["exploration_class_prob_2", "exploration_class_prob_3"]
    else:
        cols = ["grade_geochem_class_prob_3", "grade_geochem_class_prob_4"]

    if not all(col in out.columns for col in cols):
        raise KeyError(f"B2 soft-label mode '{mode_norm}' requires columns: {cols}")

    vals = out[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    vals = np.where(np.isfinite(vals) & (vals >= 0.0), vals, 0.0)
    row_sum = vals.sum(axis=1, keepdims=True)
    hard = pd.to_numeric(out.get("class2_grade_label"), errors="coerce").fillna(0).to_numpy(dtype=int)
    fallback = np.zeros((len(out), 2), dtype=float)
    fallback[np.arange(len(out)), np.clip(hard, 0, 1)] = 1.0
    vals = np.where(row_sum > 1e-12, vals / np.clip(row_sum, 1e-12, None), fallback)
    out["class2_grade_soft_low"] = vals[:, 0]
    out["class2_grade_soft_high"] = vals[:, 1]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    return str(output_path)


def _load_stage_prediction_table(run_summary: dict, split_name: str) -> pd.DataFrame:
    table_paths = ((run_summary.get("results", {}) or {}).get("table_paths", {}) or {})
    key = f"{split_name}_predictions_with_probs_csv"
    path = table_paths.get(key)
    if (not path or not os.path.exists(path)):
        artifact_paths = ((run_summary.get("artifacts", {}) or {}).get("paths", {}) or {})
        config_paths = ((run_summary.get("config", {}) or {}).get("paths", {}) or {})
        tables_dir = artifact_paths.get("tables") or config_paths.get("run_base_dir")
        if tables_dir:
            candidate = Path(tables_dir) / split_name / "predictions_with_probs.csv"
            if candidate.exists():
                path = str(candidate)
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Missing stage prediction table for split='{split_name}': {path}")
    return pd.read_csv(path)


def _merge_cascade_stage_tables(
    *,
    main_df: pd.DataFrame,
    class1_df: pd.DataFrame,
    class2_df: pd.DataFrame,
) -> pd.DataFrame:
    key_col = "cascade_global_index" if "cascade_global_index" in main_df.columns else (
        "source_index" if "source_index" in main_df.columns else "dataset_index"
    )
    if key_col not in main_df.columns:
        raise ValueError("Cascade stage tables are missing a usable row identifier.")

    out = main_df.copy()
    if key_col not in class1_df.columns or key_col not in class2_df.columns:
        raise ValueError("Cascade child prediction tables are missing the cascade row identifier.")

    c1_keep = [key_col] + [c for c in class1_df.columns if c.startswith("p_class")]
    c2_keep = [key_col] + [c for c in class2_df.columns if c.startswith("p_class")]
    c1 = class1_df[c1_keep].rename(columns={"p_class0": "p_class1_low_given_class1", "p_class1": "p_class1_high_given_class1"})
    c2 = class2_df[c2_keep].rename(columns={"p_class0": "p_class2_low_given_class2", "p_class1": "p_class2_high_given_class2"})
    out = out.merge(c1, on=key_col, how="left")
    out = out.merge(c2, on=key_col, how="left")

    for col in [
        "p_class1_low_given_class1",
        "p_class1_high_given_class1",
        "p_class2_low_given_class2",
        "p_class2_high_given_class2",
    ]:
        if col not in out.columns:
            out[col] = np.nan

    main_carbonate = pd.to_numeric(out.get("p_class0"), errors="coerce").to_numpy(dtype=float)
    main_class1 = pd.to_numeric(out.get("p_class1"), errors="coerce").to_numpy(dtype=float)
    main_class2 = pd.to_numeric(out.get("p_class2"), errors="coerce").to_numpy(dtype=float)
    c1_low = pd.to_numeric(out.get("p_class1_low_given_class1"), errors="coerce").fillna(0.5).to_numpy(dtype=float)
    c1_high = pd.to_numeric(out.get("p_class1_high_given_class1"), errors="coerce").fillna(0.5).to_numpy(dtype=float)
    c2_low = pd.to_numeric(out.get("p_class2_low_given_class2"), errors="coerce").fillna(0.5).to_numpy(dtype=float)
    c2_high = pd.to_numeric(out.get("p_class2_high_given_class2"), errors="coerce").fillna(0.5).to_numpy(dtype=float)

    final_prob = np.column_stack(
        [
            main_carbonate,
            main_class1 * c1_low,
            main_class1 * c1_high,
            main_class2 * c2_low,
            main_class2 * c2_high,
        ]
    )
    row_sum = np.clip(np.sum(final_prob, axis=1, keepdims=True), 1e-12, None)
    final_prob = final_prob / row_sum

    for class_id in range(final_prob.shape[1]):
        out[f"cascade_prob_class{class_id}"] = final_prob[:, class_id]
    out["cascade_y_pred"] = np.argmax(final_prob, axis=1).astype(int)
    return out


def _merge_exploration_cascade_stage_tables(
    *,
    main_df: pd.DataFrame,
    class2_df: pd.DataFrame,
) -> pd.DataFrame:
    key_col = "cascade_global_index" if "cascade_global_index" in main_df.columns else (
        "source_index" if "source_index" in main_df.columns else "dataset_index"
    )
    if key_col not in main_df.columns or key_col not in class2_df.columns:
        raise ValueError("Exploration cascade stage tables are missing a usable row identifier.")

    out = main_df.copy()
    c2_keep = [key_col] + [c for c in class2_df.columns if c.startswith("p_class")]
    c2 = class2_df[c2_keep].rename(columns={"p_class0": "p_class2_low_given_class2", "p_class1": "p_class2_high_given_class2"})
    out = out.merge(c2, on=key_col, how="left")
    for col in ["p_class2_low_given_class2", "p_class2_high_given_class2"]:
        if col not in out.columns:
            out[col] = np.nan

    if "exploration_class_id" not in out.columns:
        mech = pd.to_numeric(out.get("mech_class_id"), errors="coerce")
        class2_sub = pd.to_numeric(out.get("class2_grade_subclass_id"), errors="coerce")
        grade5 = pd.to_numeric(out.get("grade_geochem_class_id"), errors="coerce")
        exploration = pd.Series(np.nan, index=out.index, dtype="float64")
        exploration.loc[mech == 0] = 0
        exploration.loc[mech == 1] = 1
        exploration.loc[(mech == 2) & (class2_sub == 0)] = 2
        exploration.loc[(mech == 2) & (class2_sub == 1)] = 3
        exploration.loc[grade5 == 0] = 0
        exploration.loc[grade5.isin([1, 2])] = 1
        exploration.loc[grade5 == 3] = 2
        exploration.loc[grade5 == 4] = 3
        if exploration.isna().any():
            missing = int(exploration.isna().sum())
            raise ValueError(f"Cannot derive exploration_class_id for {missing} cascade rows.")
        out["exploration_class_id"] = exploration.astype(int)
    if "exploration_class_name" not in out.columns:
        names = CASCADE_EXPLORATION_4CLASS_NAMES
        out["exploration_class_name"] = [
            names[i] if 0 <= int(i) < len(names) else f"class_{int(i)}"
            for i in pd.to_numeric(out["exploration_class_id"], errors="coerce").fillna(0).astype(int).tolist()
        ]

    main_carbonate = pd.to_numeric(out.get("p_class0"), errors="coerce").to_numpy(dtype=float)
    main_class1 = pd.to_numeric(out.get("p_class1"), errors="coerce").to_numpy(dtype=float)
    main_class2 = pd.to_numeric(out.get("p_class2"), errors="coerce").to_numpy(dtype=float)
    c2_low = pd.to_numeric(out.get("p_class2_low_given_class2"), errors="coerce").fillna(0.5).to_numpy(dtype=float)
    c2_high = pd.to_numeric(out.get("p_class2_high_given_class2"), errors="coerce").fillna(0.5).to_numpy(dtype=float)

    final_prob = np.column_stack(
        [
            main_carbonate,
            main_class1,
            main_class2 * c2_low,
            main_class2 * c2_high,
        ]
    )
    row_sum = np.clip(np.sum(final_prob, axis=1, keepdims=True), 1e-12, None)
    final_prob = final_prob / row_sum

    for class_id in range(final_prob.shape[1]):
        out[f"cascade_prob_class{class_id}"] = final_prob[:, class_id]
    out["cascade_y_pred"] = np.argmax(final_prob, axis=1).astype(int)
    return out


def _compute_cascade_multiclass_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss

    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    prob = np.asarray(prob, dtype=float)
    y_pred = np.argmax(prob, axis=1)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y_true, prob, labels=list(range(prob.shape[1])))),
    }
    return metrics


def _apply_any_high_threshold_override(
    *,
    prob: np.ndarray,
    threshold: float,
    high_grade_classes: list[int],
) -> np.ndarray:
    prob = np.asarray(prob, dtype=float)
    if prob.ndim != 2 or prob.size == 0:
        return np.array([], dtype=int)
    high = [int(i) for i in high_grade_classes if 0 <= int(i) < prob.shape[1]]
    if not high:
        return np.argmax(prob, axis=1).astype(int)
    low = [i for i in range(prob.shape[1]) if i not in high]
    if not low:
        return np.argmax(prob, axis=1).astype(int)

    high_idx = np.asarray(high, dtype=int)
    low_idx = np.asarray(low, dtype=int)
    p_high = np.clip(np.sum(prob[:, high_idx], axis=1), 0.0, 1.0)
    best_high = high_idx[np.argmax(prob[:, high_idx], axis=1)]
    best_low = low_idx[np.argmax(prob[:, low_idx], axis=1)]
    return np.where(p_high >= float(threshold), best_high, best_low).astype(int)


def _load_high_grade_threshold_recommendation(split_dir: Path, metric_name: str = "F1") -> dict[str, float] | None:
    path = split_dir / "high_grade_threshold_recommendations.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        rec = ((data.get("best") or {}).get(str(metric_name)) or {})
        threshold = float(rec.get("threshold"))
        if not np.isfinite(threshold):
            return None
        return {
            "metric": str(metric_name),
            "threshold": float(np.clip(threshold, 0.0, 1.0)),
            "precision": float(rec.get("precision", float("nan"))),
            "recall": float(rec.get("recall", float("nan"))),
            "score": float(rec.get("score", float("nan"))),
        }
    except Exception as e:
        logger.warning(f"Cascade: failed to load high-grade threshold recommendation from {path}: {e}")
        return None


def _discover_env_feature_names(env_grid_path: str) -> list[str]:
    with open_dataset(env_grid_path) as env_ds:
        names = [str(x) for x in env_ds.data_vars]
    return [name for name in names if name != "land_mask"]


def apply_cascade_training_profile(hyperparams: dict[str, Any], training_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Normalize cascade profile defaults without removing explicit caller overrides."""

    hp = hyperparams
    training_profile = str(
        hp.get("training_profile", hp.get("cascade_objective", "classification_precision"))
        or "classification_precision"
    ).strip().lower()
    if training_profile in {"macro_f1", "classification_accuracy", "classification_precision", "accuracy"}:
        training_profile = "classification_precision"
    elif training_profile in {"exploration", "exploration_screening", "exploration_screening_4class"}:
        training_profile = "exploration_screening_4class"
    elif training_profile in {"high_grade", "high_grade_f2", "high_grade_screening"}:
        training_profile = "high_grade_recall"
    hp["training_profile"] = training_profile
    if training_profile == "high_grade_recall":
        hp["cascade_objective"] = "high_grade_recall"
    elif training_profile == "exploration_screening_4class":
        hp["cascade_objective"] = "exploration_screening"
    else:
        hp["cascade_objective"] = "macro_f1"

    if training_profile in {"classification_precision", "exploration_screening_4class"}:
        hp.setdefault("early_stopping_metric", "val_macro_f1")
        hp.setdefault("cascade_stage_a_early_stopping_metric", "val_macro_f1")
        hp.setdefault("cascade_b1_early_stopping_metric", "val_macro_f1")
        hp.setdefault("cascade_b2_early_stopping_metric", "val_macro_f1")
        hp.setdefault("label_smoothing", 0.0)
        hp.setdefault("class_balance_power", 0.6)
        hp.setdefault("class_balance_max_factor", 6.0)
        hp.setdefault("cascade_b2_class_balance_power", 0.6)
        hp.setdefault("cascade_b2_class_balance_max_factor", 6.0)
        if training_config is not None:
            training_config.setdefault("train_sampling_strategy", "block_and_class_balanced")
    else:
        hp.setdefault("early_stopping_metric", "val_macro_f1")
        hp.setdefault("cascade_stage_a_early_stopping_metric", "val_macro_f1")
        hp.setdefault("cascade_b1_early_stopping_metric", "val_macro_f1")
        hp.setdefault("cascade_b2_early_stopping_metric", "val_class1_ap")
        hp.setdefault("label_smoothing", 0.0)
        hp.setdefault("class_balance_power", 0.7)
        hp.setdefault("class_balance_max_factor", 8.0)
        hp.setdefault("cascade_b2_class_balance_power", 0.85)
        hp.setdefault("cascade_b2_class_balance_max_factor", 10.0)
    return hp


def _write_b2_threshold_stability_report(*, class2_run_dir: Path, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    rows: list[pd.DataFrame] = []
    for split_name in ("train", "val", "test"):
        split_dir = class2_run_dir / "tables" / split_name
        fixed_csv = split_dir / "high_grade_fixed_threshold_metrics.csv"
        rec_json = split_dir / "high_grade_threshold_recommendations.json"
        if fixed_csv.exists():
            try:
                df = pd.read_csv(fixed_csv)
                df.insert(0, "split", split_name)
                rows.append(df)
            except Exception as e:
                logger.warning(f"Cascade: failed to read B2 fixed threshold metrics {fixed_csv}: {e}")
        if rec_json.exists():
            try:
                out_json = output_dir / f"{split_name}_class2_high_threshold_recommendations.json"
                out_json.write_text(rec_json.read_text(encoding="utf-8"), encoding="utf-8")
                paths[f"{split_name}_class2_high_threshold_recommendations_json"] = str(out_json)
            except Exception as e:
                logger.warning(f"Cascade: failed to copy B2 threshold recommendation {rec_json}: {e}")
    if rows:
        all_df = pd.concat(rows, ignore_index=True)
        csv_path = output_dir / "class2_high_fixed_threshold_metrics_by_split.csv"
        all_df.to_csv(csv_path, index=False)
        paths["class2_high_fixed_threshold_metrics_by_split_csv"] = str(csv_path)
        summary_rows = []
        for threshold, g in all_df.groupby("threshold", dropna=False):
            row: dict[str, Any] = {"threshold": float(threshold)}
            for metric in ["precision", "recall", "f1", "f2", "n_pred_positive", "average_precision"]:
                vals = pd.to_numeric(g.get(metric), errors="coerce").dropna().to_numpy(dtype=float)
                if vals.size:
                    row[f"{metric}_mean"] = float(np.mean(vals))
                    row[f"{metric}_std"] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
                    row[f"{metric}_min"] = float(np.min(vals))
                    row[f"{metric}_max"] = float(np.max(vals))
            summary_rows.append(row)
        summary_path = output_dir / "class2_high_fixed_threshold_stability_summary.csv"
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        paths["class2_high_fixed_threshold_stability_summary_csv"] = str(summary_path)
    return paths


def _build_cascade_stage_config(
    *,
    base_config: dict[str, Any],
    stage_name: str,
    stage_base_dir: Path,
    points_path: str,
    label_col: str,
    class_names: list[str],
    num_classes: int,
    forced_indices: dict[str, list[int]],
    filtered_class_id: int | None = None,
) -> dict[str, Any]:
    cfg = json.loads(json.dumps(base_config, cls=NumpyJSONEncoder))
    cfg.setdefault("paths", {})
    cfg["paths"]["run_base_dir"] = str(stage_base_dir)
    cfg["paths"]["processed_rey_points"] = str(points_path)
    cfg.setdefault("training", {})
    cfg["training"]["hyperparams"] = cfg["training"].get("hyperparams", {}) or {}
    hp = cfg["training"]["hyperparams"]
    hp["task_type"] = "multiclass_classification"
    hp["classification_label_col"] = str(label_col)
    hp["classification_class_names"] = list(class_names)
    hp["num_classes"] = int(num_classes)
    hp["hierarchical_multiclass"] = False
    hp["hierarchical_grade_multitask"] = False
    hp.pop("ce_endmember_class_weights", None)
    hp.pop("ce_subclass_class_weights", None)
    hp.pop("ce_grade_class_weights", None)
    hp["cascade_stage_name"] = str(stage_name)
    hp["training_profile"] = str(hp.get("training_profile", hp.get("cascade_objective", "classification_precision")) or "classification_precision")
    hp.pop("fixed_test_indices", None)
    if filtered_class_id is not None:
        hp["cascade_parent_class_id"] = int(filtered_class_id)
        hp["hierarchical_multiclass"] = False
        hp["hierarchical_grade_multitask"] = False
        hp["early_stopping_metric"] = str(
            hp.get("cascade_b1_early_stopping_metric" if int(filtered_class_id) == 1 else "cascade_b2_early_stopping_metric", "val_macro_f1")
            or "val_macro_f1"
        )
        if int(filtered_class_id) == 1:


            hp["classification_loss"] = "ce"
            hp["label_smoothing"] = float(hp.get("label_smoothing", 0.0) or 0.0)
            hp["cnn_filters"] = [16, 32]
            hp["mlp_hidden_layers"] = [64, 32]
            hp["class_head_hidden_dim"] = 32
            hp["dropout_prob"] = max(float(hp.get("dropout_prob", 0.35) or 0.35), 0.45)
            hp["early_stopping_patience"] = min(int(hp.get("early_stopping_patience", 50) or 50), 30)
            hp["batch_size"] = min(int(hp.get("batch_size", 128) or 128), 64)
            hp["use_class_weights"] = False
            hp["class_balance_power"] = 0.0
            hp["class_balance_max_factor"] = 1.0
            cfg["training"]["train_sampling_strategy"] = "block_balanced"
        elif int(filtered_class_id) == 2:


            hp["explain_target_class_index"] = 1
            if bool(hp.get("cascade_b2_use_soft_labels", False)):
                hp["use_soft_labels"] = True
                hp["soft_label_cols"] = [
                    "class2_grade_soft_low",
                    "class2_grade_soft_high",
                ]
            hp["classification_loss"] = str(hp.get("cascade_b2_classification_loss", hp.get("classification_loss", "ce")) or "ce")
            hp["use_class_weights"] = bool(hp.get("cascade_b2_use_class_weights", hp.get("use_class_weights", True)))
            hp["class_balance_power"] = float(hp.get("cascade_b2_class_balance_power", float(hp.get("class_balance_power", 0.6) or 0.6)))
            hp["class_balance_max_factor"] = float(hp.get("cascade_b2_class_balance_max_factor", float(hp.get("class_balance_max_factor", 6.0) or 6.0)))
            aux_weight = float(hp.get("cascade_b2_aux_high_grade_bce_weight", hp.get("aux_high_grade_bce_weight", 0.0)) or 0.0)
            if aux_weight > 0.0:
                hp["aux_high_grade_bce_weight"] = aux_weight
                hp["aux_high_grade_class_index"] = 1
                hp["aux_high_grade_auto_pos_weight"] = bool(
                    hp.get("cascade_b2_aux_high_grade_auto_pos_weight", hp.get("aux_high_grade_auto_pos_weight", True))
                )

    cfg["training"]["forced_split_indices"] = {
        split_name: [int(i) for i in indices]
        for split_name, indices in forced_indices.items()
    }
    return cfg


def _ensure_nonempty_cascade_split(forced_indices: dict[str, list[int]]) -> dict[str, list[int]]:
    out = {k: [int(i) for i in v] for k, v in forced_indices.items()}
    if not out.get("train"):
        raise ValueError("Cascade stage train split is empty; cannot continue.")
    if not out.get("validation"):
        if len(out["train"]) < 2:
            raise ValueError("Cascade stage validation split is empty and train split is too small to borrow one sample.")
        out["validation"] = [out["train"].pop()]
    if not out.get("test"):
        if len(out["train"]) < 2:
            raise ValueError("Cascade stage test split is empty and train split is too small to borrow one sample.")
        out["test"] = [out["train"].pop()]
    return out


def _run_cascade_training_pipeline(config, progress_callback=None, stop_flag=None, history_callback=None):
    requested_task_type = str(
        (((config.get("training", {}) or {}).get("hyperparams", {}) or {}).get("task_type", CASCADE_GRADE_TASK_TYPE))
        or CASCADE_GRADE_TASK_TYPE
    )
    is_exploration_4class = requested_task_type == CASCADE_EXPLORATION_4CLASS_TASK_TYPE
    logger.info(f"Pipeline: Starting cascade training pipeline ({requested_task_type})...")
    start_time = time.time()
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')

    cfg_paths = config.get("paths", {}) if isinstance(config, dict) else {}
    cfg_run_base_dir = (cfg_paths.get("run_base_dir") if isinstance(cfg_paths, dict) else None) or None
    if cfg_run_base_dir:
        run_base_dir = Path(str(cfg_run_base_dir))
        run_id = run_base_dir.name
    else:
        run_id = f"point_model_run_{timestamp_str}_cascade_temp"
        run_base_dir = Path(RESULTS_GUI_DIR) / run_id
    run_base_dir.mkdir(parents=True, exist_ok=True)

    output_dirs = {
        "base": str(run_base_dir),
        "logs": str(run_base_dir / "logs"),
        "plots": str(run_base_dir / "plots"),
        "tables": str(run_base_dir / "tables"),
        "models": str(run_base_dir / "models"),
        "reports": str(run_base_dir / "reports"),
        "splits": str(run_base_dir / "splits"),
        "diagnostics": str(run_base_dir / "plots" / "diagnostics"),
        "classification": str(run_base_dir / "plots" / "diagnostics" / "classification"),
    }
    for dir_path in output_dirs.values():
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    run_summary = {
        "run_id": run_id,
        "run_type": "training",
        "timestamp_start": datetime.now().isoformat(),
        "status": "started",
        "results_schema_version": RESULTS_SCHEMA_VERSION,
        "config": config,
        "inputs": {},
        "results": {"task_type": requested_task_type},
        "artifacts": {"paths": output_dirs},
    }

    try:
        paths = config.get("paths", {}) or {}
        training_config = config.get("training", {}) or {}
        hyperparams = training_config.get("hyperparams", {}) or {}
        hyperparams["task_type"] = requested_task_type
        apply_cascade_training_profile(hyperparams, training_config)
        run_summary["results"]["training_profile"] = str(hyperparams.get("training_profile"))
        run_summary["results"]["cascade_objective"] = str(hyperparams.get("cascade_objective"))
        rey_points_path = paths.get("processed_rey_points")
        env_grid_path = paths.get("processed_env_grid")
        if not rey_points_path or not os.path.exists(rey_points_path):
            raise FileNotFoundError(f"REY points data path missing or file not found: {rey_points_path}")
        if not env_grid_path or not os.path.exists(env_grid_path):
            raise FileNotFoundError(f"Environment grid data path missing or file not found: {env_grid_path}")

        selected_env_variables = _discover_env_feature_names(env_grid_path)
        patch_size = tuple(training_config.get("patch_size", [3, 3]))

        cascade_points_path = _prepare_cascade_labels(rey_points_path, os.path.join(output_dirs["tables"], "cascade_inputs"))
        run_summary["inputs"]["processed_rey_points_path"] = cascade_points_path
        run_summary["inputs"]["processed_env_grid_path"] = env_grid_path
        run_summary["inputs"]["env_feature_names"] = selected_env_variables

        base_main_cfg = json.loads(json.dumps(config, cls=NumpyJSONEncoder))
        base_main_cfg["paths"]["processed_rey_points"] = cascade_points_path
        base_main_cfg["paths"]["run_base_dir"] = str(run_base_dir / "main_model")
        base_main_cfg["training"]["hyperparams"] = dict(base_main_cfg["training"].get("hyperparams", {}) or {})
        base_main_cfg["training"]["hyperparams"]["task_type"] = "multiclass_classification"
        base_main_cfg["training"]["hyperparams"]["classification_label_col"] = "main_class_id"
        base_main_cfg["training"]["hyperparams"]["classification_class_names"] = [
            "carbonate_background",
            "noncarbonate_metal_enriched",
            "noncarbonate_detrital_background",
        ]
        base_main_cfg["training"]["hyperparams"]["num_classes"] = 3
        base_main_cfg["training"]["hyperparams"]["hierarchical_multiclass"] = False
        base_main_cfg["training"]["hyperparams"]["hierarchical_grade_multitask"] = False
        base_main_cfg["training"]["hyperparams"]["early_stopping_metric"] = str(
            hyperparams.get("cascade_stage_a_early_stopping_metric", "val_macro_f1") or "val_macro_f1"
        )
        if training_config.get("forced_split_indices"):
            base_main_cfg["training"]["hyperparams"].pop("fixed_test_indices", None)
            base_main_cfg["training"]["forced_split_indices"] = {
                split_name: [int(i) for i in indices]
                for split_name, indices in (training_config.get("forced_split_indices") or {}).items()
            }

        if progress_callback:
            progress_callback(2, "Cascade: training Stage A main classifier...")
        ok_main, summary_main = run_training_pipeline(base_main_cfg, progress_callback=None, stop_flag=stop_flag, history_callback=None)
        if not ok_main:
            raise RuntimeError(summary_main.get("error_message") or "Stage A training failed.")

        splits_dir = Path((summary_main.get("artifacts", {}) or {}).get("paths", {}).get("splits_dir") or (Path(base_main_cfg["paths"]["run_base_dir"]) / "splits"))
        train_idx = np.load(splits_dir / "train_indices.npy").astype(int).tolist()
        val_idx = np.load(splits_dir / "val_indices.npy").astype(int).tolist()
        test_idx = np.load(splits_dir / "test_indices.npy").astype(int).tolist()

        cascade_df = pd.read_parquet(cascade_points_path)
        mech_series = pd.to_numeric(cascade_df["mech_class_id"], errors="coerce")
        stage1_mask = mech_series == 1
        stage2_mask = mech_series == 2
        forced_stage1 = {}
        if not is_exploration_4class:
            forced_stage1 = {
                "train": [i for i in train_idx if bool(stage1_mask.iloc[i])],
                "validation": [i for i in val_idx if bool(stage1_mask.iloc[i])],
                "test": [i for i in test_idx if bool(stage1_mask.iloc[i])],
            }
        forced_stage2 = {
            "train": [i for i in train_idx if bool(stage2_mask.iloc[i])],
            "validation": [i for i in val_idx if bool(stage2_mask.iloc[i])],
            "test": [i for i in test_idx if bool(stage2_mask.iloc[i])],
        }
        if not is_exploration_4class:
            forced_stage1 = _ensure_nonempty_cascade_split(forced_stage1)
        forced_stage2 = _ensure_nonempty_cascade_split(forced_stage2)

        stage_inputs_dir = run_base_dir / "cascade_stage_inputs"
        stage_inputs_dir.mkdir(parents=True, exist_ok=True)
        stage1_points_path = None
        stage1_index_map = {}
        if not is_exploration_4class:
            stage1_points_path, stage1_index_map = _write_cascade_stage_points(
                cascade_df=cascade_df,
                mask=stage1_mask.fillna(False),
                output_path=stage_inputs_dir / "class1_points.parquet",
            )
        stage2_points_path, stage2_index_map = _write_cascade_stage_points(
            cascade_df=cascade_df,
            mask=stage2_mask.fillna(False),
            output_path=stage_inputs_dir / "class2_points.parquet",
        )
        if bool(hyperparams.get("cascade_b2_use_soft_labels", False)):
            stage2_points_path = _add_class2_soft_label_columns(
                pd.read_parquet(stage2_points_path),
                output_path=stage_inputs_dir / "class2_points_soft_labels.parquet",
                mode=str(hyperparams.get("cascade_b2_soft_label_mode", "geochem_probs") or "geochem_probs"),
            )
        forced_stage1_local = {}
        if not is_exploration_4class:
            forced_stage1_local = {
                split_name: [stage1_index_map[int(i)] for i in indices if int(i) in stage1_index_map]
                for split_name, indices in forced_stage1.items()
            }
        forced_stage2_local = {
            split_name: [stage2_index_map[int(i)] for i in indices if int(i) in stage2_index_map]
            for split_name, indices in forced_stage2.items()
        }
        if not is_exploration_4class:
            forced_stage1_local = _ensure_nonempty_cascade_split(forced_stage1_local)
        forced_stage2_local = _ensure_nonempty_cascade_split(forced_stage2_local)

        stage1_cfg = None
        if not is_exploration_4class:
            stage1_cfg = _build_cascade_stage_config(
                base_config=config,
                stage_name="class1_grade_model",
                stage_base_dir=run_base_dir / "class1_grade_model",
                points_path=str(stage1_points_path),
                label_col="class1_grade_label",
                class_names=["class1_low_grade", "class1_high_grade"],
                num_classes=2,
                forced_indices=forced_stage1_local,
                filtered_class_id=1,
            )
        stage2_cfg = _build_cascade_stage_config(
            base_config=config,
            stage_name="class2_grade_model",
            stage_base_dir=run_base_dir / "class2_grade_model",
            points_path=stage2_points_path,
            label_col="class2_grade_label",
            class_names=["class2_low_grade", "class2_high_grade"],
            num_classes=2,
            forced_indices=forced_stage2_local,
            filtered_class_id=2,
        )

        ok_b1 = True
        summary_b1 = {}
        if not is_exploration_4class:
            if progress_callback:
                progress_callback(30, "Cascade: training Stage B1 class1 low/high...")
            ok_b1, summary_b1 = run_training_pipeline(stage1_cfg, progress_callback=None, stop_flag=stop_flag, history_callback=None)
            if not ok_b1:
                raise RuntimeError(summary_b1.get("error_message") or "Stage B1 training failed.")

        if progress_callback:
            progress_callback(42 if is_exploration_4class else 58, "Cascade: training Stage B2 class2 low/high...")
        ok_b2, summary_b2 = run_training_pipeline(stage2_cfg, progress_callback=None, stop_flag=stop_flag, history_callback=None)
        if not ok_b2:
            raise RuntimeError(summary_b2.get("error_message") or "Stage B2 training failed.")

        split_probs: dict[str, np.ndarray] = {}
        split_y_true: dict[str, np.ndarray] = {}
        split_dirs: dict[str, Path] = {}
        split_pred_tables: dict[str, pd.DataFrame] = {}
        final_class_names = CASCADE_EXPLORATION_4CLASS_NAMES if is_exploration_4class else CASCADE_FINAL_CLASS_NAMES
        final_label_col = "exploration_class_id" if is_exploration_4class else "grade_geochem_class_id"
        final_num_classes = len(final_class_names)
        high_grade_classes = infer_high_grade_classes(final_class_names, final_num_classes)
        for split_name in ("train", "val", "test"):
            main_pred = _load_stage_prediction_table(summary_main, split_name)
            c2_pred = _load_stage_prediction_table(summary_b2, split_name)
            if is_exploration_4class:
                merged = _merge_exploration_cascade_stage_tables(main_df=main_pred, class2_df=c2_pred)
            else:
                c1_pred = _load_stage_prediction_table(summary_b1, split_name)
                merged = _merge_cascade_stage_tables(main_df=main_pred, class1_df=c1_pred, class2_df=c2_pred)
            split_dir = Path(output_dirs["tables"]) / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            merged.to_csv(split_dir / "cascade_predictions_with_probs.csv", index=False)
            split_pred_tables[split_name] = merged.copy()

            y_true = pd.to_numeric(merged[final_label_col], errors="coerce").to_numpy(dtype=int)
            prob = merged[[f"cascade_prob_class{i}" for i in range(final_num_classes)]].to_numpy(dtype=float)
            y_pred = merged["cascade_y_pred"].to_numpy(dtype=int)
            split_probs[split_name] = prob
            split_y_true[split_name] = y_true
            split_dirs[split_name] = split_dir

            save_classification_tables(
                y_true=y_true,
                y_pred=y_pred,
                prob=prob,
                class_names=final_class_names,
                output_dir=str(split_dir),
            )
            save_grade_class_diagnostics(
                y_true=y_true,
                y_pred=y_pred,
                prob=prob,
                class_names=final_class_names,
                output_dir=str(split_dir),
                rey_values=pd.to_numeric(merged.get("rey"), errors="coerce").to_numpy(dtype=float) if "rey" in merged.columns else None,
            )
            save_hierarchical_classification_diagnostics(
                y_true=y_true,
                y_pred=y_pred,
                prob=prob,
                class_names=final_class_names,
                output_dir=str(split_dir / "hierarchical"),
            )

            metrics = _compute_cascade_multiclass_metrics(y_true, prob)
            hg = compute_high_grade_metrics(
                y_true=y_true,
                prob=prob,
                class_names=final_class_names,
                high_grade_classes=high_grade_classes,
            )
            if hg:
                metrics["high_grade_recall"] = float(hg.get("recall", float("nan")))
                metrics["high_grade_precision"] = float(hg.get("precision", float("nan")))
                metrics["high_grade_f1"] = float(hg.get("f1", float("nan")))
                metrics["high_grade_ap"] = float(hg.get("average_precision", float("nan")))
            run_summary["results"][f"{split_name}_metrics"] = metrics

        val_threshold_rec = _load_high_grade_threshold_recommendation(split_dirs["val"], metric_name="F1") if "val" in split_dirs else None
        if val_threshold_rec:
            run_summary["results"]["high_grade_threshold_from_val"] = val_threshold_rec
            for split_name in ("train", "val", "test"):
                if split_name not in split_probs or split_name not in split_y_true:
                    continue
                prob = split_probs[split_name]
                y_true = split_y_true[split_name]
                y_pred_thr = _apply_any_high_threshold_override(
                    prob=prob,
                    threshold=float(val_threshold_rec["threshold"]),
                    high_grade_classes=high_grade_classes,
                )
                split_dir = split_dirs[split_name] / "high_grade_thresholded"
                split_dir.mkdir(parents=True, exist_ok=True)
                save_classification_tables(
                    y_true=y_true,
                    y_pred=y_pred_thr,
                    prob=prob,
                    class_names=final_class_names,
                    output_dir=str(split_dir),
                )
                save_grade_class_diagnostics(
                    y_true=y_true,
                    y_pred=y_pred_thr,
                    prob=prob,
                    class_names=final_class_names,
                    output_dir=str(split_dir),
                    pred_prefix="high_grade_thresholded_pred",
                )
                save_hierarchical_classification_diagnostics(
                    y_true=y_true,
                    y_pred=y_pred_thr,
                    prob=prob,
                    class_names=final_class_names,
                    output_dir=str(split_dir / "hierarchical"),
                )
                hg_thr = compute_high_grade_metrics(
                    y_true=y_true,
                    prob=prob,
                    class_names=final_class_names,
                    high_grade_classes=high_grade_classes,
                    threshold=float(val_threshold_rec["threshold"]),
                )
                run_summary["results"][f"{split_name}_high_grade_thresholded_metrics"] = {
                    "threshold_source": "val_F1",
                    **{k: v for k, v in hg_thr.items() if k not in ("high_grade_classes",)},
                }

        try:
            env_missing_paths = save_env_patch_missingness_diagnostics(
                points_df=cascade_df,
                env_grid_path=str(env_grid_path),
                feature_names=selected_env_variables,
                patch_size=(int(patch_size[0]), int(patch_size[1])),
                split_indices={
                    "train": train_idx,
                    "val": val_idx,
                    "test": test_idx,
                },
                output_dir=str(Path(output_dirs["tables"]) / "env_missingness"),
                label_col=final_label_col,
                pred_tables=split_pred_tables,
            )
            if env_missing_paths:
                run_summary["artifacts"]["paths"]["env_patch_missingness"] = env_missing_paths
        except Exception as e_env_miss:
            logger.warning(f"Cascade: failed to write environmental missingness diagnostics: {e_env_miss}")

        try:
            b2_sep_paths = save_b2_separability_diagnostics(
                points_df=cascade_df,
                output_dir=str(Path(output_dirs["tables"]) / "b2_separability"),
                env_grid_path=str(env_grid_path),
                feature_names=selected_env_variables,
            )
            if b2_sep_paths:
                run_summary["artifacts"]["paths"]["b2_separability"] = b2_sep_paths
        except Exception as e_b2_sep:
            logger.warning(f"Cascade: failed to write B2 separability diagnostics: {e_b2_sep}")

        try:
            b2_run_dir = run_base_dir / "class2_grade_model"
            b2_thr_paths = _write_b2_threshold_stability_report(
                class2_run_dir=b2_run_dir,
                output_dir=Path(output_dirs["tables"]) / "b2_thresholds",
            )
            if b2_thr_paths:
                run_summary["artifacts"]["paths"]["b2_threshold_stability"] = b2_thr_paths
        except Exception as e_b2_thr:
            logger.warning(f"Cascade: failed to write B2 threshold stability diagnostics: {e_b2_thr}")

        main_artifacts_path = ((summary_main.get("artifacts", {}) or {}).get("paths", {}) or {}).get("training_artifacts")
        b1_artifacts_path = None if is_exploration_4class else ((summary_b1.get("artifacts", {}) or {}).get("paths", {}) or {}).get("training_artifacts")
        b2_artifacts_path = ((summary_b2.get("artifacts", {}) or {}).get("paths", {}) or {}).get("training_artifacts")
        if not main_artifacts_path or not b2_artifacts_path or ((not is_exploration_4class) and not b1_artifacts_path):
            raise RuntimeError("Cascade training failed to resolve one or more stage training_artifacts.json paths.")

        cascade_artifacts = CascadeTrainingArtifacts(
            artifact_type=CASCADE_EXPLORATION_4CLASS_TASK_TYPE if is_exploration_4class else CASCADE_GRADE_TASK_TYPE,
            final_label_col=final_label_col,
            final_class_names=list(final_class_names),
            final_num_classes=int(final_num_classes),
            combination_rule=(
                "P0=P(main=carbonate); P1=P(main=class1); "
                "P2=P(main=class2)*P(low|class2); P3=P(main=class2)*P(high|class2)"
                if is_exploration_4class
                else "P0=P(main=carbonate); P1=P(main=class1)*P(low|class1); P2=P(main=class1)*P(high|class1); P3=P(main=class2)*P(low|class2); P4=P(main=class2)*P(high|class2)"
            ),
            stage_main=CascadeStageArtifacts(
                stage_name="stage_main",
                training_artifacts_path=str(main_artifacts_path),
                model_path=str((summary_main.get("results", {}) or {}).get("model_path") or summary_main.get("results", {}).get("plot_paths", {}).get("model_path") or (Path(main_artifacts_path).parent / "final_trained_model.pth")),
                classification_label_col="main_class_id",
                class_names=["carbonate_background", "noncarbonate_metal_enriched", "noncarbonate_detrital_background"],
                num_classes=3,
            ),
            stage_class1_grade=(
                None
                if is_exploration_4class
                else CascadeStageArtifacts(
                    stage_name="stage_class1_grade",
                    training_artifacts_path=str(b1_artifacts_path),
                    model_path=str((summary_b1.get("results", {}) or {}).get("model_path") or (Path(str(b1_artifacts_path)).parent / "final_trained_model.pth")),
                    classification_label_col="class1_grade_label",
                    class_names=["class1_low_grade", "class1_high_grade"],
                    num_classes=2,
                )
            ),
            stage_class2_grade=CascadeStageArtifacts(
                stage_name="stage_class2_grade",
                training_artifacts_path=str(b2_artifacts_path),
                model_path=str((summary_b2.get("results", {}) or {}).get("model_path") or (Path(b2_artifacts_path).parent / "final_trained_model.pth")),
                classification_label_col="class2_grade_label",
                class_names=["class2_low_grade", "class2_high_grade"],
                num_classes=2,
            ),
        )
        cascade_artifacts_path = Path(output_dirs["models"]) / "cascade_training_artifacts.json"
        cascade_artifacts_path.write_text(cascade_artifacts.model_dump_json(indent=4), encoding="utf-8")

        run_summary["artifacts"]["paths"]["cascade_training_artifacts"] = str(cascade_artifacts_path)
        run_summary["artifacts"]["paths"]["stage_main_run"] = str(Path(main_artifacts_path).resolve().parents[1])
        if b1_artifacts_path:
            run_summary["artifacts"]["paths"]["stage_class1_run"] = str(Path(b1_artifacts_path).resolve().parents[1])
        run_summary["artifacts"]["paths"]["stage_class2_run"] = str(Path(b2_artifacts_path).resolve().parents[1])
        stage_runs = {
            "stage_main": summary_main.get("run_id"),
            "stage_class2_grade": summary_b2.get("run_id"),
        }
        if not is_exploration_4class:
            stage_runs["stage_class1_grade"] = summary_b1.get("run_id")
        run_summary["results"]["stage_runs"] = stage_runs
        run_summary["results"]["high_grade_classes"] = high_grade_classes
        run_summary["MODEL_CONFIG"] = {"artifact_type": requested_task_type}
        run_summary["NORMALIZATION_PARAMS"] = {"artifact_type": requested_task_type}
        run_summary["status"] = "completed"
        return True, run_summary
    except Exception as e:
        error_message = f"Cascade pipeline failed: {e}"
        logger.opt(exception=True).error(error_message)
        run_summary["status"] = "error"
        run_summary["error_message"] = error_message
        return False, run_summary
    finally:
        run_summary["timestamp_end"] = datetime.now().isoformat()
        try:
            run_summary["duration_seconds"] = (
                datetime.fromisoformat(run_summary["timestamp_end"]) - datetime.fromisoformat(run_summary["timestamp_start"])
            ).total_seconds()
        except Exception:
            run_summary["duration_seconds"] = time.time() - start_time
        try:
            pipeline_config_path = run_base_dir / "pipeline_config.json"
            write_json(pipeline_config_path, run_summary.get("config", {}))
            run_summary["artifacts"]["paths"]["pipeline_config"] = str(pipeline_config_path)
            report_path = run_base_dir / "RUN_REPORT.md"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(make_training_run_report_md(run_summary))
            run_summary["artifacts"]["paths"]["run_report"] = str(report_path)
            manifest_path = run_base_dir / "results_manifest.json"
            write_json(manifest_path, build_paths_manifest(run_base_dir, run_summary))
            run_summary["artifacts"]["paths"]["results_manifest"] = str(manifest_path)
        except Exception as e_aux:
            logger.warning(f"Cascade pipeline: failed to write run outputs: {e_aux}")
        try:
            save_run_summary(run_summary["run_id"], run_summary)
        except Exception as e_save:
            logger.critical(f"CRITICAL: Failed to save cascade run summary: {e_save}", exc_info=True)


def _normalize_lon_to_180(lon_deg: np.ndarray) -> np.ndarray:
    lon = np.asarray(lon_deg, dtype=float)
    return ((lon + 180.0) % 360.0) - 180.0


def _compute_spatial_block_ids(rey_df: pd.DataFrame, block_size_deg: float) -> np.ndarray:
    """
    Compute a stable spatial-block id per sample using a regular lat/lon degree grid.

    Notes
    - Intended for leakage-resistant splitting, not for geostatistical modeling.
    - Uses a fixed origin at (-180, -90) to keep ids stable across runs.
    - lon is normalized to [-180, 180).
    """
    if block_size_deg is None or not np.isfinite(float(block_size_deg)) or float(block_size_deg) <= 0:
        raise ValueError(f"block_size_deg must be positive; got {block_size_deg}")

    if "lon" not in rey_df.columns or "lat" not in rey_df.columns:
        raise KeyError("rey_df must contain 'lon' and 'lat' columns for spatial block splitting.")

    lon = _normalize_lon_to_180(rey_df["lon"].to_numpy(dtype=float))
    lat = rey_df["lat"].to_numpy(dtype=float)


    lon0 = -180.0
    lat0 = -90.0
    bs = float(block_size_deg)

    bx = np.floor((lon - lon0) / bs).astype(np.int64)
    by = np.floor((lat - lat0) / bs).astype(np.int64)

    n_lon_blocks = int(np.ceil(360.0 / bs))
    block_id = by * n_lon_blocks + bx


    block_id = np.where(np.isfinite(block_id), block_id, -1)
    return block_id.astype(np.int64)


def _make_class_balanced_sampler(
    *,
    train_indices: list[int],
    labels_all: np.ndarray,
    n_classes: int,
    power: float = 1.0,
    max_factor: float = 10.0,
    seed: int = 42,
) -> WeightedSubsetRandomSampler:
    """
    Build a sampler that oversamples minority classes (on the training subset).

    Weights are proportional to (mean_count / count(class)) ** power, optionally clipped.
    """
    idx = np.asarray(train_indices, dtype=int)
    y = np.asarray(labels_all, dtype=int).reshape(-1)
    if idx.size == 0:
        raise ValueError("train_indices cannot be empty for class-balanced sampling.")
    if y.shape[0] <= int(idx.max(initial=-1)):
        raise ValueError("labels_all length is smaller than max(train_indices).")
    y_tr = y[idx]
    n_classes = int(max(2, n_classes))
    counts = np.bincount(y_tr, minlength=n_classes).astype(float)
    denom = np.where(counts > 0, counts, 1.0)
    mean_count = float(np.mean(denom))
    raw = (mean_count / denom) ** float(max(0.0, power))
    raw = np.where(counts > 0, raw, 0.0)
    if np.isfinite(max_factor) and float(max_factor) > 0:
        raw = np.clip(raw, 0.0, float(max_factor))
    weights = raw[y_tr]
    if not np.isfinite(weights).all() or float(np.sum(weights)) <= 0:
        raise ValueError("Class-balanced weights are invalid (check labels/counts).")
    return WeightedSubsetRandomSampler(
        indices=idx.tolist(),
        weights=weights.tolist(),
        num_samples=len(idx),
        replacement=True,
        seed=int(seed),
    )


def _make_block_and_class_balanced_sampler(
    *,
    train_indices: list[int],
    block_ids_all: np.ndarray,
    labels_all: np.ndarray,
    n_classes: int,
    block_alpha: float,
    block_min_count: int,
    class_power: float,
    class_max_factor: float,
    seed: int,
) -> WeightedSubsetRandomSampler:
    """
    Combine block-balanced and class-balanced weights multiplicatively.

    This is a pragmatic compromise: reduce dominance of large spatial/core groups
    while still oversampling rare classes.
    """
    idx = np.asarray(train_indices, dtype=int)
    b_all = np.asarray(block_ids_all, dtype=np.int64).reshape(-1)
    y_all = np.asarray(labels_all, dtype=int).reshape(-1)
    if idx.size == 0:
        raise ValueError("train_indices cannot be empty for combined sampling.")
    if b_all.shape[0] <= int(idx.max(initial=-1)) or y_all.shape[0] <= int(idx.max(initial=-1)):
        raise ValueError("block_ids_all/labels_all length mismatch for combined sampling.")

    b_tr = b_all[idx]
    unique_b, b_counts = np.unique(b_tr, return_counts=True)
    b_count_map = dict(zip(unique_b.tolist(), b_counts.tolist()))
    alpha = float(max(0.0, block_alpha))
    min_c = int(block_min_count) if int(block_min_count) > 0 else 1
    w_block = []
    for b in b_tr.tolist():
        c = max(int(b_count_map.get(int(b), 1)), min_c)
        w = (1.0 / float(c)) ** alpha if alpha > 0 else 1.0
        w_block.append(w)
    w_block = np.asarray(w_block, dtype=float)

    y_tr = y_all[idx]
    n_classes = int(max(2, n_classes))
    y_counts = np.bincount(y_tr, minlength=n_classes).astype(float)
    denom = np.where(y_counts > 0, y_counts, 1.0)
    mean_count = float(np.mean(denom))
    w_class_per = (mean_count / denom) ** float(max(0.0, class_power))
    w_class_per = np.where(y_counts > 0, w_class_per, 0.0)
    if np.isfinite(class_max_factor) and float(class_max_factor) > 0:
        w_class_per = np.clip(w_class_per, 0.0, float(class_max_factor))
    w_class = w_class_per[y_tr]

    weights = w_block * w_class
    if not np.isfinite(weights).all() or float(np.sum(weights)) <= 0:
        raise ValueError("Combined block+class weights are invalid.")
    return WeightedSubsetRandomSampler(
        indices=idx.tolist(),
        weights=weights.tolist(),
        num_samples=len(idx),
        replacement=True,
        seed=int(seed),
    )


def _group_train_val_test_split(
    dataset_size: int,
    groups: np.ndarray,
    test_split_ratio: float,
    validation_split_ratio: float,
    random_state: int,
):
    """
    Leakage-resistant split by groups.

    Returns (train_indices, val_indices, test_indices).
    """
    if dataset_size <= 0:
        raise ValueError("dataset_size must be > 0")
    if groups is None:
        raise ValueError("groups is required for group-based splitting")

    groups = np.asarray(groups)
    if groups.shape[0] != dataset_size:
        raise ValueError(f"groups length mismatch: {groups.shape[0]} vs dataset_size={dataset_size}")

    test_ratio = float(test_split_ratio)
    val_ratio = float(validation_split_ratio)
    if not (0.0 < test_ratio < 1.0):
        raise ValueError(f"Invalid test_split_ratio={test_ratio}")
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"Invalid validation_split_ratio={val_ratio}")

    try:
        from sklearn.model_selection import GroupShuffleSplit
    except Exception as e:
        raise RuntimeError(
            "Group-based splitting requires scikit-learn. "
            "Install scikit-learn or switch to random_point split strategy."
        ) from e

    all_idx = np.arange(dataset_size, dtype=int)

    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=int(random_state))
    train_val_pos, test_pos = next(gss_test.split(all_idx, groups=groups))


    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=int(random_state) + 1)
    train_pos_rel, val_pos_rel = next(gss_val.split(train_val_pos, groups=groups[train_val_pos]))

    train_pos = train_val_pos[train_pos_rel]
    val_pos = train_val_pos[val_pos_rel]

    return train_pos.tolist(), val_pos.tolist(), test_pos.tolist()


def _stratified_group_train_val_test_split(
    *,
    dataset_size: int,
    groups: np.ndarray,
    targets: np.ndarray,
    test_split_ratio: float,
    validation_split_ratio: float,
    random_state: int,
    n_bins: int,
    statistic: str = "median",
):
    """
    Group-aware split with stratification on a per-group target statistic.

    This reduces split variance when groups have very different target distributions
    (e.g., some cores dominated by high-REY intervals).

    Falls back to plain group split if stratification is not feasible.
    """
    if n_bins is None or int(n_bins) <= 1:
        return _group_train_val_test_split(
            dataset_size=dataset_size,
            groups=groups,
            test_split_ratio=test_split_ratio,
            validation_split_ratio=validation_split_ratio,
            random_state=random_state,
        )

    groups = np.asarray(groups)
    targets = np.asarray(targets, dtype=float)
    if groups.shape[0] != dataset_size or targets.shape[0] != dataset_size:
        raise ValueError("groups/targets length mismatch for stratified group split.")

    try:
        from sklearn.model_selection import train_test_split
    except Exception as e:
        raise RuntimeError(
            "Stratified group split requires scikit-learn. "
            "Install scikit-learn or disable group_split_stratify_bins."
        ) from e


    df = pd.DataFrame({"g": groups, "y": targets})
    stat = str(statistic or "median").lower().strip()
    if stat == "mean":
        group_stats = df.groupby("g", sort=False)["y"].mean()
    elif stat == "sum":
        group_stats = df.groupby("g", sort=False)["y"].sum()
    elif stat == "max":
        group_stats = df.groupby("g", sort=False)["y"].max()
    else:
        group_stats = df.groupby("g", sort=False)["y"].median()
    unique_groups = group_stats.index.to_numpy()
    stat_values = group_stats.to_numpy(dtype=float)


    if unique_groups.size < 6:
        return _group_train_val_test_split(
            dataset_size=dataset_size,
            groups=groups,
            test_split_ratio=test_split_ratio,
            validation_split_ratio=validation_split_ratio,
            random_state=random_state,
        )


    bins = int(n_bins)
    try:
        bin_labels = pd.qcut(stat_values, q=min(bins, unique_groups.size), duplicates="drop").codes
    except Exception:
        return _group_train_val_test_split(
            dataset_size=dataset_size,
            groups=groups,
            test_split_ratio=test_split_ratio,
            validation_split_ratio=validation_split_ratio,
            random_state=random_state,
        )


    if np.unique(bin_labels).size < 2:
        return _group_train_val_test_split(
            dataset_size=dataset_size,
            groups=groups,
            test_split_ratio=test_split_ratio,
            validation_split_ratio=validation_split_ratio,
            random_state=random_state,
        )


    g_train_val, g_test = train_test_split(
        unique_groups,
        test_size=float(test_split_ratio),
        random_state=int(random_state),
        stratify=bin_labels,
    )


    tv_mask = np.isin(unique_groups, g_train_val)
    tv_bins = bin_labels[tv_mask]
    try:
        g_train, g_val = train_test_split(
            g_train_val,
            test_size=float(validation_split_ratio),
            random_state=int(random_state) + 1,
            stratify=tv_bins if np.unique(tv_bins).size >= 2 else None,
        )
    except Exception:

        g_train, g_val = train_test_split(
            g_train_val,
            test_size=float(validation_split_ratio),
            random_state=int(random_state) + 1,
            shuffle=True,
        )


    all_idx = np.arange(dataset_size, dtype=int)
    train_idx = all_idx[np.isin(groups, g_train)]
    val_idx = all_idx[np.isin(groups, g_val)]
    test_idx = all_idx[np.isin(groups, g_test)]


    if train_idx.size == 0 or val_idx.size == 0 or test_idx.size == 0:
        return _group_train_val_test_split(
            dataset_size=dataset_size,
            groups=groups,
            test_split_ratio=test_split_ratio,
            validation_split_ratio=validation_split_ratio,
            random_state=random_state,
        )

    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


def _spatial_balance_labels(rey_df: pd.DataFrame | None, block_deg: float = 10.0) -> np.ndarray | None:
    """Coarse spatial labels for split balancing, computed only from lon/lat."""
    if rey_df is None or "lon" not in rey_df.columns or "lat" not in rey_df.columns:
        return None
    try:
        lon = pd.to_numeric(rey_df["lon"], errors="coerce").to_numpy(dtype=float)
        lat = pd.to_numeric(rey_df["lat"], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(lon) & np.isfinite(lat)
        if int(ok.sum()) < 2:
            return None
        block = max(float(block_deg or 10.0), 1.0)
        lon_norm = np.mod(lon, 360.0)
        lon_bin = np.floor(lon_norm / block).astype(int)
        lat_bin = np.floor((lat + 90.0) / block).astype(int)
        labels = np.array([f"{a}:{b}" if valid else "missing" for a, b, valid in zip(lon_bin, lat_bin, ok)], dtype=object)
        return labels
    except Exception:
        return None


def _categorical_l1(values: np.ndarray, ref_values: np.ndarray) -> float:
    values = np.asarray(values)
    ref_values = np.asarray(ref_values)
    cats = pd.Index(pd.unique(ref_values))
    if cats.size <= 1:
        return 0.0
    ref = pd.Series(ref_values).value_counts(normalize=True)
    cur = pd.Series(values).value_counts(normalize=True) if values.size else pd.Series(dtype=float)
    return float(sum(abs(float(cur.get(c, 0.0)) - float(ref.get(c, 0.0))) for c in cats))


def _score_balanced_group_split(
    *,
    train_indices: list[int] | np.ndarray,
    val_indices: list[int] | np.ndarray,
    test_indices: list[int] | np.ndarray,
    groups: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    spatial_labels: np.ndarray | None,
    test_split_ratio: float,
    validation_split_ratio: float,
) -> dict[str, float]:
    """Lower score is better for class/spatial/sample balanced group splits."""
    tr = np.asarray(train_indices, dtype=int)
    va = np.asarray(val_indices, dtype=int)
    te = np.asarray(test_indices, dtype=int)
    n = max(int(labels.shape[0]), 1)
    test_target = float(test_split_ratio)
    val_target = float(1.0 - test_target) * float(validation_split_ratio)
    train_target = max(0.0, 1.0 - test_target - val_target)

    split_items = [("train", tr, train_target), ("val", va, val_target), ("test", te, test_target)]
    total_counts = np.bincount(labels.astype(int), minlength=int(n_classes)).astype(float)
    total_props = total_counts / max(float(total_counts.sum()), 1.0)
    total_groups = max(int(pd.Series(groups).nunique()), 1)

    sample_ratio_error = 0.0
    group_ratio_error = 0.0
    class_l1_error = 0.0
    class_count_error = 0.0
    spatial_l1_error = 0.0
    dominance_penalty = 0.0

    for split_name, idx, target_ratio in split_items:
        split_weight = 1.0 if split_name == "train" else 1.4
        sample_ratio_error += abs(float(idx.size) / float(n) - float(target_ratio))
        split_groups = np.unique(groups[idx]) if idx.size else np.asarray([], dtype=object)
        group_ratio_error += abs(float(split_groups.size) / float(total_groups) - float(target_ratio))

        counts = np.bincount(labels[idx].astype(int), minlength=int(n_classes)).astype(float) if idx.size else np.zeros(int(n_classes), dtype=float)
        props = counts / max(float(counts.sum()), 1.0)
        class_l1_error += split_weight * float(np.abs(props - total_props).sum())
        expected = total_counts * float(target_ratio)
        class_count_error += split_weight * float(np.mean(np.abs(counts - expected) / np.maximum(total_counts, 1.0)))

        if spatial_labels is not None and idx.size:
            spatial_l1_error += split_weight * _categorical_l1(spatial_labels[idx], spatial_labels)

        if idx.size:
            group_counts = pd.Series(groups[idx]).value_counts()
            dominance_penalty += split_weight * max(0.0, float(group_counts.iloc[0]) / float(idx.size) - 0.35)

    score = (
        3.0 * class_l1_error
        + 2.0 * class_count_error
        + 1.5 * sample_ratio_error
        + 1.0 * group_ratio_error
        + 0.8 * spatial_l1_error
        + 1.2 * dominance_penalty
    )
    return {
        "score": float(score),
        "class_l1_error": float(class_l1_error),
        "class_count_error": float(class_count_error),
        "sample_ratio_error": float(sample_ratio_error),
        "group_ratio_error": float(group_ratio_error),
        "spatial_l1_error": float(spatial_l1_error),
        "dominance_penalty": float(dominance_penalty),
    }


def _run_spatial_block_cv_diagnostic(
    *,
    dataset: REYPointDataset,
    groups: np.ndarray,
    output_root: Path,
    hyperparams: dict,
    test_split_ratio: float,
    validation_split_ratio: float,
    random_state: int,
    max_folds_to_run: int,
    n_splits: int,
):
    """
    Run a spatial-block GroupKFold diagnostic CV.

    This is intended to assess spatial generalization. It runs training for up to
    `max_folds_to_run` folds and writes fold summaries under `output_root/cv/`.
    """
    try:
        from sklearn.model_selection import GroupKFold, GroupShuffleSplit
    except Exception as e:
        raise RuntimeError("Spatial block CV requires scikit-learn.") from e

    n = len(dataset)
    if n <= 0:
        raise ValueError("Empty dataset for CV.")
    groups = np.asarray(groups)
    if groups.shape[0] != n:
        raise ValueError("groups length mismatch for CV.")

    n_splits = int(n_splits)
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2 for CV.")
    max_folds_to_run = int(max_folds_to_run)
    if max_folds_to_run <= 0:
        max_folds_to_run = n_splits


    gkf = GroupKFold(n_splits=n_splits)
    all_idx = np.arange(n, dtype=int)

    cv_dir = output_root / "cv"
    cv_dir.mkdir(parents=True, exist_ok=True)

    fold_results = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(hyperparams.get("batch_size", 128))
    num_workers = int(hyperparams.get("dataloader_num_workers", hyperparams.get("num_workers", 0)) or 0)
    pin_memory = torch.cuda.is_available()


    gss_val = GroupShuffleSplit(n_splits=1, test_size=float(validation_split_ratio), random_state=int(random_state) + 123)

    train_sampling_strategy = str(hyperparams.get("train_sampling_strategy", "random") or "random").lower()
    block_balance_alpha = float(hyperparams.get("block_balance_alpha", 1.0) or 1.0)
    block_balance_min_count = int(hyperparams.get("block_balance_min_count", 5) or 5)

    class_balance_power = float(hyperparams.get("class_balance_power", 0.7) or 0.7)
    class_balance_max_factor = float(hyperparams.get("class_balance_max_factor", 8.0) or 8.0)
    task_type = str(hyperparams.get("task_type", "regression") or "regression").lower()
    label_col = hyperparams.get("classification_label_col", None)
    n_classes = int(hyperparams.get("num_classes", 0) or 0)

    labels_all = None
    if task_type == "binary_classification":
        thr_ppm = float(hyperparams.get("classification_threshold_ppm", 1000.0) or 1000.0)
        if getattr(dataset, "rey_data", None) is not None and "rey" in dataset.rey_data.columns:
            labels_all = (dataset.rey_data["rey"].astype(float).to_numpy() >= thr_ppm).astype(int)
            n_classes = 2
    elif task_type == "multiclass_classification":
        if getattr(dataset, "rey_data", None) is not None and label_col and label_col in dataset.rey_data.columns:
            labels_all = dataset.rey_data[label_col].astype(int).to_numpy()
            if n_classes < 2 and labels_all.size:
                n_classes = int(np.max(labels_all)) + 1

    for fold_idx, (train_val_idx, test_idx) in enumerate(gkf.split(all_idx, groups=groups), start=1):
        if fold_idx > max_folds_to_run:
            break


        tr_rel, va_rel = next(gss_val.split(train_val_idx, groups=groups[train_val_idx]))
        train_idx = train_val_idx[tr_rel]
        val_idx = train_val_idx[va_rel]

        fold_dir = cv_dir / f"fold_{fold_idx:02d}"
        fold_dirs = {
            "base": str(fold_dir),
            "logs": str(fold_dir / "logs"),
            "plots": str(fold_dir / "plots"),
            "models": str(fold_dir / "models"),
            "reports": str(fold_dir / "reports"),
            "splits": str(fold_dir / "splits"),
            "diagnostics": str(fold_dir / "plots" / "diagnostics"),
            "scatter": str(fold_dir / "plots" / "scatter"),
            "plotly_interactive": str(fold_dir / "plots" / "plotly_interactive"),
        }
        for p in fold_dirs.values():
            Path(p).mkdir(parents=True, exist_ok=True)

        if train_sampling_strategy == "block_balanced":
            train_sampler = make_block_balanced_sampler(
                train_indices=train_idx.tolist(),
                block_ids=groups,
                config=BlockBalancedSamplingConfig(
                    alpha=block_balance_alpha,
                    min_count=block_balance_min_count,
                    num_samples=len(train_idx),
                    replacement=True,
                    seed=int(random_state) + int(fold_idx),
                ),
            )
        elif train_sampling_strategy == "class_balanced" and labels_all is not None and task_type in ("binary_classification", "multiclass_classification"):
            train_sampler = _make_class_balanced_sampler(
                train_indices=train_idx.tolist(),
                labels_all=labels_all,
                n_classes=n_classes,
                power=class_balance_power,
                max_factor=class_balance_max_factor,
                seed=int(random_state) + int(fold_idx),
            )
        elif train_sampling_strategy == "block_and_class_balanced" and labels_all is not None and task_type in ("binary_classification", "multiclass_classification"):
            train_sampler = _make_block_and_class_balanced_sampler(
                train_indices=train_idx.tolist(),
                block_ids_all=groups,
                labels_all=labels_all,
                n_classes=n_classes,
                block_alpha=block_balance_alpha,
                block_min_count=block_balance_min_count,
                class_power=class_balance_power,
                class_max_factor=class_balance_max_factor,
                seed=int(random_state) + int(fold_idx),
            )
        else:
            train_sampler = SubsetRandomSampler(train_idx.tolist())
        val_sampler = SubsetRandomSampler(val_idx.tolist())
        test_sampler = SubsetRandomSampler(test_idx.tolist())


        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers, pin_memory=pin_memory)

        add_missingness_channels = bool(hyperparams.get("add_env_missingness_channels", False))
        env_channel_count = int(len(getattr(dataset, "feature_names", []))) * (2 if add_missingness_channels else 1)
        trainer = ModelTrainer(
            model_config=hyperparams,
            input_channels=env_channel_count + 1 + (4 if bool(hyperparams.get("add_coord_channels", False)) else 0),
            device=device,
            output_dirs=fold_dirs,
            stop_flag=None,
            progress_callback=None,
            history_callback=None,
        )

        success, artifacts = trainer.train_and_evaluate(
            train_loader,
            val_loader,
            test_loader,
            epochs=int(hyperparams.get("epochs", 200)),
            early_stopping_patience=int(hyperparams.get("early_stopping_patience", 20)),
            full_dataset_for_denorm=dataset,
        )

        fold_summary = {
            "fold": int(fold_idx),
            "success": bool(success),
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "n_blocks_train": int(len(np.unique(groups[train_idx]))),
            "n_blocks_test": int(len(np.unique(groups[test_idx]))),
            "metrics": artifacts.get("test_metrics", {}) if isinstance(artifacts, dict) else {},
            "best_epoch": artifacts.get("best_epoch", None) if isinstance(artifacts, dict) else None,
            "best_val_loss": artifacts.get("best_val_loss", None) if isinstance(artifacts, dict) else None,
            "model_path": artifacts.get("model_path", None) if isinstance(artifacts, dict) else None,
        }
        fold_results.append(fold_summary)

        try:
            write_json(str(fold_dir / "cv_fold_summary.json"), fold_summary)
        except Exception:
            pass


    def _get_metric(key: str):
        vals = []
        for fr in fold_results:
            if not fr.get("success"):
                continue
            m = fr.get("metrics") or {}
            v = m.get(key, None)
            if v is None:
                continue
            try:
                vals.append(float(v))
            except Exception:
                continue
        return np.asarray(vals, dtype=float)

    task_metrics: list[str] = []
    if task_type == "regression":
        task_metrics = ["r2", "mae", "mse"]
    elif task_type == "binary_classification":
        task_metrics = ["roc_auc", "accuracy", "balanced_accuracy", "log_loss", "f1"]
    elif task_type == "multiclass_classification":
        task_metrics = ["accuracy", "balanced_accuracy", "macro_f1", "log_loss", "class1_ap"]

    cv_summary: dict[str, Any] = {
        "n_splits_requested": int(n_splits),
        "max_folds_to_run": int(max_folds_to_run),
        "n_folds_ran": int(len(fold_results)),
        "n_folds_success": int(sum(1 for fr in fold_results if fr.get("success"))),
        "task_type": str(task_type),
        "metrics": {},
        "folds": fold_results,
    }

    for key in task_metrics:
        vals = _get_metric(str(key))
        if vals.size:
            cv_summary["metrics"][f"{key}_mean"] = float(np.nanmean(vals))
            cv_summary["metrics"][f"{key}_std"] = float(np.nanstd(vals))
        else:
            cv_summary["metrics"][f"{key}_mean"] = None
            cv_summary["metrics"][f"{key}_std"] = None

    try:
        write_json(str(cv_dir / "cv_summary.json"), cv_summary)
    except Exception:
        pass

    return cv_summary


def _compute_coord_branch_diagnostics(
    *,
    model: torch.nn.Module,
    dataset: REYPointDataset,
    test_indices,
    device: torch.device,
    output_dir: str,
    batch_size: int = 256,
):
    """
    Diagnostics for the optional coordinate trend term (additive bias).

    Produces:
      - Metrics with coord branch enabled (as-is)
      - Metrics with coord branch effectively disabled (weight=0)
      - Distribution of coord_bias (normalized space)
      - CSV of per-sample coord_bias and residuals (test set)
    """
    try:
        from torch.utils.data.dataset import Subset
    except Exception as e:
        logger.warning(f"Coord-branch diagnostics: failed to import torch Subset: {e}")
        return None

    try:
        test_indices_list = [int(i) for i in list(test_indices)]
    except Exception:
        test_indices_list = [int(i) for i in test_indices]
    if not test_indices_list:
        return None

    os.makedirs(output_dir, exist_ok=True)

    subset = Subset(dataset, test_indices_list)
    loader = DataLoader(
        subset,
        batch_size=int(batch_size) if batch_size is not None else 256,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model.eval()
    preds_norm_all = []
    preds_norm_no_coord_all = []
    coord_bias_all = []
    trues_norm_all = []

    has_coord_weight = hasattr(model, "coord_branch_weight")
    orig_coord_weight = float(getattr(model, "coord_branch_weight", 0.0)) if has_coord_weight else 0.0

    with torch.no_grad():
        for combined_patches, target_reys_norm, _ in loader:
            combined_patches = combined_patches.to(device)
            target_reys_norm = target_reys_norm.to(device)


            with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
                pred_norm, comps = model(combined_patches, return_components=True)
            preds_norm_all.append(pred_norm.detach().cpu().numpy().reshape(-1))
            trues_norm_all.append(target_reys_norm.detach().cpu().numpy().reshape(-1))

            cb = comps.get("coord_bias", None) if isinstance(comps, dict) else None
            if cb is not None:
                coord_bias_all.append(cb.detach().cpu().numpy().reshape(-1))


            if has_coord_weight:
                setattr(model, "coord_branch_weight", 0.0)
            try:
                with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
                    pred_norm_no_coord = model(combined_patches)
                preds_norm_no_coord_all.append(pred_norm_no_coord.detach().cpu().numpy().reshape(-1))
            finally:
                if has_coord_weight:
                    setattr(model, "coord_branch_weight", orig_coord_weight)

    if not preds_norm_all or not trues_norm_all:
        return None

    preds_norm = np.concatenate(preds_norm_all)
    trues_norm = np.concatenate(trues_norm_all)
    preds_norm_no_coord = np.concatenate(preds_norm_no_coord_all) if preds_norm_no_coord_all else None
    coord_bias = np.concatenate(coord_bias_all) if coord_bias_all else None

    preds_phys = np.clip(np.asarray(dataset._denormalize_rey(preds_norm), dtype=float), 0.0, None)
    trues_phys = np.asarray(dataset._denormalize_rey(trues_norm), dtype=float)

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    out = {
        "coord_branch_weight": orig_coord_weight,
        "n_test": int(len(trues_phys)),
        "mae": float(mean_absolute_error(trues_phys, preds_phys)),
        "mse": float(mean_squared_error(trues_phys, preds_phys)),
        "r2": float(r2_score(trues_phys, preds_phys)),
    }

    if preds_norm_no_coord is not None:
        preds_phys_no_coord = np.clip(np.asarray(dataset._denormalize_rey(preds_norm_no_coord), dtype=float), 0.0, None)
        out["mae_no_coord"] = float(mean_absolute_error(trues_phys, preds_phys_no_coord))
        out["mse_no_coord"] = float(mean_squared_error(trues_phys, preds_phys_no_coord))
        out["r2_no_coord"] = float(r2_score(trues_phys, preds_phys_no_coord))
        out["delta_r2_coord_minus_no_coord"] = float(out["r2"] - out["r2_no_coord"])
        out["delta_mae_coord_minus_no_coord"] = float(out["mae"] - out["mae_no_coord"])


    try:
        meta = dataset.rey_data.iloc[test_indices_list].copy()
        meta = meta.reset_index(drop=False).rename(columns={"index": "source_index"})
        meta_out = pd.DataFrame(
            {
                "dataset_index": np.asarray(test_indices_list, dtype=int),
                "lon": meta.get("lon").astype(float).to_numpy() if "lon" in meta.columns else np.full(len(meta), np.nan),
                "lat": meta.get("lat").astype(float).to_numpy() if "lat" in meta.columns else np.full(len(meta), np.nan),
                "age_ka_proxy": meta.get("age_ka_proxy").astype(float).to_numpy() if "age_ka_proxy" in meta.columns else np.full(len(meta), np.nan),
                "rey_true": trues_phys,
                "rey_pred": preds_phys,
                "residual": trues_phys - preds_phys,
                "coord_bias_norm": coord_bias if coord_bias is not None and coord_bias.size == len(trues_phys) else np.full(len(trues_phys), np.nan),
            }
        )
        csv_path = os.path.join(output_dir, "coord_branch_test_diagnostics.csv")
        meta_out.to_csv(csv_path, index=False)
        out["csv"] = csv_path
    except Exception as e_csv:
        logger.warning(f"Coord-branch diagnostics: failed to write CSV: {e_csv}")


    try:
        if coord_bias is not None and np.isfinite(coord_bias).any():
            plt.figure(figsize=(8, 4))
            cb = coord_bias[np.isfinite(coord_bias)]
            plt.hist(cb, bins=40, color="#4C72B0", alpha=0.85)
            plt.title("Coordinate trend term (normalized) on test set")
            plt.xlabel("coord_bias (normalized)")
            plt.ylabel("Count")
            plt.tight_layout()
            png_path = os.path.join(output_dir, "coord_branch_bias_hist.png")
            plt.savefig(png_path, dpi=200)
            plt.close("all")
            out["bias_hist"] = png_path
    except Exception as e_plot:
        logger.warning(f"Coord-branch diagnostics: failed to plot histogram: {e_plot}")

    return out


def _compute_depth_binned_metrics(
    model,
    dataset: REYPointDataset,
    test_indices,
    device: torch.device,
    output_dir: str,
    depth_bins_m: list = None,
):
    """
    Compute depth-binned evaluation metrics (MSE, MAE, R2) in physical depth space
    for the test set, and generate richer visualizations:
      - Bar chart of R² / MAE by depth bin
      - True vs Predicted scatter colored by depth bin

    Returns a dict with file paths, or None if computation failed.
    """
    logger.warning(
        "Depth-binned metrics are deprecated (physical-depth bins are not scientifically comparable to time scale). "
        "Use age-proxy bins via dataset.rey_data['age_ka_proxy'] instead."
    )
    return None

    if depth_bins_m is None:

        max_depth_phys = None
        try:
            if hasattr(dataset, "rey_data") and "depth_physical" in dataset.rey_data.columns:
                d = dataset.rey_data["depth_physical"].astype(float).to_numpy()
                d = d[np.isfinite(d)]
                if d.size:
                    max_depth_phys = float(np.percentile(d, 99))
        except Exception:
            max_depth_phys = None

        if max_depth_phys is None or not np.isfinite(max_depth_phys) or max_depth_phys <= 0:
            max_depth_phys = 15.0

        if max_depth_phys > 15.0:
            depth_bins_m = [0.0, 3.0, 6.0, 15.0, max_depth_phys]
        else:
            depth_bins_m = [0.0, 3.0, 6.0, max_depth_phys]

        if depth_bins_m[-1] <= depth_bins_m[-2]:
            depth_bins_m[-1] = depth_bins_m[-2] + 1.0

    if not hasattr(dataset, "rey_data") or "depth_physical" not in dataset.rey_data.columns:
        logger.warning("Depth-binned metrics: dataset does not provide 'depth_physical'. Skipping.")
        return None

    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    depths_phys = []
    preds_phys = []
    trues_phys = []

    with torch.no_grad():
        for idx in test_indices:
            try:
                combined_patch, target_rey_norm, _ = dataset[idx]
            except Exception as e_fetch:
                logger.debug(f"Depth-binned metrics: failed to fetch sample {idx}: {e_fetch}")
                continue

            combined_patch = combined_patch.unsqueeze(0).to(device)
            target_rey_norm = target_rey_norm.view(1, -1).to(device)

            with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
                pred_norm = model(combined_patch)


            pred_norm_np = pred_norm.detach().cpu().numpy().squeeze()
            true_norm_np = target_rey_norm.detach().cpu().numpy().squeeze()
            pred_phys = float(dataset._denormalize_rey(np.array([pred_norm_np]))[0])
            true_phys = float(dataset._denormalize_rey(np.array([true_norm_np]))[0])

            depth_phys = float(dataset.rey_data.iloc[idx]["depth_physical"])

            preds_phys.append(pred_phys)
            trues_phys.append(true_phys)
            depths_phys.append(depth_phys)

    if not preds_phys:
        logger.warning("Depth-binned metrics: no valid predictions collected. Skipping.")
        return None

    depths_phys = np.array(depths_phys)
    preds_phys = np.array(preds_phys)
    trues_phys = np.array(trues_phys)

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    metrics_by_bin = {}
    bin_labels = []
    r2_values = []
    mae_values = []
    rmse_values = []
    bias_values = []
    frac_abs200_values = []
    frac_rel20_values = []
    skill_values = []


    cmap = plt.get_cmap("viridis")
    num_bins = max(1, len(depth_bins_m) - 1)


    for i in range(len(depth_bins_m) - 1):
        d_min = depth_bins_m[i]
        d_max = depth_bins_m[i + 1]
        mask = (depths_phys >= d_min) & (depths_phys < d_max)
        if not np.any(mask):
            logger.debug(f"Depth-binned metrics: no samples in bin [{d_min}, {d_max}) m.")
            continue

        y_true_bin = trues_phys[mask]
        y_pred_bin = preds_phys[mask]

        mse_bin = float(mean_squared_error(y_true_bin, y_pred_bin))
        mae_bin = float(mean_absolute_error(y_true_bin, y_pred_bin))
        rmse_bin = float(np.sqrt(mse_bin))

        bias_bin = float(np.mean(y_pred_bin - y_true_bin))

        abs_err = np.abs(y_pred_bin - y_true_bin)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_err = abs_err / np.maximum(np.abs(y_true_bin), 1e-6)
        frac_within_abs_200 = float(np.mean(abs_err <= 200.0))
        frac_within_rel_20 = float(np.mean(rel_err <= 0.20))

        try:
            baseline_pred = np.full_like(y_true_bin, y_true_bin.mean())
            baseline_rmse = float(np.sqrt(mean_squared_error(y_true_bin, baseline_pred)))
            skill_score = float(1.0 - (rmse_bin / baseline_rmse)) if baseline_rmse > 0 else float("nan")
        except Exception:
            baseline_rmse = float("nan")
            skill_score = float("nan")
        try:
            r2_bin = float(r2_score(y_true_bin, y_pred_bin))
        except Exception:
            r2_bin = float("nan")

        key = f"{d_min:.1f}_{d_max:.1f}m"
        num_samples = int(mask.sum())
        metrics_by_bin[key] = {
            "depth_min_m": d_min,
            "depth_max_m": d_max,
            "num_samples": num_samples,
            "mse": mse_bin,
            "rmse": rmse_bin,
            "mae": mae_bin,
            "bias": bias_bin,
            "r2": r2_bin,
            "frac_within_abs_200": frac_within_abs_200,
            "frac_within_rel_20pct": frac_within_rel_20,
            "baseline_rmse_bin_mean": baseline_rmse,
            "skill_vs_bin_mean": skill_score,
        }


        display_label = f"{d_min:.0f}-{d_max:.0f} m\n(n={num_samples})"
        bin_labels.append(display_label)
        r2_values.append(r2_bin)
        mae_values.append(mae_bin)
        rmse_values.append(rmse_bin)
        bias_values.append(bias_bin)
        frac_abs200_values.append(frac_within_abs_200)
        frac_rel20_values.append(frac_within_rel_20)
        skill_values.append(skill_score)

    if not metrics_by_bin:
        logger.warning("Depth-binned metrics: no bins had samples. Skipping save.")
        return None


    metrics_path = os.path.join(output_dir, "depth_binned_metrics.json")
    try:
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_by_bin, f, indent=4, cls=NumpyJSONEncoder)
        logger.info(f"Depth-binned metrics: saved to {metrics_path}")
    except Exception as e_save:
        logger.warning(f"Depth-binned metrics: failed to save JSON: {e_save}")


    bar_fig, ax1 = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(bin_labels))

    color_r2 = "tab:blue"
    color_mae = "tab:orange"

    ax1.bar(x_pos - 0.2, r2_values, width=0.4, color=color_r2, label="R²")
    ax1.set_ylabel("R²", color=color_r2)
    ax1.tick_params(axis="y", labelcolor=color_r2)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(bin_labels, rotation=45, ha="right")

    ax2 = ax1.twinx()
    ax2.bar(x_pos + 0.2, mae_values, width=0.4, color=color_mae, label="MAE")
    ax2.set_ylabel("MAE", color=color_mae)
    ax2.tick_params(axis="y", labelcolor=color_mae)

    plt.title("Depth-binned Metrics (Test Set)")
    plt.tight_layout()
    bar_path = os.path.join(output_dir, "depth_binned_metrics_bar.png")
    try:
        plt.savefig(bar_path, dpi=150)
        logger.info(f"Depth-binned metrics: bar chart saved to {bar_path}")
    except Exception as e_bar:
        logger.warning(f"Depth-binned metrics: failed to save bar chart: {e_bar}")
    finally:
        plt.close(bar_fig)


    scatter_fig, ax_sc = plt.subplots(figsize=(8, 6))
    for i in range(len(depth_bins_m) - 1):
        d_min = depth_bins_m[i]
        d_max = depth_bins_m[i + 1]
        mask = (depths_phys >= d_min) & (depths_phys < d_max)
        if not np.any(mask):
            continue
        color = cmap(i / max(1, num_bins - 1))
        num_samples = int(mask.sum())
        label = f"[{d_min:.1f}, {d_max:.1f}) m (n={num_samples})"
        ax_sc.scatter(
            trues_phys[mask],
            preds_phys[mask],
            s=15,
            alpha=0.7,
            color=color,
            label=label,
        )

    max_val = max(trues_phys.max(), preds_phys.max())
    ax_sc.plot([0, max_val], [0, max_val], "k--", linewidth=1.0, label="Ideal 1:1")
    ax_sc.set_xlabel("True REY (physical)")
    ax_sc.set_ylabel("Predicted REY (physical)")
    ax_sc.set_title("True vs Predicted by Depth Layer (Test Set)")
    ax_sc.legend(fontsize=8)
    ax_sc.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    scatter_path = os.path.join(output_dir, "depth_binned_scatter.png")
    try:
        plt.savefig(scatter_path, dpi=150)
        logger.info(f"Depth-binned metrics: depth-layered scatter saved to {scatter_path}")
    except Exception as e_sc:
        logger.warning(f"Depth-binned metrics: failed to save scatter plot: {e_sc}")
    finally:
        plt.close(scatter_fig)


    try:
        extra_fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()


        axes[0].bar(x_pos, rmse_values, color="tab:green")
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(bin_labels, rotation=45, ha="right")
        axes[0].set_ylabel("RMSE")
        axes[0].set_title("RMSE by Depth Layer")


        axes[1].bar(x_pos, bias_values, color="tab:red")
        axes[1].axhline(0.0, color="k", linestyle="--", linewidth=1.0)
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(bin_labels, rotation=45, ha="right")
        axes[1].set_ylabel("Bias (Pred - True)")
        axes[1].set_title("Bias by Depth Layer")


        width = 0.35
        axes[2].bar(x_pos - width / 2, frac_abs200_values, width=width, label="|error| ≤ 200")
        axes[2].bar(x_pos + width / 2, frac_rel20_values, width=width, label="rel. error ≤ 20%")
        axes[2].set_xticks(x_pos)
        axes[2].set_xticklabels(bin_labels, rotation=45, ha="right")
        axes[2].set_ylim(0.0, 1.05)
        axes[2].set_ylabel("Fraction of Samples")
        axes[2].set_title("Error Band Coverage by Depth Layer")
        axes[2].legend(fontsize=8)


        axes[3].bar(x_pos, skill_values, color="tab:purple")
        axes[3].axhline(0.0, color="k", linestyle="--", linewidth=1.0)
        axes[3].set_xticks(x_pos)
        axes[3].set_xticklabels(bin_labels, rotation=45, ha="right")
        axes[3].set_ylabel("Skill Score")
        axes[3].set_title("Skill vs. Bin-Mean Baseline")

        plt.tight_layout()
        extra_metrics_path = os.path.join(output_dir, "depth_binned_extra_metrics_bar.png")
        plt.savefig(extra_metrics_path, dpi=150)
        logger.info(f"Depth-binned metrics: extra metrics bar chart saved to {extra_metrics_path}")
    except Exception as e_extra:
        logger.warning(f"Depth-binned metrics: failed to save extra metrics charts: {e_extra}")
        extra_metrics_path = None
    finally:
        plt.close("all")

    return {
        "metrics_json": metrics_path,
        "bar_plot": bar_path,
        "scatter_plot": scatter_path,
        "extra_metrics_bar_plot": extra_metrics_path,
    }


def _compute_age_binned_metrics(
    model,
    dataset: REYPointDataset,
    test_indices,
    device: torch.device,
    output_dir: str,
    age_bins_ka: list = None,
    batch_size: int = 256,
    min_samples_per_bin: int = 25,
    auto_merge_bins: bool = True,
    output_suffix: str = "",
):
    """
    Compute age-proxy-binned evaluation metrics (MSE, MAE, R2, correlation)
    for the test set, and generate:
      - Bar chart of absolute correlation by age bin
      - True vs Predicted scatter colored by age bin

    Requires dataset.rey_data['age_ka_proxy'] (computed from sedimentation rate).
    """
    if age_bins_ka is None:

        age_bins_ka = [0.0, 200.0, 1000.0, 2580.0]

    if not hasattr(dataset, "rey_data") or "age_ka_proxy" not in dataset.rey_data.columns:
        logger.warning("Age-binned metrics: dataset does not provide 'age_ka_proxy'. Skipping.")
        return None


    try:
        age_bins = sorted({float(x) for x in age_bins_ka})
    except Exception:
        age_bins = [0.0, 200.0, 1000.0, 2580.0]

    if len(age_bins) < 2:
        logger.warning("Age-binned metrics: age_bins_ka is invalid (need at least 2 edges). Skipping.")
        return None

    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    try:
        from torch.utils.data.dataset import Subset
    except Exception as e_torch:
        logger.warning(f"Age-binned metrics: failed to import torch Subset: {e_torch}")
        return None

    try:
        test_indices_list = [int(i) for i in list(test_indices)]
    except Exception:
        test_indices_list = [int(i) for i in test_indices]

    if len(test_indices_list) == 0:
        logger.warning("Age-binned metrics: empty test_indices. Skipping.")
        return None

    try:
        ages_ka = (
            dataset.rey_data.iloc[test_indices_list]["age_ka_proxy"]
            .astype(float)
            .to_numpy()
        )
    except Exception as e_age:
        logger.warning(f"Age-binned metrics: failed to read ages for test indices: {e_age}")
        return None

    subset = Subset(dataset, test_indices_list)
    loader = DataLoader(
        subset,
        batch_size=int(batch_size) if batch_size is not None else 256,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    preds_phys_chunks = []
    trues_phys_chunks = []
    with torch.no_grad():
        for combined_patches, target_reys_norm, _ in loader:
            combined_patches = combined_patches.to(device)
            target_reys_norm = target_reys_norm.to(device)

            with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
                pred_norm = model(combined_patches)

            pred_norm_np = pred_norm.detach().cpu().numpy().reshape(-1)
            true_norm_np = target_reys_norm.detach().cpu().numpy().reshape(-1)

            pred_phys = dataset._denormalize_rey(pred_norm_np)
            true_phys = dataset._denormalize_rey(true_norm_np)
            pred_phys = np.clip(np.asarray(pred_phys, dtype=float), 0.0, None)
            true_phys = np.asarray(true_phys, dtype=float)

            preds_phys_chunks.append(pred_phys)
            trues_phys_chunks.append(true_phys)

    if not preds_phys_chunks:
        logger.warning("Age-binned metrics: no valid predictions collected. Skipping.")
        return None

    ages_ka = np.asarray(ages_ka, dtype=float)
    preds_phys = np.concatenate(preds_phys_chunks) if preds_phys_chunks else np.array([], dtype=float)
    trues_phys = np.concatenate(trues_phys_chunks) if trues_phys_chunks else np.array([], dtype=float)

    if preds_phys.size != trues_phys.size or preds_phys.size != ages_ka.size:
        n = int(min(preds_phys.size, trues_phys.size, ages_ka.size))
        logger.warning(
            "Age-binned metrics: length mismatch among preds/targets/ages "
            f"(pred={preds_phys.size}, true={trues_phys.size}, age={ages_ka.size}); truncating to n={n}."
        )
        preds_phys = preds_phys[:n]
        trues_phys = trues_phys[:n]
        ages_ka = ages_ka[:n]

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    def _bin_mask(ages: np.ndarray, a_min: float, a_max: float, is_last: bool) -> np.ndarray:
        ages = np.asarray(ages, dtype=float)
        if np.isinf(a_max):
            return np.isfinite(ages) & (ages >= a_min)
        if is_last:
            return (ages >= a_min) & (ages <= a_max)
        return (ages >= a_min) & (ages < a_max)

    def _bin_counts(ages: np.ndarray, bins: list) -> list[int]:
        bins = [float(x) for x in bins]
        out = []
        for i in range(len(bins) - 1):
            out.append(int(np.count_nonzero(_bin_mask(ages, bins[i], bins[i + 1], i == len(bins) - 2))))
        return out

    def _auto_merge_bins(ages: np.ndarray, bins_in: list, min_n: int) -> list[float]:
        if min_n is None or int(min_n) <= 0:
            return [float(x) for x in bins_in]
        bins_work = [float(x) for x in bins_in]

        for _ in range(1000):
            if len(bins_work) < 3:
                break
            counts_local = _bin_counts(ages, bins_work)
            idx_small = next((i for i, c in enumerate(counts_local) if c < int(min_n)), None)
            if idx_small is None:
                break
            if idx_small + 1 < len(bins_work) - 1:
                del bins_work[idx_small + 1]
            else:
                del bins_work[idx_small]
        return bins_work

    if auto_merge_bins:
        try:
            age_bins = _auto_merge_bins(ages_ka, age_bins, int(min_samples_per_bin))
        except Exception:
            pass

    def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if np.count_nonzero(mask) < 2:
            return float("nan")
        x = x[mask]
        y = y[mask]
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
        try:
            rx = pd.Series(x).rank(method="average").to_numpy(dtype=float)
            ry = pd.Series(y).rank(method="average").to_numpy(dtype=float)
            return _pearson_r(rx, ry)
        except Exception:
            return float("nan")

    metrics_by_bin = {}
    bin_labels = []
    abs_r_values = []
    r2_values = []
    counts = []
    mae_values = []
    rmse_values = []

    for i in range(len(age_bins) - 1):
        a_min = age_bins[i]
        a_max = age_bins[i + 1]
        mask = _bin_mask(ages_ka, a_min, a_max, i == len(age_bins) - 2)
        if not np.any(mask):
            continue

        y_true_bin = trues_phys[mask]
        y_pred_bin = preds_phys[mask]

        mse_bin = float(mean_squared_error(y_true_bin, y_pred_bin))
        mae_bin = float(mean_absolute_error(y_true_bin, y_pred_bin))
        rmse_bin = float(np.sqrt(mse_bin))
        try:
            r2_bin = float(r2_score(y_true_bin, y_pred_bin))
        except Exception:
            r2_bin = float("nan")

        pr = _pearson_r(y_true_bin, y_pred_bin)
        sr = _spearman_rho(y_true_bin, y_pred_bin)

        if np.isinf(a_max):
            key = f"{a_min:.0f}_infka"
        else:
            key = f"{a_min:.0f}_{a_max:.0f}ka"
        n = int(mask.sum())
        metrics_by_bin[key] = {
            "age_min_ka": float(a_min),
            "age_max_ka": float(a_max) if np.isfinite(a_max) else float("inf"),
            "num_samples": n,
            "mse": mse_bin,
            "rmse": rmse_bin,
            "mae": mae_bin,
            "r2": r2_bin,
            "pearson_r": float(pr),
            "spearman_rho": float(sr),
            "abs_pearson_r": float(abs(pr)) if np.isfinite(pr) else float("nan"),
            "abs_spearman_rho": float(abs(sr)) if np.isfinite(sr) else float("nan"),
        }

        if np.isinf(a_max):
            bin_labels.append(f"{a_min:.0f}+ ka\n(n={n})")
        else:
            bin_labels.append(f"{a_min:.0f}-{a_max:.0f} ka\n(n={n})")
        abs_r_values.append(float(abs(pr)) if np.isfinite(pr) else float("nan"))
        r2_values.append(r2_bin)
        counts.append(n)
        mae_values.append(mae_bin)
        rmse_values.append(rmse_bin)

    if not metrics_by_bin:
        logger.warning("Age-binned metrics: no bins had samples. Skipping save.")
        return None

    suffix = str(output_suffix or "")
    try:
        metrics_by_bin["_meta"] = {
            "age_bins_ka_used": [float(x) for x in age_bins],
            "min_samples_per_bin": int(min_samples_per_bin),
            "auto_merge_bins": bool(auto_merge_bins),
            "num_eval_samples": int(len(ages_ka)),
            "output_suffix": suffix,
            "scatter_axis_clip_quantile": 99.0,
        }
    except Exception:
        pass
    metrics_path = os.path.join(output_dir, f"age_binned_metrics{suffix}.json")
    try:
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_by_bin, f, indent=4, cls=NumpyJSONEncoder)
        logger.info(f"Age-binned metrics: saved to {metrics_path}")
    except Exception as e_save:
        logger.warning(f"Age-binned metrics: failed to save JSON: {e_save}")


    abs_fig_h = max(3.5, 1.2 + 0.85 * len(bin_labels))
    abs_fig, ax = plt.subplots(figsize=(10, abs_fig_h))
    y_pos = np.arange(len(bin_labels))
    display_labels = []
    for lbl, mae_v, rmse_v in zip(bin_labels, mae_values, rmse_values):
        lbl_one_line = lbl.replace("\n", " ")
        display_labels.append(f"{lbl_one_line} | MAE={mae_v:.0f} | RMSE={rmse_v:.0f}")

    ax.barh(y_pos, abs_r_values, color="tab:blue", alpha=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_labels)
    ax.invert_yaxis()
    ax.set_xlabel("|Pearson r| (True vs Pred)")
    ax.set_title("Age-binned |Pearson r| (evaluation)")
    ax.set_xlim(0.0, 1.0)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    for i, v in enumerate(abs_r_values):
        if not np.isfinite(v):
            continue
        x_txt = max(0.02, min(float(v) - 0.04, 0.92))
        ax.text(x_txt, i, f"{v:.2f}", va="center", ha="left", color="white", fontsize=10)
    plt.tight_layout()
    abs_bar_path = os.path.join(output_dir, f"age_binned_abs_coeff_bar{suffix}.png")
    try:
        plt.savefig(abs_bar_path, dpi=150)
        logger.info(f"Age-binned metrics: abs-correlation bar chart saved to {abs_bar_path}")
    except Exception as e_bar:
        logger.warning(f"Age-binned metrics: failed to save abs-correlation bar chart: {e_bar}")
    finally:
        plt.close(abs_fig)


    scat_fig, ax_sc = plt.subplots(figsize=(7.5, 7.5))
    cmap = plt.get_cmap("tab10")

    try:
        q = 99.0
        vmax_true = float(np.nanpercentile(trues_phys[np.isfinite(trues_phys)], q))
        vmax_pred = float(np.nanpercentile(preds_phys[np.isfinite(preds_phys)], q))
        vmax = float(max(vmax_true, vmax_pred))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = float(np.nanmax(np.concatenate([trues_phys, preds_phys])))
    except Exception:
        vmax = float(np.nanmax(np.concatenate([trues_phys, preds_phys])))
    vmax = max(1.0, float(vmax))

    for i in range(len(age_bins) - 1):
        a_min = age_bins[i]
        a_max = age_bins[i + 1]
        mask = _bin_mask(ages_ka, a_min, a_max, i == len(age_bins) - 2)
        if not np.any(mask):
            continue
        t = trues_phys[mask]
        p = preds_phys[mask]
        out = (t > vmax) | (p > vmax)
        t_plot = np.minimum(t, vmax)
        p_plot = np.minimum(p, vmax)
        n = int(mask.sum())
        n_out = int(np.count_nonzero(out))

        alpha = float(min(0.75, max(0.06, 2000.0 / max(1, n))))
        ax_sc.scatter(
            t_plot[~out],
            p_plot[~out],
            s=18 if n < 5000 else 12,
            alpha=alpha,
            color=cmap(i % 10),
            label=(
                f"{a_min:.0f}+ ka (n={n}, clipped={n_out})"
                if np.isinf(a_max)
                else f"{a_min:.0f}-{a_max:.0f} ka (n={n}, clipped={n_out})"
            ),
            edgecolors="none",
        )
        if n_out > 0:
            ax_sc.scatter(
                t_plot[out],
                p_plot[out],
                s=26,
                alpha=min(0.35, alpha),
                color=cmap(i % 10),
                marker="x",
            )


    ax_sc.plot([0.0, vmax], [0.0, vmax], "k--", linewidth=1)

    ax_sc.set_xlabel("True REY (physical)")
    ax_sc.set_ylabel("Predicted REY (physical)")
    ax_sc.set_title(f"True vs Predicted by age bin (clipped at p99={vmax:.0f})")
    ax_sc.grid(True, linestyle="--", alpha=0.25)
    ax_sc.legend(fontsize=8, loc="best")
    ax_sc.set_xlim(0.0, vmax)
    ax_sc.set_ylim(0.0, vmax)
    plt.tight_layout()
    scatter_path = os.path.join(output_dir, f"age_binned_scatter{suffix}.png")
    try:
        plt.savefig(scatter_path, dpi=150)
        logger.info(f"Age-binned metrics: scatter plot saved to {scatter_path}")
    except Exception as e_sc:
        logger.warning(f"Age-binned metrics: failed to save scatter plot: {e_sc}")
    finally:
        plt.close(scat_fig)

    return {
        "metrics_json": metrics_path,
        "abs_coeff_bar_plot": abs_bar_path,
        "scatter_plot": scatter_path,
    }


def _save_predictions_table(
    model,
    dataset: REYPointDataset,
    indices,
    device: torch.device,
    output_csv_path: str,
    split_name: str,
    age_bins_ka: list = None,
    batch_size: int = 256,
):
    """
    Save per-sample predictions and metadata (lon/lat/depth) for a given split
    (train or test) to a CSV table.

    Columns include:
      - dataset_index
      - split
      - lon, lat (if available)
      - depth_m (physical depth)
      - rey_true, rey_pred (physical space)
      - age_ka_proxy, age_bin (if available)
    """
    if age_bins_ka is not None:
        try:
            age_bins_ka = sorted({float(x) for x in age_bins_ka})
        except Exception:
            age_bins_ka = None

    if not hasattr(dataset, "rey_data"):
        logger.warning("Save predictions table: dataset has no 'rey_data'; skipping.")
        return None

    model.eval()
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    try:
        from torch.utils.data.dataset import Subset
    except Exception as e_torch:
        logger.warning(f"Save predictions table: failed to import torch Subset: {e_torch}")
        return None

    try:
        indices_list = [int(i) for i in list(indices)]
    except Exception:
        indices_list = [int(i) for i in indices]

    if len(indices_list) == 0:
        logger.warning(f"Save predictions table: empty indices for split='{split_name}'.")
        return None

    meta_df = dataset.rey_data.iloc[indices_list].copy().reset_index(drop=True)

    subset = Subset(dataset, indices_list)
    loader = DataLoader(
        subset,
        batch_size=int(batch_size) if batch_size is not None else 256,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    preds_phys_chunks = []
    trues_phys_chunks = []
    with torch.no_grad():
        for combined_patches, target_reys_norm, _ in loader:
            combined_patches = combined_patches.to(device)
            target_reys_norm = target_reys_norm.to(device)

            with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
                pred_norm = model(combined_patches)

            pred_norm_np = pred_norm.detach().cpu().numpy().reshape(-1)
            true_norm_np = target_reys_norm.detach().cpu().numpy().reshape(-1)

            pred_phys = dataset._denormalize_rey(pred_norm_np)
            true_phys = dataset._denormalize_rey(true_norm_np)
            pred_phys = np.clip(np.asarray(pred_phys, dtype=float), 0.0, None)
            true_phys = np.asarray(true_phys, dtype=float)

            preds_phys_chunks.append(pred_phys)
            trues_phys_chunks.append(true_phys)

    if not preds_phys_chunks:
        logger.warning(f"Save predictions table: no predictions collected for split='{split_name}'.")
        return None

    preds_phys = np.concatenate(preds_phys_chunks) if preds_phys_chunks else np.array([], dtype=float)
    trues_phys = np.concatenate(trues_phys_chunks) if trues_phys_chunks else np.array([], dtype=float)

    n = int(min(len(meta_df), preds_phys.size, trues_phys.size))
    if n <= 0:
        logger.warning(f"Save predictions table: no aligned rows for split='{split_name}'.")
        return None

    meta_df = meta_df.iloc[:n].copy()
    preds_phys = preds_phys[:n]
    trues_phys = trues_phys[:n]

    depth_feature_type = getattr(dataset, "depth_feature_type", "depth")

    def _pick_col(df: pd.DataFrame, candidates: list[str]):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    lon_col = _pick_col(meta_df, ["lon", "longitude", "Lon", "Longitude"])
    lat_col = _pick_col(meta_df, ["lat", "latitude", "Lat", "Latitude"])
    depth_phys_col = _pick_col(meta_df, ["depth_physical", "depth_clipped", "depth_mbsf"])
    depth_feat_col = _pick_col(meta_df, ["depth"])

    out_df = pd.DataFrame(
        {
            "dataset_index": indices_list[:n],
            "split": [split_name] * n,
            "lon": meta_df[lon_col].astype(float).to_numpy() if lon_col else np.full(n, np.nan),
            "lat": meta_df[lat_col].astype(float).to_numpy() if lat_col else np.full(n, np.nan),
            "depth_mbsf": meta_df[depth_phys_col].astype(float).to_numpy() if depth_phys_col else np.full(n, np.nan),
            "depth_feature": meta_df[depth_feat_col].astype(float).to_numpy() if depth_feat_col else np.full(n, np.nan),
            "depth_feature_type": [depth_feature_type] * n,
            "age_ka_proxy": meta_df["age_ka_proxy"].astype(float).to_numpy() if "age_ka_proxy" in meta_df.columns else np.full(n, np.nan),
            "rey_true": trues_phys,
            "rey_pred": preds_phys,
        }
    )

    if age_bins_ka is not None and "age_ka_proxy" in out_df.columns:
        try:
            bins = sorted({float(x) for x in age_bins_ka})
            if len(bins) >= 2:
                ages = out_df["age_ka_proxy"].to_numpy(dtype=float)
                labels = []
                for i in range(len(bins) - 1):
                    if np.isinf(bins[i + 1]):
                        labels.append(f"{bins[i]:.0f}_infka")
                    else:
                        labels.append(f"{bins[i]:.0f}_{bins[i+1]:.0f}ka")
                bin_idx = np.searchsorted(np.asarray(bins, dtype=float), ages, side="right") - 1
                if np.isinf(bins[-1]):
                    out_of_range = ~np.isfinite(ages) | (ages < bins[0])
                else:
                    out_of_range = ~np.isfinite(ages) | (ages < bins[0]) | (ages > bins[-1])
                bin_idx = np.clip(bin_idx, 0, len(labels) - 1)
                age_bin = np.array([labels[i] for i in bin_idx], dtype=object)
                age_bin[out_of_range] = None
                out_df["age_bin"] = age_bin
        except Exception:
            pass

    try:
        out_df.to_csv(output_csv_path, index=False)
        logger.info(f"Save predictions table: wrote {len(out_df)} rows to {output_csv_path}")
        return output_csv_path
    except Exception as e_csv:
        logger.warning(f"Save predictions table: failed to save CSV to {output_csv_path}: {e_csv}")
        return None

def run_training_pipeline(config, progress_callback=None, stop_flag=None, history_callback=None):
    """
    Orchestrates the CNN-MLP model training pipeline using point-based data.
    This version is integrated with RunManager for robust artifact and configuration handling.
    """
    logger.info("Pipeline: Starting model training pipeline...")
    start_time = time.time()

    training_config = config.get("training", {}) if isinstance(config, dict) else {}
    hyperparams = training_config.get("hyperparams", {}) if isinstance(training_config, dict) else {}
    requested_task_type = str(hyperparams.get("task_type", "regression") or "regression").lower()

    if requested_task_type in {CASCADE_GRADE_TASK_TYPE, CASCADE_EXPLORATION_4CLASS_TASK_TYPE}:
        return _run_cascade_training_pipeline(
            config=config,
            progress_callback=progress_callback,
            stop_flag=stop_flag,
            history_callback=history_callback,
        )

    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')


    cfg_paths = config.get("paths", {}) if isinstance(config, dict) else {}
    cfg_run_base_dir = (cfg_paths.get("run_base_dir") if isinstance(cfg_paths, dict) else None) or None
    if cfg_run_base_dir:
        run_base_dir = Path(str(cfg_run_base_dir))
        temp_run_id = run_base_dir.name
    else:
        temp_run_id = f"point_model_run_{timestamp_str}_temp"
        run_base_dir = Path(RESULTS_GUI_DIR) / temp_run_id

    output_dirs = {
        "base": str(run_base_dir),
        "logs": str(run_base_dir / "logs"),
        "plots": str(run_base_dir / "plots"),
        "tables": str(run_base_dir / "tables"),
        "models": str(run_base_dir / "models"),
        "reports": str(run_base_dir / "reports"),
        "splits": str(run_base_dir / "splits"),
        "diagnostics": str(run_base_dir / "plots" / "diagnostics"),
        "classification": str(run_base_dir / "plots" / "diagnostics" / "classification"),
        "scatter": str(run_base_dir / "plots" / "scatter"),
        "plotly_interactive": str(run_base_dir / "plots" / "plotly_interactive"),
        "shap_explainability": str(run_base_dir / "plots" / "shap_explainability"),
        "post_hoc_explainability": str(run_base_dir / "plots" / "post_hoc_explainability"),
        "mechanism_clustering": str(run_base_dir / "plots" / "mechanism_clustering"),
        "feature_maps": str(run_base_dir / "feature_maps"),
        "feature_maps_plots": str(run_base_dir / "feature_maps" / "plots"),
        "feature_maps_netcdf": str(run_base_dir / "feature_maps" / "netcdf"),
    }
    for dir_path in output_dirs.values():
        Path(dir_path).mkdir(parents=True, exist_ok=True)


    run_summary = {
        "run_id": temp_run_id,
        "run_type": "training",
        "timestamp_start": datetime.now().isoformat(),
        "status": "started",
        "results_schema_version": RESULTS_SCHEMA_VERSION,
        "config": config,
        "inputs": {},
        "results": {},
        "artifacts": {"paths": output_dirs}
    }


    full_dataset = None
    dataset_summary_for_report = None
    train_indices = val_indices = test_indices = None


    training_config = config.get("training", {}) if isinstance(config, dict) else {}
    hyperparams = training_config.get("hyperparams", {}) if isinstance(training_config, dict) else {}
    random_state = int(hyperparams.get("random_state", 42))
    split_random_state = int(
        training_config.get("split_random_state", hyperparams.get("split_random_state", random_state))
        if isinstance(training_config, dict)
        else random_state
    )

    try:
        logger.info(
            f"Pipeline: Using random_state={random_state} for reproducible training "
            f"(split_random_state={split_random_state})."
        )
        random.seed(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
    except Exception as e_seed:
        logger.warning(f"Pipeline: Failed to fully set random seeds: {e_seed}")

    try:

        training_run_config = config.get("training_run_config")
        if training_run_config:
            try:
                config_save_path = run_base_dir / "training_config.json"
                with open(config_save_path, 'w', encoding='utf-8') as f:
                    json.dump(training_run_config, f, indent=4, cls=NumpyJSONEncoder)
                logger.info(f"Successfully saved training configuration to {config_save_path}")
                run_summary["artifacts"]["paths"]["training_config"] = str(config_save_path)
            except Exception as e_conf:
                logger.warning(f"Could not save training_config.json: {e_conf}")


        logger.debug("Pipeline: Extracting configuration and validating inputs...")
        paths = config.get("paths", {})
        data_processing_config = config.get("data_processing", {})
        training_config = config.get("training", {})
        plot_config = dict(config.get("plotting", {}) or {})

        plot_config.setdefault("export_formats", ["png", "pdf"])
        plot_config.setdefault("publication_dpi", 600)
        plot_config.setdefault("base_fontsize", 11)
        plot_config.setdefault("shap_style", "publication")
        plot_config.setdefault("shap_max_display", 20)
        plot_config.setdefault("shap_summary_plot_size", (8.3, 6.0))

        hyperparams = training_config.get("hyperparams", {})

        rey_points_path = paths.get("processed_rey_points")
        env_grid_path = paths.get("processed_env_grid")

        run_summary["inputs"]["processed_rey_points_path"] = rey_points_path
        run_summary["inputs"]["processed_env_grid_path"] = env_grid_path

        if not rey_points_path or not os.path.exists(rey_points_path):
             raise FileNotFoundError(f"REY points data path missing or file not found: {rey_points_path}")
        if not env_grid_path or not os.path.exists(env_grid_path):
             raise FileNotFoundError(f"Environment grid data path missing or file not found: {env_grid_path}")


        logger.info(f"Pipeline: Dynamically discovering feature names from the processed grid: {env_grid_path}")
        try:
            with open_dataset(env_grid_path) as env_ds:
                # Dynamically discover features, then explicitly remove 'land_mask' if it exists.
                discovered_variables = list(env_ds.data_vars)
                if 'land_mask' in discovered_variables:
                    logger.info("Pipeline: 'land_mask' found in data variables. Explicitly removing it from the list of features for training.")
                    discovered_variables.remove('land_mask')

                selected_env_variables = discovered_variables
                logger.info(f"Pipeline: Final features for training: {selected_env_variables}")
        except Exception as e:
            raise RuntimeError(f"Failed to open or read variables from the processed env_gridded.nc file at '{env_grid_path}'. Please ensure Step 1 (Preprocessing) was successful. Error: {e}")

        if not selected_env_variables:
            raise ValueError("No data variables (features) found in the processed env_gridded.nc file.")

        patch_size = tuple(training_config.get("patch_size", [3, 3]))

        logger.debug(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")


        if not rey_points_path or not os.path.exists(rey_points_path):
             raise FileNotFoundError(f"REY points data path missing or file not found: {rey_points_path}")
        if not env_grid_path or not os.path.exists(env_grid_path):
             raise FileNotFoundError(f"Environment grid data path missing or file not found: {env_grid_path}")
        if not selected_env_variables:
            raise ValueError("No environmental variables selected.")


        logger.debug("Pipeline: Performing pre-flight check on environmental data...")
        try:
            with open_dataset(env_grid_path) as env_ds:
                for var_name in selected_env_variables:
                    if var_name not in env_ds.data_vars:
                        raise ValueError(f"Environmental variable '{var_name}' is absent from the NetCDF file.")
                    try:
                        da = env_ds[var_name]
                        indexer = {d: 0 for d in da.dims if int(da.sizes.get(d, 0) or 0) > 0}
                        _ = da.isel(indexer).values
                        logger.debug(f"Successfully checked '{var_name}'.")
                    except Exception as e_load:
                        raise RuntimeError(f"Failed to load environmental variable '{var_name}': {e_load}")
        except FileNotFoundError:
            raise
        except Exception as e_pre_check:
            raise RuntimeError(f"Data precheck failed: {e_pre_check}")
        logger.debug("Pipeline: Environmental data pre-flight check passed.")


        logger.debug("Pipeline: Loading data and creating datasets/loaders...")
        depth_transform_method = data_processing_config.get("depth_transform_method", "log1p")
        env_normalization_method = data_processing_config.get("env_normalization_method", "zscore")

        max_training_depth_mbsf = data_processing_config.get("max_training_depth_mbsf", None)


        task_type_tmp = str(hyperparams.get("task_type", "regression") or "regression").lower()
        default_max_training_age_ka_proxy = 2580.0 if task_type_tmp != "multiclass_classification" else None
        max_training_age_ka_proxy = data_processing_config.get(
            "max_training_age_ka_proxy",
            default_max_training_age_ka_proxy,
        )
        try:
            if max_training_age_ka_proxy is not None:
                max_training_age_ka_proxy = float(max_training_age_ka_proxy)
        except Exception:
            max_training_age_ka_proxy = None
        # Sedimentation-rate grid for age-proxy calculation (recommended; required if max_training_age_ka_proxy is set)
        default_sed_rate_nc_path = os.path.join("data", "external", "grids", "Average sedimentation rate.nc")
        sed_rate_nc_path = data_processing_config.get(
            "sedimentation_rate_grid_path",
            default_sed_rate_nc_path if os.path.exists(default_sed_rate_nc_path) else None,
        )
        if not sed_rate_nc_path and max_training_age_ka_proxy is not None:
            logger.warning(
                "Pipeline: max_training_age_ka_proxy is set but no sedimentation rate grid is available; "
                "disabling age-proxy filtering."
            )
            max_training_age_ka_proxy = None
        logger.info(f"Pipeline: Using depth_transform_method: {depth_transform_method}")
        logger.info(f"Pipeline: Using env_normalization_method: {env_normalization_method}")
        if max_training_depth_mbsf is not None:
            logger.info(f"Pipeline: Applying max_training_depth_mbsf={max_training_depth_mbsf} (physical depth filter).")
        if max_training_age_ka_proxy is not None:
            logger.info(f"Pipeline: Applying max_training_age_ka_proxy={max_training_age_ka_proxy} ka (age-proxy filter).")
        if sed_rate_nc_path:
            logger.info(f"Pipeline: Using sedimentation rate grid for age proxy: {sed_rate_nc_path}")
        else:
            logger.info("Pipeline: No sedimentation rate grid configured; age proxy will be disabled.")

        full_dataset = REYPointDataset(
            rey_points_path=rey_points_path,
            env_grid_path=env_grid_path,
            feature_names=selected_env_variables,
            patch_size=patch_size,
            depth_transform_method=depth_transform_method,
            env_normalization_method=env_normalization_method,
            augment_data=hyperparams.get("augment_data", False),
            augmentation_degrees=hyperparams.get("augmentation_degrees", 15.0),
            shift_augmentation_pixels=hyperparams.get("shift_augmentation_pixels", 0),
            use_weighted_loss=hyperparams.get("use_weighted_loss", False),
            weighting_strategy=hyperparams.get("weighting_strategy", "linear"),
            max_weight=hyperparams.get("max_weight", 5.0),
            max_depth_mbsf=max_training_depth_mbsf,
            max_age_ka_proxy=max_training_age_ka_proxy,
            sed_rate_nc_path=sed_rate_nc_path,
            depth_feature_type_override=(
                (data_processing_config.get("depth_feature_type_override", None) if isinstance(data_processing_config, dict) else None)
                or (data_processing_config.get("depth_feature_type", None) if isinstance(data_processing_config, dict) else None)
            ),
            depth_weighting_scheme=hyperparams.get("depth_weighting_scheme", None),
            add_coord_channels=bool(hyperparams.get("add_coord_channels", False)),
            add_env_missingness_channels=bool(hyperparams.get("add_env_missingness_channels", False)),
            task_type=str(hyperparams.get("task_type", "regression") or "regression"),
            classification_threshold_ppm=float(hyperparams.get("classification_threshold_ppm", 1000.0) or 1000.0),
            classification_label_col=hyperparams.get("classification_label_col", None),
            classification_class_names=hyperparams.get("classification_class_names", None),
            soft_label_cols=hyperparams.get("soft_label_cols", None),
            use_sample_weights=bool(
                hyperparams.get("domain_priority_enabled", False)
                or hyperparams.get("use_sample_weights", False)
                or hyperparams.get("sample_weight_col", None)
            ),
            sample_weight_col=hyperparams.get("sample_weight_col", None),
            domain_priority_enabled=bool(hyperparams.get("domain_priority_enabled", False)),
            domain_priority_mode=str(hyperparams.get("domain_priority_mode", "loss_weighting") or "loss_weighting"),
            training_age_domain_ka=float(hyperparams.get("training_age_domain_ka", 2580.0) or 2580.0),
            domain_priority_out_weight=float(hyperparams.get("domain_priority_out_weight", 0.2) or 0.2),
        )
        logger.debug(f"Pipeline: full_dataset.norm_params after REYPointDataset init: {full_dataset.norm_params}")
        dataset_size = len(full_dataset)
        logger.debug(f"Total dataset size: {dataset_size}")
        if dataset_size == 0:
            raise ValueError("Dataset is empty after loading.")

        test_split_ratio = training_config.get("test_split_ratio", 0.2)
        validation_split_ratio = training_config.get("validation_split_ratio", 0.1)
        forced_split_indices = training_config.get("forced_split_indices", None)

        split_strategy = str(training_config.get("split_strategy", "random_point") or "random_point")
        spatial_block_deg = float(training_config.get("spatial_block_deg", 5.0) or 5.0)
        run_spatial_block_cv = bool(training_config.get("run_spatial_block_cv", False))
        spatial_cv_folds = int(training_config.get("spatial_block_cv_folds", 5) or 5)
        spatial_cv_max_folds = int(training_config.get("spatial_block_cv_max_folds", spatial_cv_folds) or spatial_cv_folds)
        spatial_cv_only = bool(training_config.get("spatial_block_cv_only", False))

        train_sampling_strategy = str(training_config.get("train_sampling_strategy", "random") or "random").lower()
        block_balance_alpha = float(training_config.get("block_balance_alpha", 1.0) or 1.0)
        block_balance_min_count = int(training_config.get("block_balance_min_count", 5) or 5)
        group_split_stratify_bins = int(training_config.get("group_split_stratify_bins", 0) or 0)


        try:
            task_type_tmp = str(hyperparams.get("task_type", "regression") or "regression").lower()
            loss_tmp = str(hyperparams.get("classification_loss", "ce") or "ce").lower()
            use_cw_tmp = bool(hyperparams.get("use_class_weights", True))
            if task_type_tmp == "multiclass_classification" and train_sampling_strategy in ("class_balanced", "block_and_class_balanced"):
                if loss_tmp == "focal" and use_cw_tmp:
                    logger.warning(
                        "Pipeline: You enabled BOTH class-balanced sampling and class-weighted loss with focal loss. "
                        "This can over-correct imbalance and collapse predictions to a single class. "
                        "Recommended: disable class weights, or switch loss to CE, or reduce class_balance_power/max_factor."
                    )
        except Exception:
            pass

        hyperparams["train_sampling_strategy"] = train_sampling_strategy
        hyperparams["block_balance_alpha"] = block_balance_alpha
        hyperparams["block_balance_min_count"] = block_balance_min_count
        hyperparams["group_split_stratify_bins"] = group_split_stratify_bins

        run_summary["results"]["split_strategy"] = split_strategy
        run_summary["results"]["spatial_block_deg"] = spatial_block_deg if split_strategy == "spatial_block" else None
        run_summary["results"]["run_spatial_block_cv"] = bool(run_spatial_block_cv)
        run_summary["results"]["spatial_block_cv_folds"] = int(spatial_cv_folds) if run_spatial_block_cv else None
        run_summary["results"]["spatial_block_cv_only"] = bool(spatial_cv_only) if run_spatial_block_cv else None
        run_summary["results"]["train_sampling_strategy"] = train_sampling_strategy
        run_summary["results"]["block_balance_alpha"] = block_balance_alpha if train_sampling_strategy in ("block_balanced", "block_and_class_balanced") else None
        run_summary["results"]["block_balance_min_count"] = block_balance_min_count if train_sampling_strategy in ("block_balanced", "block_and_class_balanced") else None
        run_summary["results"]["class_balance_power"] = float(hyperparams.get("class_balance_power", 0.7) or 0.7) if train_sampling_strategy in ("class_balanced", "block_and_class_balanced") else None
        run_summary["results"]["class_balance_max_factor"] = float(hyperparams.get("class_balance_max_factor", 8.0) or 8.0) if train_sampling_strategy in ("class_balanced", "block_and_class_balanced") else None
        run_summary["results"]["group_split_stratify_bins"] = group_split_stratify_bins if split_strategy in ("core_id", "site_group") else None

        block_ids = None
        if forced_split_indices:
            train_indices = [int(i) for i in forced_split_indices.get("train", [])]
            val_indices = [int(i) for i in forced_split_indices.get("validation", [])]
            test_indices = [int(i) for i in forced_split_indices.get("test", [])]
            run_summary["results"]["forced_split_indices_used"] = True
            run_summary["results"]["split_strategy"] = "forced_indices"
        elif split_strategy == "spatial_block":
            block_ids = _compute_spatial_block_ids(full_dataset.rey_data, spatial_block_deg)

            try:
                np.save(Path(output_dirs["splits"]) / "spatial_block_ids.npy", block_ids.astype(np.int64))
                df_blocks = pd.DataFrame(
                    {
                        "lon": full_dataset.rey_data["lon"].astype(float).to_numpy(),
                        "lat": full_dataset.rey_data["lat"].astype(float).to_numpy(),
                        "spatial_block_id": block_ids.astype(np.int64),
                    }
                )
                df_blocks.to_csv(Path(output_dirs["splits"]) / "spatial_blocks.csv", index=False)
            except Exception as e_blocks:
                logger.warning(f"Pipeline: failed to persist spatial block ids: {e_blocks}")

            if run_spatial_block_cv:
                logger.info(
                    f"Pipeline: Running spatial block CV diagnostic (GroupKFold) with folds={spatial_cv_folds} "
                    f"(max_folds_to_run={spatial_cv_max_folds})."
                )
                try:
                    cv_summary = _run_spatial_block_cv_diagnostic(
                        dataset=full_dataset,
                        groups=block_ids,
                        output_root=run_base_dir,
                        hyperparams=hyperparams,
                        test_split_ratio=test_split_ratio,
                        validation_split_ratio=validation_split_ratio,
                        random_state=random_state,
                        max_folds_to_run=spatial_cv_max_folds,
                        n_splits=spatial_cv_folds,
                    )
                    run_summary["results"]["spatial_block_cv"] = cv_summary
                except Exception as e_cv:
                    logger.warning(f"Pipeline: Spatial block CV failed: {e_cv}")

                if spatial_cv_only:

                    run_summary["status"] = "success"
                    run_summary["results"]["note"] = "spatial_block_cv_only=True; no final model trained."
                    train_indices = val_indices = test_indices = []
                    return True, run_summary


            y_multi = None
            n_classes_for_split = 0
            try:
                task_type_for_split = str(hyperparams.get("task_type", "regression") or "regression").lower()
                label_col = hyperparams.get("classification_label_col", None)
                rey_df = getattr(full_dataset, "rey_data", None)
                if (
                    task_type_for_split == "multiclass_classification"
                    and rey_df is not None
                    and label_col
                    and label_col in rey_df.columns
                ):
                    y_multi = rey_df[label_col].astype(int).to_numpy(dtype=np.int64, copy=True)
                    n_classes_for_split = int(np.nanmax(y_multi)) + 1 if y_multi.size else 0
            except Exception:
                y_multi = None
                n_classes_for_split = 0

            _mcs = training_config.get("min_class_samples_per_split", 0)
            _smtm = training_config.get("split_max_tries_multiclass", 1)
            min_class_samples_per_split = int(0 if _mcs is None else _mcs)
            split_max_tries_multiclass = int(1 if _smtm is None else _smtm)
            _mcg = training_config.get("min_class_groups_per_split", 0)
            min_class_groups_per_split = int(0 if _mcg is None else _mcg)

            _mtcs = training_config.get("min_train_class_samples", None)
            _mtcg = training_config.get("min_train_class_groups", None)
            min_train_class_samples = (
                int(min_class_samples_per_split) if _mtcs is None else int(_mtcs)
            )
            min_train_class_groups = (
                int(min_class_groups_per_split) if _mtcg is None else int(_mtcg)
            )
            split_multiclass_stats = None

            def _do_block_split(rs: int):
                return _group_train_val_test_split(
                    dataset_size=dataset_size,
                    groups=block_ids,
                    test_split_ratio=test_split_ratio,
                    validation_split_ratio=validation_split_ratio,
                    random_state=rs,
                )

            if y_multi is not None and n_classes_for_split >= 2 and min_class_samples_per_split > 0 and split_max_tries_multiclass > 1:
                totals = np.bincount(y_multi, minlength=n_classes_for_split).astype(int)
                required = np.where(totals >= int(3 * min_class_samples_per_split), 1, 0).astype(int)

                def _count_classes(indices):
                    idx = np.asarray(indices, dtype=int)
                    if idx.size == 0:
                        return np.zeros((n_classes_for_split,), dtype=int)
                    return np.bincount(y_multi[idx], minlength=n_classes_for_split).astype(int)

                def _count_groups(indices):
                    idx = np.asarray(indices, dtype=int)
                    if idx.size == 0:
                        return np.zeros((n_classes_for_split,), dtype=int)
                    g = block_ids[idx]
                    y = y_multi[idx]
                    out = np.zeros((n_classes_for_split,), dtype=int)
                    for c in range(n_classes_for_split):
                        if required[c] == 0:
                            continue
                        out[c] = int(len(set(g[y == c].tolist())))
                    return out

                for attempt in range(max(1, split_max_tries_multiclass)):
                    rs = int(split_random_state) + int(attempt) * 17
                    tr, va, te = _do_block_split(rs)
                    c_tr = _count_classes(tr)
                    c_va = _count_classes(va)
                    c_te = _count_classes(te)
                    g_tr = _count_groups(tr) if min_class_groups_per_split > 0 else None
                    g_va = _count_groups(va) if min_class_groups_per_split > 0 else None
                    g_te = _count_groups(te) if min_class_groups_per_split > 0 else None
                    ok = True
                    for c in range(n_classes_for_split):
                        if required[c] == 0:
                            continue
                        if (
                            (c_tr[c] < min_train_class_samples)
                            or (c_va[c] < min_class_samples_per_split)
                            or (c_te[c] < min_class_samples_per_split)
                        ):
                            ok = False
                            break
                        if min_class_groups_per_split > 0:
                            if (
                                (g_tr[c] < min_train_class_groups)
                                or (g_va[c] < min_class_groups_per_split)
                                or (g_te[c] < min_class_groups_per_split)
                            ):
                                ok = False
                                break
                    if ok:
                        train_indices, val_indices, test_indices = tr, va, te
                        split_multiclass_stats = {
                            "class_totals": totals.tolist(),
                            "required_classes": required.tolist(),
                            "min_class_samples_per_split": int(min_class_samples_per_split),
                            "min_class_groups_per_split": int(min_class_groups_per_split),
                            "min_train_class_samples": int(min_train_class_samples),
                            "min_train_class_groups": int(min_train_class_groups),
                            "train_counts": c_tr.tolist(),
                            "val_counts": c_va.tolist(),
                            "test_counts": c_te.tolist(),
                            "train_group_counts": g_tr.tolist() if g_tr is not None else None,
                            "val_group_counts": g_va.tolist() if g_va is not None else None,
                            "test_group_counts": g_te.tolist() if g_te is not None else None,
                            "retry_used": int(attempt) + 1,
                        }
                        break

                if split_multiclass_stats is None:
                    train_indices, val_indices, test_indices = _do_block_split(int(split_random_state))
                    c_tr = _count_classes(train_indices)
                    c_va = _count_classes(val_indices)
                    c_te = _count_classes(test_indices)
                    g_tr = _count_groups(train_indices) if min_class_groups_per_split > 0 else None
                    g_va = _count_groups(val_indices) if min_class_groups_per_split > 0 else None
                    g_te = _count_groups(test_indices) if min_class_groups_per_split > 0 else None
                    split_multiclass_stats = {
                        "class_totals": totals.tolist(),
                        "required_classes": required.tolist(),
                        "min_class_samples_per_split": int(min_class_samples_per_split),
                        "min_class_groups_per_split": int(min_class_groups_per_split),
                        "train_counts": c_tr.tolist(),
                        "val_counts": c_va.tolist(),
                        "test_counts": c_te.tolist(),
                        "train_group_counts": g_tr.tolist() if g_tr is not None else None,
                        "val_group_counts": g_va.tolist() if g_va is not None else None,
                        "test_group_counts": g_te.tolist() if g_te is not None else None,
                        "retry_used": 0,
                    }
                    logger.warning(
                        "Pipeline: spatial_block split could not satisfy minimal class support after "
                        f"{split_max_tries_multiclass} attempts (min_class_samples_per_split={min_class_samples_per_split}). "
                        f"Proceeding anyway with multiclass_stats={split_multiclass_stats}."
                    )
            else:
                train_indices, val_indices, test_indices = _do_block_split(int(split_random_state))

            if split_multiclass_stats is not None:
                run_summary["results"]["spatial_block_split_multiclass_stats"] = split_multiclass_stats
        elif split_strategy in ("core_id", "site_group"):

            rey_df = getattr(full_dataset, "rey_data", None)
            if rey_df is None or len(rey_df) == 0:
                raise ValueError(f"Dataset is empty; cannot perform {split_strategy} split.")

            if split_strategy == "site_group":
                group_col = (
                    "site_group"
                    if "site_group" in rey_df.columns
                    else ("core_id" if "core_id" in rey_df.columns else ("site_hole" if "site_hole" in rey_df.columns else None))
                )
            else:
                group_col = "core_id" if "core_id" in rey_df.columns else ("site_hole" if "site_hole" in rey_df.columns else None)
            if group_col is None:
                raise ValueError(
                    f"split_strategy='{split_strategy}' requires a 'site_group', 'core_id', or 'site_hole' column in rey_points.parquet. "
                    "Re-run preprocessing and include 'Site (Hole)' (or an equivalent core-id column)."
                )

            core_ids = rey_df[group_col].astype(str).str.strip()
            missing_mask = core_ids.isna() | core_ids.isin(["", "nan", "None", "<NA>"])
            if bool(missing_mask.any()):
                n_missing = int(missing_mask.sum())
                raise ValueError(
                    f"split_strategy='{split_strategy}' cannot proceed: {n_missing} samples have missing {group_col}. "
                    "Fix the raw Excel field or preprocessing mapping."
                )


            try:
                df_cores = pd.DataFrame(
                    {
                        "lon": rey_df["lon"].astype(float).to_numpy(),
                        "lat": rey_df["lat"].astype(float).to_numpy(),
                        "core_group_col": group_col,
                        "core_id": core_ids.to_numpy(),
                    }
                )
                df_cores.to_csv(Path(output_dirs["splits"]) / f"{split_strategy}_groups.csv", index=False)
            except Exception as e_cores:
                logger.warning(f"Pipeline: failed to persist {split_strategy} groups: {e_cores}")

            task_type_for_split = str(hyperparams.get("task_type", "regression") or "regression").lower()
            thr_ppm_for_split = float(hyperparams.get("classification_threshold_ppm", 1000.0) or 1000.0)


            y_bin = None
            if task_type_for_split == "binary_classification" and "rey" in rey_df.columns:
                y_bin = (rey_df["rey"].astype(float).to_numpy() >= thr_ppm_for_split).astype(np.int64)


            y_multi = None
            n_classes_for_split = 0
            if task_type_for_split == "multiclass_classification":
                try:
                    label_col = hyperparams.get("classification_label_col", None)
                    if label_col and label_col in rey_df.columns:
                        y_multi = rey_df[label_col].astype(int).to_numpy(dtype=np.int64, copy=True)
                        n_classes_for_split = int(np.nanmax(y_multi)) + 1 if y_multi.size else 0
                except Exception:
                    y_multi = None
                    n_classes_for_split = 0

            _mps = training_config.get("min_pos_samples_per_split", 5)
            _mpg = training_config.get("min_pos_groups_per_split", 2)
            _smt = training_config.get("split_max_tries", 50)
            min_pos_samples_per_split = int(5 if _mps is None else _mps)
            min_pos_groups_per_split = int(2 if _mpg is None else _mpg)
            split_max_tries = int(50 if _smt is None else _smt)
            split_try_used = 0
            split_pos_stats = None
            split_multiclass_stats = None

            _mcs = training_config.get("min_class_samples_per_split", 1)
            _smtm = training_config.get("split_max_tries_multiclass", 25)
            min_class_samples_per_split = int(1 if _mcs is None else _mcs)
            split_max_tries_multiclass = int(25 if _smtm is None else _smtm)
            _mcg = training_config.get("min_class_groups_per_split", 0)
            min_class_groups_per_split = int(0 if _mcg is None else _mcg)

            _mtcs = training_config.get("min_train_class_samples", None)
            _mtcg = training_config.get("min_train_class_groups", None)
            min_train_class_samples = (
                int(min_class_samples_per_split) if _mtcs is None else int(_mtcs)
            )
            min_train_class_groups = (
                int(min_class_groups_per_split) if _mtcg is None else int(_mtcg)
            )

            def _count_pos_groups(indices) -> int:
                if y_bin is None:
                    return 0
                idx = np.asarray(indices, dtype=int)
                if idx.size == 0:
                    return 0
                g = core_ids.to_numpy()[idx]
                y = y_bin[idx]

                df_tmp = pd.DataFrame({"g": g, "y": y})
                return int((df_tmp.groupby("g", sort=False)["y"].max() >= 1).sum())

            def _count_pos_samples(indices) -> int:
                if y_bin is None:
                    return 0
                idx = np.asarray(indices, dtype=int)
                if idx.size == 0:
                    return 0
                return int(np.sum(y_bin[idx] >= 1))

            def _do_split(rs: int):
                if group_split_stratify_bins > 1:
                    if y_bin is not None:

                        return _stratified_group_train_val_test_split(
                            dataset_size=dataset_size,
                            groups=core_ids.to_numpy(),
                            targets=y_bin.astype(float),
                            test_split_ratio=test_split_ratio,
                            validation_split_ratio=validation_split_ratio,
                            random_state=rs,
                            n_bins=group_split_stratify_bins,
                            statistic="mean",
                        )
                    if y_multi is not None and n_classes_for_split >= 2:


                        totals = np.bincount(y_multi, minlength=n_classes_for_split).astype(int)
                        present = np.where(totals > 0)[0]
                        rare_class = int(present[np.argmin(totals[present])]) if present.size else 0
                        y_rare = (y_multi == rare_class).astype(float)
                        return _stratified_group_train_val_test_split(
                            dataset_size=dataset_size,
                            groups=core_ids.to_numpy(),
                            targets=y_rare,
                            test_split_ratio=test_split_ratio,
                            validation_split_ratio=validation_split_ratio,
                            random_state=rs,
                            n_bins=group_split_stratify_bins,
                            statistic="sum",
                        )
                    if "rey" in rey_df.columns:
                        return _stratified_group_train_val_test_split(
                            dataset_size=dataset_size,
                            groups=core_ids.to_numpy(),
                            targets=rey_df["rey"].to_numpy(dtype=float),
                            test_split_ratio=test_split_ratio,
                            validation_split_ratio=validation_split_ratio,
                            random_state=rs,
                            n_bins=group_split_stratify_bins,
                            statistic="median",
                        )
                return _group_train_val_test_split(
                    dataset_size=dataset_size,
                    groups=core_ids.to_numpy(),
                    test_split_ratio=test_split_ratio,
                    validation_split_ratio=validation_split_ratio,
                    random_state=rs,
                )


            if y_bin is not None:
                for attempt in range(max(1, split_max_tries)):
                    rs = int(split_random_state) + int(attempt) * 17
                    tr, va, te = _do_split(rs)

                    pos_s = {
                        "train_pos_samples": _count_pos_samples(tr),
                        "val_pos_samples": _count_pos_samples(va),
                        "test_pos_samples": _count_pos_samples(te),
                        "train_pos_groups": _count_pos_groups(tr),
                        "val_pos_groups": _count_pos_groups(va),
                        "test_pos_groups": _count_pos_groups(te),
                    }
                    ok = (
                        pos_s["train_pos_samples"] >= min_pos_samples_per_split
                        and pos_s["val_pos_samples"] >= min_pos_samples_per_split
                        and pos_s["test_pos_samples"] >= min_pos_samples_per_split
                        and pos_s["train_pos_groups"] >= min_pos_groups_per_split
                        and pos_s["val_pos_groups"] >= min_pos_groups_per_split
                        and pos_s["test_pos_groups"] >= min_pos_groups_per_split
                    )
                    if ok:
                        train_indices, val_indices, test_indices = tr, va, te
                        split_try_used = int(attempt) + 1
                        split_pos_stats = pos_s
                        break

                if split_pos_stats is None:
                    train_indices, val_indices, test_indices = _do_split(int(split_random_state))
                    split_try_used = 0
                    split_pos_stats = {
                        "train_pos_samples": _count_pos_samples(train_indices),
                        "val_pos_samples": _count_pos_samples(val_indices),
                        "test_pos_samples": _count_pos_samples(test_indices),
                        "train_pos_groups": _count_pos_groups(train_indices),
                        "val_pos_groups": _count_pos_groups(val_indices),
                        "test_pos_groups": _count_pos_groups(test_indices),
                    }
                    logger.warning(
                        f"Pipeline: {split_strategy} split could not satisfy minimal positive support after "
                        f"{split_max_tries} attempts (min_pos_samples_per_split={min_pos_samples_per_split}, "
                        f"min_pos_groups_per_split={min_pos_groups_per_split}). Proceeding anyway with "
                        f"pos_stats={split_pos_stats}."
                    )
            elif y_multi is not None and n_classes_for_split >= 2:
                totals = np.bincount(y_multi, minlength=n_classes_for_split).astype(int)
                required = np.where(totals >= int(3 * min_class_samples_per_split), 1, 0).astype(int)

                try:
                    req_classes = np.where(required >= 1)[0]
                    if req_classes.size:
                        rare_class = int(req_classes[np.argmin(totals[req_classes])])
                    else:
                        rare_class = int(np.argmin(totals)) if totals.size else 0
                except Exception:
                    rare_class = 0

                def _count_classes(indices):
                    idx = np.asarray(indices, dtype=int)
                    if idx.size == 0:
                        return np.zeros((n_classes_for_split,), dtype=int)
                    return np.bincount(y_multi[idx], minlength=n_classes_for_split).astype(int)

                def _count_groups(indices):
                    idx = np.asarray(indices, dtype=int)
                    if idx.size == 0:
                        return np.zeros((n_classes_for_split,), dtype=int)
                    g = core_ids.to_numpy()[idx]
                    y = y_multi[idx]
                    out = np.zeros((n_classes_for_split,), dtype=int)
                    for c in range(n_classes_for_split):
                        if required[c] == 0:
                            continue
                        out[c] = int(len(set(g[y == c].tolist())))
                    return out


                target_class_for_balanced_split = training_config.get("target_class_for_balanced_split", None)
                target_test_class_samples = training_config.get("target_test_class_samples", None)
                target_test_class_samples_tol = training_config.get(
                    "target_test_class_samples_tol",
                    training_config.get("target_test_class_samples_tolerance", 0),
                )
                try:
                    target_test_class_samples = (
                        None
                        if target_test_class_samples is None
                        else int(target_test_class_samples)
                    )
                except Exception:
                    target_test_class_samples = None
                try:
                    target_test_class_samples_tol = int(target_test_class_samples_tol or 0)
                except Exception:
                    target_test_class_samples_tol = 0
                try:
                    target_class_for_balanced_split = (
                        None
                        if target_class_for_balanced_split is None
                        else int(target_class_for_balanced_split)
                    )
                except Exception:
                    target_class_for_balanced_split = None

                selection_class = int(rare_class)
                if (
                    target_test_class_samples is not None
                    and target_class_for_balanced_split is not None
                    and 0 <= int(target_class_for_balanced_split) < int(n_classes_for_split)
                ):
                    selection_class = int(target_class_for_balanced_split)

                group_split_selection = str(
                    training_config.get("group_split_selection", "balanced")
                    or "balanced"
                ).lower().strip()
                use_balanced_group_split = group_split_selection in {
                    "balanced",
                    "constrained",
                    "class_spatial_balanced",
                    "class-space-balanced",
                }
                spatial_balance_deg = float(training_config.get("group_split_spatial_block_deg", 10.0) or 10.0)
                spatial_labels = _spatial_balance_labels(rey_df, spatial_balance_deg)
                best = None
                best_score = None
                best_key = None
                for attempt in range(max(1, split_max_tries_multiclass)):
                    rs = int(split_random_state) + int(attempt) * 17
                    tr, va, te = _do_split(rs)
                    c_tr = _count_classes(tr)
                    c_va = _count_classes(va)
                    c_te = _count_classes(te)
                    g_tr = _count_groups(tr) if min_class_groups_per_split > 0 else None
                    g_va = _count_groups(va) if min_class_groups_per_split > 0 else None
                    g_te = _count_groups(te) if min_class_groups_per_split > 0 else None
                    ok = True
                    for c in range(n_classes_for_split):
                        if required[c] == 0:
                            continue
                        if (
                            (c_tr[c] < min_train_class_samples)
                            or (c_va[c] < min_class_samples_per_split)
                            or (c_te[c] < min_class_samples_per_split)
                        ):
                            ok = False
                            break
                        if min_class_groups_per_split > 0:
                            if (
                                (g_tr[c] < min_train_class_groups)
                                or (g_va[c] < min_class_groups_per_split)
                                or (g_te[c] < min_class_groups_per_split)
                            ):
                                ok = False
                                break
                    if not ok:
                        continue


                    gtr_sel = int(g_tr[selection_class]) if g_tr is not None and 0 <= selection_class < len(g_tr) else 0
                    ctr_sel = int(c_tr[selection_class]) if 0 <= selection_class < len(c_tr) else 0
                    gva_sel = int(g_va[selection_class]) if g_va is not None and 0 <= selection_class < len(g_va) else 0
                    gte_sel = int(g_te[selection_class]) if g_te is not None and 0 <= selection_class < len(g_te) else 0
                    cva_sel = int(c_va[selection_class]) if 0 <= selection_class < len(c_va) else 0
                    cte_sel = int(c_te[selection_class]) if 0 <= selection_class < len(c_te) else 0

                    balance_components = None
                    if use_balanced_group_split:
                        balance_components = _score_balanced_group_split(
                            train_indices=tr,
                            val_indices=va,
                            test_indices=te,
                            groups=core_ids.to_numpy(),
                            labels=y_multi,
                            n_classes=n_classes_for_split,
                            spatial_labels=spatial_labels,
                            test_split_ratio=test_split_ratio,
                            validation_split_ratio=validation_split_ratio,
                        )
                        score = -float(balance_components["score"])
                    else:

                        score = (1000 * gtr_sel) + (2 * ctr_sel) - (50 * gva_sel) - (50 * gte_sel)
                        score = score - (10 * cva_sel) - (20 * cte_sel)

                    if target_test_class_samples is not None:
                        diff = abs(int(cte_sel) - int(target_test_class_samples))
                        within = 1 if diff <= int(target_test_class_samples_tol) else 0
                        key = (within, -int(diff), float(score))
                    else:
                        key = (float(score),)

                    if best_key is None or key > best_key:
                        best_key = key
                        best_score = score
                        best = (tr, va, te, c_tr, c_va, c_te, g_tr, g_va, g_te, int(attempt) + 1, balance_components)

                if best is not None:
                    tr, va, te, c_tr, c_va, c_te, g_tr, g_va, g_te, retry_used, balance_components = best
                    train_indices, val_indices, test_indices = tr, va, te
                    split_try_used = int(retry_used)
                    split_multiclass_stats = {
                        "class_totals": totals.tolist(),
                        "required_classes": required.tolist(),
                        "min_class_samples_per_split": int(min_class_samples_per_split),
                        "min_class_groups_per_split": int(min_class_groups_per_split),
                        "min_train_class_samples": int(min_train_class_samples),
                        "min_train_class_groups": int(min_train_class_groups),
                        "train_counts": c_tr.tolist(),
                        "val_counts": c_va.tolist(),
                        "test_counts": c_te.tolist(),
                        "train_group_counts": g_tr.tolist() if g_tr is not None else None,
                        "val_group_counts": g_va.tolist() if g_va is not None else None,
                        "test_group_counts": g_te.tolist() if g_te is not None else None,
                        "rare_class_for_selection": int(rare_class),
                        "selection_class": int(selection_class),
                        "target_test_class_samples": (
                            int(target_test_class_samples)
                            if target_test_class_samples is not None
                            else None
                        ),
                        "target_test_class_samples_tol": int(target_test_class_samples_tol),
                        "best_split_score": float(best_score) if best_score is not None else None,
                        "group_split_selection": group_split_selection,
                        "group_split_spatial_block_deg": float(spatial_balance_deg),
                        "balance_components": balance_components,
                    }

                if split_multiclass_stats is None:
                    train_indices, val_indices, test_indices = _do_split(int(split_random_state))
                    c_tr = _count_classes(train_indices)
                    c_va = _count_classes(val_indices)
                    c_te = _count_classes(test_indices)
                    split_multiclass_stats = {
                        "class_totals": totals.tolist(),
                        "required_classes": required.tolist(),
                        "min_class_samples_per_split": int(min_class_samples_per_split),
                        "min_class_groups_per_split": int(min_class_groups_per_split),
                        "train_counts": c_tr.tolist(),
                        "val_counts": c_va.tolist(),
                        "test_counts": c_te.tolist(),
                        "train_group_counts": _count_groups(train_indices).tolist() if min_class_groups_per_split > 0 else None,
                        "val_group_counts": _count_groups(val_indices).tolist() if min_class_groups_per_split > 0 else None,
                        "test_group_counts": _count_groups(test_indices).tolist() if min_class_groups_per_split > 0 else None,
                    }
                    logger.warning(
                        f"Pipeline: {split_strategy} split could not satisfy minimal class support after "
                        f"{split_max_tries_multiclass} attempts (min_class_samples_per_split={min_class_samples_per_split}). "
                        f"Proceeding anyway with multiclass_stats={split_multiclass_stats}."
                    )
            else:
                train_indices, val_indices, test_indices = _do_split(int(split_random_state))

            if split_try_used:
                run_summary["results"][f"{split_strategy}_split_retry_used"] = int(split_try_used)
            if split_pos_stats is not None:
                run_summary["results"][f"{split_strategy}_split_pos_stats"] = split_pos_stats
            if split_multiclass_stats is not None:
                run_summary["results"][f"{split_strategy}_split_multiclass_stats"] = split_multiclass_stats
        elif not forced_split_indices:

            indices = list(range(dataset_size))
            np.random.seed(split_random_state)
            np.random.shuffle(indices)
            test_split_idx = int(np.floor(test_split_ratio * dataset_size))
            train_val_indices, test_indices = indices[test_split_idx:], indices[:test_split_idx]
            val_split_idx = int(np.floor(validation_split_ratio * len(train_val_indices)))
            train_indices, val_indices = train_val_indices[val_split_idx:], train_val_indices[:val_split_idx]

        fixed_test_indices = hyperparams.get("fixed_test_indices", None)
        if fixed_test_indices:
            fixed_test = np.asarray([int(i) for i in fixed_test_indices], dtype=int)
            fixed_test = np.unique(fixed_test[(fixed_test >= 0) & (fixed_test < dataset_size)])
            if fixed_test.size == 0:
                logger.warning("Pipeline: fixed_test_indices was provided but no valid dataset indices were found.")
            else:
                remaining = np.asarray([i for i in range(dataset_size) if i not in set(fixed_test.tolist())], dtype=int)
                if remaining.size == 0:
                    raise ValueError("fixed_test_indices contains all samples; cannot build train/validation splits.")
                try:
                    if split_strategy in ("core_id", "site_group"):
                        rey_df_fixed = getattr(full_dataset, "rey_data", None)
                        if split_strategy == "site_group":
                            group_col_fixed = (
                                "site_group"
                                if rey_df_fixed is not None and "site_group" in rey_df_fixed.columns
                                else ("core_id" if rey_df_fixed is not None and "core_id" in rey_df_fixed.columns else "site_hole")
                            )
                        else:
                            group_col_fixed = "core_id" if rey_df_fixed is not None and "core_id" in rey_df_fixed.columns else "site_hole"
                        from sklearn.model_selection import GroupShuffleSplit

                        groups_fixed = rey_df_fixed[group_col_fixed].astype(str).to_numpy()[remaining]
                        group_split_selection_fixed = str(
                            training_config.get("group_split_selection", "balanced") or "balanced"
                        ).lower().strip()
                        y_fixed_multi = None
                        n_fixed_classes = 0
                        label_col_fixed = hyperparams.get("classification_label_col", None)
                        if label_col_fixed and label_col_fixed in rey_df_fixed.columns:
                            try:
                                y_fixed_multi = rey_df_fixed[label_col_fixed].astype(int).to_numpy(dtype=np.int64, copy=True)
                                n_fixed_classes = int(np.nanmax(y_fixed_multi)) + 1 if y_fixed_multi.size else 0
                            except Exception:
                                y_fixed_multi = None
                                n_fixed_classes = 0
                        if (
                            group_split_selection_fixed in {"balanced", "constrained", "class_spatial_balanced", "class-space-balanced"}
                            and y_fixed_multi is not None
                            and n_fixed_classes >= 2
                        ):
                            best_fixed = None
                            best_fixed_score = None
                            spatial_fixed = _spatial_balance_labels(
                                rey_df_fixed,
                                float(training_config.get("group_split_spatial_block_deg", 10.0) or 10.0),
                            )
                            n_candidates_fixed = max(1, int(training_config.get("split_max_tries_multiclass", 25) or 25))
                            for attempt_fixed in range(n_candidates_fixed):
                                gss_fixed = GroupShuffleSplit(
                                    n_splits=1,
                                    test_size=float(validation_split_ratio),
                                    random_state=int(split_random_state) + 1 + int(attempt_fixed) * 17,
                                )
                                tr_rel, va_rel = next(gss_fixed.split(remaining, groups=groups_fixed))
                                tr_candidate = remaining[tr_rel].astype(int).tolist()
                                va_candidate = remaining[va_rel].astype(int).tolist()
                                components_fixed = _score_balanced_group_split(
                                    train_indices=tr_candidate,
                                    val_indices=va_candidate,
                                    test_indices=fixed_test.astype(int).tolist(),
                                    groups=rey_df_fixed[group_col_fixed].astype(str).to_numpy(),
                                    labels=y_fixed_multi,
                                    n_classes=n_fixed_classes,
                                    spatial_labels=spatial_fixed,
                                    test_split_ratio=float(test_split_ratio),
                                    validation_split_ratio=float(validation_split_ratio),
                                )
                                score_fixed = float(components_fixed["score"])
                                if best_fixed_score is None or score_fixed < best_fixed_score:
                                    best_fixed_score = score_fixed
                                    best_fixed = (tr_candidate, va_candidate, components_fixed)
                            if best_fixed is None:
                                raise RuntimeError("balanced fixed-test validation split produced no candidate")
                            train_indices, val_indices, fixed_val_balance_components = best_fixed
                            run_summary["results"]["fixed_test_balanced_val_components"] = fixed_val_balance_components
                        else:
                            gss_fixed = GroupShuffleSplit(
                                n_splits=1,
                                test_size=float(validation_split_ratio),
                                random_state=int(split_random_state) + 1,
                            )
                            tr_rel, va_rel = next(gss_fixed.split(remaining, groups=groups_fixed))
                            train_indices = remaining[tr_rel].astype(int).tolist()
                            val_indices = remaining[va_rel].astype(int).tolist()
                    else:
                        rng_fixed = np.random.default_rng(int(split_random_state) + 1)
                        shuffled = remaining.copy()
                        rng_fixed.shuffle(shuffled)
                        val_n = max(1, int(np.floor(float(validation_split_ratio) * shuffled.size)))
                        val_indices = shuffled[:val_n].astype(int).tolist()
                        train_indices = shuffled[val_n:].astype(int).tolist()
                    test_indices = fixed_test.astype(int).tolist()
                    run_summary["results"]["fixed_test_indices_used"] = True
                    run_summary["results"]["fixed_test_n_samples"] = int(len(test_indices))
                except Exception as e_fixed:
                    raise RuntimeError(f"Failed to apply fixed_test_indices: {e_fixed}") from e_fixed

        logger.debug(f"Data split: Train={len(train_indices)}, Validation={len(val_indices)}, Test={len(test_indices)}")
        dataset_sizes = {'train': len(train_indices), 'validation': len(val_indices), 'test': len(test_indices)}
        run_summary["results"]["split_sizes"] = dataset_sizes
        run_summary["results"]["split_ratios"] = {
            "test_split_ratio": float(test_split_ratio),
            "validation_split_ratio": float(validation_split_ratio),
        }
        run_summary["results"]["random_state"] = int(hyperparams.get("random_state", 42))
        run_summary["results"]["split_random_state"] = int(split_random_state)  


        try:
            hyperparams["_split_val_indices"] = [int(i) for i in val_indices]
            hyperparams["_split_test_indices"] = [int(i) for i in test_indices]
        except Exception:
            hyperparams["_split_val_indices"] = None
            hyperparams["_split_test_indices"] = None

        try:
            split_diag_paths = save_split_group_diagnostics(
                rey_df=getattr(full_dataset, "rey_data", pd.DataFrame()),
                split_indices={
                    "train": np.asarray(train_indices, dtype=int),
                    "validation": np.asarray(val_indices, dtype=int),
                    "test": np.asarray(test_indices, dtype=int),
                },
                output_dir=output_dirs["splits"],
                label_col=hyperparams.get("classification_label_col")
                if str(hyperparams.get("task_type", "regression") or "regression").lower() == "multiclass_classification"
                else None,
                group_col=(
                    "site_group"
                    if split_strategy == "site_group"
                    else ("core_id" if split_strategy == "core_id" else None)
                ),
                class_names=hyperparams.get("classification_class_names", None),
                relaxed_overlap_col="site_group" if split_strategy == "core_id" else None,
            )
            if split_diag_paths:
                run_summary["artifacts"]["paths"]["split_group_diagnostics"] = split_diag_paths
                diag_json = split_diag_paths.get("split_group_diagnostics_json")
                if diag_json and os.path.exists(diag_json):
                    with open(diag_json, "r", encoding="utf-8") as f:
                        run_summary["results"]["split_group_diagnostics"] = json.load(f)
        except Exception as e_split_diag:
            logger.warning(f"Pipeline: Failed to write split group diagnostics: {e_split_diag}")


        try:
            rey_df = getattr(full_dataset, "rey_data", None)
            if rey_df is not None and len(rey_df):
                age_vals = rey_df["age_ka_proxy"].to_numpy(dtype=float) if "age_ka_proxy" in rey_df.columns else None
                depth_phys = rey_df["depth_physical"].to_numpy(dtype=float) if "depth_physical" in rey_df.columns else None
                dataset_summary_for_report = {
                    "num_points_total": int(getattr(full_dataset, "num_points_loaded", len(rey_df))),
                    "num_points_after_filters": int(len(rey_df)),
                    "depth_feature_type": getattr(full_dataset, "depth_feature_type", "depth"),
                    "max_training_depth_mbsf": max_training_depth_mbsf,
                    "max_training_age_ka_proxy": max_training_age_ka_proxy,
                    "sed_rate_grid_path": sed_rate_nc_path,
                    "depth_mbsf_min": float(np.nanmin(depth_phys)) if depth_phys is not None and np.isfinite(depth_phys).any() else None,
                    "depth_mbsf_max": float(np.nanmax(depth_phys)) if depth_phys is not None and np.isfinite(depth_phys).any() else None,
                    "age_ka_proxy_min": float(np.nanmin(age_vals)) if age_vals is not None and np.isfinite(age_vals).any() else None,
                    "age_ka_proxy_p50": float(np.nanpercentile(age_vals[np.isfinite(age_vals)], 50)) if age_vals is not None and np.isfinite(age_vals).any() else None,
                    "age_ka_proxy_max": float(np.nanmax(age_vals)) if age_vals is not None and np.isfinite(age_vals).any() else None,
                    "split_sizes": dataset_sizes,
                }
                run_summary["results"]["dataset_summary"] = dataset_summary_for_report
        except Exception as e_ds_sum:
            logger.warning(f"Pipeline: Failed to build dataset summary: {e_ds_sum}")

        if not all(len(s) > 0 for s in [train_indices, val_indices, test_indices]):
            raise ValueError("One or more data splits are empty. Check data size and split ratios.")


        sampler_group_ids = None
        if split_strategy == "spatial_block" and block_ids is not None:
            sampler_group_ids = block_ids
        elif split_strategy in ("core_id", "site_group"):
            rey_df = getattr(full_dataset, "rey_data", None)
            if rey_df is not None:
                if split_strategy == "site_group":
                    group_col = (
                        "site_group"
                        if "site_group" in rey_df.columns
                        else ("core_id" if "core_id" in rey_df.columns else ("site_hole" if "site_hole" in rey_df.columns else None))
                    )
                else:
                    group_col = "core_id" if "core_id" in rey_df.columns else ("site_hole" if "site_hole" in rey_df.columns else None)
                if group_col is not None:
                    codes, _ = pd.factorize(rey_df[group_col].astype(str).str.strip(), sort=True)
                    sampler_group_ids = codes.astype(np.int64)

        task_type_for_sampling = str(hyperparams.get("task_type", "regression") or "regression").lower()
        label_col_for_sampling = hyperparams.get("classification_label_col", None)
        n_classes_for_sampling = int(hyperparams.get("num_classes", 0) or 0)
        class_balance_power = float(hyperparams.get("class_balance_power", 0.7) or 0.7)
        class_balance_max_factor = float(hyperparams.get("class_balance_max_factor", 8.0) or 8.0)

        labels_all = None
        rey_df = getattr(full_dataset, "rey_data", None)
        if task_type_for_sampling == "binary_classification":
            thr_ppm = float(hyperparams.get("classification_threshold_ppm", 1000.0) or 1000.0)
            if rey_df is not None and "rey" in rey_df.columns:
                labels_all = (rey_df["rey"].astype(float).to_numpy() >= thr_ppm).astype(int)
                n_classes_for_sampling = 2
        elif task_type_for_sampling == "multiclass_classification":
            if rey_df is not None and label_col_for_sampling and label_col_for_sampling in rey_df.columns:
                labels_all = rey_df[label_col_for_sampling].astype(int).to_numpy()
                if n_classes_for_sampling < 2 and labels_all.size:
                    n_classes_for_sampling = int(np.max(labels_all)) + 1

        if train_sampling_strategy == "block_balanced" and sampler_group_ids is not None:
            train_sampler = make_block_balanced_sampler(
                train_indices=train_indices,
                block_ids=sampler_group_ids,
                config=BlockBalancedSamplingConfig(
                    alpha=block_balance_alpha,
                    min_count=block_balance_min_count,
                    num_samples=len(train_indices),
                    replacement=True,
                    seed=int(hyperparams.get("random_state", 42)),
                ),
            )
        elif train_sampling_strategy == "class_balanced" and labels_all is not None and task_type_for_sampling in ("binary_classification", "multiclass_classification"):
            train_sampler = _make_class_balanced_sampler(
                train_indices=train_indices,
                labels_all=labels_all,
                n_classes=n_classes_for_sampling,
                power=class_balance_power,
                max_factor=class_balance_max_factor,
                seed=int(hyperparams.get("random_state", 42)),
            )
        elif (
            train_sampling_strategy == "block_and_class_balanced"
            and sampler_group_ids is not None
            and labels_all is not None
            and task_type_for_sampling in ("binary_classification", "multiclass_classification")
        ):
            train_sampler = _make_block_and_class_balanced_sampler(
                train_indices=train_indices,
                block_ids_all=sampler_group_ids,
                labels_all=labels_all,
                n_classes=n_classes_for_sampling,
                block_alpha=block_balance_alpha,
                block_min_count=block_balance_min_count,
                class_power=class_balance_power,
                class_max_factor=class_balance_max_factor,
                seed=int(hyperparams.get("random_state", 42)),
            )
        else:
            train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        num_workers = hyperparams.get("dataloader_num_workers", 0)
        pin_memory = torch.cuda.is_available()
        batch_size = hyperparams.get("batch_size", 128)


        train_loader = DataLoader(
            full_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
        try:
            from torch.utils.data.dataset import Subset
        except Exception:
            Subset = None
        if Subset is not None:
            train_eval_dataset = Subset(full_dataset, [int(i) for i in train_indices])
            val_dataset = Subset(full_dataset, [int(i) for i in val_indices])
            test_dataset = Subset(full_dataset, [int(i) for i in test_indices])
            train_eval_loader = DataLoader(train_eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        else:
            train_eval_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices), num_workers=num_workers, pin_memory=pin_memory)
            val_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers, pin_memory=pin_memory)
            test_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers, pin_memory=pin_memory)
        logger.debug(f"DataLoaders created (Batch Size: {batch_size}, Num Workers: {num_workers}, Pin Memory: {pin_memory}).")


        task_type = str(hyperparams.get("task_type", "regression") or "regression").lower()
        if task_type == "binary_classification":
            thr_ppm = float(hyperparams.get("classification_threshold_ppm", 1000.0) or 1000.0)
            try:
                rey_df = getattr(full_dataset, "rey_data", None)
                if rey_df is not None and "rey" in rey_df.columns:
                    y_train = (rey_df["rey"].astype(float).to_numpy()[np.asarray(train_indices, dtype=int)] >= thr_ppm).astype(int)
                    n_pos = int(np.sum(y_train == 1))
                    n_neg = int(np.sum(y_train == 0))
                    if n_pos > 0 and n_neg > 0:
                        pos_weight = float(n_neg) / float(n_pos)
                        hyperparams["bce_pos_weight"] = pos_weight
                        logger.info(
                            "Pipeline: Binary classification enabled. "
                            f"Threshold={thr_ppm:g} ppm, train pos={n_pos}, neg={n_neg}, pos_weight={pos_weight:.3f}."
                        )
                    else:
                        hyperparams["bce_pos_weight"] = None
                        logger.warning(
                            "Pipeline: Binary classification enabled but train split has a single class "
                            f"(pos={n_pos}, neg={n_neg}). AUC will be undefined; consider adjusting threshold/split."
                        )
            except Exception as e_pw:
                logger.warning(f"Pipeline: Failed to compute BCE pos_weight: {e_pw}")
        elif task_type == "multiclass_classification":

            n_classes_total = int(getattr(full_dataset, "num_classes", 0) or 0)
            if n_classes_total < 2:
                try:
                    rey_df = getattr(full_dataset, "rey_data", None)
                    label_col = hyperparams.get("classification_label_col", None)
                    if rey_df is not None and label_col and label_col in rey_df.columns:
                        labels_all = pd.to_numeric(rey_df[label_col], errors="coerce").to_numpy(dtype=float)
                        labels_all = labels_all[np.isfinite(labels_all)]
                        if labels_all.size:
                            n_classes_total = int(np.nanmax(labels_all)) + 1
                except Exception:
                    n_classes_total = 0
            if n_classes_total >= 2:
                hyperparams["num_classes"] = int(n_classes_total)
            else:

                hyperparams["num_classes"] = None

            if float(hyperparams.get("aux_high_grade_bce_weight", 0.0) or 0.0) > 0.0:
                try:
                    target_k = int(hyperparams.get("aux_high_grade_class_index", hyperparams.get("explain_target_class_index", n_classes_total - 1)) or 0)
                    target_k = int(np.clip(target_k, 0, max(int(n_classes_total) - 1, 0)))
                    hyperparams["aux_high_grade_class_index"] = target_k
                    if bool(hyperparams.get("aux_high_grade_auto_pos_weight", True)):
                        rey_df = getattr(full_dataset, "rey_data", None)
                        label_col = hyperparams.get("classification_label_col", None)
                        if rey_df is not None and label_col and label_col in rey_df.columns:
                            y_train_aux = rey_df[label_col].astype(int).to_numpy()[np.asarray(train_indices, dtype=int)]
                            n_pos_aux = int(np.sum(y_train_aux == target_k))
                            n_neg_aux = int(np.sum(y_train_aux != target_k))
                            if n_pos_aux > 0 and n_neg_aux > 0:
                                pos_weight_aux = float(n_neg_aux) / float(n_pos_aux)
                                max_pos_weight_aux = float(hyperparams.get("aux_high_grade_max_pos_weight", 8.0) or 8.0)
                                if max_pos_weight_aux > 0:
                                    pos_weight_aux = min(pos_weight_aux, max_pos_weight_aux)
                                hyperparams["aux_high_grade_pos_weight"] = float(pos_weight_aux)
                                logger.info(
                                    "Pipeline: Auxiliary high-grade BCE enabled. "
                                    f"target_class={target_k}, train pos={n_pos_aux}, neg={n_neg_aux}, "
                                    f"pos_weight={pos_weight_aux:.3f}."
                                )
                            else:
                                hyperparams["aux_high_grade_pos_weight"] = None
                                logger.warning(
                                    "Pipeline: Auxiliary high-grade BCE enabled but train split has a single "
                                    f"one-vs-rest class for target={target_k} (pos={n_pos_aux}, neg={n_neg_aux})."
                                )
                except Exception as e_aux:
                    hyperparams["aux_high_grade_pos_weight"] = None
                    logger.warning(f"Pipeline: Failed to compute auxiliary high-grade BCE pos_weight: {e_aux}")

            if bool(hyperparams.get("use_high_grade_ranking_head", False)):
                try:
                    target_k = int(
                        hyperparams.get(
                            "high_grade_ranking_target_class_index",
                            hyperparams.get("aux_high_grade_class_index", hyperparams.get("explain_target_class_index", n_classes_total - 1)),
                        )
                        or 0
                    )
                    target_k = int(np.clip(target_k, 0, max(int(n_classes_total) - 1, 0)))
                    hyperparams["high_grade_ranking_target_class_index"] = target_k
                    if bool(hyperparams.get("high_grade_ranking_auto_pos_weight", True)):
                        rey_df = getattr(full_dataset, "rey_data", None)
                        label_col = hyperparams.get("classification_label_col", None)
                        if rey_df is not None and label_col and label_col in rey_df.columns:
                            y_train_rank = rey_df[label_col].astype(int).to_numpy()[np.asarray(train_indices, dtype=int)]
                            n_pos_rank = int(np.sum(y_train_rank == target_k))
                            n_neg_rank = int(np.sum(y_train_rank != target_k))
                            if n_pos_rank > 0 and n_neg_rank > 0:
                                pos_weight_rank = float(n_neg_rank) / float(n_pos_rank)
                                max_pos_weight_rank = float(hyperparams.get("high_grade_ranking_max_pos_weight", 8.0) or 8.0)
                                if max_pos_weight_rank > 0:
                                    pos_weight_rank = min(pos_weight_rank, max_pos_weight_rank)
                                hyperparams["high_grade_ranking_pos_weight"] = float(pos_weight_rank)
                                logger.info(
                                    "Pipeline: Learned high-grade ranking head enabled. "
                                    f"target_class={target_k}, train pos={n_pos_rank}, neg={n_neg_rank}, "
                                    f"pos_weight={pos_weight_rank:.3f}."
                                )
                            else:
                                hyperparams["high_grade_ranking_pos_weight"] = None
                                logger.warning(
                                    "Pipeline: Ranking head enabled but train split has a single "
                                    f"one-vs-rest class for target={target_k} (pos={n_pos_rank}, neg={n_neg_rank})."
                                )
                except Exception as e_rank:
                    hyperparams["high_grade_ranking_pos_weight"] = None
                    logger.warning(f"Pipeline: Failed to compute ranking-head pos_weight: {e_rank}")


            try:
                eval_age_domain_ka = data_processing_config.get("training_age_domain_ka", None)
                if eval_age_domain_ka is None:
                    eval_age_domain_ka = data_processing_config.get("max_training_age_ka_proxy", None)
                eval_age_domain_ka = 2580.0 if eval_age_domain_ka is None else float(eval_age_domain_ka)
            except Exception:
                eval_age_domain_ka = 2580.0
            hyperparams["training_age_domain_ka"] = float(eval_age_domain_ka)

            if (
                task_type_tmp == "multiclass_classification"
                and int(hyperparams.get("num_classes", 0) or 0) == 4
                and float(hyperparams.get("aux_main_class_ce_weight", 0.0) or 0.0) > 0.0
                and bool(hyperparams.get("aux_main_class_auto_weights", True))
            ):
                try:
                    rey_df = getattr(full_dataset, "rey_data", None)
                    label_col_aux = hyperparams.get("classification_label_col", None)
                    if rey_df is not None and label_col_aux and label_col_aux in rey_df.columns:
                        y_aux = rey_df[label_col_aux].astype(int).to_numpy()[np.asarray(train_indices, dtype=int)]
                        y_main_aux = np.where(y_aux >= 2, 2, y_aux).astype(int)
                        counts_main = np.bincount(y_main_aux, minlength=3).astype(float)
                        denom = np.where(counts_main > 0, counts_main, 1.0)
                        weights_main = denom.sum() / (denom * 3.0)
                        finite = counts_main > 0
                        if np.any(finite):
                            mean_w = float(np.mean(weights_main[finite]))
                            if np.isfinite(mean_w) and mean_w > 0:
                                weights_main = weights_main / mean_w
                        weights_main = np.clip(weights_main, 0.5, 3.0)
                        weights_main = np.where(counts_main > 0, weights_main, 0.0)
                        hyperparams["aux_main_class_ce_weights"] = weights_main.tolist()
                        logger.info(
                            "Pipeline: Auxiliary main-class CE weights enabled. "
                            f"counts={counts_main.tolist()}, weights={weights_main.tolist()}."
                        )
                except Exception as e_aux_main:
                    hyperparams["aux_main_class_ce_weights"] = None
                    logger.warning(f"Pipeline: Failed to compute auxiliary main-class CE weights: {e_aux_main}")

            for aux_task_name, default_positive in [
                ("aux_high_value", [1, 3]),
                ("aux_noncarbonate", [1, 2, 3]),
                ("aux_class2", [2, 3]),
            ]:
                if float(hyperparams.get(f"{aux_task_name}_bce_weight", 0.0) or 0.0) <= 0.0:
                    continue
                if not bool(hyperparams.get(f"{aux_task_name}_auto_pos_weight", True)):
                    continue
                try:
                    rey_df = getattr(full_dataset, "rey_data", None)
                    label_col_aux = hyperparams.get("classification_label_col", None)
                    raw_positive = hyperparams.get(f"{aux_task_name}_positive_classes", default_positive)
                    if isinstance(raw_positive, str):
                        positive = [int(x.strip()) for x in raw_positive.split(",") if x.strip()]
                    else:
                        positive = [int(x) for x in raw_positive]
                    if rey_df is not None and label_col_aux and label_col_aux in rey_df.columns:
                        y_aux = rey_df[label_col_aux].astype(int).to_numpy()[np.asarray(train_indices, dtype=int)]
                        pos_mask = np.isin(y_aux, np.asarray(positive, dtype=int))
                        n_pos_aux = int(np.sum(pos_mask))
                        n_neg_aux = int(y_aux.size - n_pos_aux)
                        if n_pos_aux > 0 and n_neg_aux > 0:
                            pos_weight_aux = float(n_neg_aux) / float(n_pos_aux)
                            max_pos_weight_aux = float(hyperparams.get(f"{aux_task_name}_max_pos_weight", 8.0) or 8.0)
                            if max_pos_weight_aux > 0:
                                pos_weight_aux = min(pos_weight_aux, max_pos_weight_aux)
                            hyperparams[f"{aux_task_name}_pos_weight"] = float(pos_weight_aux)
                            logger.info(
                                "Pipeline: Auxiliary binary BCE pos_weight enabled. "
                                f"name={aux_task_name}, positive={positive}, pos={n_pos_aux}, neg={n_neg_aux}, "
                                f"pos_weight={pos_weight_aux:.3f}."
                            )
                        else:
                            hyperparams[f"{aux_task_name}_pos_weight"] = None
                except Exception as e_aux_binary:
                    hyperparams[f"{aux_task_name}_pos_weight"] = None
                    logger.warning(f"Pipeline: Failed to compute auxiliary binary pos_weight for {aux_task_name}: {e_aux_binary}")

            use_class_weights = bool(hyperparams.get("use_class_weights", True))
            hierarchical_multiclass = bool(hyperparams.get("hierarchical_multiclass", False))
            hierarchical_grade_multitask = bool(hyperparams.get("hierarchical_grade_multitask", False))


            hierarchical_endmember_use_class_weights = bool(
                hyperparams.get("hierarchical_endmember_use_class_weights", False)
            )


            classification_loss_name = str(hyperparams.get("classification_loss", "ce") or "ce").lower()
            if (not hierarchical_multiclass) and classification_loss_name == "focal" and use_class_weights:
                logger.warning(
                    "Pipeline: Disabling CE class weights because classification_loss='focal' in non-hierarchical "
                    "multiclass. Using both often destabilizes training for rare classes."
                )
                use_class_weights = False
                hyperparams["use_class_weights"] = False
            if not use_class_weights:
                hyperparams["ce_class_weights"] = None
                hyperparams["ce_endmember_class_weights"] = None
                hyperparams["ce_subclass_class_weights"] = None
                hyperparams["ce_grade_class_weights"] = None
            else:
                try:
                    rey_df = getattr(full_dataset, "rey_data", None)
                    label_col = hyperparams.get("classification_label_col", None)
                    if rey_df is None or not label_col or label_col not in rey_df.columns:
                        hyperparams["ce_class_weights"] = None
                        hyperparams["ce_endmember_class_weights"] = None
                        hyperparams["ce_subclass_class_weights"] = None
                        hyperparams["ce_grade_class_weights"] = None
                    else:
                        y_train = rey_df[label_col].astype(int).to_numpy()[np.asarray(train_indices, dtype=int)]
                        n_classes = int(hyperparams.get("num_classes", 0) or 0)
                        if n_classes < 2:
                            n_classes = int(np.max(y_train)) + 1 if y_train.size else 0

                        def _inv_freq_weights(counts_arr: np.ndarray) -> tuple[list[float], list[float]]:
                            counts_arr = np.asarray(counts_arr, dtype=float)
                            missing_arr = counts_arr <= 0
                            denom_arr = np.where(counts_arr > 0, counts_arr, 1.0)
                            inv_arr = denom_arr.sum() / (denom_arr * float(max(len(counts_arr), 1)))
                            mean_inv_arr = float(np.mean(inv_arr[~missing_arr])) if np.any(~missing_arr) else float(np.mean(inv_arr))
                            if np.isfinite(mean_inv_arr) and mean_inv_arr > 0:
                                inv_arr = inv_arr / mean_inv_arr
                            inv_arr = np.clip(inv_arr, 0.5, 3.0)
                            inv_arr = np.where(missing_arr, 0.0, inv_arr)
                            return inv_arr.tolist(), counts_arr.tolist()

                        counts_full = np.bincount(y_train, minlength=max(n_classes, 1)).astype(float)
                        hyperparams["class_counts"] = counts_full.tolist()
                        if float(np.sum(counts_full)) > 0:
                            hyperparams["class_priors"] = (counts_full / np.clip(np.sum(counts_full), 1e-12, None)).tolist()
                        w_full, _ = _inv_freq_weights(counts_full)
                        hyperparams["ce_class_weights"] = w_full

                        if hierarchical_multiclass and n_classes >= 3:
                            y_end = (y_train > 0).astype(int)
                            counts_end = np.bincount(y_end, minlength=2).astype(float)
                            if hierarchical_endmember_use_class_weights:
                                w_end, _ = _inv_freq_weights(counts_end)
                                hyperparams["ce_endmember_class_weights"] = w_end
                            else:
                                w_end = None
                                hyperparams["ce_endmember_class_weights"] = None

                            y_non = y_train[y_train > 0] - 1
                            if hierarchical_grade_multitask and n_classes == 5:
                                y_sub = y_non // 2
                                y_grade = y_non % 2
                                n_sub = 2
                                counts_grade = np.bincount(y_grade, minlength=2).astype(float)
                                w_grade, _ = _inv_freq_weights(counts_grade)
                                hyperparams["ce_grade_class_weights"] = w_grade
                            else:
                                y_sub = y_non
                                n_sub = int(n_classes - 1)
                                hyperparams["ce_grade_class_weights"] = None
                            counts_sub = np.bincount(y_sub, minlength=max(n_sub, 1)).astype(float)
                            w_sub, _ = _inv_freq_weights(counts_sub)
                            hyperparams["ce_subclass_class_weights"] = w_sub
                            logger.info(
                                "Pipeline: Hierarchical multiclass weights enabled. "
                                f"mode={'grade_multitask_3head' if (hierarchical_grade_multitask and n_classes == 5) else 'legacy_2head'}, "
                                f"endmember_weights={w_end}, subclass_weights={w_sub}, "
                                f"grade_weights={hyperparams.get('ce_grade_class_weights')}"
                            )
                        else:
                            hyperparams["ce_endmember_class_weights"] = None
                            hyperparams["ce_subclass_class_weights"] = None
                            hyperparams["ce_grade_class_weights"] = None

                        logger.info(
                            "Pipeline: Multiclass classification enabled. "
                            f"label_col={label_col}, n_classes={n_classes}, class_weights={hyperparams['ce_class_weights']}"
                        )
                except Exception as e_cw:
                    hyperparams["ce_class_weights"] = None
                    hyperparams["ce_endmember_class_weights"] = None
                    hyperparams["ce_subclass_class_weights"] = None
                    hyperparams["ce_grade_class_weights"] = None
                    logger.warning(f"Pipeline: Failed to compute CE class weights: {e_cw}")


        add_missingness_channels = bool(hyperparams.get("add_env_missingness_channels", False))
        env_missingness_channel_count = len(selected_env_variables) if add_missingness_channels else 0
        input_channel_count = (
            len(selected_env_variables)
            + env_missingness_channel_count
            + 1
            + (4 if bool(hyperparams.get("add_coord_channels", False)) else 0)
        )
        trainer = ModelTrainer(
            model_config=hyperparams,
            # Input channels: env (+ optional missingness masks) + depth (+ optional coord channels)
            input_channels=input_channel_count,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            output_dirs=output_dirs,
            stop_flag=stop_flag,
            progress_callback=progress_callback,
            history_callback=history_callback
        )


        training_successful, training_artifacts = trainer.train_and_evaluate(
            train_loader, val_loader, test_loader,
            epochs=hyperparams.get('epochs', 200),
            early_stopping_patience=hyperparams.get('early_stopping_patience', 20),
            full_dataset_for_denorm=full_dataset,
            train_eval_loader=train_eval_loader,
        )

        run_summary["results"].update(training_artifacts)
        run_summary["artifacts"]["paths"].update(training_artifacts.get("plot_paths", {}))

        if not training_successful:
            raise RuntimeError(f"Training failed. Reason: {training_artifacts.get('error_message', 'Unknown')}")


        diagnostics_results = {}
        task_type_for_diag = str(hyperparams.get("task_type", "regression") or "regression").lower()


        try:
            if task_type_for_diag in ("binary_classification", "multiclass_classification"):
                logger.info("Pipeline: Skipping depth-feature-only regression baseline for classification.")
                raise RuntimeError("skip_classification")
            logger.info("Pipeline: Running depth-feature-only baseline diagnostic...")
            rey_df = getattr(full_dataset, "rey_data", None)
            if rey_df is not None and all(col in rey_df.columns for col in ["depth", "rey"]):
                depth_values = rey_df["depth"].to_numpy(dtype=float)
                rey_values = rey_df["rey"].to_numpy(dtype=float)

                train_idx_for_baseline = np.array(train_indices, dtype=int)
                test_idx_for_baseline = np.array(test_indices, dtype=int)

                if train_idx_for_baseline.size > 1 and test_idx_for_baseline.size > 0:
                    x_train = depth_values[train_idx_for_baseline]
                    y_train = rey_values[train_idx_for_baseline]
                    x_test = depth_values[test_idx_for_baseline]
                    y_test = rey_values[test_idx_for_baseline]


                    coeffs = np.polyfit(x_train, y_train, deg=2)
                    y_pred_baseline = np.polyval(coeffs, x_test)

                    residuals_baseline = y_test - y_pred_baseline
                    mse_baseline = float(np.mean(residuals_baseline ** 2))
                    mae_baseline = float(np.mean(np.abs(residuals_baseline)))
                    if np.allclose(y_test, y_test.mean()):
                        r2_baseline = float("nan")
                    else:
                        ss_res = float(np.sum(residuals_baseline ** 2))
                        ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
                        r2_baseline = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

                    diagnostics_results["depth_only_baseline"] = {
                        "depth_feature_type": getattr(full_dataset, "depth_feature_type", "depth"),
                        "mse": mse_baseline,
                        "mae": mae_baseline,
                        "r2": r2_baseline
                    }
                    logger.info(
                        f"Pipeline: Depth-feature-only baseline metrics - MSE={mse_baseline:.4f}, "
                        f"MAE={mae_baseline:.4f}, R2={r2_baseline:.4f}"
                    )
                else:
                    logger.warning("Pipeline: Not enough samples for depth-feature-only baseline diagnostic.")
            else:
                logger.warning("Pipeline: Skipping depth-feature-only baseline; 'depth' or 'rey' column missing in rey_data.")
        except Exception as e_depth_diag:
            if str(e_depth_diag) != "skip_classification":
                logger.warning(f"Pipeline: Depth-feature-only baseline diagnostic failed: {e_depth_diag}")


        try:

            device_for_eval = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if bool(hyperparams.get("add_coord_channels", False)) and bool(hyperparams.get("use_coord_branch", False)):
                logger.info("Pipeline: Running coord-branch diagnostics...")
                coord_diag = _compute_coord_branch_diagnostics(
                    model=trainer.model,
                    dataset=full_dataset,
                    test_indices=test_indices,
                    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                    output_dir=output_dirs["diagnostics"],
                    batch_size=int(batch_size) if batch_size is not None else 256,
                )
                if coord_diag:
                    diagnostics_results["coord_branch"] = coord_diag
        except Exception as e_coord_diag:
            logger.warning(f"Pipeline: Coord-branch diagnostics failed: {e_coord_diag}")


        try:
            if task_type_for_diag in ("binary_classification", "multiclass_classification"):
                logger.info("Pipeline: Skipping age-binned regression metrics for classification.")
                raise RuntimeError("skip_classification")
            logger.info("Pipeline: Computing age-binned evaluation metrics...")
            predictions_dir = os.path.join(output_dirs["diagnostics"], "predictions_tables")

            age_bins_ka = hyperparams.get("age_bins_ka", None) or [0.0, 200.0, 1000.0, 2580.0]
            min_age_bin_samples = int(hyperparams.get("min_age_bin_samples", 25))
            auto_merge_age_bins = bool(hyperparams.get("auto_merge_age_bins", True))
            include_older_eval = bool(hyperparams.get("include_older_samples_in_age_eval", True))
            merge_under_ka = float(hyperparams.get("merge_age_bins_under_ka", 1000.0))
            force_merge_under = bool(hyperparams.get("force_merge_age_bins_under_ka", True))

            def _force_merge_under_threshold(bins_in: list, threshold: float) -> list[float]:
                try:
                    bins = sorted({float(x) for x in bins_in})
                except Exception:
                    return [0.0, float(threshold)]
                if len(bins) < 2:
                    return [0.0, float(threshold)]
                thr = float(threshold)
                left = bins[0]
                keep = [left]
                if thr > left:
                    keep.append(thr)
                keep.extend([b for b in bins[1:] if (b >= thr) or np.isinf(b)])
                keep = sorted({float(x) for x in keep})
                if len(keep) < 2:
                    keep = [left, thr]
                return keep

            if force_merge_under and np.isfinite(merge_under_ka):
                age_bins_ka = _force_merge_under_threshold(age_bins_ka, merge_under_ka)
            run_age_binned = bool(hyperparams.get("run_age_binned_metrics", True))
            if run_age_binned:
                age_binned_results = _compute_age_binned_metrics(
                    model=trainer.model,
                    dataset=full_dataset,
                    test_indices=test_indices,
                    device=device_for_eval,
                    output_dir=output_dirs["diagnostics"],
                    age_bins_ka=age_bins_ka,
                    batch_size=batch_size,
                    min_samples_per_bin=min_age_bin_samples,
                    auto_merge_bins=auto_merge_age_bins,
                    output_suffix="",
                )
                if age_binned_results:
                    diagnostics_results["age_binned_metrics"] = age_binned_results
            else:
                logger.info("Pipeline: Skipping age-binned metrics as per config.")


            if run_age_binned and include_older_eval and (max_training_age_ka_proxy is not None) and sed_rate_nc_path:
                try:
                    eval_dataset = REYPointDataset(
                        rey_points_path=rey_points_path,
                        env_grid_path=env_grid_path,
                        feature_names=selected_env_variables,
                        patch_size=patch_size,
                        depth_transform_method=depth_transform_method,
                        env_normalization_method=env_normalization_method,
                        augment_data=False,
                        augmentation_degrees=0.0,
                        shift_augmentation_pixels=0,
                        use_weighted_loss=False,
                        weighting_strategy=hyperparams.get("weighting_strategy", "linear"),
                        max_weight=hyperparams.get("max_weight", 5.0),
                        max_depth_mbsf=max_training_depth_mbsf,
                        max_age_ka_proxy=None,
                        sed_rate_nc_path=sed_rate_nc_path,
                        depth_weighting_scheme=None,
                        norm_params_override=getattr(full_dataset, "norm_params", None),
                        add_coord_channels=bool(hyperparams.get("add_coord_channels", False)),
                        add_env_missingness_channels=bool(hyperparams.get("add_env_missingness_channels", False)),
                    )


                    src_index_vals = full_dataset.rey_data.index.to_numpy()
                    test_src_index_vals = src_index_vals[np.asarray(test_indices, dtype=int)]
                    mapped_test_pos = eval_dataset.rey_data.index.get_indexer(test_src_index_vals)
                    mapped_test_pos = mapped_test_pos[mapped_test_pos >= 0]

                    ages_eval = eval_dataset.rey_data["age_ka_proxy"].astype(float).to_numpy()
                    older_pos = np.where(np.isfinite(ages_eval) & (ages_eval > float(max_training_age_ka_proxy)))[0]
                    eval_indices = np.unique(np.concatenate([mapped_test_pos, older_pos])).tolist()


                    eval_bins = sorted({float(x) for x in age_bins_ka} | {float(max_training_age_ka_proxy), float("inf")})
                    if force_merge_under and np.isfinite(merge_under_ka):
                        eval_bins = _force_merge_under_threshold(eval_bins, merge_under_ka)

                    age_binned_extended = _compute_age_binned_metrics(
                        model=trainer.model,
                        dataset=eval_dataset,
                        test_indices=eval_indices,
                        device=device_for_eval,
                        output_dir=output_dirs["diagnostics"],
                        age_bins_ka=eval_bins,
                        batch_size=batch_size,
                        min_samples_per_bin=min_age_bin_samples,
                        auto_merge_bins=auto_merge_age_bins,
                        output_suffix="_extended",
                    )
                    if age_binned_extended:
                        diagnostics_results["age_binned_metrics_extended"] = age_binned_extended


                    if older_pos.size:
                        older_pred_csv = _save_predictions_table(
                            model=trainer.model,
                            dataset=eval_dataset,
                            indices=older_pos.tolist(),
                            device=device_for_eval,
                            output_csv_path=os.path.join(predictions_dir, "older_than_training_age_predictions_with_metadata.csv"),
                            split_name="older_than_training_age",
                            age_bins_ka=eval_bins,
                            batch_size=batch_size,
                        )
                        diagnostics_results.setdefault("prediction_tables", {})
                        diagnostics_results["prediction_tables"]["older_csv"] = older_pred_csv
                except Exception as e_ext:
                    logger.warning(f"Pipeline: Extended age-binned evaluation failed: {e_ext}")


            train_pred_csv = _save_predictions_table(
                model=trainer.model,
                dataset=full_dataset,
                indices=train_indices,
                device=device_for_eval,
                output_csv_path=os.path.join(predictions_dir, "train_predictions_with_metadata.csv"),
                split_name="train",
                age_bins_ka=age_bins_ka,
                batch_size=batch_size,
            )
            test_pred_csv = _save_predictions_table(
                model=trainer.model,
                dataset=full_dataset,
                indices=test_indices,
                device=device_for_eval,
                output_csv_path=os.path.join(predictions_dir, "test_predictions_with_metadata.csv"),
                split_name="test",
                age_bins_ka=age_bins_ka,
                batch_size=batch_size,
            )
            diagnostics_results.setdefault("prediction_tables", {})
            diagnostics_results["prediction_tables"]["train_csv"] = train_pred_csv
            diagnostics_results["prediction_tables"]["test_csv"] = test_pred_csv

        except Exception as e_age_bins:
            if str(e_age_bins) != "skip_classification":
                logger.warning(f"Pipeline: Age-binned metrics / prediction tables computation failed: {e_age_bins}")


        run_env_shuffle_eval = hyperparams.get("run_spatial_shuffle_eval", True)
        if task_type_for_diag in ("binary_classification", "multiclass_classification"):
            run_env_shuffle_eval = False
        if run_env_shuffle_eval:
            try:
                logger.info("Pipeline: Running env-shuffle evaluation diagnostic...")
                trainer.model.eval()
                all_targets_denorm_shuffle, all_preds_denorm_shuffle = [], []

                with torch.no_grad():
                    for data in test_loader:
                        combined_patches, target_reys, _ = data[0], data[1], data[2]
                        current_batch_size = combined_patches.size(0)
                        if current_batch_size == 0:
                            continue

                        combined_patches = combined_patches.to(
                            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        )


                        num_channels_total = combined_patches.size(1)
                        if num_channels_total <= 1:
                            shuffled_patches = combined_patches
                        else:
                            num_env_channels = num_channels_total - 1
                            env_part = combined_patches[:, :num_env_channels, :, :]
                            depth_part = combined_patches[:, num_env_channels:, :, :]
                            perm = torch.randperm(current_batch_size, device=combined_patches.device)
                            env_shuffled = env_part[perm, :, :, :]
                            shuffled_patches = torch.cat([env_shuffled, depth_part], dim=1)

                        with torch.amp.autocast(
                            device_type=shuffled_patches.device.type,
                            enabled=torch.cuda.is_available()
                        ):
                            preds_norm_shuffle = trainer.model(shuffled_patches)

                        preds_norm_np = preds_norm_shuffle.squeeze(-1).cpu().numpy()
                        targets_norm_np = target_reys.squeeze(-1).numpy()

                        preds_denorm_batch = full_dataset._denormalize_rey(preds_norm_np)
                        targets_denorm_batch = full_dataset._denormalize_rey(targets_norm_np)

                        all_preds_denorm_shuffle.append(preds_denorm_batch)
                        all_targets_denorm_shuffle.append(targets_denorm_batch)

                if all_preds_denorm_shuffle and all_targets_denorm_shuffle:
                    y_pred = np.concatenate(all_preds_denorm_shuffle)
                    y_true = np.concatenate(all_targets_denorm_shuffle)

                    residuals_shuffle = y_true - y_pred
                    mse_shuffle = float(np.mean(residuals_shuffle ** 2))
                    mae_shuffle = float(np.mean(np.abs(residuals_shuffle)))
                    if np.allclose(y_true, y_true.mean()):
                        r2_shuffle = float("nan")
                    else:
                        ss_res_s = float(np.sum(residuals_shuffle ** 2))
                        ss_tot_s = float(np.sum((y_true - y_true.mean()) ** 2))
                        r2_shuffle = float(1.0 - ss_res_s / ss_tot_s) if ss_tot_s > 0 else float("nan")

                    diagnostics_results["env_shuffle_eval"] = {
                        "mse": mse_shuffle,
                        "mae": mae_shuffle,
                        "r2": r2_shuffle
                    }
                    logger.info(
                        f"Pipeline: Env-shuffle evaluation metrics - MSE={mse_shuffle:.4f}, "
                        f"MAE={mae_shuffle:.4f}, R2={r2_shuffle:.4f}"
                    )
                else:
                    logger.warning("Pipeline: Env-shuffle evaluation skipped due to empty predictions/targets.")
            except Exception as e_shuffle:
                logger.warning(f"Pipeline: Env-shuffle evaluation diagnostic failed: {e_shuffle}")

        if diagnostics_results:
            run_summary["results"].setdefault("diagnostics", {}).update(diagnostics_results)


        training_history = training_artifacts.get("training_history")
        if training_history:
            try:
                history_df = pd.DataFrame(training_history)


                if "epoch" in history_df.columns:
                    history_df = history_df.sort_values("epoch").groupby("epoch", as_index=False).last()
                history_csv_path = os.path.join(output_dirs["diagnostics"], "training_history.csv")
                history_df.to_csv(history_csv_path, index=False)
                run_summary["artifacts"]["paths"]["training_history_csv"] = history_csv_path
                logger.info(f"Pipeline: Training history saved to {history_csv_path}")


                plt.figure(figsize=(10, 6))
                plt.plot(history_df['epoch'], history_df['train_loss'], label='Train Loss', linewidth=2.0)
                plt.plot(history_df['epoch'], history_df['val_loss'], label='Validation Loss', linewidth=2.0)
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Training and Validation Loss Curve')
                plt.legend()
                plt.grid(True)
                loss_curve_path = os.path.join(output_dirs["diagnostics"], "loss_curve.png")
                plt.tight_layout()
                plt.savefig(loss_curve_path, dpi=220)
                plt.close()
                run_summary["artifacts"]["paths"]["loss_curve_plot"] = loss_curve_path
                logger.info(f"Pipeline: Loss curve plot saved to {loss_curve_path}")


                try:
                    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                    ax = axes[0, 0]
                    ax.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss", linewidth=2.0)
                    ax.plot(history_df["epoch"], history_df["val_loss"], label="Validation Loss", linewidth=2.0)
                    ax.set_title("Loss")
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("Loss")
                    ax.grid(True, linestyle="--", alpha=0.25)
                    ax.legend(loc="best", frameon=True)

                    ax = axes[0, 1]
                    if "val_macro_f1" in history_df.columns:
                        ax.plot(history_df["epoch"], history_df["val_macro_f1"], linewidth=2.0, color="#377eb8")
                    ax.set_title("Val macro-F1")
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("F1")
                    ax.set_ylim(0.0, 1.0)
                    ax.grid(True, linestyle="--", alpha=0.25)

                    ax = axes[1, 0]
                    if "val_target_class_ap" in history_df.columns:
                        ax.plot(history_df["epoch"], history_df["val_target_class_ap"], linewidth=2.0, color="#e41a1c")
                    elif "val_class1_ap" in history_df.columns:
                        ax.plot(history_df["epoch"], history_df["val_class1_ap"], linewidth=2.0, color="#e41a1c")
                    ax.set_title("Val target-class AP")
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("AP")
                    ax.set_ylim(0.0, 1.0)
                    ax.grid(True, linestyle="--", alpha=0.25)

                    ax = axes[1, 1]
                    if "lr" in history_df.columns:
                        ax.plot(history_df["epoch"], history_df["lr"], linewidth=2.0, color="#4daf4a")
                        try:
                            ax.set_yscale("log")
                        except Exception:
                            pass
                    ax.set_title("Learning rate")
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("LR")
                    ax.grid(True, linestyle="--", alpha=0.25)

                    fig.tight_layout()
                    curves_path = os.path.join(output_dirs["diagnostics"], "training_curves.png")
                    fig.savefig(curves_path, dpi=220)
                    plt.close(fig)
                    run_summary["artifacts"]["paths"]["training_curves_plot"] = curves_path
                    logger.info(f"Pipeline: Training curves plot saved to {curves_path}")
                except Exception as e_curves:
                    logger.debug(f"Pipeline: Failed to save additional training curves: {e_curves}")

            except Exception as e_hist:
                logger.warning(f"Pipeline: Failed to process or save training history: {e_hist}")


        logger.info("Pipeline: Consolidating and saving training artifacts...")


        coord_feature_names = ["sin_lat", "cos_lat", "sin_lon", "cos_lon"] if bool(hyperparams.get("add_coord_channels", False)) else []
        coord_feature_dim = 4 if coord_feature_names else 0
        add_env_missingness_channels = bool(hyperparams.get("add_env_missingness_channels", False))
        env_missingness_feature_names = (
            [f"{name}__missing" for name in selected_env_variables]
            if add_env_missingness_channels
            else []
        )

        categorical_channel_indices = [
            int(i)
            for i, n in enumerate(selected_env_variables)
            if "__cat_" in str(n)
        ]


        hyperparams["categorical_channel_indices"] = (
            categorical_channel_indices if categorical_channel_indices else None
        )
        model_init_config = ModelInitConfig(
            model_type=hyperparams.get('model_type', 'mixture_profiles'),
            input_channels=len(selected_env_variables) + len(env_missingness_feature_names) + 1 + coord_feature_dim,
            cnn_filters=hyperparams.get('cnn_filters', [64, 128, 128]),
            kernel_size=hyperparams.get('kernel_size', 3),
            mlp_hidden_layers=hyperparams.get('mlp_hidden_layers', [128, 64]),
            dropout_prob=hyperparams.get('dropout_prob', 0.3),
            linear_attention_heads=int(hyperparams.get("linear_attention_heads", 4) or 4),
            use_linear_attention=bool(hyperparams.get("use_linear_attention", True)),
            attention_bottleneck_factor=hyperparams.get('attention_bottleneck_factor', 2),
            patch_size=list(patch_size),
            feature_names=selected_env_variables,
            categorical_channel_indices=categorical_channel_indices if categorical_channel_indices else None,
            categorical_compress_dim=hyperparams.get("categorical_compress_dim", None),
            categorical_compress_dropout=hyperparams.get("categorical_compress_dropout", None),

            use_multiscale_env=hyperparams.get('use_multiscale_env', False),
            multiscale_kernel_size=hyperparams.get('multiscale_kernel_size', 5),
            use_film=hyperparams.get('use_film', False),
            film_embedding_dim=hyperparams.get('film_embedding_dim', 32),
            depth_feature_order=hyperparams.get('depth_feature_order', 1),

            num_profiles=hyperparams.get('num_profiles'),
            env_embedding_dim=hyperparams.get('env_embedding_dim'),
            profile_hidden_layers=hyperparams.get('profile_hidden_layers'),
            num_classes=int(getattr(full_dataset, "num_classes", None) or 0) or None,
            class_head_hidden_dim=(
                int(hyperparams.get("class_head_hidden_dim", 64) or 64)
                if (str(hyperparams.get("task_type", "regression") or "regression").lower() == "multiclass_classification"
                    and not bool(hyperparams.get("hierarchical_multiclass", False)))
                else None
            ),
            use_high_grade_ranking_head=(
                bool(hyperparams.get("use_high_grade_ranking_head", False))
                if (str(hyperparams.get("task_type", "regression") or "regression").lower() == "multiclass_classification"
                    and not bool(hyperparams.get("hierarchical_multiclass", False)))
                else None
            ),
            high_grade_ranking_head_hidden_dim=(
                int(hyperparams.get("high_grade_ranking_head_hidden_dim", 64) or 64)
                if (str(hyperparams.get("task_type", "regression") or "regression").lower() == "multiclass_classification"
                    and not bool(hyperparams.get("hierarchical_multiclass", False))
                    and bool(hyperparams.get("use_high_grade_ranking_head", False)))
                else None
            ),
            hierarchical_multiclass=bool(hyperparams.get("hierarchical_multiclass", False)) if str(hyperparams.get("task_type", "regression") or "regression").lower() == "multiclass_classification" else None,
            hierarchical_grade_multitask=(
                bool(hyperparams.get("hierarchical_grade_multitask", False))
                if (
                    str(hyperparams.get("task_type", "regression") or "regression").lower() == "multiclass_classification"
                    and bool(hyperparams.get("hierarchical_multiclass", False))
                )
                else None
            ),
            endmember_head_hidden_dim=int(hyperparams.get("endmember_head_hidden_dim", 64) or 64) if bool(hyperparams.get("hierarchical_multiclass", False)) else None,
            subclass_head_hidden_dim=int(hyperparams.get("subclass_head_hidden_dim", 64) or 64) if bool(hyperparams.get("hierarchical_multiclass", False)) else None,
            grade_head_hidden_dim=int(hyperparams.get("grade_head_hidden_dim", 64) or 64) if bool(hyperparams.get("hierarchical_grade_multitask", False)) else None,
            coord_feature_dim=coord_feature_dim if coord_feature_dim else None,
            coord_feature_names=coord_feature_names if coord_feature_names else None,
            add_env_missingness_channels=add_env_missingness_channels if add_env_missingness_channels else None,
            env_missingness_feature_names=env_missingness_feature_names if env_missingness_feature_names else None,
            use_coord_branch=bool(hyperparams.get("use_coord_branch", False)) if coord_feature_dim else None,
            coord_branch_hidden_dim=int(hyperparams.get("coord_branch_hidden_dim", 8)) if coord_feature_dim else None,
            coord_branch_dropout=float(hyperparams.get("coord_branch_dropout", 0.5)) if coord_feature_dim else None,
            coord_branch_weight=float(hyperparams.get("coord_branch_weight", 1.0)) if coord_feature_dim else None,
        )


        source_norm_params = full_dataset.norm_params

        task_type = str(hyperparams.get("task_type", "regression") or "regression").lower()
        if 'rey' in source_norm_params and isinstance(source_norm_params['rey'], dict):
            if task_type == "binary_classification":
                source_norm_params["rey"]["transform"] = None
                source_norm_params["rey"]["task_type"] = "binary_classification"
                source_norm_params["rey"]["threshold_ppm"] = float(hyperparams.get("classification_threshold_ppm", 1000.0) or 1000.0)

                source_norm_params["rey"]["mean"] = float(source_norm_params["rey"].get("mean", 0.0) or 0.0)
                source_norm_params["rey"]["std"] = float(source_norm_params["rey"].get("std", 1.0) or 1.0)
            elif task_type == "multiclass_classification":
                source_norm_params["rey"]["transform"] = None
                source_norm_params["rey"]["task_type"] = "multiclass_classification"

                source_norm_params["rey"]["mean"] = float(source_norm_params["rey"].get("mean", 0.0) or 0.0)
                source_norm_params["rey"]["std"] = float(source_norm_params["rey"].get("std", 1.0) or 1.0)

                try:
                    source_norm_params["rey"]["num_classes"] = int(getattr(full_dataset, "num_classes", None) or 0) or None
                except Exception:
                    source_norm_params["rey"]["num_classes"] = None
                source_norm_params["rey"]["class_names"] = hyperparams.get("classification_class_names", None)
            else:

                source_norm_params["rey"]["transform"] = "log1p"
                source_norm_params["rey"]["task_type"] = "regression"


        source_norm_params["depth_transform_method"] = depth_transform_method
        try:
            rey_df = getattr(full_dataset, "rey_data", None)
            if rey_df is not None and "depth" in rey_df.columns:
                depth_feature_min = float(rey_df["depth"].min())
                depth_feature_max = float(rey_df["depth"].max())
                depth_feature_type = getattr(full_dataset, "depth_feature_type", None) or "depth_mbsf"

                source_norm_params["depth_feature_type"] = depth_feature_type
                source_norm_params["depth_feature_min"] = depth_feature_min
                source_norm_params["depth_feature_max"] = depth_feature_max

                # Backward-compatible aliases
                source_norm_params["depth_min"] = depth_feature_min
                source_norm_params["depth_max"] = depth_feature_max

                logger.info(
                    "Pipeline: Recorded depth-feature range from dataset: "
                    f"type={depth_feature_type}, min={depth_feature_min}, max={depth_feature_max}"
                )
            else:
                logger.warning("Pipeline: Could not record depth range; 'depth' column missing in rey_data.")
        except Exception as e_depth_range:
            logger.warning(f"Pipeline: Failed to compute depth range from dataset: {e_depth_range}")

        norm_params = NormalizationParams(**source_norm_params)


        training_artifacts_obj = TrainingArtifacts(
            model_path=training_artifacts.get("model_path"),
            model_init_config=model_init_config,
            norm_params=norm_params
        )


        artifacts_path = os.path.join(output_dirs["models"], "training_artifacts.json")
        with open(artifacts_path, 'w') as f:
            f.write(training_artifacts_obj.model_dump_json(indent=4))

        run_summary["artifacts"]["paths"]["training_artifacts"] = artifacts_path
        logger.info(f"Pipeline: Training artifacts saved to {artifacts_path}")

        task_type_for_explain = str(hyperparams.get("task_type", "regression") or "regression").lower()


        if hyperparams.get("run_shap_analysis", True):
            logger.info("Pipeline: Starting SHAP analysis...")
            if progress_callback: progress_callback(90, "Starting SHAP analysis...")

            explain_target_class_index = None
            try:
                explain_target_class_index = hyperparams.get("explain_target_class_index", None)
                if explain_target_class_index is None:

                    explain_target_class_index = hyperparams.get("multiclass_target_class_index", 1)
                explain_target_class_index = int(explain_target_class_index)
            except Exception:
                explain_target_class_index = 1

            shap_results = analyze_shap_values(
                model=trainer.model,
                train_loader=train_loader,
                test_loader=test_loader,
                feature_names=selected_env_variables,
                output_dir=output_dirs["shap_explainability"],
                plot_config=plot_config
                ,
                depth_feature_type=getattr(full_dataset, "depth_feature_type", None),
                add_coord_channels=bool(hyperparams.get("add_coord_channels", False)),
                task_type=task_type_for_explain,
                target_class_index=explain_target_class_index,
            )
            run_summary["artifacts"]["paths"]["shap_plots"] = shap_results
            if "error_message_shap" in shap_results and shap_results["error_message_shap"]:
               logger.warning(f"Pipeline: SHAP analysis completed with errors. Message: {shap_results['error_message_shap']}")
            else:
               logger.debug("Pipeline: Captured SHAP plot paths.")

            if progress_callback: progress_callback(95, "SHAP analysis completed.")
        else:
            logger.info("Pipeline: Skipping SHAP analysis as per config.")


        if hyperparams.get("run_post_hoc_analysis", True):
            logger.info("Pipeline: Starting Post-hoc analysis (PDP, Permutation Importance)...")
            if progress_callback: progress_callback(95, "Starting Post-hoc analysis...")


            depth_feature_name = "age_ka_proxy" if getattr(full_dataset, "depth_feature_type", "") == "age_ka_proxy" else "depth"
            coord_feature_names = ["sin_lat", "cos_lat", "sin_lon", "cos_lon"] if bool(hyperparams.get("add_coord_channels", False)) else []
            env_missingness_feature_names = (
                [f"{name}__missing" for name in selected_env_variables]
                if bool(hyperparams.get("add_env_missingness_channels", False))
                else []
            )
            post_hoc_feature_names = selected_env_variables + env_missingness_feature_names + [depth_feature_name] + coord_feature_names


            plot_config_with_patch = plot_config.copy()
            plot_config_with_patch['patch_size'] = tuple(training_config.get("patch_size", [5, 5]))

            plot_config_with_patch.setdefault("export_formats", ["png", "pdf"])
            plot_config_with_patch.setdefault("publication_dpi", 600)
            plot_config_with_patch.setdefault("base_fontsize", 11)

            try:
                plot_config_with_patch["norm_params"] = training_artifacts_obj.norm_params.model_dump()
            except Exception:
                plot_config_with_patch["norm_params"] = None

            post_hoc_results = run_post_hoc_analysis(
                model=trainer.model,
                test_loader=test_loader,
                feature_names=post_hoc_feature_names,
                output_dir=output_dirs["post_hoc_explainability"],
                plot_config=plot_config_with_patch,
                task_type=task_type_for_explain,
                target_class_index=int(hyperparams.get("explain_target_class_index", hyperparams.get("multiclass_target_class_index", 1)) or 1),
                class_names=hyperparams.get("classification_class_names", None),
                group_categorical_features=True,
            )
            run_summary["artifacts"]["paths"]["post_hoc_plots"] = post_hoc_results
            if "error_message_post_hoc" in post_hoc_results and post_hoc_results["error_message_post_hoc"]:
                logger.warning(f"Pipeline: Post-hoc analysis completed with errors. Message: {post_hoc_results['error_message_post_hoc']}")
            else:
                logger.debug("Pipeline: Captured post-hoc plot paths.")


            model_type_lower = hyperparams.get('model_type', 'mixture_profiles').lower()
            if model_type_lower == "mixture_profiles":
                logger.info("Pipeline: Running Mixture-of-Profiles depth profile analysis...")
                mop_results = analyze_mop_depth_profiles(
                    model=trainer.model,
                    training_artifacts=training_artifacts_obj,
                    depth_transform_method=depth_transform_method,
                    output_dir=output_dirs["post_hoc_explainability"],
                )
                run_summary["artifacts"]["paths"].setdefault("post_hoc_plots", {}).update(mop_results)
                logger.debug("Pipeline: Captured MoP depth profile plot paths.")


            try:
                max_depth_m_plot = None
                try:
                    if hasattr(full_dataset, "rey_data") and "depth_physical" in full_dataset.rey_data.columns:
                        depth_phys = full_dataset.rey_data["depth_physical"].astype(float).to_numpy()
                        depth_phys = depth_phys[np.isfinite(depth_phys)]
                        if depth_phys.size:

                            max_depth_m_plot = float(np.percentile(depth_phys, 99))
                except Exception:
                    max_depth_m_plot = None
                if (max_depth_m_plot is None or not np.isfinite(max_depth_m_plot) or max_depth_m_plot <= 0) and max_training_depth_mbsf:
                    max_depth_m_plot = float(max_training_depth_mbsf)
                if max_depth_m_plot is None or not np.isfinite(max_depth_m_plot) or max_depth_m_plot <= 0:
                    raise ValueError("Could not determine a valid max_depth_m for profile diagnostics.")
                if hyperparams.get("run_peak_profile_analysis", True):
                    peak_results = analyze_peak_profiles(
                        model=trainer.model,
                        dataset=full_dataset,
                        test_indices=test_indices,
                        device=device_for_eval,
                        output_dir=output_dirs["post_hoc_explainability"],
                        max_depth_m=max_depth_m_plot,
                    )
                    if peak_results:
                        run_summary["artifacts"]["paths"].setdefault("post_hoc_plots", {}).update(peak_results)
                        logger.debug("Pipeline: Captured peak profile analysis artifacts.")
                else:
                    logger.info("Pipeline: Skipping peak-profile analysis as per config.")
            except Exception as e_peak:
                logger.warning(f"Pipeline: Peak-profile analysis failed: {e_peak}")


            try:
                if hyperparams.get("run_env_profile_analysis", True):
                    env_prof_results = analyze_env_profile_relationships(
                        model=trainer.model,
                        dataset=full_dataset,
                        test_indices=test_indices,
                        device=device_for_eval,
                        output_dir=output_dirs["post_hoc_explainability"],
                        max_depth_m=max_depth_m_plot,
                        max_samples=hyperparams.get("post_hoc_max_samples", 3000),
                        batch_size=hyperparams.get("post_hoc_batch_size", 256),
                    )
                    if env_prof_results:
                        run_summary["artifacts"]["paths"].setdefault("post_hoc_plots", {}).update(env_prof_results)
                        logger.debug("Pipeline: Captured env–profile relationship analysis artifacts.")
                else:
                    logger.info("Pipeline: Skipping env–profile analysis as per config.")
            except Exception as e_env_prof:
                logger.warning(f"Pipeline: Env–profile relationship analysis failed: {e_env_prof}")


            try:
                if hyperparams.get("run_profile_cluster_analysis", True):
                    profile_cluster_results = analyze_profile_clusters(
                        model=trainer.model,
                        dataset=full_dataset,
                        train_indices=train_indices,
                        test_indices=test_indices,
                        device=device_for_eval,
                        output_dir=output_dirs["post_hoc_explainability"],
                        max_depth_m=max_depth_m_plot,
                        max_samples_per_split=hyperparams.get("post_hoc_max_samples", 3000),
                        batch_size=hyperparams.get("post_hoc_batch_size", 256),
                    )
                    if profile_cluster_results:
                        run_summary["artifacts"]["paths"].setdefault("post_hoc_plots", {}).update(profile_cluster_results)
                        logger.debug("Pipeline: Captured profile cluster comparison artifacts.")
                else:
                    logger.info("Pipeline: Skipping profile-cluster analysis as per config.")
            except Exception as e_prof_cluster:
                logger.warning(f"Pipeline: Profile cluster analysis failed: {e_prof_cluster}")

            if progress_callback:
                progress_callback(98, "Post-hoc analysis completed.")
        else:
            logger.info("Pipeline: Skipping Post-hoc analysis as per config.")

        run_summary["MODEL_CONFIG"] = model_init_config.model_dump()
        run_summary["NORMALIZATION_PARAMS"] = norm_params.model_dump()

        run_summary["status"] = "completed"
        return True, run_summary

    except (StopExecutionError, FileNotFoundError, ValueError, RuntimeError) as e:
        error_message = f"Pipeline failed: {e}"
        logger.error(error_message, exc_info=True)
        run_summary["status"] = "error"
        run_summary["error_message"] = error_message
        return False, run_summary
    except Exception as e:
        error_message = f"An unexpected error occurred in the training pipeline: {e}"
        logger.error(error_message, exc_info=True)
        run_summary["status"] = "error"
        run_summary["error_message"] = error_message
        return False, run_summary
    finally:
        run_summary["timestamp_end"] = datetime.now().isoformat()
        run_summary["duration_seconds"] = (datetime.fromisoformat(run_summary["timestamp_end"]) - datetime.fromisoformat(run_summary["timestamp_start"])).total_seconds()

        final_run_id = run_summary["run_id"]
        final_run_base_dir = run_base_dir

        if run_summary["status"] == "completed":

            try:
                skip_rename = bool(
                    (run_summary.get("config", {}) or {})
                    .get("training", {})
                    .get("hyperparams", {})
                    .get("skip_run_dir_rename", False)
                )
            except Exception:
                skip_rename = False

            if skip_rename:
                logger.info("Skipping run directory rename as per config (skip_run_dir_rename=True).")

                pass
            else:
                try:
                    hp = (run_summary.get("config", {}) or {}).get("training", {}).get("hyperparams", {}) or {}
                    task_type = str(hp.get("task_type", "regression") or "regression").lower()

                    if task_type == "binary_classification":
                        test_auc = run_summary.get("results", {}).get("test_metrics", {}).get("roc_auc")
                        if test_auc is not None and np.isfinite(test_auc):
                            final_run_id = f"point_model_run_{timestamp_str}_AUC_{float(test_auc):.4f}"
                        else:
                            final_run_id = f"point_model_run_{timestamp_str}_AUC_NA"
                    elif task_type == "multiclass_classification":
                        test_f1 = run_summary.get("results", {}).get("test_metrics", {}).get("macro_f1")
                        test_bal_acc = run_summary.get("results", {}).get("test_metrics", {}).get("balanced_accuracy")
                        metric = None
                        metric_name = None
                        if test_f1 is not None and np.isfinite(test_f1):
                            metric = float(test_f1)
                            metric_name = "F1"
                        elif test_bal_acc is not None and np.isfinite(test_bal_acc):
                            metric = float(test_bal_acc)
                            metric_name = "BalAcc"
                        if metric is not None and metric_name is not None:
                            final_run_id = f"point_model_run_{timestamp_str}_{metric_name}_{metric:.4f}"
                        else:
                            final_run_id = f"point_model_run_{timestamp_str}_F1_NA"
                    else:
                        test_r2 = run_summary.get("results", {}).get("test_metrics", {}).get("r2")
                        if test_r2 is not None and np.isfinite(test_r2):
                            final_run_id = f"point_model_run_{timestamp_str}_R2_{test_r2:.4f}"
                        else:
                            final_run_id = f"point_model_run_{timestamp_str}_R2_NA"

                    final_run_base_dir = Path(RESULTS_GUI_DIR) / final_run_id

                    import shutil

                    for attempt in range(6):
                        try:
                            if final_run_base_dir.exists():
                                final_run_base_dir = Path(RESULTS_GUI_DIR) / f"{final_run_id}__dup{attempt+1}"
                            run_base_dir.rename(final_run_base_dir)
                            logger.info(f"Renamed run directory to: {final_run_base_dir}")
                            break
                        except PermissionError:
                            if attempt >= 5:
                                raise
                            time.sleep(0.4 * (attempt + 1))
                            continue
                        except OSError:

                            if attempt >= 5:
                                raise
                            time.sleep(0.4 * (attempt + 1))
                            try:
                                shutil.move(str(run_base_dir), str(final_run_base_dir))
                                logger.info(f"Moved run directory to: {final_run_base_dir}")
                                break
                            except Exception:
                                continue


                    run_summary["run_id"] = final_run_id

                    temp_base_path_str = str(run_base_dir)
                    final_base_path_str = str(final_run_base_dir)

                    def update_paths_recursive(d):
                        for k, v in d.items():
                            if isinstance(v, str) and v.startswith(temp_base_path_str):
                                d[k] = v.replace(temp_base_path_str, final_base_path_str, 1)
                            elif isinstance(v, dict):
                                update_paths_recursive(v)
                            elif isinstance(v, list):


                                if not v:
                                    continue
                                first = v[0]
                                if not isinstance(first, (str, dict, list)):
                                    continue

                                new_list = []
                                for item in v:
                                    if isinstance(item, str) and item.startswith(temp_base_path_str):
                                        new_list.append(item.replace(temp_base_path_str, final_base_path_str, 1))
                                    elif isinstance(item, dict):
                                        update_paths_recursive(item)
                                        new_list.append(item)
                                    else:
                                        new_list.append(item)
                                d[k] = new_list

                    update_paths_recursive(run_summary)


                    try:
                        logger.info("Post-rename: Directly updating model_path in training_artifacts.json...")


                        artifacts_json_path_final = final_run_base_dir / "models" / "training_artifacts.json"

                        if artifacts_json_path_final.exists():

                            with open(artifacts_json_path_final, "r", encoding="utf-8") as f:
                                artifacts_data = json.load(f)


                            model_path_final = final_run_base_dir / "models" / "final_trained_model.pth"


                            artifacts_data["model_path"] = str(model_path_final)


                            with open(artifacts_json_path_final, "w", encoding="utf-8") as f:
                                json.dump(artifacts_data, f, indent=4, ensure_ascii=False)

                            logger.info(f"Successfully updated model_path in {artifacts_json_path_final}")
                        else:
                            logger.error(f"Could not find training_artifacts.json at new path: {artifacts_json_path_final}")

                    except Exception as e_update_artifact:
                        logger.error(
                            f"Failed to directly update training_artifacts.json after rename: {e_update_artifact}",
                            exc_info=True,
                        )


                except Exception as e_rename:
                    logger.error(f"Failed to rename run directory or update paths: {e_rename}", exc_info=True)

                    final_run_id = run_summary["run_id"]


        try:
            run_dir_for_outputs = final_run_base_dir if Path(str(final_run_base_dir)).exists() else run_base_dir


            pipeline_config_path = run_dir_for_outputs / "pipeline_config.json"
            write_json(pipeline_config_path, run_summary.get("config", {}))
            run_summary["artifacts"]["paths"]["pipeline_config"] = str(pipeline_config_path)


            env_path = run_dir_for_outputs / "environment_snapshot.json"
            write_json(env_path, snapshot_environment(extra={"run_id": run_summary.get("run_id"), "run_type": "training"}))
            run_summary["artifacts"]["paths"]["environment_snapshot"] = str(env_path)


            ds_summary_path = run_dir_for_outputs / "dataset_summary.json"
            write_json(ds_summary_path, dataset_summary_for_report or run_summary.get("results", {}).get("dataset_summary", {}))
            run_summary["artifacts"]["paths"]["dataset_summary"] = str(ds_summary_path)


            try:
                splits_dir = run_dir_for_outputs / "splits"
                splits_dir.mkdir(parents=True, exist_ok=True)
                if train_indices is not None:
                    np.save(splits_dir / "train_indices.npy", np.asarray(train_indices, dtype=int))
                if val_indices is not None:
                    np.save(splits_dir / "val_indices.npy", np.asarray(val_indices, dtype=int))
                if test_indices is not None:
                    np.save(splits_dir / "test_indices.npy", np.asarray(test_indices, dtype=int))
                if full_dataset is not None and hasattr(full_dataset, "rey_data"):
                    np.save(splits_dir / "dataset_source_index.npy", full_dataset.rey_data.index.to_numpy())
                run_summary["artifacts"]["paths"]["splits_dir"] = str(splits_dir)
            except Exception as e_splits:
                logger.warning(f"Pipeline: Failed to save split indices: {e_splits}")


            report_path = run_dir_for_outputs / "RUN_REPORT.md"
            report_md = make_training_run_report_md(run_summary, dataset_summary=dataset_summary_for_report)
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_md)
            run_summary["artifacts"]["paths"]["run_report"] = str(report_path)


            manifest_path = run_dir_for_outputs / "results_manifest.json"
            write_json(manifest_path, build_paths_manifest(run_dir_for_outputs, run_summary))
            run_summary["artifacts"]["paths"]["results_manifest"] = str(manifest_path)

        except Exception as e_aux:
            logger.warning(f"Pipeline: Failed to write journal-ready outputs: {e_aux}")

        try:

            save_run_summary(final_run_id, run_summary)
        except Exception as e_save:
            logger.critical(f"CRITICAL: Failed to save final run summary for run {final_run_id}: {e_save}", exc_info=True)

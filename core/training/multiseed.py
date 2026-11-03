import copy
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from core.training.pipeline import (
    CASCADE_EXPLORATION_4CLASS_TASK_TYPE,
    CASCADE_GRADE_TASK_TYPE,
    _score_balanced_group_split,
    _spatial_balance_labels,
    apply_cascade_training_profile,
    run_training_pipeline,
)
from core.utils.constants import RESULTS_GUI_DIR
from core.utils.config_models import CascadeEnsembleArtifacts, CascadeEnsembleMember, CascadeTrainingArtifacts
from core.utils.run_manager import save_run_summary
from core.utils.serialization_utils import NumpyJSONEncoder


ProgressCallback = Callable[[float, str], None]
HistoryCallback = Callable[[list[dict[str, Any]]], None]


def parse_seed_list(seed_text: str | list[int] | tuple[int, ...] | None) -> list[int]:
    if seed_text is None:
        return []
    if isinstance(seed_text, (list, tuple)):
        return [int(x) for x in seed_text]
    seeds: list[int] = []
    for part in str(seed_text).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = [int(x.strip()) for x in part.split("-", 1)]
            step = 1 if b >= a else -1
            seeds.extend(list(range(a, b + step, step)))
        else:
            seeds.append(int(part))
    return list(dict.fromkeys(seeds))


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        return out if math.isfinite(out) else None
    except Exception:
        return None


def _get_path(d: dict[str, Any], path: list[str], default: Any = None) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _metric_value(row: dict[str, Any], metric: str) -> float | None:
    if str(metric) in {"val_composite", "val_screening_composite"}:
        macro = _safe_float(row.get("val_macro_f1")) or 0.0
        balanced = _safe_float(row.get("val_balanced_accuracy")) or 0.0
        high_ap = _safe_float(row.get("val_high_grade_ap")) or 0.0
        high_f2 = _safe_float(row.get("val_high_grade_f2_at_0_10")) or 0.0
        return float(0.45 * macro + 0.25 * balanced + 0.20 * high_ap + 0.10 * high_f2)
    aliases = {
        "best_val_metric": "best_val_metric",
        "val_macro_f1": "val_macro_f1",
        "val_balanced_accuracy": "val_balanced_accuracy",
        "val_target_class_ap": "val_target_class_ap",
        "val_high_grade_ap": "val_high_grade_ap",
        "val_high_grade_f1": "val_high_grade_f1",
        "val_high_grade_f2_at_0_10": "val_high_grade_f2_at_0_10",
        "val_high_grade_f1_at_0_10": "val_high_grade_f1_at_0_10",
        "val_high_grade_thresholded_f1": "val_high_grade_thresholded_f1",
        "val_high_grade_thresholded_f2": "val_high_grade_thresholded_f2",
        "val_composite": "val_composite",
        "val_screening_composite": "val_composite",
        "test_macro_f1": "test_macro_f1",
        "test_high_grade_ap": "test_high_grade_ap",
        "test_high_grade_f1": "test_high_grade_f1",
        "test_balanced_accuracy": "test_balanced_accuracy",
    }
    return _safe_float(row.get(aliases.get(metric, metric)))


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _fixed_threshold_row(run_dir: Path, split_name: str, threshold: float) -> dict[str, Any]:
    df = _load_csv(run_dir / "tables" / split_name / "high_grade_fixed_threshold_metrics.csv")
    if df.empty or "threshold" not in df.columns:
        return {}
    t = pd.to_numeric(df["threshold"], errors="coerce")
    if t.isna().all():
        return {}
    idx = int((t - float(threshold)).abs().idxmin())
    row = df.iloc[idx].to_dict()
    return {str(k): v for k, v in row.items()}


def _per_class_row(run_dir: Path, split_name: str, class_name_contains: str) -> dict[str, Any]:
    df = _load_csv(run_dir / "tables" / split_name / "per_class_probability_metrics.csv")
    if df.empty or "class_name" not in df.columns:
        return {}
    m = df["class_name"].astype(str).str.contains(class_name_contains, case=False, na=False)
    if not bool(m.any()):
        return {}
    return df.loc[m].iloc[0].to_dict()


def _stage_metric_row(run_dir: Path, split_name: str, stage_name_contains: str) -> dict[str, Any]:
    df = _load_csv(run_dir / "tables" / split_name / "hierarchical" / "hierarchical_stage_metrics.csv")
    if df.empty:
        return {}
    stage_col = "stage" if "stage" in df.columns else ("level" if "level" in df.columns else None)
    if not stage_col:
        return {}
    m = df[stage_col].astype(str).str.contains(stage_name_contains, case=False, na=False)
    if not bool(m.any()):
        return {}
    return df.loc[m].iloc[0].to_dict()


def _extract_row(summary: dict[str, Any], seed: int, run_dir: Path, phase: str) -> dict[str, Any]:
    results = summary.get("results", {}) or {}
    test = results.get("test_metrics", {}) or {}
    val = results.get("val_metrics", {}) or {}
    split_diag = results.get("split_group_diagnostics", {}) or {}
    threshold = results.get("threshold_override", {}) or {}

    row = {
        "phase": phase,
        "seed": int(seed),
        "run_id": summary.get("run_id"),
        "run_dir": str(run_dir),
        "status": summary.get("status"),
        "random_state": results.get("random_state"),
        "split_random_state": results.get("split_random_state"),
        "early_stopping_metric": results.get("early_stopping_metric"),
        "best_val_metric": results.get("best_val_metric"),
        "best_val_loss": results.get("best_val_loss"),
        "best_epoch": results.get("best_epoch"),
        "val_accuracy": val.get("accuracy"),
        "val_balanced_accuracy": val.get("balanced_accuracy"),
        "val_macro_f1": val.get("macro_f1"),
        "val_log_loss": val.get("log_loss"),
        "val_high_grade_recall": val.get("high_grade_recall"),
        "val_high_grade_ap": val.get("high_grade_ap"),
        "val_target_class_ap": results.get("val_target_class_ap", results.get("val_class1_ap")),
        "test_accuracy": test.get("accuracy"),
        "test_balanced_accuracy": test.get("balanced_accuracy"),
        "test_macro_f1": test.get("macro_f1"),
        "test_log_loss": test.get("log_loss"),
        "test_high_grade_recall": test.get("high_grade_recall"),
        "test_high_grade_precision": test.get("high_grade_precision"),
        "test_high_grade_f1": test.get("high_grade_f1"),
        "test_high_grade_ap": test.get("high_grade_ap"),
        "high_grade_threshold_from_val": (results.get("high_grade_threshold_from_val") or {}).get("threshold")
        if isinstance(results.get("high_grade_threshold_from_val"), dict)
        else None,
        "val_high_grade_thresholded_precision": (results.get("val_high_grade_thresholded_metrics") or {}).get("precision")
        if isinstance(results.get("val_high_grade_thresholded_metrics"), dict)
        else None,
        "val_high_grade_thresholded_recall": (results.get("val_high_grade_thresholded_metrics") or {}).get("recall")
        if isinstance(results.get("val_high_grade_thresholded_metrics"), dict)
        else None,
        "val_high_grade_thresholded_f1": (results.get("val_high_grade_thresholded_metrics") or {}).get("f1")
        if isinstance(results.get("val_high_grade_thresholded_metrics"), dict)
        else None,
        "test_high_grade_thresholded_precision": (results.get("test_high_grade_thresholded_metrics") or {}).get("precision")
        if isinstance(results.get("test_high_grade_thresholded_metrics"), dict)
        else None,
        "test_high_grade_thresholded_recall": (results.get("test_high_grade_thresholded_metrics") or {}).get("recall")
        if isinstance(results.get("test_high_grade_thresholded_metrics"), dict)
        else None,
        "test_high_grade_thresholded_f1": (results.get("test_high_grade_thresholded_metrics") or {}).get("f1")
        if isinstance(results.get("test_high_grade_thresholded_metrics"), dict)
        else None,
        "target_class": threshold.get("target_class"),
        "threshold": threshold.get("threshold"),
        "split_leakage_detected": split_diag.get("leakage_detected"),
        "split_leaking_groups": split_diag.get("n_leaking_groups"),
        "error_message": summary.get("error_message"),
    }
    for split_name in ("val", "test"):
        fixed = _fixed_threshold_row(run_dir, split_name, 0.10)
        if fixed:
            for metric in ("precision", "recall", "f1", "f2", "n_pred_positive", "average_precision"):
                row[f"{split_name}_high_grade_{metric}_at_0_10"] = fixed.get(metric)
        class2_high = _per_class_row(run_dir, split_name, "class2_high")
        if class2_high:
            row[f"{split_name}_class2_high_recall"] = class2_high.get("recall")
            row[f"{split_name}_class2_high_f1"] = class2_high.get("f1")
            row[f"{split_name}_class2_high_ap"] = class2_high.get("average_precision")
        c2_stage = _stage_metric_row(run_dir, split_name, "c2")
        if c2_stage:
            for metric in ("balanced_accuracy", "macro_f1", "average_precision_macro"):
                if metric in c2_stage:
                    row[f"{split_name}_c2_{metric}"] = c2_stage.get(metric)
    row["val_composite"] = _metric_value(row, "val_composite")
    return row


def _summarize_rows(rows: list[dict[str, Any]], metric_names: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_runs": len(rows), "metrics": {}}
    for metric in metric_names:
        vals = [_safe_float(r.get(metric)) for r in rows]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        out["metrics"][metric] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
            "min": float(np.min(arr)),
            "median": float(np.median(arr)),
            "max": float(np.max(arr)),
        }
    return out


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, cls=NumpyJSONEncoder, indent=2, ensure_ascii=False)


def _as_artifact_path(run_dir: str | Path | None, artifact_name: str) -> Path | None:
    if not run_dir:
        return None
    path = Path(str(run_dir)) / "models" / artifact_name
    return path if path.exists() else None


def _stable_threshold_from_rows(
    rows: list[dict[str, Any]],
    *,
    split_name: str = "val",
    min_recall: float = 0.75,
    metric: str = "f2",
) -> dict[str, Any] | None:
    """Choose a fixed high-grade threshold using validation rows only.

    The threshold is selected from the fixed-threshold tables produced by each
    seed run. We prefer thresholds whose mean recall clears `min_recall`; among
    those, maximize mean F2/F1. If none clear recall, maximize mean recall then
    mean metric.
    """

    frames: list[pd.DataFrame] = []
    for row in rows:
        run_dir = row.get("run_dir")
        if not run_dir or str(row.get("status")) != "completed":
            continue
        df = _load_csv(Path(str(run_dir)) / "tables" / split_name / "high_grade_fixed_threshold_metrics.csv")
        if df.empty or "threshold" not in df.columns:
            continue
        df = df.copy()
        df["seed"] = row.get("seed")
        frames.append(df)
    if not frames:
        return None

    all_df = pd.concat(frames, ignore_index=True)
    summary_rows = []
    for threshold, group in all_df.groupby("threshold", dropna=False):
        row_out: dict[str, Any] = {"threshold": float(threshold), "n_runs": int(group["seed"].nunique())}
        for col in ("precision", "recall", "f1", "f2", "average_precision", "n_pred_positive"):
            vals = pd.to_numeric(group.get(col), errors="coerce").dropna().to_numpy(dtype=float)
            if vals.size:
                row_out[f"{col}_mean"] = float(np.mean(vals))
                row_out[f"{col}_std"] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
                row_out[f"{col}_min"] = float(np.min(vals))
                row_out[f"{col}_max"] = float(np.max(vals))
        summary_rows.append(row_out)
    if not summary_rows:
        return None

    metric_key = f"{metric}_mean"
    candidates = [r for r in summary_rows if _safe_float(r.get("recall_mean")) is not None]
    candidates_met = [r for r in candidates if float(r.get("recall_mean") or 0.0) >= float(min_recall)]
    if candidates_met:
        best = sorted(
            candidates_met,
            key=lambda r: (
                -float(r.get(metric_key) or -1.0),
                -float(r.get("precision_mean") or -1.0),
                abs(float(r.get("threshold") or 0.0) - 0.1),
            ),
        )[0]
        selection_reason = f"max_{metric}_mean_with_recall_ge_{min_recall:g}"
    else:
        best = sorted(
            candidates,
            key=lambda r: (
                -float(r.get("recall_mean") or -1.0),
                -float(r.get(metric_key) or -1.0),
                abs(float(r.get("threshold") or 0.0) - 0.1),
            ),
        )[0]
        selection_reason = f"max_recall_mean_no_threshold_met_recall_ge_{min_recall:g}"

    return {
        "split": split_name,
        "metric": metric,
        "min_recall": float(min_recall),
        "selection_reason": selection_reason,
        "selected": best,
        "summary_rows": summary_rows,
    }


def _write_cascade_ensemble_artifacts(
    *,
    parent: Path,
    rows: list[dict[str, Any]],
    stable_threshold: dict[str, Any] | None,
) -> dict[str, str]:
    members: list[CascadeEnsembleMember] = []
    for row in rows:
        if row.get("phase") != "seed_screen" or str(row.get("status")) != "completed":
            continue
        artifact_path = _as_artifact_path(row.get("run_dir"), "cascade_training_artifacts.json")
        if artifact_path is None:
            continue
        members.append(
            CascadeEnsembleMember(
                seed=int(row.get("seed")),
                split_random_state=int(row["split_random_state"]) if row.get("split_random_state") is not None else None,
                run_dir=str(row.get("run_dir")),
                cascade_artifacts_path=str(artifact_path),
                weight=1.0,
            )
        )
    if not members:
        return {}

    first_artifact = None
    try:
        first_artifact = CascadeTrainingArtifacts.model_validate_json(
            Path(str(members[0].cascade_artifacts_path)).read_text(encoding="utf-8")
        )
    except Exception:
        first_artifact = None

    selected_threshold = None
    selected_source = None
    if stable_threshold:
        selected = stable_threshold.get("selected") or {}
        selected_threshold = _safe_float(selected.get("threshold"))
        selected_source = stable_threshold.get("selection_reason")

    first_artifact_type = str(getattr(first_artifact, "artifact_type", "") or "")
    ensemble_artifact_type = (
        "cascade_exploration_4class_ensemble"
        if first_artifact_type == CASCADE_EXPLORATION_4CLASS_TASK_TYPE
        else "cascade_grade_geochem_ensemble"
    )
    artifact = CascadeEnsembleArtifacts(
        artifact_type=ensemble_artifact_type,
        final_label_col=str(getattr(first_artifact, "final_label_col", None) or "grade_geochem_class_id"),
        final_class_names=list(
            getattr(first_artifact, "final_class_names", None)
            or [
                "carbonate_background",
                "class1_low_grade",
                "class1_high_grade",
                "class2_low_grade",
                "class2_high_grade",
            ]
        ),
        final_num_classes=int(getattr(first_artifact, "final_num_classes", None) or 5),
        combination_rule=(
            str(getattr(first_artifact, "combination_rule", None))
            if first_artifact is not None
            else (
                "Each member composes cascade probabilities as P0=P(main=carbonate); "
                "P1=P(main=class1)*P(low|class1); P2=P(main=class1)*P(high|class1); "
                "P3=P(main=class2)*P(low|class2); P4=P(main=class2)*P(high|class2)."
            )
        ),
        members=members,
        recommended_high_grade_threshold=selected_threshold,
        recommended_threshold_metric=str(stable_threshold.get("metric")) if stable_threshold else None,
        recommended_threshold_source=str(selected_source) if selected_source else None,
    )
    artifact_path = parent / "models" / "cascade_ensemble_artifacts.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(artifact.model_dump_json(indent=4), encoding="utf-8")
    return {"cascade_ensemble_artifacts": str(artifact_path)}


def _load_fixed_test_indices(base_config: dict[str, Any], base_split_seed: int) -> list[int] | None:
    paths = base_config.get("paths", {}) if isinstance(base_config, dict) else {}
    training = base_config.get("training", {}) if isinstance(base_config, dict) else {}
    hp = training.get("hyperparams", {}) if isinstance(training, dict) else {}
    points_path = paths.get("processed_rey_points")
    if not points_path or not Path(str(points_path)).exists():
        return None
    try:
        from core.datasets.rey_dataset import REYPointDataset
        from core.training.pipeline import _discover_env_feature_names, _group_train_val_test_split, _stratified_group_train_val_test_split

        env_grid = paths.get("processed_env_grid")
        if not env_grid or not Path(str(env_grid)).exists():
            return None
        data_processing = base_config.get("data_processing", {}) or {}
        dataset = REYPointDataset(
            rey_points_path=str(points_path),
            env_grid_path=str(env_grid),
            feature_names=_discover_env_feature_names(str(env_grid)),
            patch_size=tuple(training.get("patch_size", [3, 3])),
            depth_transform_method=data_processing.get("depth_transform_method", "log1p"),
            env_normalization_method=data_processing.get("env_normalization_method", "zscore"),
            max_age_ka_proxy=data_processing.get("max_training_age_ka_proxy", None),
            sed_rate_nc_path=data_processing.get("sedimentation_rate_grid_path", None),
            depth_feature_type_override=data_processing.get("depth_feature_type_override", data_processing.get("depth_feature_type", None)),
            add_coord_channels=bool(hp.get("add_coord_channels", False)),
            task_type="multiclass_classification"
            if str(hp.get("task_type", "") or "") in {CASCADE_GRADE_TASK_TYPE, CASCADE_EXPLORATION_4CLASS_TASK_TYPE}
            else str(hp.get("task_type", "regression") or "regression"),
            classification_threshold_ppm=float(hp.get("classification_threshold_ppm", 1000.0) or 1000.0),
            classification_label_col=hp.get(
                "classification_label_col",
                "exploration_class_id" if str(hp.get("task_type", "") or "") == CASCADE_EXPLORATION_4CLASS_TASK_TYPE else "grade_geochem_class_id",
            ),
            classification_class_names=hp.get("classification_class_names", None),
        )
        rey_df = dataset.rey_data
        split_strategy = str(training.get("split_strategy", "core_id") or "core_id")
        if split_strategy == "site_group":
            group_col = "site_group" if "site_group" in rey_df.columns else ("core_id" if "core_id" in rey_df.columns else "site_hole")
        elif split_strategy == "core_id":
            group_col = "core_id" if "core_id" in rey_df.columns else "site_hole"
        else:
            return None
        groups = rey_df[group_col].astype(str).to_numpy()
        label_col = hp.get("classification_label_col", None)
        y = None
        n_classes = 0
        if label_col and label_col in rey_df.columns:
            y = pd.to_numeric(rey_df[label_col], errors="coerce").to_numpy(dtype=np.int64)
            n_classes = int(np.nanmax(y)) + 1 if y.size else 0
        group_split_selection = str(training.get("group_split_selection", "balanced") or "balanced").lower().strip()
        use_balanced = group_split_selection in {"balanced", "constrained", "class_spatial_balanced", "class-space-balanced"}
        n_bins = int(training.get("group_split_stratify_bins", 0) or 0)
        test_ratio = float(training.get("test_split_ratio", 0.15) or 0.15)
        val_ratio = float(training.get("validation_split_ratio", 0.1765) or 0.1765)
        if use_balanced and y is not None and n_classes >= 2:
            spatial_labels = _spatial_balance_labels(
                rey_df,
                float(training.get("group_split_spatial_block_deg", 10.0) or 10.0),
            )
            best = None
            best_score = None
            n_candidates = max(1, int(training.get("split_max_tries_multiclass", 25) or 25))
            rare_target = (y == int(np.argmin(np.bincount(y, minlength=n_classes)))).astype(float)
            for attempt in range(n_candidates):
                rs = int(base_split_seed) + int(attempt) * 17
                if n_bins > 1:
                    train_idx, val_idx, test_idx = _stratified_group_train_val_test_split(
                        dataset_size=len(dataset),
                        groups=groups,
                        targets=rare_target,
                        test_split_ratio=test_ratio,
                        validation_split_ratio=val_ratio,
                        random_state=rs,
                        n_bins=n_bins,
                        statistic="sum",
                    )
                else:
                    train_idx, val_idx, test_idx = _group_train_val_test_split(
                        dataset_size=len(dataset),
                        groups=groups,
                        test_split_ratio=test_ratio,
                        validation_split_ratio=val_ratio,
                        random_state=rs,
                    )
                components = _score_balanced_group_split(
                    train_indices=train_idx,
                    val_indices=val_idx,
                    test_indices=test_idx,
                    groups=groups,
                    labels=y,
                    n_classes=n_classes,
                    spatial_labels=spatial_labels,
                    test_split_ratio=test_ratio,
                    validation_split_ratio=val_ratio,
                )
                score = float(components["score"])
                if best_score is None or score < best_score:
                    best_score = score
                    best = (train_idx, val_idx, test_idx)
            if best is None:
                return None
            train_idx, val_idx, test_idx = best
        elif n_bins > 1 and y is not None and n_classes >= 2:
            train_idx, val_idx, test_idx = _stratified_group_train_val_test_split(
                dataset_size=len(dataset),
                groups=groups,
                targets=(y == int(np.argmin(np.bincount(y, minlength=n_classes)))).astype(float),
                test_split_ratio=test_ratio,
                validation_split_ratio=val_ratio,
                random_state=int(base_split_seed),
                n_bins=n_bins,
                statistic="sum",
            )
        else:
            train_idx, val_idx, test_idx = _group_train_val_test_split(
                dataset_size=len(dataset),
                groups=groups,
                test_split_ratio=test_ratio,
                validation_split_ratio=val_ratio,
                random_state=int(base_split_seed),
            )
        return [int(i) for i in test_idx]
    except Exception:
        return None


def _make_config_for_seed(
    base_config: dict[str, Any],
    *,
    run_dir: Path,
    seed: int,
    split_seed: int,
    run_explainability: bool,
    skip_run_dir_rename: bool,
    screening_epochs: int | None = None,
    screening_patience: int | None = None,
    fixed_test_indices: list[int] | None = None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_config)
    cfg.setdefault("paths", {})
    cfg["paths"]["run_base_dir"] = str(run_dir)
    cfg.setdefault("plotting", {})
    cfg.setdefault("training", {})
    cfg["training"]["split_random_state"] = int(split_seed)
    hp = cfg["training"].setdefault("hyperparams", {})
    hp["random_state"] = int(seed)
    hp["split_random_state"] = int(split_seed)
    hp["run_shap_analysis"] = bool(run_explainability and hp.get("run_shap_analysis", True))
    hp["run_post_hoc_analysis"] = bool(run_explainability and hp.get("run_post_hoc_analysis", True))
    hp["skip_run_dir_rename"] = bool(skip_run_dir_rename)
    if fixed_test_indices:
        hp["fixed_test_indices"] = [int(i) for i in fixed_test_indices]
    if not run_explainability:
        cfg["plotting"]["plot_scatter"] = False
        cfg["plotting"]["plot_feature_maps"] = False
        cfg["plotting"]["num_feature_map_samples"] = 0
        hp["run_spatial_shuffle_eval"] = False
        hp["run_age_binned_metrics"] = False
        hp["enable_extended_age_evaluation"] = False
        hp["include_older_samples_in_age_eval"] = False
        if screening_epochs is not None:
            hp["epochs"] = int(screening_epochs)
        if screening_patience is not None:
            hp["early_stopping_patience"] = int(screening_patience)
    return cfg


def run_multiseed_training(
    config: dict[str, Any],
    *,
    seeds: list[int],
    split_seed_mode: str = "same",
    split_seed: int | None = None,
    selection_metric: str = "best_val_metric",
    run_best_explainability: bool = True,
    screening_epochs: int | None = None,
    screening_patience: int | None = None,
    parent_run_dir: str | Path | None = None,
    progress_callback: ProgressCallback | None = None,
    history_callback: HistoryCallback | None = None,
) -> tuple[bool, dict[str, Any]]:
    if not seeds:
        raise ValueError("run_multiseed_training requires at least one seed.")
    if split_seed_mode not in {"same", "vary_with_seed", "fixed_test_vary_val"}:
        raise ValueError("split_seed_mode must be 'same', 'vary_with_seed', or 'fixed_test_vary_val'.")
    try:
        hp0 = (config.get("training", {}) or {}).get("hyperparams", {}) or {}
        apply_cascade_training_profile(hp0, config.get("training", {}) or {})
    except Exception:
        pass

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if parent_run_dir is None:
        parent = Path(RESULTS_GUI_DIR) / f"multiseed_training_{stamp}"
    else:
        parent = Path(parent_run_dir)
    parent.mkdir(parents=True, exist_ok=True)

    base_split_seed = int(
        split_seed
        if split_seed is not None
        else _get_path(config, ["training", "split_random_state"], _get_path(config, ["training", "hyperparams", "split_random_state"], seeds[0]))
    )
    rows: list[dict[str, Any]] = []
    child_summaries: list[dict[str, Any]] = []
    n_total = len(seeds) + (1 if run_best_explainability else 0)
    fixed_test_indices = (
        _load_fixed_test_indices(config, int(base_split_seed))
        if split_seed_mode == "fixed_test_vary_val"
        else None
    )

    for i, seed in enumerate(seeds):
        if progress_callback:
            progress_callback(i / max(n_total, 1), f"Multi-seed: training seed {seed} ({i + 1}/{len(seeds)})")
        seed_split = int(seed) if split_seed_mode in {"vary_with_seed", "fixed_test_vary_val"} else int(base_split_seed)
        run_dir = parent / f"seed_{int(seed)}"
        cfg_i = _make_config_for_seed(
            config,
            run_dir=run_dir,
            seed=int(seed),
            split_seed=seed_split,
            run_explainability=False,
            skip_run_dir_rename=True,
            screening_epochs=screening_epochs,
            screening_patience=screening_patience,
            fixed_test_indices=fixed_test_indices,
        )
        ok_i, summary_i = run_training_pipeline(cfg_i, progress_callback=None, history_callback=None)
        row = _extract_row(summary_i, int(seed), run_dir, phase="seed_screen")
        row["success"] = bool(ok_i)
        rows.append(row)
        child_summaries.append(summary_i)
        _write_rows_csv(parent / "multiseed_runs.csv", rows)

    success_rows = [r for r in rows if str(r.get("status")) == "completed"]
    if not success_rows:
        summary = {
            "run_id": parent.name,
            "run_type": "multiseed_training",
            "timestamp_end": datetime.now().isoformat(),
            "status": "error",
            "config": config,
            "results": {
                "seeds": seeds,
                "selection_metric": selection_metric,
                "rows": rows,
                "error_message": "No seed run completed successfully.",
            },
            "artifacts": {"paths": {"base": str(parent), "multiseed_runs_csv": str(parent / "multiseed_runs.csv")}},
        }
        save_run_summary(parent.name, summary)
        return False, summary

    def sort_key(row: dict[str, Any]) -> tuple[int, float]:
        value = _metric_value(row, selection_metric)
        return (0 if value is not None else 1, -float(value) if value is not None else 0.0)

    best_row = sorted(success_rows, key=sort_key)[0]
    best_seed = int(best_row["seed"])
    best_split_seed = int(best_row.get("split_random_state") or base_split_seed)
    final_summary = None
    final_ok = True
    if run_best_explainability:
        if progress_callback:
            progress_callback(len(seeds) / max(n_total, 1), f"Multi-seed: running explainability for best seed {best_seed}")
        final_run_dir = parent / f"best_seed_{best_seed}_explainability"
        cfg_best = _make_config_for_seed(
            config,
            run_dir=final_run_dir,
            seed=best_seed,
            split_seed=best_split_seed,
            run_explainability=True,
            skip_run_dir_rename=True,
            screening_epochs=None,
            screening_patience=None,
            fixed_test_indices=fixed_test_indices,
        )
        final_ok, final_summary = run_training_pipeline(
            cfg_best,
            progress_callback=None,
            history_callback=history_callback,
        )
        final_row = _extract_row(final_summary, best_seed, final_run_dir, phase="best_explainability")
        final_row["success"] = bool(final_ok)
        rows.append(final_row)
        _write_rows_csv(parent / "multiseed_runs.csv", rows)

    stability = _summarize_rows(
        success_rows,
        [
            "best_val_metric",
            "val_macro_f1",
            "val_balanced_accuracy",
            "val_target_class_ap",
            "val_high_grade_ap",
            "val_high_grade_f1",
            "val_high_grade_f1_at_0_10",
            "val_high_grade_f2_at_0_10",
            "test_macro_f1",
            "test_balanced_accuracy",
            "test_high_grade_recall",
            "test_high_grade_f1",
            "test_high_grade_ap",
            "test_high_grade_f1_at_0_10",
            "test_high_grade_f2_at_0_10",
            "test_class2_high_recall",
            "test_class2_high_ap",
            "test_c2_macro_f1",
        ],
    )
    summary = {
        "run_id": parent.name,
        "run_type": "multiseed_training",
        "timestamp_end": datetime.now().isoformat(),
        "status": "completed" if final_ok else "completed_with_best_explainability_error",
        "config": config,
        "results": {
            "seeds": [int(s) for s in seeds],
            "split_seed_mode": split_seed_mode,
            "base_split_seed": int(base_split_seed),
            "selection_metric": selection_metric,
            "screening_epochs": int(screening_epochs) if screening_epochs is not None else None,
            "screening_patience": int(screening_patience) if screening_patience is not None else None,
            "best_seed": int(best_seed),
            "best_seed_screen_run_dir": best_row.get("run_dir"),
            "best_explainability_run_dir": str(parent / f"best_seed_{best_seed}_explainability") if run_best_explainability else None,
            "best_explainability_run_id": final_summary.get("run_id") if isinstance(final_summary, dict) else None,
            "stability": stability,
            "rows": rows,
        },
        "artifacts": {
            "paths": {
                "base": str(parent),
                "multiseed_runs_csv": str(parent / "multiseed_runs.csv"),
                "multiseed_summary_json": str(parent / "multiseed_summary.json"),
                "best_run_summary": str(parent / f"best_seed_{best_seed}_explainability" / "run_summary.json")
                if run_best_explainability
                else str(Path(str(best_row.get("run_dir"))) / "run_summary.json"),
            }
        },
    }
    _write_json(parent / "multiseed_summary.json", summary)
    try:
        task_type = str(_get_path(config, ["training", "hyperparams", "task_type"], "") or "")
        if task_type in {CASCADE_GRADE_TASK_TYPE, CASCADE_EXPLORATION_4CLASS_TASK_TYPE}:
            cascade_runs_csv = parent / "cascade_repeated_runs.csv"
            cascade_summary_json = parent / "cascade_repeated_summary.json"
            _write_rows_csv(cascade_runs_csv, rows)
            _write_json(cascade_summary_json, summary)
            threshold_rows = []
            for r in rows:
                for split_name in ("val", "test"):
                    threshold_rows.append(
                        {
                            "phase": r.get("phase"),
                            "seed": r.get("seed"),
                            "split_random_state": r.get("split_random_state"),
                            "split": split_name,
                            "fixed_threshold": 0.10,
                            "precision": r.get(f"{split_name}_high_grade_precision_at_0_10"),
                            "recall": r.get(f"{split_name}_high_grade_recall_at_0_10"),
                            "f1": r.get(f"{split_name}_high_grade_f1_at_0_10"),
                            "f2": r.get(f"{split_name}_high_grade_f2_at_0_10"),
                            "n_pred_positive": r.get(f"{split_name}_high_grade_n_pred_positive_at_0_10"),
                            "average_precision": r.get(f"{split_name}_high_grade_average_precision_at_0_10"),
                            "val_selected_threshold": r.get("high_grade_threshold_from_val"),
                        }
                    )
            threshold_csv = parent / "cascade_threshold_stability.csv"
            _write_rows_csv(threshold_csv, threshold_rows)
            stable_threshold = _stable_threshold_from_rows(
                success_rows,
                split_name="val",
                min_recall=0.75,
                metric="f2",
            )
            if stable_threshold:
                stable_threshold_path = parent / "cascade_stable_threshold_recommendation.json"
                _write_json(stable_threshold_path, stable_threshold)
                threshold_summary_csv = parent / "cascade_stable_threshold_summary.csv"
                pd.DataFrame(stable_threshold.get("summary_rows", [])).to_csv(threshold_summary_csv, index=False)
            else:
                stable_threshold_path = None
                threshold_summary_csv = None
            ensemble_paths = _write_cascade_ensemble_artifacts(
                parent=parent,
                rows=success_rows,
                stable_threshold=stable_threshold,
            )
            summary["artifacts"]["paths"].update(
                {
                    "cascade_repeated_runs_csv": str(cascade_runs_csv),
                    "cascade_repeated_summary_json": str(cascade_summary_json),
                    "cascade_threshold_stability_csv": str(threshold_csv),
                    **(
                        {"cascade_stable_threshold_recommendation_json": str(stable_threshold_path)}
                        if stable_threshold_path
                        else {}
                    ),
                    **(
                        {"cascade_stable_threshold_summary_csv": str(threshold_summary_csv)}
                        if threshold_summary_csv
                        else {}
                    ),
                    **ensemble_paths,
                }
            )
            if stable_threshold:
                summary["results"]["cascade_stable_high_grade_threshold"] = stable_threshold
            if ensemble_paths:
                summary["results"]["cascade_ensemble"] = {
                    "artifact_path": ensemble_paths.get("cascade_ensemble_artifacts"),
                    "n_members": len([r for r in success_rows if _as_artifact_path(r.get("run_dir"), "cascade_training_artifacts.json")]),
                    "rule": "arithmetic_mean_final_probabilities",
                }
            _write_json(parent / "multiseed_summary.json", summary)
            _write_json(cascade_summary_json, summary)
    except Exception:
        pass
    save_run_summary(parent.name, summary)
    if progress_callback:
        progress_callback(1.0, f"Multi-seed complete. Best seed={best_seed}")
    return True, summary

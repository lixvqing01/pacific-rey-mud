from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

from scripts.run_rey_continuous_multitask import (
    MECHANISM3_CLASS_NAMES,
    _classification_metrics,
    _core_presence_metrics,
    _evaluate_predictions,
    _sample_threshold_calibration_metrics,
)
from core.utils.serialization_utils import NumpyJSONEncoder


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, cls=NumpyJSONEncoder, indent=2, ensure_ascii=False)


def _parse_thresholds(value: str) -> list[float]:
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def _run_prediction_path(run_dir: Path, split: str) -> Path:
    path = run_dir / "tables" / split / "mechanism_aware_rey_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction table: {path}")
    return path


def _load_split(run_dir: Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(_run_prediction_path(run_dir, split))
    df["source_run"] = run_dir.name
    return df


def _summarize_single_runs(run_dirs: list[Path], thresholds: list[float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        test_df = _load_split(run_dir, "test")
        metrics = _evaluate_predictions(test_df, thresholds)
        row: dict[str, Any] = {
            "run_id": run_dir.name,
            "n_test": int(len(test_df)),
            "log_mae": metrics["regression"]["log_mae"],
            "log_rmse": metrics["regression"]["log_rmse"],
            "spearman_log": metrics["regression"]["spearman_log"],
            "r2_log": metrics["regression"]["r2_log"],
            "mechanism_accuracy": metrics["mechanism3"]["accuracy"],
            "mechanism_balanced_accuracy": metrics["mechanism3"]["balanced_accuracy"],
            "mechanism_macro_f1": metrics["mechanism3"]["macro_f1"],
            "class1_precision": metrics["mechanism3"]["class1_enriched_precision"],
            "class1_recall": metrics["mechanism3"]["class1_enriched_recall"],
            "class1_f1": metrics["mechanism3"]["class1_enriched_f1"],
            "class2_precision": metrics["mechanism3"]["class2_slow_sedimentation_precision"],
            "class2_recall": metrics["mechanism3"]["class2_slow_sedimentation_recall"],
            "class2_f1": metrics["mechanism3"]["class2_slow_sedimentation_f1"],
        }
        for threshold in thresholds:
            prefix = f"rey_ge_{int(threshold)}"
            row[f"{prefix}_ap"] = metrics["thresholds"][f"{prefix}_ap"]
            row[f"{prefix}_roc_auc"] = metrics["thresholds"][f"{prefix}_roc_auc"]
            row[f"{prefix}_f1_at_0_5"] = metrics["thresholds"][f"{prefix}_f1_at_0_5"]
            row[f"{prefix}_recall_at_0_5"] = metrics["thresholds"][f"{prefix}_recall_at_0_5"]
            row[f"{prefix}_precision_at_0_5"] = metrics["thresholds"][f"{prefix}_precision_at_0_5"]
        rows.append(row)
    return pd.DataFrame(rows)


def _normalize_weights(weights: list[float] | None, n_runs: int) -> np.ndarray:
    if weights is None:
        return np.full(n_runs, 1.0 / float(n_runs), dtype=float)
    if len(weights) != n_runs:
        raise ValueError(f"Expected {n_runs} ensemble weights, got {len(weights)}")
    arr = np.asarray(weights, dtype=float)
    if np.any(arr < 0.0):
        raise ValueError("Ensemble weights must be non-negative.")
    total = float(np.sum(arr))
    if total <= 0.0:
        raise ValueError("At least one ensemble weight must be positive.")
    return arr / total


def _average_predictions(
    run_dirs: list[Path],
    split: str,
    thresholds: list[float],
    weights: np.ndarray,
) -> pd.DataFrame:
    frames = [_load_split(run_dir, split) for run_dir in run_dirs]
    base_cols = ["dataset_index", "core_id", "site_group", "mechanism_true", "rey_true"]
    base = frames[0][base_cols].copy()
    base = base.sort_values("dataset_index").reset_index(drop=True)
    prob_cols = [f"prob_{name}" for name in MECHANISM3_CLASS_NAMES]
    threshold_cols = [f"prob_rey_ge_{int(threshold)}" for threshold in thresholds]
    log_preds = []
    mech_probs = []
    threshold_probs = []
    sigmas = []
    for frame in frames:
        frame = frame.sort_values("dataset_index").reset_index(drop=True)
        if not np.array_equal(base["dataset_index"].to_numpy(), frame["dataset_index"].to_numpy()):
            raise ValueError("Prediction tables do not share the same dataset_index ordering.")
        log_preds.append(frame["log_rey_pred"].to_numpy(dtype=float))
        mech_probs.append(frame[prob_cols].to_numpy(dtype=float))
        threshold_probs.append(frame[threshold_cols].to_numpy(dtype=float))
        if "pred_sigma_norm" in frame.columns:
            sigmas.append(frame["pred_sigma_norm"].to_numpy(dtype=float))
    avg_log = np.average(np.vstack(log_preds), axis=0, weights=weights)
    avg_mech = np.average(np.stack(mech_probs, axis=0), axis=0, weights=weights)
    avg_threshold = np.average(np.stack(threshold_probs, axis=0), axis=0, weights=weights)
    avg_threshold = np.minimum.accumulate(avg_threshold, axis=1)
    out = base.copy()
    out["mechanism_pred"] = np.argmax(avg_mech, axis=1).astype(int)
    out["rey_pred"] = np.clip(np.expm1(avg_log), 0.0, None)
    out["log_rey_pred"] = avg_log
    out["pred_sigma_norm"] = np.average(np.vstack(sigmas), axis=0, weights=weights) if sigmas else np.nan
    for i, col in enumerate(prob_cols):
        out[col] = avg_mech[:, i]
    for i, col in enumerate(threshold_cols):
        out[col] = avg_threshold[:, i]
    return out


def _calibrated_threshold_metrics(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    thresholds: list[float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    methods = ["raw", "platt", "isotonic"]
    for rey_threshold in thresholds:
        score_col = f"prob_rey_ge_{int(rey_threshold)}"
        y_val = (val_df["rey_true"].to_numpy(dtype=float) >= float(rey_threshold)).astype(int)
        y_test = (test_df["rey_true"].to_numpy(dtype=float) >= float(rey_threshold)).astype(int)
        val_score_raw = val_df[score_col].to_numpy(dtype=float)
        test_score_raw = test_df[score_col].to_numpy(dtype=float)
        if np.unique(y_val).size < 2:
            continue
        transformed: dict[str, tuple[np.ndarray, np.ndarray]] = {"raw": (val_score_raw, test_score_raw)}
        try:
            model = LogisticRegression(solver="lbfgs")
            model.fit(val_score_raw.reshape(-1, 1), y_val)
            transformed["platt"] = (
                model.predict_proba(val_score_raw.reshape(-1, 1))[:, 1],
                model.predict_proba(test_score_raw.reshape(-1, 1))[:, 1],
            )
        except Exception:
            pass
        try:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(val_score_raw, y_val)
            transformed["isotonic"] = (iso.transform(val_score_raw), iso.transform(test_score_raw))
        except Exception:
            pass
        for method in methods:
            if method not in transformed:
                continue
            val_score, test_score = transformed[method]
            calibrated_val = val_df.copy()
            calibrated_test = test_df.copy()
            calibrated_val[score_col] = val_score
            calibrated_test[score_col] = test_score
            table = _sample_threshold_calibration_metrics(calibrated_val, calibrated_test, [rey_threshold])
            table.insert(0, "calibration_method", method)
            rows.extend(table.to_dict("records"))
    return pd.DataFrame(rows)


def _class1_threshold_scan(val_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    val_y = (val_df["mechanism_true"].to_numpy(dtype=int) == 1).astype(int)
    test_y = (test_df["mechanism_true"].to_numpy(dtype=int) == 1).astype(int)
    val_score = val_df["prob_class1_enriched"].to_numpy(dtype=float)
    test_score = test_df["prob_class1_enriched"].to_numpy(dtype=float)
    if np.unique(val_y).size < 2:
        return pd.DataFrame(rows)
    candidates = np.unique(np.quantile(val_score, np.linspace(0.0, 1.0, 101)))
    objectives = {
        "max_f1": lambda p, r, f: (f, p, r),
        "max_precision_recall_ge_0p50": lambda p, r, f: (p if r >= 0.50 else -1.0, f, r),
        "max_precision_recall_ge_0p70": lambda p, r, f: (p if r >= 0.70 else -1.0, f, r),
    }
    selected: dict[str, float] = {}
    for name, objective in objectives.items():
        best_key = None
        best_thr = float(candidates[0])
        for threshold in candidates:
            pred = (val_score >= float(threshold)).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(val_y, pred, labels=[1], zero_division=0)
            key = (*objective(float(precision[0]), float(recall[0]), float(f1[0])), float(threshold))
            if best_key is None or key > best_key:
                best_key = key
                best_thr = float(threshold)
        selected[name] = best_thr
    for objective, threshold in selected.items():
        for split_name, labels, score in (("val", val_y, val_score), ("test", test_y, test_score)):
            pred = (score >= float(threshold)).astype(int)
            precision, recall, f1, support = precision_recall_fscore_support(labels, pred, labels=[1], zero_division=0)
            rows.append(
                {
                    "objective": objective,
                    "selected_probability_threshold": float(threshold),
                    "split": split_name,
                    "n_samples": int(labels.size),
                    "positive_support": int(support[0]),
                    "accuracy": float(accuracy_score(labels, pred)),
                    "balanced_accuracy": float(balanced_accuracy_score(labels, pred)) if np.unique(labels).size > 1 else float("nan"),
                    "precision": float(precision[0]),
                    "recall": float(recall[0]),
                    "f1": float(f1[0]),
                }
            )
    return pd.DataFrame(rows)


def _aggregate_summary(single: pd.DataFrame) -> pd.DataFrame:
    numeric = single.select_dtypes(include=[np.number])
    rows = []
    for stat_name, values in (("mean", numeric.mean()), ("std", numeric.std(ddof=1)), ("min", numeric.min()), ("max", numeric.max())):
        row = {"stat": stat_name}
        row.update(values.to_dict())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate mechanism-aware REY multitask runs and seed ensembles.")
    parser.add_argument("--run-dirs", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--thresholds", default="600,800,1000,1500")
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="Optional non-negative ensemble weights matching --run-dirs; normalized internally.",
    )
    args = parser.parse_args()

    run_dirs = [Path(path) for path in args.run_dirs]
    thresholds = _parse_thresholds(args.thresholds)
    weights = _normalize_weights(args.weights, len(run_dirs))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    single = _summarize_single_runs(run_dirs, thresholds)
    single.to_csv(output_dir / "single_seed_test_summary.csv", index=False)
    aggregate = _aggregate_summary(single)
    aggregate.to_csv(output_dir / "single_seed_test_summary_stats.csv", index=False)

    ensemble_val = _average_predictions(run_dirs, "val", thresholds, weights)
    ensemble_test = _average_predictions(run_dirs, "test", thresholds, weights)
    ensemble_val.to_csv(output_dir / "ensemble_val_predictions.csv", index=False)
    ensemble_test.to_csv(output_dir / "ensemble_test_predictions.csv", index=False)

    ensemble_metrics = {
        "run_ids": [path.name for path in run_dirs],
        "weights": weights.tolist(),
        "n_runs": int(len(run_dirs)),
        "val_metrics": _evaluate_predictions(ensemble_val, thresholds),
        "test_metrics": _evaluate_predictions(ensemble_test, thresholds),
        "ensemble_mechanism3_with_argmax": _classification_metrics(
            ensemble_test["mechanism_true"].to_numpy(dtype=int),
            ensemble_test["mechanism_pred"].to_numpy(dtype=int),
        ),
    }
    _write_json(output_dir / "ensemble_metrics.json", ensemble_metrics)

    raw_calibrated = _sample_threshold_calibration_metrics(ensemble_val, ensemble_test, thresholds)
    raw_calibrated.to_csv(output_dir / "ensemble_sample_threshold_val_selected_metrics.csv", index=False)
    calibrated = _calibrated_threshold_metrics(ensemble_val, ensemble_test, thresholds)
    calibrated.to_csv(output_dir / "ensemble_sample_threshold_calibrated_methods.csv", index=False)
    core = _core_presence_metrics(ensemble_val, ensemble_test, thresholds)
    core.to_csv(output_dir / "ensemble_core_presence_val_selected_metrics.csv", index=False)
    class1 = _class1_threshold_scan(ensemble_val, ensemble_test)
    class1.to_csv(output_dir / "ensemble_class1_threshold_scan.csv", index=False)

    print(
        json.dumps(
            {
                "status": "completed",
                "output_dir": str(output_dir),
                "n_runs": len(run_dirs),
                "test_log_mae": ensemble_metrics["test_metrics"]["regression"]["log_mae"],
                "test_spearman_log": ensemble_metrics["test_metrics"]["regression"]["spearman_log"],
                "test_mechanism_macro_f1": ensemble_metrics["test_metrics"]["mechanism3"]["macro_f1"],
                "test_class1_f1": ensemble_metrics["test_metrics"]["mechanism3"]["class1_enriched_f1"],
            },
            cls=NumpyJSONEncoder,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

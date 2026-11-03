from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.utils.serialization_utils import NumpyJSONEncoder
from scripts.run_rey_continuous_multitask import _evaluate_predictions


def _parse_thresholds(value: str) -> list[float]:
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, cls=NumpyJSONEncoder, indent=2, ensure_ascii=False)


def _load_predictions(ensemble_dir: Path) -> pd.DataFrame:
    frames = []
    for split in ("val", "test"):
        path = ensemble_dir / f"ensemble_{split}_predictions.csv"
        if not path.exists():
            single_run_path = ensemble_dir / "tables" / split / "mechanism_aware_rey_predictions.csv"
            if single_run_path.exists():
                path = single_run_path
            else:
                raise FileNotFoundError(
                    f"Missing prediction table: expected {path} or {single_run_path}"
                )
        df = pd.read_csv(path)
        if "cnn_split" not in df.columns:
            df["cnn_split"] = split
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    if out["dataset_index"].duplicated().any():
        dupes = out.loc[out["dataset_index"].duplicated(), "dataset_index"].head(10).tolist()
        raise ValueError(f"Duplicate dataset_index values across val/test predictions: {dupes}")
    return out


def _log_bias(df: pd.DataFrame, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    true_log = np.log1p(np.clip(df.loc[mask, "rey_true"].to_numpy(dtype=float), 0.0, None))
    pred_log = np.log1p(np.clip(df.loc[mask, "rey_pred"].to_numpy(dtype=float), 0.0, None))
    return float(np.mean(pred_log - true_log))


def _repeat_row(repeat_name: str, test_df: pd.DataFrame, thresholds: list[float]) -> dict[str, Any]:
    metrics = _evaluate_predictions(test_df, thresholds)
    threshold_metrics = metrics["thresholds"]
    rey = test_df["rey_true"].to_numpy(dtype=float)
    return {
        "repeat": repeat_name,
        "n_samples": int(len(test_df)),
        "n_cores": int(test_df["core_id"].astype(str).nunique()),
        "log_mae": metrics["regression"]["log_mae"],
        "log_rmse": metrics["regression"]["log_rmse"],
        "spearman_log": metrics["regression"]["spearman_log"],
        "r2_log": metrics["regression"]["r2_log"],
        "mechanism_macro_f1": metrics["mechanism3"]["macro_f1"],
        "class1_f1": metrics["mechanism3"]["class1_enriched_f1"],
        "ap1000": threshold_metrics.get("rey_ge_1000_ap", float("nan")),
        "ap1500": threshold_metrics.get("rey_ge_1500_ap", float("nan")),
        "high1000_support": int(np.sum(rey >= 1000.0)),
        "high1500_support": int(np.sum(rey >= 1500.0)),
        "high1000_log_bias": _log_bias(test_df, rey >= 1000.0),
        "high1500_log_bias": _log_bias(test_df, rey >= 1500.0),
        "low600_log_bias": _log_bias(test_df, rey < 600.0),
    }


def evaluate_repeated_splits(args: argparse.Namespace) -> dict[str, Any]:
    ensemble_dir = Path(args.ensemble_dir)
    split_dir = Path(args.repeated_split_dir)
    output_dir = Path(args.output_dir)
    thresholds = _parse_thresholds(args.thresholds)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = _load_predictions(ensemble_dir)
    prediction_index = predictions.set_index("dataset_index", drop=False)
    repeat_dirs = sorted(path for path in split_dir.glob("repeat_*") if path.is_dir())
    if not repeat_dirs:
        raise FileNotFoundError(f"No repeat_* directories found under {split_dir}")

    rows: list[dict[str, Any]] = []
    for repeat_dir in repeat_dirs:
        assignment_path = repeat_dir / "split_assignments.csv"
        if not assignment_path.exists():
            raise FileNotFoundError(f"Missing split assignment file: {assignment_path}")
        assignments = pd.read_csv(assignment_path)
        if not {"dataset_index", "split"}.issubset(assignments.columns):
            raise ValueError(f"{assignment_path} must contain dataset_index and split columns.")
        test_ids = assignments.loc[assignments["split"].astype(str) == "test", "dataset_index"].astype(int).to_numpy()
        missing = sorted(set(test_ids) - set(prediction_index.index.astype(int)))
        if missing:
            raise ValueError(
                f"{repeat_dir.name} has {len(missing)} test rows missing from ensemble val/test predictions; "
                f"first missing indices: {missing[:10]}"
            )
        test_df = prediction_index.loc[test_ids].copy().reset_index(drop=True)
        test_df.to_csv(output_dir / f"{repeat_dir.name}_test_predictions.csv", index=False)
        rows.append(_repeat_row(repeat_dir.name, test_df, thresholds))

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "repeated_test_metrics.csv", index=False)
    numeric = metrics.select_dtypes(include=[np.number])
    stats_rows = []
    for stat_name, values in (
        ("mean", numeric.mean()),
        ("std", numeric.std(ddof=1)),
        ("min", numeric.min()),
        ("max", numeric.max()),
    ):
        row: dict[str, Any] = {"stat": stat_name}
        row.update(values.to_dict())
        stats_rows.append(row)
    stats = pd.DataFrame(stats_rows)
    stats.to_csv(output_dir / "repeated_test_metric_stats.csv", index=False)

    mean_row = stats.loc[stats["stat"] == "mean"].iloc[0].to_dict()
    std_row = stats.loc[stats["stat"] == "std"].iloc[0].to_dict()
    result = {
        "status": "completed",
        "ensemble_dir": str(ensemble_dir),
        "repeated_split_dir": str(split_dir),
        "output_dir": str(output_dir),
        "n_repeats": int(len(metrics)),
        "mean": mean_row,
        "std": std_row,
    }
    _write_json(output_dir / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a REY multitask ensemble on existing repeated held-out core split assignments."
    )
    parser.add_argument("--ensemble-dir", required=True, type=Path)
    parser.add_argument("--repeated-split-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--thresholds", default="600,800,1000,1500")
    args = parser.parse_args()
    print(json.dumps(evaluate_repeated_splits(args), cls=NumpyJSONEncoder, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

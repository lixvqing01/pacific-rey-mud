from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from core.utils.netcdf_io import open_dataset

from core.utils.serialization_utils import NumpyJSONEncoder
from scripts.evaluate_rey_multitask_ensemble import _average_predictions


DEFAULT_ENSEMBLE_DIR = Path("models/ensemble_evaluation")
DEFAULT_MODEL_RUNS = [
    Path("models/seed310"),
    Path("models/seed411"),
    Path("models/seed512"),
    Path("models/seed613"),
    Path("models/seed714"),
]
DEFAULT_POINTS = Path("data/geochem_class_table.parquet")
DEFAULT_ENV_NC = Path("data/env_gridded.nc")
HARD_CORES = ["76", "74", "U1370D7", "MR14-E02 PC11", "316", "163", "851B", "U1367B2"]


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, cls=NumpyJSONEncoder, indent=2, ensure_ascii=False)


def _load_predictions(args: argparse.Namespace) -> pd.DataFrame:
    ensemble_dir = Path(args.ensemble_dir) if args.ensemble_dir else None
    pred_path = ensemble_dir / "ensemble_test_predictions.csv" if ensemble_dir else None
    if pred_path and pred_path.exists():
        return pd.read_csv(pred_path)
    run_dirs = [Path(path) for path in args.model_run_dirs]
    weights = np.full(len(run_dirs), 1.0 / max(len(run_dirs), 1), dtype=float)
    return _average_predictions(run_dirs, "test", [600.0, 800.0, 1000.0, 1500.0], weights)


def _attach_point_metadata(pred: pd.DataFrame, points_path: Path) -> pd.DataFrame:
    points = pd.read_parquet(points_path).reset_index(drop=True)
    if "dataset_index" not in pred.columns:
        return pred
    meta_cols = [
        col
        for col in ["lon", "lat", "depth", "core_id", "site_group", "exploration_class_id", "exploration_class_name"]
        if col in points.columns
    ]
    meta = points.loc[pred["dataset_index"].astype(int).to_numpy(), meta_cols].reset_index(drop=True)
    out = pred.reset_index(drop=True).copy()
    for col in meta.columns:
        if col not in out.columns:
            out[col] = meta[col]
    return out


def _core_residual_table(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["log_true"] = np.log1p(np.clip(pd.to_numeric(work["rey_true"], errors="coerce"), 0.0, None))
    work["log_pred"] = np.log1p(np.clip(pd.to_numeric(work["rey_pred"], errors="coerce"), 0.0, None))
    work["log_residual"] = work["log_pred"] - work["log_true"]
    rows = []
    for core, group in work.groupby("core_id", dropna=False):
        true = group["rey_true"].to_numpy(dtype=float)
        pred = group["rey_pred"].to_numpy(dtype=float)
        rows.append(
            {
                "core_id": core,
                "n": int(len(group)),
                "mean_rey_true": float(np.nanmean(true)),
                "mean_rey_pred": float(np.nanmean(pred)),
                "max_rey_true": float(np.nanmax(true)),
                "max_rey_pred": float(np.nanmax(pred)),
                "log_mae": float(np.nanmean(np.abs(group["log_residual"]))),
                "log_bias": float(np.nanmean(group["log_residual"])),
                "high1000_support": int(np.sum(true >= 1000.0)),
                "high1500_support": int(np.sum(true >= 1500.0)),
                "high1000_log_bias": float(np.nanmean(group.loc[true >= 1000.0, "log_residual"])) if np.any(true >= 1000.0) else np.nan,
                "high1500_log_bias": float(np.nanmean(group.loc[true >= 1500.0, "log_residual"])) if np.any(true >= 1500.0) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["log_mae", "n"], ascending=[False, False])


def _mechanism_posthoc(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mech, group in df.groupby("mechanism_true", dropna=False):
        true = group["rey_true"].to_numpy(dtype=float)
        pred = group["rey_pred"].to_numpy(dtype=float)
        rows.append(
            {
                "mechanism_true": int(mech) if pd.notna(mech) else -1,
                "n": int(len(group)),
                "mean_true_rey": float(np.nanmean(true)),
                "mean_pred_rey": float(np.nanmean(pred)),
                "median_true_rey": float(np.nanmedian(true)),
                "median_pred_rey": float(np.nanmedian(pred)),
                "log_mae": float(np.nanmean(np.abs(np.log1p(np.clip(pred, 0.0, None)) - np.log1p(np.clip(true, 0.0, None))))),
                "positive_1000_rate_true": float(np.nanmean(true >= 1000.0)),
                "positive_1000_rate_pred_p05": float(np.nanmean(group.get("prob_rey_ge_1000", pd.Series(0.0, index=group.index)).to_numpy(dtype=float) >= 0.5)),
                "positive_1500_rate_true": float(np.nanmean(true >= 1500.0)),
                "positive_1500_rate_pred_p05": float(np.nanmean(group.get("prob_rey_ge_1500", pd.Series(0.0, index=group.index)).to_numpy(dtype=float) >= 0.5)),
            }
        )
    return pd.DataFrame(rows)


def _coord_names(ds: xr.Dataset) -> tuple[str, str]:
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    return lat_name, lon_name


def _env_center_table(df: pd.DataFrame, env_nc: Path, max_features: int) -> pd.DataFrame:
    if not {"lon", "lat"}.issubset(df.columns):
        return pd.DataFrame()
    with open_dataset(env_nc) as ds:
        lat_name, lon_name = _coord_names(ds)
        features = [str(name) for name in ds.data_vars if str(name) != "land_mask"][: int(max_features)]
        lon = ((df["lon"].to_numpy(dtype=float) + 180.0) % 360.0) - 180.0
        lat = df["lat"].to_numpy(dtype=float)
        rows = []
        target = np.log1p(np.clip(df["rey_pred"].to_numpy(dtype=float), 0.0, None))
        residual = np.log1p(np.clip(df["rey_pred"].to_numpy(dtype=float), 0.0, None)) - np.log1p(np.clip(df["rey_true"].to_numpy(dtype=float), 0.0, None))
        for feature in features:
            try:
                field = ds[feature]
                coords = {}
                if lon_name in field.coords:
                    coords[lon_name] = ("points", lon)
                if lat_name in field.coords:
                    coords[lat_name] = ("points", lat)
                if len(coords) != 2:
                    continue
                vals = field.interp(coords, method="nearest").values.astype(float)
                finite = np.isfinite(vals) & np.isfinite(target) & np.isfinite(residual)
                if int(finite.sum()) < 10 or float(np.nanstd(vals[finite])) < 1e-12:
                    continue
                rows.append(
                    {
                        "feature": feature,
                        "n": int(finite.sum()),
                        "corr_with_pred_logrey": float(np.corrcoef(vals[finite], target[finite])[0, 1]),
                        "corr_with_log_residual": float(np.corrcoef(vals[finite], residual[finite])[0, 1]),
                    }
                )
            except Exception:
                continue
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["abs_corr_with_pred_logrey"] = out["corr_with_pred_logrey"].abs()
    return out.sort_values("abs_corr_with_pred_logrey", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight explainability/post-hoc diagnostics for continuous REY model.")
    parser.add_argument("--ensemble-dir", type=Path, default=DEFAULT_ENSEMBLE_DIR)
    parser.add_argument("--model-run-dirs", nargs="+", default=[str(path) for path in DEFAULT_MODEL_RUNS])
    parser.add_argument("--points", type=Path, default=DEFAULT_POINTS)
    parser.add_argument("--env-nc", type=Path, default=DEFAULT_ENV_NC)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-env-features", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred = _load_predictions(args)
    pred = _attach_point_metadata(pred, Path(args.points))
    pred.to_csv(out_dir / "test_predictions_with_metadata.csv", index=False)

    core_table = _core_residual_table(pred)
    core_table.to_csv(out_dir / "core_residual_audit.csv", index=False)
    hard_mask = core_table["core_id"].astype(str).apply(lambda x: any(pattern in x for pattern in HARD_CORES))
    core_table.loc[hard_mask].to_csv(out_dir / "hard_core_residual_audit.csv", index=False)

    mechanism_table = _mechanism_posthoc(pred)
    mechanism_table.to_csv(out_dir / "mechanism_posthoc_summary.csv", index=False)
    env_table = _env_center_table(pred, Path(args.env_nc), int(args.max_env_features))
    env_table.to_csv(out_dir / "env_feature_univariate_association.csv", index=False)

    summary = {
        "status": "completed",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(out_dir),
        "n_test_predictions": int(len(pred)),
        "artifacts": {
            "test_predictions_with_metadata": str(out_dir / "test_predictions_with_metadata.csv"),
            "core_residual_audit": str(out_dir / "core_residual_audit.csv"),
            "hard_core_residual_audit": str(out_dir / "hard_core_residual_audit.csv"),
            "mechanism_posthoc_summary": str(out_dir / "mechanism_posthoc_summary.csv"),
            "env_feature_univariate_association": str(out_dir / "env_feature_univariate_association.csv"),
        },
        "placeholder_next": {
            "shap": "Deep/Kernel SHAP for logREY, P1000, P1500, and mechanism probabilities can be added under this output directory after the fixed GUI workflow is frozen.",
            "profile_explainability": "Depth-profile attributions for 76* and MR14-E02 PC11 should be added after hard-core profile reruns are finalized.",
        },
    }
    _write_json(out_dir / "continuous_rey_explainability_summary.json", summary)
    print(json.dumps(summary, cls=NumpyJSONEncoder, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

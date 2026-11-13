from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from core.utils.serialization_utils import NumpyJSONEncoder


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, cls=NumpyJSONEncoder, indent=2, ensure_ascii=False)


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table suffix for {path}")


def _parse_thresholds(value: str) -> list[float]:
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def _load_predictions(ensemble_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for split in ("val", "test"):
        path = ensemble_dir / f"ensemble_{split}_predictions.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing ensemble prediction table: {path}")
        df = pd.read_csv(path)
        if "cnn_split" not in df.columns:
            df["cnn_split"] = split
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["dataset_index"] = out["dataset_index"].astype(int)
    if out["dataset_index"].duplicated().any():
        dupes = out.loc[out["dataset_index"].duplicated(), "dataset_index"].head(10).tolist()
        raise ValueError(f"Duplicate dataset_index values across val/test predictions: {dupes}")
    return out


def _load_geochem_context(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    df = _read_table(path).reset_index(drop=True)
    df = df.copy()
    df["dataset_index"] = np.arange(len(df), dtype=int)
    rename = {}
    if "lon" in df.columns and "longitude" not in df.columns:
        rename["lon"] = "longitude"
    if "lat" in df.columns and "latitude" not in df.columns:
        rename["lat"] = "latitude"
    if rename:
        df = df.rename(columns=rename)
    return df


def _merge_context(predictions: pd.DataFrame, context: pd.DataFrame | None) -> pd.DataFrame:
    if context is None:
        return predictions.copy()
    keep_cols = [
        "dataset_index",
        "longitude",
        "latitude",
        "depth",
        "core_id",
        "site_group",
        "rey",
        "mech_class_id",
        "exploration_class_id",
        "exploration_class_name",
        "grade_geochem_class_id",
        "grade_geochem_class_name",
    ]
    keep_cols = [col for col in keep_cols if col in context.columns]
    merged = predictions.merge(
        context[keep_cols],
        on="dataset_index",
        how="left",
        suffixes=("", "_context"),
    )
    for col in ("core_id", "site_group"):
        ctx = f"{col}_context"
        if ctx in merged.columns:
            merged[col] = merged[col].combine_first(merged[ctx])
            merged = merged.drop(columns=[ctx])
    if "rey_context" in merged.columns:
        merged["rey_true"] = merged["rey_true"].combine_first(merged["rey_context"])
        merged = merged.drop(columns=["rey_context"])
    if "mech_class_id" in merged.columns and "mechanism_true" in merged.columns:
        merged["mechanism_true"] = merged["mechanism_true"].combine_first(merged["mech_class_id"])
    return merged


def _log_values(values: pd.Series | np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(np.asarray(values, dtype=float), 0.0, None))


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    sx = pd.Series(x).rank(method="average")
    sy = pd.Series(y).rank(method="average")
    if sx.nunique(dropna=True) < 2 or sy.nunique(dropna=True) < 2:
        return float("nan")
    return float(sx.corr(sy))


def _safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def _haversine_km(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    lon1_rad = np.radians(lon1)
    lat1_rad = np.radians(lat1)
    lon2_rad = np.radians(lon2)
    lat2_rad = np.radians(lat2)
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _split_summary(repeat: str, split: str, df: pd.DataFrame) -> dict[str, Any]:
    rey = df["rey_true"].to_numpy(dtype=float)
    row: dict[str, Any] = {
        "repeat": repeat,
        "split": split,
        "n_samples": int(len(df)),
        "n_cores": int(df["core_id"].astype(str).nunique()) if "core_id" in df.columns else 0,
        "n_sites": int(df["site_group"].astype(str).nunique()) if "site_group" in df.columns else 0,
    }
    for threshold in (600.0, 800.0, 1000.0, 1500.0, 2000.0):
        mask = rey >= threshold
        key = str(int(threshold))
        row[f"high{key}_samples"] = int(mask.sum())
        row[f"high{key}_cores"] = int(df.loc[mask, "core_id"].astype(str).nunique()) if np.any(mask) else 0
    for q in (0.5, 0.75, 0.9, 0.95, 0.99):
        row[f"rey_q{int(q * 100)}"] = float(np.quantile(rey, q)) if len(rey) else float("nan")
    if "mechanism_true" in df.columns:
        mech_counts = df["mechanism_true"].value_counts(dropna=False)
        for label, count in mech_counts.items():
            row[f"mechanism_{label}_samples"] = int(count)
    return row


def _core_profile_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for core_id, core_df in df.groupby(df["core_id"].astype(str), dropna=False):
        rey = core_df["rey_true"].to_numpy(dtype=float)
        row: dict[str, Any] = {
            "core_id": str(core_id),
            "n_samples": int(len(core_df)),
            "site_group": str(core_df["site_group"].dropna().astype(str).mode().iloc[0])
            if "site_group" in core_df.columns and not core_df["site_group"].dropna().empty
            else "",
            "rey_mean": float(np.mean(rey)) if len(rey) else float("nan"),
            "rey_median": float(np.median(rey)) if len(rey) else float("nan"),
            "rey_max": float(np.max(rey)) if len(rey) else float("nan"),
            "high1000_samples": int(np.sum(rey >= 1000.0)),
            "high1500_samples": int(np.sum(rey >= 1500.0)),
        }
        for col in ("longitude", "latitude", "depth"):
            if col in core_df.columns:
                vals = pd.to_numeric(core_df[col], errors="coerce").dropna()
                row[f"{col}_mean"] = float(vals.mean()) if len(vals) else float("nan")
                row[f"{col}_min"] = float(vals.min()) if len(vals) else float("nan")
                row[f"{col}_max"] = float(vals.max()) if len(vals) else float("nan")
        if "mechanism_true" in core_df.columns:
            mech = core_df["mechanism_true"].value_counts(normalize=True)
            for label in (0, 1, 2):
                row[f"mechanism_{label}_frac"] = float(mech.get(label, 0.0))
        rows.append(row)
    return pd.DataFrame(rows)


def _nearest_core_features(
    test_profile: pd.Series,
    train_profiles: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[str, float]:
    if train_profiles.empty or not feature_columns:
        return "", float("nan")
    train = train_profiles[feature_columns].astype(float)
    test = test_profile[feature_columns].astype(float)
    combined = pd.concat([train, test.to_frame().T], ignore_index=True)
    std = combined.std(axis=0, ddof=0).replace(0, 1.0)
    mean = combined.mean(axis=0)
    train_z = (train - mean) / std
    test_z = (test - mean) / std
    dist = np.sqrt(np.sum((train_z.to_numpy(dtype=float) - test_z.to_numpy(dtype=float)) ** 2, axis=1))
    idx = int(np.nanargmin(dist))
    return str(train_profiles.iloc[idx]["core_id"]), float(dist[idx])


def _nearest_core_geo(test_profile: pd.Series, train_profiles: pd.DataFrame) -> tuple[str, float]:
    required = {"longitude_mean", "latitude_mean"}
    if train_profiles.empty or not required.issubset(train_profiles.columns) or not required.issubset(test_profile.index):
        return "", float("nan")
    if pd.isna(test_profile["longitude_mean"]) or pd.isna(test_profile["latitude_mean"]):
        return "", float("nan")
    train_valid = train_profiles.dropna(subset=["longitude_mean", "latitude_mean"])
    if train_valid.empty:
        return "", float("nan")
    dist = _haversine_km(
        np.full(len(train_valid), float(test_profile["longitude_mean"])),
        np.full(len(train_valid), float(test_profile["latitude_mean"])),
        train_valid["longitude_mean"].to_numpy(dtype=float),
        train_valid["latitude_mean"].to_numpy(dtype=float),
    )
    idx = int(np.nanargmin(dist))
    return str(train_valid.iloc[idx]["core_id"]), float(dist[idx])


def _same_site_support(test_profile: pd.Series, train_df: pd.DataFrame) -> dict[str, Any]:
    site = str(test_profile.get("site_group", ""))
    if not site or "site_group" not in train_df.columns:
        return {"same_site_train_samples": 0, "same_site_train_cores": 0}
    mask = train_df["site_group"].astype(str) == site
    return {
        "same_site_train_samples": int(mask.sum()),
        "same_site_train_cores": int(train_df.loc[mask, "core_id"].astype(str).nunique()) if mask.any() else 0,
    }


def _core_metrics(
    repeat: str,
    core_df: pd.DataFrame,
    thresholds: list[float],
    train_df: pd.DataFrame,
    train_profiles: pd.DataFrame,
    cnn_train_df: pd.DataFrame | None,
    cnn_train_profiles: pd.DataFrame | None,
) -> dict[str, Any]:
    true = core_df["rey_true"].to_numpy(dtype=float)
    pred = core_df["rey_pred"].to_numpy(dtype=float)
    true_log = _log_values(true)
    pred_log = _log_values(pred)
    err = pred_log - true_log
    profile = _core_profile_table(core_df).iloc[0]
    row: dict[str, Any] = {
        "repeat": repeat,
        "core_id": str(profile["core_id"]),
        "site_group": str(profile.get("site_group", "")),
        "n_samples": int(len(core_df)),
        "cnn_split_values": ";".join(sorted(core_df["cnn_split"].astype(str).unique())) if "cnn_split" in core_df.columns else "",
        "rey_true_mean": float(np.mean(true)),
        "rey_true_median": float(np.median(true)),
        "rey_true_max": float(np.max(true)),
        "rey_pred_mean": float(np.mean(pred)),
        "rey_pred_median": float(np.median(pred)),
        "rey_pred_max": float(np.max(pred)),
        "log_mae": float(np.mean(np.abs(err))),
        "log_bias": float(np.mean(err)),
        "spearman_log": _safe_spearman(true_log, pred_log),
        "high1000_support": int(np.sum(true >= 1000.0)),
        "high1500_support": int(np.sum(true >= 1500.0)),
        "low600_support": int(np.sum(true < 600.0)),
    }
    for col in ("longitude", "latitude", "depth"):
        if col in core_df.columns:
            vals = pd.to_numeric(core_df[col], errors="coerce").dropna()
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            row[f"{col}_min"] = float(vals.min()) if len(vals) else float("nan")
            row[f"{col}_max"] = float(vals.max()) if len(vals) else float("nan")
    for threshold in thresholds:
        key = str(int(threshold))
        label = (true >= threshold).astype(int)
        score_col = f"prob_rey_ge_{key}"
        score = core_df[score_col].to_numpy(dtype=float) if score_col in core_df.columns else np.full(len(core_df), np.nan)
        row[f"ap{key}"] = _safe_ap(label, score) if not np.isnan(score).all() else float("nan")
        row[f"prob_ge_{key}_mean"] = float(np.nanmean(score)) if not np.isnan(score).all() else float("nan")
        row[f"prob_ge_{key}_max"] = float(np.nanmax(score)) if not np.isnan(score).all() else float("nan")
        mask = true >= threshold
        row[f"high{key}_support"] = int(mask.sum())
        row[f"high{key}_log_bias"] = float(np.mean(err[mask])) if np.any(mask) else float("nan")
    low_mask = true < 600.0
    row["low600_log_bias"] = float(np.mean(err[low_mask])) if np.any(low_mask) else float("nan")
    row["error_pattern"] = "mixed"
    if row["high1000_support"] > 0 and row.get("high1000_log_bias", 0.0) < -0.5:
        row["error_pattern"] = "high_tail_underprediction"
    elif row["low600_support"] > 0 and row["low600_log_bias"] > 0.3:
        row["error_pattern"] = "low_background_overprediction"

    feature_cols = [
        col
        for col in [
            "rey_mean",
            "rey_median",
            "rey_max",
            "high1000_samples",
            "high1500_samples",
            "depth_mean",
            "depth_max",
            "mechanism_0_frac",
            "mechanism_1_frac",
            "mechanism_2_frac",
        ]
        if col in train_profiles.columns and col in profile.index
    ]
    nearest_core, profile_dist = _nearest_core_features(profile, train_profiles, feature_cols)
    nearest_geo_core, geo_km = _nearest_core_geo(profile, train_profiles)
    row.update(
        {
            "nearest_repeat_train_core_profile": nearest_core,
            "nearest_repeat_train_profile_dist": profile_dist,
            "nearest_repeat_train_core_geo": nearest_geo_core,
            "nearest_repeat_train_geo_km": geo_km,
        }
    )
    row.update({f"repeat_{k}": v for k, v in _same_site_support(profile, train_df).items()})

    if cnn_train_df is not None and cnn_train_profiles is not None:
        nearest_core, profile_dist = _nearest_core_features(profile, cnn_train_profiles, feature_cols)
        nearest_geo_core, geo_km = _nearest_core_geo(profile, cnn_train_profiles)
        row.update(
            {
                "nearest_cnn_train_core_profile": nearest_core,
                "nearest_cnn_train_profile_dist": profile_dist,
                "nearest_cnn_train_core_geo": nearest_geo_core,
                "nearest_cnn_train_geo_km": geo_km,
            }
        )
        row.update({f"cnn_{k}": v for k, v in _same_site_support(profile, cnn_train_df).items()})
    return row


def _load_cnn_train_context(
    assignment_path: Path | None,
    context: pd.DataFrame | None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if assignment_path is None:
        return None, None
    if context is None:
        raise ValueError("--cnn-split-assignments requires --geochem-table so train rows can be reconstructed.")
    assignments = pd.read_csv(assignment_path)
    if not {"dataset_index", "split"}.issubset(assignments.columns):
        raise ValueError(f"{assignment_path} must contain dataset_index and split columns.")
    train_ids = assignments.loc[assignments["split"].astype(str) == "train", "dataset_index"].astype(int)
    ctx = context.copy()
    ctx = ctx.merge(assignments[["dataset_index", "split"]], on="dataset_index", how="inner")
    train_df = ctx.loc[ctx["split"].astype(str) == "train"].copy()
    if len(train_df) != len(train_ids):
        missing = sorted(set(train_ids) - set(train_df["dataset_index"].astype(int)))
        raise ValueError(f"CNN train context is missing {len(missing)} rows; first missing: {missing[:10]}")
    if "rey_true" not in train_df.columns and "rey" in train_df.columns:
        train_df["rey_true"] = train_df["rey"]
    if "mechanism_true" not in train_df.columns and "mech_class_id" in train_df.columns:
        train_df["mechanism_true"] = train_df["mech_class_id"]
    return train_df, _core_profile_table(train_df)


def _prepare_context_for_summary(context: pd.DataFrame | None) -> pd.DataFrame | None:
    if context is None:
        return None
    out = context.copy()
    if "rey_true" not in out.columns and "rey" in out.columns:
        out["rey_true"] = out["rey"]
    if "mechanism_true" not in out.columns and "mech_class_id" in out.columns:
        out["mechanism_true"] = out["mech_class_id"]
    return out


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    ensemble_dir = Path(args.ensemble_dir)
    repeated_split_dir = Path(args.repeated_split_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = _parse_thresholds(args.thresholds)

    context = _load_geochem_context(Path(args.geochem_table) if args.geochem_table else None)
    context_for_summary = _prepare_context_for_summary(context)
    predictions = _merge_context(_load_predictions(ensemble_dir), context)
    predictions.to_csv(output_dir / "prediction_pool_with_context.csv", index=False)
    pred_index = predictions.set_index("dataset_index", drop=False)
    context_index = (
        context_for_summary.set_index("dataset_index", drop=False)
        if context_for_summary is not None
        else None
    )

    cnn_train_df, cnn_train_profiles = _load_cnn_train_context(
        Path(args.cnn_split_assignments) if args.cnn_split_assignments else None,
        context,
    )

    repeat_dirs = sorted(path for path in repeated_split_dir.glob("repeat_*") if path.is_dir())
    if not repeat_dirs:
        raise FileNotFoundError(f"No repeat_* directories found under {repeated_split_dir}")

    repeat_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    core_rows: list[dict[str, Any]] = []

    for repeat_dir in repeat_dirs:
        assignments = pd.read_csv(repeat_dir / "split_assignments.csv")
        if not {"dataset_index", "split"}.issubset(assignments.columns):
            raise ValueError(f"{repeat_dir / 'split_assignments.csv'} must contain dataset_index and split columns.")
        repeat_name = repeat_dir.name
        available_ids = set(pred_index.index.astype(int))
        context_ids = set(context_index.index.astype(int)) if context_index is not None else set()
        for split_name in ("train", "val", "test"):
            ids = assignments.loc[assignments["split"].astype(str) == split_name, "dataset_index"].astype(int).to_numpy()
            if split_name == "test":
                missing = sorted(set(ids) - available_ids)
                if missing:
                    raise ValueError(f"{repeat_name}/{split_name} has rows missing from prediction pool: {missing[:10]}")
                split_df = pred_index.loc[ids].copy().reset_index(drop=True)
            elif set(ids).issubset(available_ids):
                split_df = pred_index.loc[ids].copy().reset_index(drop=True)
            elif context_index is not None and set(ids).issubset(context_ids):
                split_df = context_index.loc[ids].copy().reset_index(drop=True)
            else:
                missing = sorted(set(ids) - available_ids - context_ids)
                raise ValueError(
                    f"{repeat_name}/{split_name} has rows missing from both prediction pool and context: {missing[:10]}"
                )
            split_rows.append(_split_summary(repeat_name, split_name, split_df))
            if split_name == "test":
                test_df = split_df
            if split_name == "train":
                train_df = split_df

        true = test_df["rey_true"].to_numpy(dtype=float)
        pred = test_df["rey_pred"].to_numpy(dtype=float)
        true_log = _log_values(true)
        pred_log = _log_values(pred)
        err = pred_log - true_log
        repeat_row = _split_summary(repeat_name, "test", test_df)
        repeat_row.update(
            {
                "log_mae": float(np.mean(np.abs(err))),
                "log_bias": float(np.mean(err)),
                "spearman_log": _safe_spearman(true_log, pred_log),
                "high1000_log_bias": float(np.mean(err[true >= 1000.0])) if np.any(true >= 1000.0) else float("nan"),
                "high1500_log_bias": float(np.mean(err[true >= 1500.0])) if np.any(true >= 1500.0) else float("nan"),
                "low600_log_bias": float(np.mean(err[true < 600.0])) if np.any(true < 600.0) else float("nan"),
            }
        )
        for threshold in thresholds:
            key = str(int(threshold))
            score_col = f"prob_rey_ge_{key}"
            if score_col in test_df.columns:
                repeat_row[f"ap{key}"] = _safe_ap(
                    (true >= threshold).astype(int),
                    test_df[score_col].to_numpy(dtype=float),
                )
        repeat_rows.append(repeat_row)

        train_profiles = _core_profile_table(train_df)
        for _, core_df in test_df.groupby(test_df["core_id"].astype(str), dropna=False):
            core_rows.append(
                _core_metrics(
                    repeat_name,
                    core_df,
                    thresholds,
                    train_df,
                    train_profiles,
                    cnn_train_df,
                    cnn_train_profiles,
                )
            )

    repeat_summary = pd.DataFrame(repeat_rows)
    split_summary = pd.DataFrame(split_rows)
    core_repeat = pd.DataFrame(core_rows)
    repeat_summary.to_csv(output_dir / "repeat_summary.csv", index=False)
    split_summary.to_csv(output_dir / "split_distribution_summary.csv", index=False)
    core_repeat.to_csv(output_dir / "core_repeat_audit.csv", index=False)

    numeric_cols = [
        col
        for col in core_repeat.select_dtypes(include=[np.number]).columns
        if col not in {"n_samples", "high1000_support", "high1500_support", "low600_support"}
    ]
    aggregations: dict[str, Any] = {
        "repeat": "nunique",
        "n_samples": "mean",
        "high1000_support": "mean",
        "high1500_support": "mean",
        "low600_support": "mean",
        "site_group": lambda x: ";".join(sorted(set(str(v) for v in x if pd.notna(v))))[:200],
        "error_pattern": lambda x: x.value_counts().index[0] if len(x) else "",
    }
    for col in numeric_cols:
        aggregations[col] = "mean"
    core_summary = core_repeat.groupby("core_id", dropna=False).agg(aggregations).reset_index()
    core_summary = core_summary.rename(columns={"repeat": "n_test_repeats"})
    core_summary = core_summary.sort_values(["log_mae", "high1500_support", "high1000_support"], ascending=[False, False, False])
    core_summary.to_csv(output_dir / "core_summary.csv", index=False)
    core_summary.head(int(args.top_n)).to_csv(output_dir / "worst_core_summary.csv", index=False)

    result = {
        "status": "completed",
        "ensemble_dir": str(ensemble_dir),
        "repeated_split_dir": str(repeated_split_dir),
        "output_dir": str(output_dir),
        "n_repeats": int(len(repeat_dirs)),
        "n_core_repeat_rows": int(len(core_repeat)),
        "worst_cores_by_log_mae": core_summary.head(int(args.top_n))[
            ["core_id", "n_test_repeats", "log_mae", "log_bias", "high1000_support", "high1500_support", "error_pattern"]
        ].to_dict(orient="records"),
        "repeat_mean": repeat_summary.select_dtypes(include=[np.number]).mean().to_dict(),
    }
    _write_json(output_dir / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose repeated held-out core failures for a REY continuous multitask ensemble."
    )
    parser.add_argument("--ensemble-dir", required=True, type=Path)
    parser.add_argument("--repeated-split-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--geochem-table", type=Path, default=None)
    parser.add_argument(
        "--cnn-split-assignments",
        type=Path,
        default=None,
        help="Optional original CNN split_assignments.csv for same-site/nearest original-train diagnostics.",
    )
    parser.add_argument("--thresholds", default="600,800,1000,1500")
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(run_audit(args), cls=NumpyJSONEncoder, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.utils.serialization_utils import NumpyJSONEncoder


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, cls=NumpyJSONEncoder, indent=2, ensure_ascii=False)


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table suffix: {path}")


def _load_prediction_pool(ensemble_dir: Path, splits: set[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for split in ("train", "val", "test"):
        if split not in splits:
            continue
        path = ensemble_dir / f"ensemble_{split}_predictions.csv"
        if not path.exists():
            if split == "train":
                continue
            raise FileNotFoundError(f"Missing prediction table: {path}")
        df = pd.read_csv(path)
        if "cnn_split" not in df.columns:
            df["cnn_split"] = split
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No requested prediction splits found under {ensemble_dir}")
    out = pd.concat(frames, ignore_index=True)
    out["dataset_index"] = out["dataset_index"].astype(int)
    if out["dataset_index"].duplicated().any():
        dupes = out.loc[out["dataset_index"].duplicated(), "dataset_index"].head(10).tolist()
        raise ValueError(f"Duplicate dataset_index values in split pool: {dupes}")
    return out


def _merge_geochem(pool: pd.DataFrame, geochem_path: Path) -> pd.DataFrame:
    geochem = _read_table(geochem_path).reset_index(drop=True).copy()
    geochem["dataset_index"] = np.arange(len(geochem), dtype=int)
    if "lon" in geochem.columns and "longitude" not in geochem.columns:
        geochem = geochem.rename(columns={"lon": "longitude"})
    if "lat" in geochem.columns and "latitude" not in geochem.columns:
        geochem = geochem.rename(columns={"lat": "latitude"})
    keep = [
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
    ]
    keep = [col for col in keep if col in geochem.columns]
    merged = pool.merge(geochem[keep], on="dataset_index", how="left", suffixes=("", "_geo"))
    for col in ("core_id", "site_group"):
        geo = f"{col}_geo"
        if geo in merged.columns:
            merged[col] = merged[col].combine_first(merged[geo])
            merged = merged.drop(columns=[geo])
    if "rey_true" not in merged.columns and "rey" in merged.columns:
        merged["rey_true"] = merged["rey"]
    elif "rey" in merged.columns:
        merged["rey_true"] = merged["rey_true"].combine_first(merged["rey"])
    if "mechanism_true" not in merged.columns and "mech_class_id" in merged.columns:
        merged["mechanism_true"] = merged["mech_class_id"]
    elif "mech_class_id" in merged.columns:
        merged["mechanism_true"] = merged["mechanism_true"].combine_first(merged["mech_class_id"])
    required = {"dataset_index", "core_id", "rey_true"}
    missing = required - set(merged.columns)
    if missing:
        raise ValueError(f"Split pool is missing required columns after merge: {sorted(missing)}")
    return merged


def _spatial_block(df: pd.DataFrame, block_deg: float) -> pd.Series:
    if "longitude" not in df.columns or "latitude" not in df.columns:
        return pd.Series(["missing"] * len(df), index=df.index)
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    ok = lon.notna() & lat.notna()
    block = max(float(block_deg), 1.0)
    lon_bin = np.floor(np.mod(lon.fillna(0.0).to_numpy(dtype=float), 360.0) / block).astype(int)
    lat_bin = np.floor((lat.fillna(0.0).to_numpy(dtype=float) + 90.0) / block).astype(int)
    labels = [f"{a}:{b}" if valid else "missing" for a, b, valid in zip(lon_bin, lat_bin, ok.to_numpy(), strict=False)]
    return pd.Series(labels, index=df.index)


def _categorical_l1(values: pd.Series, ref: pd.Series) -> float:
    cats = pd.Index(pd.unique(ref.astype(str)))
    if len(cats) <= 1:
        return 0.0
    ref_props = ref.astype(str).value_counts(normalize=True)
    cur_props = values.astype(str).value_counts(normalize=True) if len(values) else pd.Series(dtype=float)
    return float(sum(abs(float(cur_props.get(cat, 0.0)) - float(ref_props.get(cat, 0.0))) for cat in cats))


def _core_profile(pool: pd.DataFrame, thresholds: list[float], spatial_block_deg: float) -> pd.DataFrame:
    rows = []
    pool = pool.copy()
    pool["spatial_block"] = _spatial_block(pool, spatial_block_deg)
    for core_id, sub in pool.groupby(pool["core_id"].astype(str), dropna=False):
        rey = pd.to_numeric(sub["rey_true"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        row: dict[str, Any] = {
            "core_id": str(core_id),
            "n_samples": int(len(sub)),
            "site_group": str(sub["site_group"].dropna().astype(str).mode().iloc[0])
            if "site_group" in sub.columns and not sub["site_group"].dropna().empty
            else str(core_id),
            "spatial_block": str(sub["spatial_block"].dropna().astype(str).mode().iloc[0]),
            "rey_mean": float(np.mean(rey)),
            "rey_median": float(np.median(rey)),
            "rey_max": float(np.max(rey)),
            "rey_q90": float(np.quantile(rey, 0.90)),
            "rey_q95": float(np.quantile(rey, 0.95)),
        }
        for threshold in thresholds:
            key = str(int(threshold))
            row[f"high{key}_samples"] = int(np.sum(rey >= threshold))
            row[f"has_high{key}"] = int(np.any(rey >= threshold))
        if "mechanism_true" in sub.columns:
            mech = pd.to_numeric(sub["mechanism_true"], errors="coerce")
            for label in (0, 1, 2):
                row[f"mechanism{label}_samples"] = int((mech == label).sum())
                row[f"has_mechanism{label}"] = int((mech == label).any())
        if "exploration_class_id" in sub.columns:
            cls = pd.to_numeric(sub["exploration_class_id"], errors="coerce")
            for label in (0, 1, 2, 3):
                row[f"exploration{label}_samples"] = int((cls == label).sum())
                row[f"has_exploration{label}"] = int((cls == label).any())
        rows.append(row)
    return pd.DataFrame(rows)


def _split_rows(pool: pd.DataFrame, profiles: pd.DataFrame, groups: set[str]) -> pd.DataFrame:
    group_list = list(groups)
    return pool[pool["core_id"].astype(str).isin(group_list)].copy()


def _quantile_error(values: np.ndarray, ref: np.ndarray) -> float:
    if len(values) == 0 or len(ref) == 0:
        return 1.0
    qs = np.array([0.5, 0.75, 0.9, 0.95], dtype=float)
    ref_q = np.quantile(ref, qs)
    cur_q = np.quantile(values, qs)
    scale = max(float(np.quantile(ref, 0.95) - np.quantile(ref, 0.05)), 1.0)
    return float(np.mean(np.abs(cur_q - ref_q) / scale))


def _score_split(
    pool: pd.DataFrame,
    profiles: pd.DataFrame,
    train_groups: set[str],
    val_groups: set[str],
    test_groups: set[str],
    target_ratios: dict[str, float],
    thresholds: list[float],
    min_test_high1000_cores: int,
    min_test_high1500_cores: int,
    min_val_high1000_cores: int,
    min_val_high1500_cores: int,
) -> dict[str, float]:
    full_rey = pd.to_numeric(pool["rey_true"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    full_n = max(len(pool), 1)
    full_group_n = max(profiles["core_id"].nunique(), 1)
    full_threshold_props = {
        int(thr): float(np.mean(full_rey >= thr))
        for thr in thresholds
    }
    full_mech = pool["mechanism_true"] if "mechanism_true" in pool.columns else pd.Series([], dtype=object)
    full_space = pool["spatial_block"] if "spatial_block" in pool.columns else pd.Series([], dtype=object)
    split_groups = {"train": train_groups, "val": val_groups, "test": test_groups}

    sample_ratio_error = 0.0
    group_ratio_error = 0.0
    threshold_error = 0.0
    mechanism_error = 0.0
    spatial_error = 0.0
    quantile_error = 0.0
    dominance_penalty = 0.0
    support_penalty = 0.0

    for split_name, groups in split_groups.items():
        weight = 1.0 if split_name == "train" else 1.6
        df = _split_rows(pool, profiles, groups)
        rey = pd.to_numeric(df["rey_true"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        sample_ratio_error += weight * abs(len(df) / full_n - target_ratios[split_name])
        group_ratio_error += weight * abs(len(groups) / full_group_n - target_ratios[split_name])
        for thr in thresholds:
            key = int(thr)
            threshold_error += weight * abs(float(np.mean(rey >= thr)) - full_threshold_props[key]) if len(rey) else weight
        quantile_error += weight * _quantile_error(rey, full_rey)
        if len(full_mech):
            mechanism_error += weight * _categorical_l1(df["mechanism_true"], full_mech)
        if len(full_space):
            spatial_error += weight * _categorical_l1(df["spatial_block"], full_space)
        if len(df):
            largest_core_frac = df["core_id"].astype(str).value_counts(normalize=True).iloc[0]
            largest_site_frac = df["site_group"].astype(str).value_counts(normalize=True).iloc[0] if "site_group" in df.columns else 0.0
            dominance_penalty += weight * max(0.0, float(largest_core_frac) - 0.35)
            dominance_penalty += 0.5 * weight * max(0.0, float(largest_site_frac) - 0.45)

    test_profiles = profiles[profiles["core_id"].isin(test_groups)]
    val_profiles = profiles[profiles["core_id"].isin(val_groups)]
    for split_name, split_profiles, min1000, min1500 in (
        ("test", test_profiles, min_test_high1000_cores, min_test_high1500_cores),
        ("val", val_profiles, min_val_high1000_cores, min_val_high1500_cores),
    ):
        high1000 = int(split_profiles.get("has_high1000", pd.Series(dtype=int)).sum())
        high1500 = int(split_profiles.get("has_high1500", pd.Series(dtype=int)).sum())
        split_weight = 2.5 if split_name == "test" else 1.5
        support_penalty += split_weight * max(0, int(min1000) - high1000)
        support_penalty += 2.0 * split_weight * max(0, int(min1500) - high1500)

    score = (
        1.5 * sample_ratio_error
        + 1.0 * group_ratio_error
        + 2.5 * threshold_error
        + 1.2 * mechanism_error
        + 0.8 * spatial_error
        + 1.3 * quantile_error
        + 1.2 * dominance_penalty
        + 5.0 * support_penalty
    )
    return {
        "score": float(score),
        "sample_ratio_error": float(sample_ratio_error),
        "group_ratio_error": float(group_ratio_error),
        "threshold_error": float(threshold_error),
        "mechanism_error": float(mechanism_error),
        "spatial_error": float(spatial_error),
        "quantile_error": float(quantile_error),
        "dominance_penalty": float(dominance_penalty),
        "support_penalty": float(support_penalty),
    }


def _summarize_split(pool: pd.DataFrame, groups: set[str], split_name: str, repeat: str) -> dict[str, Any]:
    df = _split_rows(pool, pd.DataFrame(), groups)
    rey = pd.to_numeric(df["rey_true"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    row: dict[str, Any] = {
        "repeat": repeat,
        "split": split_name,
        "n_samples": int(len(df)),
        "n_cores": int(df["core_id"].astype(str).nunique()),
        "n_sites": int(df["site_group"].astype(str).nunique()) if "site_group" in df.columns else 0,
        "rey_q50": float(np.quantile(rey, 0.50)) if len(rey) else float("nan"),
        "rey_q90": float(np.quantile(rey, 0.90)) if len(rey) else float("nan"),
        "rey_q95": float(np.quantile(rey, 0.95)) if len(rey) else float("nan"),
    }
    for thr in (600.0, 800.0, 1000.0, 1500.0, 2000.0):
        key = str(int(thr))
        mask = rey >= thr
        row[f"high{key}_samples"] = int(mask.sum())
        row[f"high{key}_cores"] = int(df.loc[mask, "core_id"].astype(str).nunique()) if np.any(mask) else 0
    if "mechanism_true" in df.columns:
        mech = pd.to_numeric(df["mechanism_true"], errors="coerce")
        for label in (0, 1, 2):
            row[f"mechanism{label}_samples"] = int((mech == label).sum())
    return row


def _write_assignments(output_dir: Path, repeat_name: str, pool: pd.DataFrame, split_groups: dict[str, set[str]]) -> None:
    rows: list[dict[str, Any]] = []
    for split_name, groups in split_groups.items():
        df = _split_rows(pool, pd.DataFrame(), groups)
        for _, row in df.iterrows():
            rows.append(
                {
                    "dataset_index": int(row["dataset_index"]),
                    "split": split_name,
                    "core_id": str(row["core_id"]),
                }
            )
    repeat_dir = output_dir / repeat_name
    repeat_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("dataset_index").to_csv(repeat_dir / "split_assignments.csv", index=False)


def generate_splits(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = [float(x) for x in str(args.thresholds).split(",") if str(x).strip()]
    splits = {part.strip() for part in str(args.prediction_splits).split(",") if part.strip()}
    pool = _load_prediction_pool(Path(args.ensemble_dir), splits)
    pool = _merge_geochem(pool, Path(args.geochem_table))
    pool = pool.copy()
    pool["core_id"] = pool["core_id"].astype(str)
    pool["site_group"] = pool.get("site_group", pool["core_id"]).astype(str)
    pool["spatial_block"] = _spatial_block(pool, float(args.spatial_block_deg))
    profiles = _core_profile(pool, thresholds, float(args.spatial_block_deg))
    profiles.to_csv(output_dir / "core_profiles.csv", index=False)
    pool.to_csv(output_dir / "split_pool.csv", index=False)

    groups = profiles["core_id"].astype(str).to_numpy()
    n_groups = int(len(groups))
    if n_groups < 5:
        raise ValueError(f"Need at least 5 cores for repeated split generation, got {n_groups}")
    test_group_n = max(1, int(round(n_groups * float(args.test_ratio))))
    val_group_n = max(1, int(round((n_groups - test_group_n) * float(args.validation_ratio))))
    train_group_n = n_groups - test_group_n - val_group_n
    if train_group_n <= 0:
        raise ValueError("Requested ratios leave no training cores.")
    target_ratios = {
        "test": float(args.test_ratio),
        "val": (1.0 - float(args.test_ratio)) * float(args.validation_ratio),
    }
    target_ratios["train"] = max(0.0, 1.0 - target_ratios["test"] - target_ratios["val"])

    selected_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    selected_test_sets: list[set[str]] = []
    rng = np.random.default_rng(int(args.seed))
    all_candidate_rows: list[dict[str, Any]] = []

    for repeat_idx in range(int(args.n_repeats)):
        best: tuple[float, dict[str, Any], dict[str, set[str]]] | None = None
        for candidate_idx in range(int(args.candidates_per_repeat)):
            shuffled = rng.permutation(groups)
            test_groups = set(str(x) for x in shuffled[:test_group_n])
            val_groups = set(str(x) for x in shuffled[test_group_n : test_group_n + val_group_n])
            train_groups = set(str(x) for x in shuffled[test_group_n + val_group_n :])
            components = _score_split(
                pool,
                profiles,
                train_groups,
                val_groups,
                test_groups,
                target_ratios,
                thresholds,
                int(args.min_test_high1000_cores),
                int(args.min_test_high1500_cores),
                int(args.min_val_high1000_cores),
                int(args.min_val_high1500_cores),
            )
            max_overlap = 0.0
            for prev in selected_test_sets:
                union = len(test_groups | prev)
                if union:
                    max_overlap = max(max_overlap, len(test_groups & prev) / union)
            adjusted_score = float(components["score"]) + float(args.test_overlap_penalty) * max_overlap
            row = {
                "repeat": f"repeat_{repeat_idx:03d}",
                "candidate": int(candidate_idx),
                "raw_score": float(components["score"]),
                "adjusted_score": adjusted_score,
                "max_test_jaccard": float(max_overlap),
                **components,
            }
            if repeat_idx < int(args.keep_candidate_rows_for_first_repeats):
                all_candidate_rows.append(row)
            if best is None or adjusted_score < best[0]:
                best = (adjusted_score, row, {"train": train_groups, "val": val_groups, "test": test_groups})
        if best is None:
            raise RuntimeError(f"No candidate split generated for repeat {repeat_idx}")
        _adjusted_score, selected_row, split_groups = best
        selected_test_sets.append(set(split_groups["test"]))
        selected_rows.append(selected_row)
        repeat_name = str(selected_row["repeat"])
        _write_assignments(output_dir, repeat_name, pool, split_groups)
        for split_name, group_set in split_groups.items():
            summary_rows.append(_summarize_split(pool, group_set, split_name, repeat_name))

    selected = pd.DataFrame(selected_rows)
    split_summary = pd.DataFrame(summary_rows)
    selected.to_csv(output_dir / "selected_split_scores.csv", index=False)
    split_summary.to_csv(output_dir / "split_distribution_summary.csv", index=False)
    if all_candidate_rows:
        pd.DataFrame(all_candidate_rows).to_csv(output_dir / "candidate_scores_sample.csv", index=False)

    config = {
        "ensemble_dir": str(args.ensemble_dir),
        "geochem_table": str(args.geochem_table),
        "output_dir": str(output_dir),
        "prediction_splits": sorted(splits),
        "n_repeats": int(args.n_repeats),
        "candidates_per_repeat": int(args.candidates_per_repeat),
        "seed": int(args.seed),
        "thresholds": thresholds,
        "n_pool_samples": int(len(pool)),
        "n_pool_cores": int(n_groups),
        "test_group_n": int(test_group_n),
        "val_group_n": int(val_group_n),
        "train_group_n": int(train_group_n),
        "target_ratios": target_ratios,
        "min_test_high1000_cores": int(args.min_test_high1000_cores),
        "min_test_high1500_cores": int(args.min_test_high1500_cores),
        "min_val_high1000_cores": int(args.min_val_high1000_cores),
        "min_val_high1500_cores": int(args.min_val_high1500_cores),
        "mean_selected_score": float(selected["raw_score"].mean()),
        "mean_selected_max_test_jaccard": float(selected["max_test_jaccard"].mean()),
    }
    _write_json(output_dir / "split_generation_summary.json", config)
    return {"status": "completed", **config}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate balanced repeated held-out core splits for continuous REY ensemble evaluation."
    )
    parser.add_argument("--ensemble-dir", required=True, type=Path)
    parser.add_argument("--geochem-table", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prediction-splits", default="val,test")
    parser.add_argument("--n-repeats", type=int, default=20)
    parser.add_argument("--candidates-per-repeat", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=310)
    parser.add_argument("--test-ratio", type=float, default=0.25)
    parser.add_argument("--validation-ratio", type=float, default=0.25)
    parser.add_argument("--thresholds", default="600,800,1000,1500")
    parser.add_argument("--spatial-block-deg", type=float, default=10.0)
    parser.add_argument("--min-test-high1000-cores", type=int, default=5)
    parser.add_argument("--min-test-high1500-cores", type=int, default=3)
    parser.add_argument("--min-val-high1000-cores", type=int, default=4)
    parser.add_argument("--min-val-high1500-cores", type=int, default=2)
    parser.add_argument("--test-overlap-penalty", type=float, default=1.5)
    parser.add_argument("--keep-candidate-rows-for-first-repeats", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(generate_splits(args), cls=NumpyJSONEncoder, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import average_precision_score, f1_score, r2_score
from scipy.stats import spearmanr

from core.utils.serialization_utils import NumpyJSONEncoder
from scripts.run_rey_paper_reproduction import (
    DEFAULT_ENV_NC,
    DEFAULT_POINTS,
    DEFAULT_REPEATED_SPLITS,
    DEFAULT_SED_RATE_NC,
    _baseline_train_cmd,
    _python_executable,
)


SEEDS = [310, 411, 512, 613, 714]
THRESHOLDS = [600.0, 800.0, 1000.0, 1500.0]
BASELINE_ENSEMBLE = Path("models/ensemble_evaluation")
BASELINE_RUN = Path("models/seed310")

VARIANTS: list[dict[str, Any]] = [
    {"id": "A1_no_stratigraphy", "label": "No stratigraphy", "kind": "ablation", "options": {"--depth-feature-type": "zero"}},
    {"id": "A2_physical_depth", "label": "Physical depth", "kind": "sensitivity", "options": {"--depth-feature-type": "depth_mbsf"}},
    {
        "id": "A3_no_mechanism_block",
        "label": "No mechanism block",
        "kind": "ablation",
        "options": {
            "--regression-head-mode": "global",
            "--mechanism-loss-weight": "0",
            "--gate-balance-weight": "0",
            "--selection-mechanism-f1-weight": "0",
            "--selection-class1-recall-weight": "0",
        },
        "mechanism_valid": False,
    },
    {
        "id": "A4_no_threshold_block",
        "label": "No threshold block",
        "kind": "ablation",
        "options": {
            "--threshold-loss-weight": "0",
            "--monotonic-loss-weight": "0",
            "--selection-threshold-f1-weight": "0",
        },
        "threshold_head_valid": False,
    },
    {"id": "A5_no_rank_loss", "label": "No rank loss", "kind": "ablation", "options": {"--rank-loss-weight": "0"}},
    {"id": "A6_single_scale", "label": "Single-scale CNN", "kind": "ablation", "options": {}, "flags": ["--no-multiscale"]},
    {
        "id": "A7_no_spatial_neighbourhood",
        "label": "No spatial neighbourhood",
        "kind": "ablation",
        "options": {"--spatial-context-mode": "center_broadcast"},
    },
    {"id": "A8_huber_regression", "label": "Huber regression", "kind": "sensitivity", "options": {"--regression-loss": "huber"}},
]


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, cls=NumpyJSONEncoder, indent=2, ensure_ascii=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    result: dict[str, Any] = {"path": str(path), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}
    if stat.st_size <= 64 * 1024 * 1024:
        result["sha256"] = _sha256(path)
    return result


def _successful_command_history(log_dir: Path) -> list[dict[str, Any]]:
    """Recover successful commands so resumed/table-only runs preserve provenance."""
    records: list[dict[str, Any]] = []
    for path in sorted(log_dir.glob("*.json"), key=lambda item: item.stat().st_mtime_ns):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if int(record.get("returncode", -1)) == 0 and record.get("cmd"):
            record["record_file"] = str(path)
            records.append(record)
    return records


def _deduplicate_command_history(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[tuple[str, ...], dict[str, Any]] = {}
    for record in records:
        key = tuple(str(part) for part in record.get("cmd", []))
        if key:
            unique[key] = record
    return sorted(unique.values(), key=lambda record: str(record.get("started", "")))


def _git_state(repo_root: Path) -> dict[str, Any]:
    def _output(args: list[str]) -> str:
        proc = subprocess.run(args, cwd=str(repo_root), text=True, capture_output=True, encoding="utf-8", errors="replace")
        return (proc.stdout or "").strip() if proc.returncode == 0 else ""

    status = _output(["git", "status", "--porcelain"])
    return {
        "commit": _output(["git", "rev-parse", "HEAD"]),
        "branch": _output(["git", "branch", "--show-current"]),
        "dirty": bool(status),
        "dirty_file_count": len(status.splitlines()) if status else 0,
    }


def _replace_option(cmd: list[str], option: str, value: str) -> None:
    if option in cmd:
        idx = cmd.index(option)
        if idx + 1 >= len(cmd) or cmd[idx + 1].startswith("--"):
            raise ValueError(f"Cannot replace flag-like option {option}")
        cmd[idx + 1] = str(value)
    else:
        cmd.extend([option, str(value)])


def _emit(stage: str, completed: int, total: int, **extra: Any) -> None:
    payload = {"stage": stage, "event": stage, "percent": completed / max(total, 1), **extra}
    print("PROGRESS_JSON " + json.dumps(payload, ensure_ascii=False), flush=True)


def _run(cmd: list[str], *, cwd: Path, log_path: Path, dry_run: bool) -> dict[str, Any]:
    started = datetime.now().isoformat(timespec="seconds")
    if dry_run:
        result = {"cmd": cmd, "returncode": 0, "dry_run": True, "started": started}
        _write_json(log_path.with_suffix(".json"), result)
        return result
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, encoding="utf-8", errors="replace", env=env)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text((proc.stdout or "") + "\n--- STDERR ---\n" + (proc.stderr or ""), encoding="utf-8")
    result = {
        "cmd": cmd,
        "returncode": int(proc.returncode),
        "dry_run": False,
        "started": started,
        "finished": datetime.now().isoformat(timespec="seconds"),
        "log": str(log_path),
    }
    _write_json(log_path.with_suffix(".json"), result)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}); see {log_path}")
    return result


def _load_predictions(model: dict[str, Any], split: str) -> pd.DataFrame:
    if model["type"] == "rf":
        path = Path(model["path"]) / "tables" / split / "random_forest_rey_predictions.csv"
    else:
        path = Path(model["path"]) / f"ensemble_{split}_predictions.csv"
    frame = pd.read_csv(path)
    return frame.sort_values("dataset_index").reset_index(drop=True)


def _safe_ap(y: np.ndarray, score: np.ndarray) -> float:
    return float(average_precision_score(y, score)) if np.unique(y).size > 1 else float("nan")


def _common_metrics(df: pd.DataFrame, *, mechanism_valid: bool, threshold_head_valid: bool) -> dict[str, float]:
    true_rey = df["rey_true"].to_numpy(dtype=float)
    pred_rey = np.clip(df["rey_pred"].to_numpy(dtype=float), 0.0, None)
    true_log = np.log1p(np.clip(true_rey, 0.0, None))
    pred_log = np.log1p(pred_rey)
    out = {
        "n_samples": float(len(df)),
        "log_mae": float(np.mean(np.abs(pred_log - true_log))),
        "spearman_log": float(spearmanr(true_log, pred_log).statistic),
        "r2_log": float(r2_score(true_log, pred_log)),
    }
    for threshold in (1000.0, 1500.0):
        label = (true_rey >= threshold).astype(int)
        out[f"grade_ap{int(threshold)}"] = _safe_ap(label, pred_log)
        head_col = f"prob_rey_ge_{int(threshold)}"
        out[f"head_ap{int(threshold)}"] = (
            _safe_ap(label, df[head_col].to_numpy(dtype=float))
            if threshold_head_valid and head_col in df.columns
            else float("nan")
        )
        mask = true_rey >= threshold
        out[f"high{int(threshold)}_log_bias"] = float(np.mean(pred_log[mask] - true_log[mask])) if mask.any() else float("nan")
    low = true_rey < 600.0
    out["low600_log_bias"] = float(np.mean(pred_log[low] - true_log[low])) if low.any() else float("nan")
    if mechanism_valid and {"mechanism_true", "mechanism_pred"}.issubset(df.columns):
        y_true = df["mechanism_true"].to_numpy(dtype=int)
        y_pred = df["mechanism_pred"].to_numpy(dtype=int)
        out["mechanism_macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        out["facies_ii_f1"] = float(f1_score(y_true == 1, y_pred == 1, zero_division=0))
    else:
        out["mechanism_macro_f1"] = float("nan")
        out["facies_ii_f1"] = float("nan")
    return out


def _markdown(df: pd.DataFrame) -> str:
    # Human-facing tables cite the supplementary sample inventory instead of
    # duplicating a potentially version-sensitive sample-count statement.
    drop = [c for c in ["n_samples", "mean_n_samples", "std_n_samples", "delta_mean_n_samples"] if c in df.columns]
    df = df.drop(columns=drop)
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_csv(index=False)


def _core_bootstrap(
    models: list[dict[str, Any]],
    prediction_cache: dict[str, dict[str, pd.DataFrame]],
    *,
    n_boot: int = 500,
) -> pd.DataFrame:
    baseline = prediction_cache["A0_full_model"]["test"].sort_values("dataset_index").reset_index(drop=True)
    core_ids = baseline["core_id"].astype(str).unique()
    rng = np.random.default_rng(310)
    rows: list[dict[str, Any]] = []
    metric_names = ["log_mae", "spearman_log", "r2_log", "grade_ap1000", "grade_ap1500"]
    for model in models:
        if model["id"] == "A0_full_model":
            continue
        candidate = prediction_cache[model["id"]]["test"].sort_values("dataset_index").reset_index(drop=True)
        if not np.array_equal(baseline["dataset_index"].to_numpy(), candidate["dataset_index"].to_numpy()):
            raise ValueError(f"Bootstrap alignment failed for {model['id']}")
        delta_values = {name: [] for name in metric_names}
        for _ in range(int(n_boot)):
            sampled = rng.choice(core_ids, size=len(core_ids), replace=True)
            base_parts = [baseline.loc[baseline["core_id"].astype(str) == core] for core in sampled]
            cand_parts = [candidate.loc[candidate["core_id"].astype(str) == core] for core in sampled]
            base_metrics = _common_metrics(pd.concat(base_parts, ignore_index=True), mechanism_valid=True, threshold_head_valid=True)
            cand_metrics = _common_metrics(
                pd.concat(cand_parts, ignore_index=True),
                mechanism_valid=model["mechanism_valid"],
                threshold_head_valid=model["threshold_head_valid"],
            )
            for name in metric_names:
                delta_values[name].append(float(cand_metrics[name] - base_metrics[name]))
        for name, values in delta_values.items():
            arr = np.asarray(values, dtype=float)
            rows.append(
                {
                    "model_id": model["id"],
                    "model": model["label"],
                    "metric": name,
                    "n_boot": int(n_boot),
                    "delta_mean": float(np.nanmean(arr)),
                    "delta_ci_low": float(np.nanpercentile(arr, 2.5)),
                    "delta_ci_high": float(np.nanpercentile(arr, 97.5)),
                }
            )
    return pd.DataFrame(rows)


def _build_tables(models: list[dict[str, Any]], repeated_split_dir: Path, output_dir: Path) -> None:
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    fixed_rows: list[dict[str, Any]] = []
    prediction_cache: dict[str, dict[str, pd.DataFrame]] = {}
    for model in models:
        val = _load_predictions(model, "val")
        test = _load_predictions(model, "test")
        prediction_cache[model["id"]] = {"val": val, "test": test}
        row = {"model_id": model["id"], "model": model["label"], "experiment_type": model["kind"]}
        row.update(_common_metrics(test, mechanism_valid=model["mechanism_valid"], threshold_head_valid=model["threshold_head_valid"]))
        fixed_rows.append(row)
    fixed = pd.DataFrame(fixed_rows)
    baseline = fixed.loc[fixed["model_id"] == "A0_full_model"].iloc[0]
    metric_cols = [c for c in fixed.columns if c not in ("model_id", "model", "experiment_type", "n_samples")]
    for col in metric_cols:
        fixed[f"delta_{col}"] = pd.to_numeric(fixed[col], errors="coerce") - float(baseline[col])
    fixed.to_csv(table_dir / "rey_ablation_fixed_test.csv", index=False)
    (table_dir / "rey_ablation_fixed_test.md").write_text(_markdown(fixed), encoding="utf-8")

    single_seed_rows: list[dict[str, Any]] = []
    for model in models:
        if model["type"] != "cnn":
            continue
        ensemble_dir = Path(model["path"])
        ensemble_metrics = json.loads((ensemble_dir / "ensemble_metrics.json").read_text(encoding="utf-8"))
        for run_id in ensemble_metrics.get("run_ids", []):
            run_dir = (
                ensemble_dir.parent / str(run_id)
                if model["id"] == "A0_full_model"
                else output_dir / "runs" / model["id"] / str(run_id)
            )
            predictions = pd.read_csv(run_dir / "tables" / "test" / "mechanism_aware_rey_predictions.csv")
            row = {
                "model_id": model["id"],
                "model": model["label"],
                "experiment_type": model["kind"],
                "run_id": str(run_id),
            }
            row.update(
                _common_metrics(
                    predictions,
                    mechanism_valid=model["mechanism_valid"],
                    threshold_head_valid=model["threshold_head_valid"],
                )
            )
            single_seed_rows.append(row)
    single_seed = pd.DataFrame(single_seed_rows)
    single_seed.to_csv(table_dir / "rey_ablation_single_seed_metrics.csv", index=False)
    single_seed_summary_rows: list[dict[str, Any]] = []
    for model_id, group in single_seed.groupby("model_id", sort=False):
        summary_row: dict[str, Any] = {
            "model_id": model_id,
            "model": group["model"].iloc[0],
            "experiment_type": group["experiment_type"].iloc[0],
            "n_seeds": int(len(group)),
        }
        for col in [c for c in group.columns if c not in ("model_id", "model", "experiment_type", "run_id")]:
            values = pd.to_numeric(group[col], errors="coerce")
            summary_row[f"mean_{col}"] = float(values.mean())
            summary_row[f"std_{col}"] = float(values.std(ddof=1))
        single_seed_summary_rows.append(summary_row)
    single_seed_summary = pd.DataFrame(single_seed_summary_rows)
    single_seed_summary.to_csv(table_dir / "rey_ablation_single_seed_summary.csv", index=False)
    (table_dir / "rey_ablation_single_seed_summary.md").write_text(_markdown(single_seed_summary), encoding="utf-8")

    repeat_rows: list[dict[str, Any]] = []
    repeat_dirs = sorted(p for p in repeated_split_dir.glob("repeat_*") if p.is_dir())
    for model in models:
        pool = pd.concat([prediction_cache[model["id"]]["val"], prediction_cache[model["id"]]["test"]], ignore_index=True)
        pool = pool.drop_duplicates("dataset_index").set_index("dataset_index", drop=False)
        per_repeat: list[dict[str, Any]] = []
        for repeat_dir in repeat_dirs:
            assignments = pd.read_csv(repeat_dir / "split_assignments.csv")
            ids = assignments.loc[assignments["split"].astype(str) == "test", "dataset_index"].astype(int).to_numpy()
            missing = sorted(set(ids) - set(pool.index.astype(int)))
            if missing:
                raise ValueError(f"{model['id']} missing repeated-test ids: {missing[:10]}")
            metrics = _common_metrics(
                pool.loc[ids].reset_index(drop=True),
                mechanism_valid=model["mechanism_valid"],
                threshold_head_valid=model["threshold_head_valid"],
            )
            per_repeat.append({"repeat": repeat_dir.name, **metrics})
        per = pd.DataFrame(per_repeat)
        per.to_csv(table_dir / f"{model['id']}_repeated_metrics.csv", index=False)
        row = {"model_id": model["id"], "model": model["label"], "experiment_type": model["kind"], "n_repeats": len(per)}
        for col in per.columns:
            if col != "repeat":
                row[f"mean_{col}"] = float(pd.to_numeric(per[col], errors="coerce").mean())
                row[f"std_{col}"] = float(pd.to_numeric(per[col], errors="coerce").std(ddof=1))
        repeat_rows.append(row)
    repeated = pd.DataFrame(repeat_rows)
    base_rep = repeated.loc[repeated["model_id"] == "A0_full_model"].iloc[0]
    for col in [c for c in repeated.columns if c.startswith("mean_")]:
        repeated[f"delta_{col}"] = pd.to_numeric(repeated[col], errors="coerce") - float(base_rep[col])
    baseline_per = pd.read_csv(table_dir / "A0_full_model_repeated_metrics.csv").set_index("repeat")
    for row_idx, model_id in repeated["model_id"].items():
        model_per = pd.read_csv(table_dir / f"{model_id}_repeated_metrics.csv").set_index("repeat")
        shared = baseline_per.index.intersection(model_per.index)
        for metric in ("log_mae", "spearman_log", "r2_log", "grade_ap1000", "grade_ap1500"):
            base_values = pd.to_numeric(baseline_per.loc[shared, metric], errors="coerce")
            model_values = pd.to_numeric(model_per.loc[shared, metric], errors="coerce")
            wins = model_values < base_values if metric == "log_mae" else model_values > base_values
            repeated.loc[row_idx, f"wins_vs_full_{metric}"] = int(wins.fillna(False).sum())
    repeated.to_csv(table_dir / "rey_ablation_repeated.csv", index=False)
    (table_dir / "rey_ablation_repeated.md").write_text(_markdown(repeated), encoding="utf-8")
    _write_json(
        table_dir / "rey_ablation_full_metrics.json",
        {
            "fixed_test": fixed_rows,
            "single_seed": single_seed_rows,
            "single_seed_summary": single_seed_summary_rows,
            "repeated": repeat_rows,
        },
    )
    bootstrap = _core_bootstrap(models, prediction_cache, n_boot=500)
    bootstrap.to_csv(table_dir / "rey_ablation_core_bootstrap.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fixed OFAT continuous-REY ablation suite and RF baseline.")
    parser.add_argument("--env-nc", type=Path, default=DEFAULT_ENV_NC)
    parser.add_argument("--points", type=Path, default=DEFAULT_POINTS)
    parser.add_argument("--sed-rate-nc", type=Path, default=DEFAULT_SED_RATE_NC)
    parser.add_argument("--repeated-split-dir", type=Path, default=DEFAULT_REPEATED_SPLITS)
    parser.add_argument("--baseline-ensemble-dir", type=Path, default=BASELINE_ENSEMBLE)
    parser.add_argument("--baseline-run-dir", type=Path, default=BASELINE_RUN)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/rey_continuous_ofat_ablation"))
    parser.add_argument("--mode", choices=["smoke", "publication"], default="publication")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--tables-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path.cwd()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    py = _python_executable(repo_root)
    seeds = [310] if args.mode == "smoke" else list(SEEDS)
    epochs = 1 if args.mode == "smoke" else 100
    patience = 1 if args.mode == "smoke" else 20
    split_dir = Path(args.baseline_run_dir) / "splits"
    required = [args.env_nc, args.points, args.sed_rate_nc, args.repeated_split_dir, args.baseline_ensemble_dir, split_dir]
    missing = [str(p) for p in required if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required ablation inputs: {missing}")

    prior_manifest_path = output_dir / "manifest.json"
    prior_manifest = (
        json.loads(prior_manifest_path.read_text(encoding="utf-8"))
        if prior_manifest_path.exists()
        else {}
    )
    checkpoint_path = Path(args.baseline_run_dir) / "models" / "mechanism_aware_rey_regressor_best.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    feature_names = [str(name) for name in checkpoint.get("feature_names", [])]
    if not feature_names:
        raise ValueError(f"No feature_names found in baseline checkpoint: {checkpoint_path}")
    now = datetime.now().isoformat(timespec="seconds")

    manifest = {
        "suite": "continuous_rey_ofat_ablation_v1",
        "status": "running",
        "mode": args.mode,
        "publication_ready": args.mode == "publication",
        "created": prior_manifest.get("created", now),
        "last_started": now,
        "suite_command": prior_manifest.get(
            "suite_command",
            [sys.executable, "-B", "-m", "scripts.run_rey_ablation_suite", *sys.argv[1:]],
        ),
        "last_command": [sys.executable, "-B", "-m", "scripts.run_rey_ablation_suite", *sys.argv[1:]],
        "seeds": seeds,
        "split_seed": 310,
        "split_indices_dir": str(split_dir),
        "baseline_ensemble_dir": str(args.baseline_ensemble_dir),
        "baseline_checkpoint": str(checkpoint_path),
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "input_channels": len(feature_names) + 1,
        "land_mask_excluded": "land_mask" not in feature_names,
        "variants": VARIANTS,
        "random_forest": {"id": "B0_random_forest", "n_estimators": 500, "random_state": 310},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "git": _git_state(repo_root),
        "inputs": [_fingerprint(Path(p)) for p in (args.env_nc, args.points, args.sed_rate_nc)],
        "split_files": [_fingerprint(split_dir / f) for f in ("train_indices.npy", "val_indices.npy", "test_indices.npy", "split_assignments.csv")],
    }
    _write_json(output_dir / "manifest.json", manifest)

    models: list[dict[str, Any]] = [
        {"id": "A0_full_model", "label": "Full model", "kind": "reference", "type": "cnn", "path": str(args.baseline_ensemble_dir), "mechanism_valid": True, "threshold_head_valid": True}
    ]
    command_log: list[dict[str, Any]] = _successful_command_history(output_dir / "logs")
    total = len(VARIANTS) * len(seeds) + len(VARIANTS) + 1
    completed = 0

    if not args.tables_only:
        for variant in VARIANTS:
            run_dirs: list[Path] = []
            for seed in seeds:
                run_dir = output_dir / "runs" / variant["id"] / f"seed_{seed}"
                run_dirs.append(run_dir)
                summary_path = run_dir / "run_summary.json"
                if not (args.skip_existing and summary_path.exists()):
                    cmd = _baseline_train_cmd(
                        py,
                        env_nc=Path(args.env_nc),
                        points=Path(args.points),
                        sed_rate_nc=Path(args.sed_rate_nc),
                        output_dir=run_dir,
                        seed=seed,
                        split_seed=310,
                        epochs=epochs,
                        patience=patience,
                    )
                    cmd.extend(["--split-indices-dir", str(split_dir)])
                    for option, value in variant.get("options", {}).items():
                        _replace_option(cmd, option, str(value))
                    cmd.extend(variant.get("flags", []))
                    command_log.append(_run(cmd, cwd=repo_root, log_path=output_dir / "logs" / f"{variant['id']}_seed{seed}.log", dry_run=bool(args.dry_run)))
                completed += 1
                _emit("training", completed, total, variant=variant["id"], seed=seed)
            ensemble_dir = output_dir / "ensembles" / variant["id"]
            if not (args.skip_existing and (ensemble_dir / "ensemble_metrics.json").exists()):
                cmd = [py, "-B", "-m", "scripts.evaluate_rey_multitask_ensemble", "--run-dirs", *[str(p) for p in run_dirs], "--output-dir", str(ensemble_dir)]
                command_log.append(_run(cmd, cwd=repo_root, log_path=output_dir / "logs" / f"{variant['id']}_ensemble.log", dry_run=bool(args.dry_run)))
            completed += 1
            _emit("ensemble", completed, total, variant=variant["id"])
            models.append(
                {
                    "id": variant["id"], "label": variant["label"], "kind": variant["kind"], "type": "cnn", "path": str(ensemble_dir),
                    "mechanism_valid": bool(variant.get("mechanism_valid", True)),
                    "threshold_head_valid": bool(variant.get("threshold_head_valid", True)),
                }
            )

        rf_dir = output_dir / "runs" / "B0_random_forest"
        if not (args.skip_existing and (rf_dir / "run_summary.json").exists()):
            cmd = [
                py, "-B", "-m", "scripts.train_rey_random_forest_baseline",
                "--env-nc", str(args.env_nc), "--points", str(args.points), "--sed-rate-nc", str(args.sed_rate_nc),
                "--reference-run-dir", str(args.baseline_run_dir), "--split-indices-dir", str(split_dir), "--output-dir", str(rf_dir),
            ]
            command_log.append(_run(cmd, cwd=repo_root, log_path=output_dir / "logs" / "B0_random_forest.log", dry_run=bool(args.dry_run)))
        completed += 1
        _emit("random_forest", completed, total)
    else:
        for variant in VARIANTS:
            models.append(
                {
                    "id": variant["id"], "label": variant["label"], "kind": variant["kind"], "type": "cnn",
                    "path": str(output_dir / "ensembles" / variant["id"]),
                    "mechanism_valid": bool(variant.get("mechanism_valid", True)),
                    "threshold_head_valid": bool(variant.get("threshold_head_valid", True)),
                }
            )
    models.append({"id": "B0_random_forest", "label": "Random forest", "kind": "traditional_baseline", "type": "rf", "path": str(output_dir / "runs" / "B0_random_forest"), "mechanism_valid": False, "threshold_head_valid": False})

    if not args.dry_run:
        _build_tables(models, Path(args.repeated_split_dir), output_dir)
    _write_json(output_dir / "commands.json", _deduplicate_command_history(command_log))
    manifest.update({"status": "completed", "completed": datetime.now().isoformat(timespec="seconds"), "models": models})
    _write_json(output_dir / "manifest.json", manifest)
    _emit("completed", total, total)
    print(json.dumps({"status": "completed", "output_dir": str(output_dir), "fixed_table": str(output_dir / "tables" / "rey_ablation_fixed_test.csv")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

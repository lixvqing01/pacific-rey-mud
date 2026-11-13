from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from core.utils.serialization_utils import NumpyJSONEncoder


DEFAULT_ENV_NC = Path("data/env_gridded.nc")
DEFAULT_POINTS = Path("data/geochem_class_table.parquet")
DEFAULT_SED_RATE_NC = Path("data/Average sedimentation rate.nc")
DEFAULT_SEEDS = [310, 411, 512, 613, 714]
DEFAULT_SPLIT_SEED = 310
DEFAULT_RUN_PREFIX = "rey_continuous_multitask_midloss_patch11"
DEFAULT_ENSEMBLE_NAME = "rey_continuous_multitask_midloss_5seed_ensemble_split310_20260602"
DEFAULT_REPEATED_SPLITS = Path("splits/balanced_repeated")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, cls=NumpyJSONEncoder, indent=2, ensure_ascii=False)


def _python_executable(repo_root: Path) -> str:
    path_file = repo_root / "python_path.txt"
    if path_file.exists():
        value = path_file.read_text(encoding="utf-8").strip()
        if value:
            return value
    return sys.executable


def _run(cmd: list[str], *, cwd: Path, dry_run: bool = False) -> dict[str, Any]:
    started = datetime.now().isoformat(timespec="seconds")
    if dry_run:
        return {
            "cmd": cmd,
            "started": started,
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "dry_run": True,
        }
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    return {
        "cmd": cmd,
        "started": started,
        "finished": datetime.now().isoformat(timespec="seconds"),
        "returncode": int(proc.returncode),
        "stdout": proc.stdout[-12000:],
        "stderr": proc.stderr[-12000:],
        "dry_run": False,
    }


def _baseline_train_cmd(
    py: str,
    *,
    env_nc: Path,
    points: Path,
    sed_rate_nc: Path,
    output_dir: Path,
    seed: int,
    split_seed: int,
    epochs: int,
    patience: int,
) -> list[str]:
    return [
        py,
        "-B",
        "-m",
        "scripts.run_rey_continuous_multitask",
        "--env-nc",
        str(env_nc),
        "--points",
        str(points),
        "--sed-rate-nc",
        str(sed_rate_nc),
        "--output-dir",
        str(output_dir),
        "--depth-feature-type",
        "age_ka_proxy",
        "--group-col",
        "core_id",
        "--seed",
        str(int(seed)),
        "--split-seed",
        str(int(split_seed)),
        "--test-split-ratio",
        "0.15",
        "--validation-split-ratio",
        "0.1765",
        "--group-split-stratify-bins",
        "6",
        "--patch-size",
        "11",
        "--thresholds",
        "600,800,1000,1500",
        "--epochs",
        str(int(epochs)),
        "--patience",
        str(int(patience)),
        "--batch-size",
        "128",
        "--eval-batch-size",
        "256",
        "--lr",
        "0.001",
        "--weight-decay",
        "0.0001",
        "--grad-clip",
        "5.0",
        "--cnn-filters",
        "64,128,128",
        "--hidden-dim",
        "128",
        "--dropout",
        "0.25",
        "--regression-loss",
        "student_t",
        "--student-t-nu",
        "4.0",
        "--threshold-loss-weight",
        "0.4",
        "--mechanism-loss-weight",
        "0.24",
        "--mechanism-class-weighting",
        "sqrt",
        "--use-threshold-pos-weight",
        "--threshold-pos-weight-cap",
        "3.0",
        "--threshold-loss-multipliers",
        "0.8,1.0,1.2,1.4",
        "--threshold-focal-gamma",
        "0.35",
        "--high-rey-loss-threshold",
        "1000.0",
        "--high-rey-loss-weight",
        "1.2",
        "--high-rey-sampler-threshold",
        "1000.0",
        "--high-rey-sampler-weight",
        "1.1",
        "--rank-loss-weight",
        "0.08",
        "--rank-loss-margin",
        "0.25",
        "--selection-metric",
        "composite",
        "--selection-spearman-weight",
        "0.14",
        "--selection-threshold-f1-weight",
        "0.1",
        "--selection-mechanism-f1-weight",
        "0.03",
        "--selection-class1-recall-weight",
        "0.025",
        "--monotonic-loss-weight",
        "0.05",
        "--gate-balance-weight",
        "0.02",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed paper-reproduction workflow for the continuous REY baseline.")
    parser.add_argument("--env-nc", type=Path, default=DEFAULT_ENV_NC)
    parser.add_argument("--points", type=Path, default=DEFAULT_POINTS)
    parser.add_argument("--sed-rate-nc", type=Path, default=DEFAULT_SED_RATE_NC)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seeds", default="310,411,512,613,714")
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--skip-training-if-exists", action="store_true")
    parser.add_argument("--train", action="store_true", help="Train missing official seed runs.")
    parser.add_argument("--evaluate-ensemble", action="store_true", help="Build/evaluate the fixed 5-seed ensemble.")
    parser.add_argument("--evaluate-repeated", action="store_true", help="Run balanced repeated split evaluation when split artifacts exist.")
    parser.add_argument("--failure-audit", action="store_true", help="Run balanced repeated failure audit when split artifacts exist.")
    parser.add_argument("--run-ablation", action="store_true", help="Run the fixed OFAT ablation suite and RF baseline.")
    parser.add_argument("--ablation-mode", choices=["smoke", "publication"], default="publication")
    parser.add_argument("--ablation-output-dir", type=Path, default=None)
    parser.add_argument("--ablation-tables-only", action="store_true")
    parser.add_argument("--repeated-split-dir", type=Path, default=DEFAULT_REPEATED_SPLITS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path.cwd()
    py = _python_executable(repo_root)
    run_id = args.run_id or f"rey_paper_reproduction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(part.strip()) for part in str(args.seeds).split(",") if part.strip()]

    commands: list[dict[str, Any]] = []
    missing_inputs = [str(path) for path in (args.env_nc, args.points, args.sed_rate_nc) if not Path(path).exists()]
    if missing_inputs:
        summary = {
            "status": "failed",
            "run_id": run_id,
            "missing_inputs": missing_inputs,
            "output_dir": str(output_dir),
        }
        _write_json(output_dir / "paper_reproduction_summary.json", summary)
        raise FileNotFoundError(f"Missing required input files: {missing_inputs}")

    seed_run_dirs: list[Path] = []
    for seed in seeds:
        seed_dir = Path(args.output_root) / f"{DEFAULT_RUN_PREFIX}_seed{seed}_split{int(args.split_seed)}_20260602"
        seed_run_dirs.append(seed_dir)
        best_model = seed_dir / "models" / "mechanism_aware_rey_regressor_best.pth"
        if args.train and not (args.skip_training_if_exists and best_model.exists()):
            cmd = _baseline_train_cmd(
                py,
                env_nc=args.env_nc,
                points=args.points,
                sed_rate_nc=args.sed_rate_nc,
                output_dir=seed_dir,
                seed=seed,
                split_seed=int(args.split_seed),
                epochs=int(args.epochs),
                patience=int(args.patience),
            )
            result = _run(cmd, cwd=repo_root, dry_run=bool(args.dry_run))
            commands.append(result)
            if result["returncode"] != 0:
                _write_json(output_dir / "paper_reproduction_summary.json", {"status": "failed", "commands": commands})
                raise SystemExit(result["returncode"])

    ensemble_dir = Path(args.output_root) / DEFAULT_ENSEMBLE_NAME
    if args.evaluate_ensemble:
        cmd = [
            py,
            "-B",
            "-m",
            "scripts.evaluate_rey_multitask_ensemble",
            "--run-dirs",
            *[str(path) for path in seed_run_dirs],
            "--output-dir",
            str(ensemble_dir),
        ]
        result = _run(cmd, cwd=repo_root, dry_run=bool(args.dry_run))
        commands.append(result)
        if result["returncode"] != 0:
            _write_json(output_dir / "paper_reproduction_summary.json", {"status": "failed", "commands": commands})
            raise SystemExit(result["returncode"])

    repeated_eval_dir = Path(args.output_root) / "rey_continuous_multitask_midloss_5seed_balanced_repeated_eval_20260603"
    if args.evaluate_repeated and Path(args.repeated_split_dir).exists() and ensemble_dir.exists():
        cmd = [
            py,
            "-B",
            "-m",
            "scripts.evaluate_rey_repeated_core_splits",
            "--ensemble-dir",
            str(ensemble_dir),
            "--repeated-split-dir",
            str(args.repeated_split_dir),
            "--output-dir",
            str(repeated_eval_dir),
        ]
        result = _run(cmd, cwd=repo_root, dry_run=bool(args.dry_run))
        commands.append(result)
        if result["returncode"] != 0:
            _write_json(output_dir / "paper_reproduction_summary.json", {"status": "failed", "commands": commands})
            raise SystemExit(result["returncode"])

    audit_dir = Path(args.output_root) / "rey_continuous_multitask_midloss_5seed_balanced_failure_audit_20260603"
    split_assignments = seed_run_dirs[0] / "splits" / "split_assignments.csv"
    if args.failure_audit and Path(args.repeated_split_dir).exists() and ensemble_dir.exists():
        cmd = [
            py,
            "-B",
            "-m",
            "scripts.diagnose_rey_repeated_core_failures",
            "--ensemble-dir",
            str(ensemble_dir),
            "--repeated-split-dir",
            str(args.repeated_split_dir),
            "--geochem-table",
            str(args.points),
            "--cnn-split-assignments",
            str(split_assignments),
            "--output-dir",
            str(audit_dir),
        ]
        result = _run(cmd, cwd=repo_root, dry_run=bool(args.dry_run))
        commands.append(result)
        if result["returncode"] != 0:
            _write_json(output_dir / "paper_reproduction_summary.json", {"status": "failed", "commands": commands})
            raise SystemExit(result["returncode"])

    ablation_dir = args.ablation_output_dir or (Path(args.output_root) / "rey_continuous_ofat_ablation_v1_20260713")
    if args.run_ablation:
        cmd = [
            py,
            "-B",
            "-m",
            "scripts.run_rey_ablation_suite",
            "--env-nc",
            str(args.env_nc),
            "--points",
            str(args.points),
            "--sed-rate-nc",
            str(args.sed_rate_nc),
            "--repeated-split-dir",
            str(args.repeated_split_dir),
            "--output-dir",
            str(ablation_dir),
            "--mode",
            str(args.ablation_mode),
            "--skip-existing",
        ]
        if args.ablation_tables_only:
            cmd.append("--tables-only")
        result = _run(cmd, cwd=repo_root, dry_run=bool(args.dry_run))
        commands.append(result)
        if result["returncode"] != 0:
            _write_json(output_dir / "paper_reproduction_summary.json", {"status": "failed", "commands": commands})
            raise SystemExit(result["returncode"])

    summary = {
        "status": "completed",
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "fixed_workflow": {
            "env_nc": str(args.env_nc),
            "points": str(args.points),
            "sed_rate_nc": str(args.sed_rate_nc),
            "seeds": seeds,
            "split_seed": int(args.split_seed),
            "group_col": "core_id",
            "patch_size": 11,
            "thresholds": [600, 800, 1000, 1500],
        },
        "seed_run_dirs": [str(path) for path in seed_run_dirs],
        "ensemble_dir": str(ensemble_dir),
        "balanced_repeated_eval_dir": str(repeated_eval_dir),
        "failure_audit_dir": str(audit_dir),
        "ablation_dir": str(ablation_dir),
        "commands": commands,
    }
    _write_json(output_dir / "paper_reproduction_summary.json", summary)
    print(json.dumps(summary, cls=NumpyJSONEncoder, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import average_precision_score

from core.utils.serialization_utils import NumpyJSONEncoder
from scripts.run_rey_continuous_multitask import (
    _load_split_indices,
    _make_base_dataset,
    _prepare_points,
    _regression_metrics,
)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, cls=NumpyJSONEncoder, indent=2, ensure_ascii=False)


def _matrix(dataset) -> np.ndarray:
    rows: list[np.ndarray] = []
    for idx in range(len(dataset)):
        patch = dataset[idx][0]
        rows.append(patch[:, 0, 0].detach().cpu().numpy().astype(np.float32, copy=False))
    return np.asarray(rows, dtype=np.float32)


def _metrics(df: pd.DataFrame, thresholds: list[float]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "regression": _regression_metrics(df["rey_true"].to_numpy(), df["rey_pred"].to_numpy()),
        "grade_score_thresholds": {},
    }
    score = df["log_rey_pred"].to_numpy(dtype=float)
    for threshold in thresholds:
        y = (df["rey_true"].to_numpy(dtype=float) >= threshold).astype(int)
        out["grade_score_thresholds"][f"rey_ge_{int(threshold)}_ap"] = (
            float(average_precision_score(y, score)) if np.unique(y).size > 1 else float("nan")
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a center-pixel environment + age-proxy RandomForestRegressor baseline for continuous REY."
    )
    parser.add_argument("--env-nc", required=True, type=Path)
    parser.add_argument("--points", required=True, type=Path)
    parser.add_argument("--sed-rate-nc", required=True, type=Path)
    parser.add_argument("--reference-run-dir", required=True, type=Path)
    parser.add_argument("--split-indices-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--random-state", type=int, default=310)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--thresholds", default="600,800,1000,1500")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.reference_run_dir) / "models" / "mechanism_aware_rey_regressor_best.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    feature_names = [str(x) for x in checkpoint.get("feature_names", [])]
    norm_params = dict(checkpoint.get("norm_params", {}) or {})
    if not feature_names:
        raise ValueError(f"No feature_names found in {checkpoint_path}")

    prepared_points = _prepare_points(Path(args.points), output_dir)
    dataset = _make_base_dataset(
        points_path=prepared_points,
        env_nc=Path(args.env_nc),
        feature_names=feature_names,
        patch_size=1,
        sed_rate_nc=Path(args.sed_rate_nc),
        depth_feature_type="age_ka_proxy",
        target_transform_name="log1p",
        target_transform_scale=1000.0,
        target_transform_hinge_beta=0.25,
        norm_params_override=norm_params,
        use_target_weights=False,
        max_weight=5.0,
        augment=False,
        shift_pixels=0,
        add_coord_channels=False,
    )
    train_idx, val_idx, test_idx = _load_split_indices(Path(args.split_indices_dir), len(dataset))
    x = _matrix(dataset)
    rey = dataset.rey_data["rey"].astype(float).to_numpy()
    y = np.log1p(np.clip(rey, 0.0, None))

    model = RandomForestRegressor(
        n_estimators=int(args.n_estimators),
        random_state=int(args.random_state),
        n_jobs=int(args.n_jobs),
    )
    model.fit(x[np.asarray(train_idx, dtype=int)], y[np.asarray(train_idx, dtype=int)])
    joblib.dump(model, output_dir / "random_forest_regressor.joblib")

    thresholds = [float(x) for x in str(args.thresholds).split(",") if x.strip()]
    split_metrics: dict[str, Any] = {}
    table_dir = output_dir / "tables"
    for split_name, indices in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        idx = np.asarray(indices, dtype=int)
        pred_log = np.asarray(model.predict(x[idx]), dtype=float)
        pred_rey = np.clip(np.expm1(pred_log), 0.0, None)
        meta = dataset.rey_data.iloc[idx]
        frame = pd.DataFrame(
            {
                "dataset_index": idx,
                "core_id": meta["core_id"].astype(str).to_numpy(),
                "site_group": meta.get("site_group", pd.Series("", index=meta.index)).astype(str).to_numpy(),
                "rey_true": rey[idx],
                "rey_pred": pred_rey,
                "log_rey_pred": pred_log,
            }
        )
        split_out = table_dir / split_name
        split_out.mkdir(parents=True, exist_ok=True)
        frame.to_csv(split_out / "random_forest_rey_predictions.csv", index=False)
        split_metrics[split_name] = _metrics(frame, thresholds)
        _write_json(split_out / "metrics.json", split_metrics[split_name])

    summary = {
        "status": "completed",
        "model_id": "B0_random_forest",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "input_definition": "42 normalized center-pixel environment features + normalized age_ka_proxy",
        "target": "log1p(REY)",
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "input_columns": [*feature_names, "age_ka_proxy"],
        "split_indices_dir": str(args.split_indices_dir),
        "reference_checkpoint": str(checkpoint_path),
        "model_config": {
            "class": "sklearn.ensemble.RandomForestRegressor",
            "n_estimators": int(args.n_estimators),
            "random_state": int(args.random_state),
            "n_jobs": int(args.n_jobs),
            "sklearn_version": sklearn.__version__,
        },
        "metrics": split_metrics,
    }
    _write_json(output_dir / "run_summary.json", summary)
    print(json.dumps(summary, cls=NumpyJSONEncoder, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

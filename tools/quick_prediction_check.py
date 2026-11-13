from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.prediction.pipeline_multiclass import run_prediction_pipeline_multiclass
from core.utils.config_models import (
    GeospatialBounds,
    PlottingConfig,
    PredictionConfig,
    PredictionGridSettings,
)


def main() -> int:
    p = argparse.ArgumentParser(description="Quick prediction check (small-area) for multiclass 2D summary maps.")
    p.add_argument(
        "--training-artifacts",
        default="outputs/training_artifacts.json",
        help="Path to training_artifacts.json",
    )
    p.add_argument(
        "--env-nc",
        default="data/env_gridded.nc",
        help="Path to env_gridded.nc",
    )
    p.add_argument("--min-lon", type=float, default=120.0)
    p.add_argument("--max-lon", type=float, default=160.0)
    p.add_argument("--min-lat", type=float, default=-10.0)
    p.add_argument("--max-lat", type=float, default=10.0)
    args = p.parse_args()

    training_artifacts_path = str(args.training_artifacts)
    env_nc = str(args.env_nc)

    bounds = GeospatialBounds(min_lon=args.min_lon, max_lon=args.max_lon, min_lat=args.min_lat, max_lat=args.max_lat)
    grid = PredictionGridSettings(
        geospatial_bounds=bounds,
        lon_res=2.0,
        lat_res=2.0,
        vertical_axis_mode="age_ka",
        age_levels_ka=[0.0, 100.0, 1000.0],
        depth_levels=[],
    )
    plot = PlottingConfig(figsize_pred_map=(6, 4), pred_map_qmin=2, pred_map_qmax=98, dpi_pred_map=120)

    conf = PredictionConfig(
        training_artifacts_path=training_artifacts_path,
        env_gridded_nc_path=env_nc,
        batch_size=128,
        prediction_grid_settings=grid,
        plotting=plot,
        output_mode="summaries_2d",
        enable_2d_age_summaries=True,
        enable_multiclass_target_threshold_maps=True,
        multiclass_target_class_index=1,
        multiclass_target_threshold_source="manual",
        multiclass_target_threshold=0.2,
        multiclass_target_include_freq=True,
        multiclass_target_include_any=True,
        summary_sampling_scheme="linear",
        summary_n_samples_holocene=3,
        summary_n_samples_lgc=3,
        summary_n_samples_quaternary=3,
    )

    ok, summary = run_prediction_pipeline_multiclass(pred_config_obj=conf, progress_callback=None)
    print("ok", bool(ok))
    print("run_id", summary.get("run_id"))
    print("predicted_summary_2d_nc_path", (summary.get("results", {}) or {}).get("predicted_summary_2d_nc_path"))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from core.datasets.rey_dataset import REYPointDataset
from core.utils.logger_config import logger
from core.visualization.publication_style import publication_style, save_figure


def _get_env_channel_indices(
    feature_names: Sequence[str],
    target_features: Sequence[str],
) -> Dict[str, int]:
    """
    Map selected environmental feature names to their channel indices.
    Only features that are present in `feature_names` are returned.
    """
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    result: Dict[str, int] = {}
    for feat in target_features:
        if feat in name_to_idx:
            result[feat] = name_to_idx[feat]
        else:
            logger.warning(
                f"PeakProfileAnalyzer: Feature '{feat}' not found in dataset.feature_names; "
                "skipping it in peak-profile analysis."
            )
    return result


def _env_values_at_location(
    dataset: REYPointDataset,
    lon: float,
    lat: float,
    channel_indices: Dict[str, int],
) -> Dict[str, float]:
    """
    Sample environmental variables at the nearest grid point to (lon, lat).
    Returns a dict mapping feature name -> raw (physical) value.
    """
    env_lat = dataset.env_lat
    env_lon = dataset.env_lon
    lat_res = dataset.lat_res if getattr(dataset, "lat_res", 0) else 0.0
    lon_res = dataset.lon_res if getattr(dataset, "lon_res", 0) else 0.0


    lat_idx = np.searchsorted(env_lat, lat - lat_res / 2.0)
    lon_idx = np.searchsorted(env_lon, lon - lon_res / 2.0)
    lat_idx = int(np.clip(lat_idx, 0, len(env_lat) - 1))
    lon_idx = int(np.clip(lon_idx, 0, len(env_lon) - 1))

    values: Dict[str, float] = {}
    for name, ch_idx in channel_indices.items():
        try:
            val = float(dataset.env_data_np[ch_idx, lat_idx, lon_idx])
        except Exception as e:
            logger.debug(
                f"PeakProfileAnalyzer: Failed to sample '{name}' "
                f"at (lat_idx={lat_idx}, lon_idx={lon_idx}): {e}"
            )
            val = np.nan
        values[name] = val

    return values


def analyze_peak_profiles(
    model: torch.nn.Module,
    dataset: REYPointDataset,
    test_indices: Sequence[int],
    device: torch.device,
    output_dir: str,
    max_depth_m: float = 15.0,
) -> Dict[str, object]:
    """
    Analyze, for each sediment core (approximated by unique lon/lat),
    the peak REY value and its depth within 0–max_depth_m, and relate
    these peaks to key environmental variables.

    The analysis operates on the *test set* only to avoid bias.

    Outputs:
      - CSV summary of per-core peak statistics
      - Scatter plots of:
          * peak depth vs selected env variables (colored by peak REY)
          * peak REY vs selected env variables
    """
    results: Dict[str, object] = {}

    if not isinstance(dataset, REYPointDataset):
        logger.warning(
            "PeakProfileAnalyzer: Dataset is not REYPointDataset; "
            "skipping peak-profile analysis."
        )
        return results

    if not hasattr(dataset, "rey_data") or "depth_physical" not in dataset.rey_data.columns:
        logger.warning(
            "PeakProfileAnalyzer: Dataset lacks 'depth_physical'; "
            "skipping peak-profile analysis."
        )
        return results

    if not hasattr(dataset, "env_data_np") or not hasattr(dataset, "feature_names"):
        logger.warning(
            "PeakProfileAnalyzer: Dataset does not provide env_data_np/feature_names; "
            "cannot relate peaks to environment."
        )
        return results

    model.eval()
    out_dir = Path(output_dir) / "peak_profiles"
    out_dir.mkdir(parents=True, exist_ok=True)


    target_env_features: List[str] = [
        "Average sedimentation rate",
        "Water depth",
        "Bottom water dissolved oxygen",
        "Summer primary productivity",
    ]
    env_channel_indices = _get_env_channel_indices(dataset.feature_names, target_env_features)
    if not env_channel_indices:
        logger.warning(
            "PeakProfileAnalyzer: None of the requested environmental features were "
            "found in dataset.feature_names; peak-env plots will not be generated."
        )


    samples: List[Dict[str, float]] = []
    with torch.no_grad():
        for idx in test_indices:
            try:
                combined_patch, target_rey_norm, _ = dataset[idx]
            except Exception as e_fetch:
                logger.debug(f"PeakProfileAnalyzer: failed to fetch sample {idx}: {e_fetch}")
                continue

            row = dataset.rey_data.iloc[idx]
            depth_phys = float(row.get("depth_physical", row.get("depth", 0.0)))
            if depth_phys < 0.0 or depth_phys > max_depth_m:

                continue

            lon = float(row["lon"])
            lat = float(row["lat"])

            combined_patch = combined_patch.unsqueeze(0).to(device)
            target_rey_norm = target_rey_norm.view(1, -1).to(device)

            with torch.amp.autocast(
                device_type=device.type, enabled=torch.cuda.is_available()
            ):
                pred_norm = model(combined_patch)

            pred_norm_np = pred_norm.detach().cpu().numpy().squeeze()
            true_norm_np = target_rey_norm.detach().cpu().numpy().squeeze()


            pred_phys = float(dataset._denormalize_rey(np.array([pred_norm_np]))[0])
            true_phys = float(dataset._denormalize_rey(np.array([true_norm_np]))[0])

            samples.append(
                {
                    "lon": lon,
                    "lat": lat,
                    "depth_m": depth_phys,
                    "pred_rey": pred_phys,
                    "true_rey": true_phys,
                }
            )

    if not samples:
        logger.warning(
            "PeakProfileAnalyzer: No valid test samples collected for peak analysis."
        )
        return results

    samples_df = pd.DataFrame(samples)


    samples_df["lon_round"] = samples_df["lon"].round(4)
    samples_df["lat_round"] = samples_df["lat"].round(4)

    peak_records: List[Dict[str, float]] = []
    grouped = samples_df.groupby(["lon_round", "lat_round"], sort=False)

    for (lon_r, lat_r), group in grouped:
        if group.empty:
            continue


        idx_peak_pred = group["pred_rey"].idxmax()
        row_peak = group.loc[idx_peak_pred]

        peak_depth_m = float(row_peak["depth_m"])
        peak_pred_rey = float(row_peak["pred_rey"])
        peak_true_rey = float(group["true_rey"].max())

        env_vals = (
            _env_values_at_location(
                dataset=dataset,
                lon=float(row_peak["lon"]),
                lat=float(row_peak["lat"]),
                channel_indices=env_channel_indices,
            )
            if env_channel_indices
            else {}
        )

        record: Dict[str, float] = {
            "lon": float(row_peak["lon"]),
            "lat": float(row_peak["lat"]),
            "peak_depth_m": peak_depth_m,
            "peak_pred_rey": peak_pred_rey,
            "peak_true_rey": peak_true_rey,
            "num_samples_in_core": float(len(group)),
        }
        record.update(env_vals)
        peak_records.append(record)

    if not peak_records:
        logger.warning(
            "PeakProfileAnalyzer: No per-core peaks could be computed; "
            "skipping peak-profile plots."
        )
        return results

    peak_df = pd.DataFrame(peak_records)
    csv_path = out_dir / "peak_profiles_summary.csv"
    peak_df.to_csv(csv_path, index=False)
    results["peak_profiles_csv"] = str(csv_path)
    logger.info(f"PeakProfileAnalyzer: Saved peak-profile summary to {csv_path}")


    plot_paths: Dict[str, str] = {}

    def _scatter_env_vs_peak(
        env_name: str,
        x_label: str,
        y: str,
        y_label: str,
        fname: str,
        invert_y: bool = False,
    ) -> Optional[str]:
        if env_name not in peak_df.columns:
            return None
        series_env = peak_df[env_name].to_numpy()
        series_y = peak_df[y].to_numpy()

        mask = np.isfinite(series_env) & np.isfinite(series_y)
        if not np.any(mask):
            return None

        x = series_env[mask]
        yy = series_y[mask]
        cvals = peak_df.loc[mask, "peak_pred_rey"].to_numpy()

        with publication_style():
            fig, ax = plt.subplots(figsize=(7.4, 5.4))
            sc = ax.scatter(
                x,
                yy,
                c=cvals,
                cmap="viridis",
                s=28,
                alpha=0.85,
                edgecolors="none",
            )
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label("Predicted peak ΣREY (ppm)")

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            if invert_y:
                ax.invert_yaxis()
            ax.grid(True, linestyle="--", alpha=0.35)
            fig.tight_layout()

            out_path = out_dir / fname
            save_paths = save_figure(
                fig,
                path_no_ext=out_path.with_suffix(""),
                plot_config=None,
                formats=["png", "pdf"],
                dpi=600,
            )
            logger.info(f"PeakProfileAnalyzer: Saved plot to {save_paths}")
            plt.close(fig)
            return next((p for p in save_paths if p.lower().endswith(".png")), save_paths[0])


    depth_y = "peak_depth_m"
    for env_name, x_label in [
        ("Average sedimentation rate", "Sedimentation rate (cm/ka)"),
        ("Water depth", "Water depth (m)"),
        ("Bottom water dissolved oxygen", "Bottom water dissolved oxygen"),
        ("Summer primary productivity", "Summer primary productivity"),
    ]:
        path = _scatter_env_vs_peak(
            env_name=env_name,
            x_label=x_label,
            y=depth_y,
            y_label="Peak depth (m bsf)",
            fname=f"peak_depth_vs_{env_name.replace(' ', '_')}.png",
            invert_y=True,
        )
        if path:
            plot_paths[f"peak_depth_vs_{env_name}"] = path


    rey_y = "peak_pred_rey"
    for env_name, x_label in [
        ("Average sedimentation rate", "Sedimentation rate (cm/ka)"),
        ("Water depth", "Water depth (m)"),
    ]:
        path = _scatter_env_vs_peak(
            env_name=env_name,
            x_label=x_label,
            y=rey_y,
            y_label="Predicted peak REY (physical)",
            fname=f"peak_rey_vs_{env_name.replace(' ', '_')}.png",
            invert_y=False,
        )
        if path:
            plot_paths[f"peak_rey_vs_{env_name}"] = path


    try:
        depths = peak_df["peak_depth_m"].to_numpy()
        depths = depths[np.isfinite(depths)]
        if depths.size > 0:
            with publication_style():
                fig, ax = plt.subplots(figsize=(7.4, 4.2))
                ax.hist(depths, bins=15, color="steelblue", edgecolor="black", alpha=0.85)
                ax.set_xlabel("Peak depth (m bsf)")
                ax.set_ylabel("Number of cores")
                ax.set_title("Peak depth distribution (test set cores)")
                ax.grid(True, linestyle="--", alpha=0.35)
                fig.tight_layout()

                hist_path = out_dir / "peak_depth_histogram.png"
                save_paths = save_figure(
                    fig,
                    path_no_ext=hist_path.with_suffix(""),
                    plot_config=None,
                    formats=["png", "pdf"],
                    dpi=600,
                )
                plot_paths["peak_depth_histogram"] = next(
                    (p for p in save_paths if p.lower().endswith(".png")), save_paths[0]
                )
                logger.info(f"PeakProfileAnalyzer: Saved peak depth histogram to {save_paths}")
                plt.close(fig)
    except Exception as e_hist:
        logger.warning(f"PeakProfileAnalyzer: Failed to generate peak depth histogram: {e_hist}")

    results["peak_profile_plot_paths"] = plot_paths

    return results


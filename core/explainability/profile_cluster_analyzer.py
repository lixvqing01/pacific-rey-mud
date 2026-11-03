import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from core.datasets.rey_dataset import REYPointDataset
from core.utils.logger_config import logger
from core.visualization.publication_style import publication_style, save_figure


def _build_core_profiles_from_df(
    df: pd.DataFrame,
    depth_col: str,
    value_col: str,
    depth_grid: np.ndarray,
    min_points_per_core: int = 3,
) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """
    Build core-level vertical profiles on a common depth grid by linear interpolation.

    Returns:
      profiles: ndarray of shape [num_cores, num_depth_points]
      core_keys: list of (lon_round, lat_round) identifying each core
    """
    if df.empty:
        return np.empty((0, depth_grid.size), dtype=float), []

    df = df.copy()
    df = df[np.isfinite(df[depth_col]) & np.isfinite(df[value_col])]
    if df.empty:
        return np.empty((0, depth_grid.size), dtype=float), []

    df["lon_round"] = df["lon"].round(4)
    df["lat_round"] = df["lat"].round(4)

    profiles: List[np.ndarray] = []
    core_keys: List[Tuple[float, float]] = []

    grouped = df.groupby(["lon_round", "lat_round"], sort=False)
    for (lon_r, lat_r), group in grouped:
        group = group.sort_values(depth_col)
        if group.shape[0] < min_points_per_core:
            continue

        depths = group[depth_col].to_numpy(dtype=float)
        values = group[value_col].to_numpy(dtype=float)


        d_min = max(depth_grid.min(), depths.min())
        d_max = min(depth_grid.max(), depths.max())
        if d_max <= d_min:
            continue

        mask_grid = (depth_grid >= d_min) & (depth_grid <= d_max)
        if not np.any(mask_grid):
            continue

        try:
            interp_vals = np.interp(depth_grid[mask_grid], depths, values)
        except Exception as e_interp:
            logger.debug(f"ProfileClusterAnalyzer: interpolation failed for core {(lon_r, lat_r)}: {e_interp}")
            continue

        profile = np.full_like(depth_grid, np.nan, dtype=float)
        profile[mask_grid] = interp_vals


        if np.count_nonzero(np.isfinite(profile)) < 0.7 * depth_grid.size:
            continue


        finite_mask = np.isfinite(profile)
        if not np.all(finite_mask):
            if np.any(finite_mask):
                first_valid = np.argmax(finite_mask)
                last_valid = len(profile) - 1 - np.argmax(finite_mask[::-1])
                profile[:first_valid] = profile[first_valid]
                profile[last_valid + 1 :] = profile[last_valid]
                for i in range(first_valid + 1, last_valid + 1):
                    if not finite_mask[i]:
                        profile[i] = profile[i - 1]
            else:
                continue

        profiles.append(profile)
        core_keys.append((float(lon_r), float(lat_r)))

    if not profiles:
        return np.empty((0, depth_grid.size), dtype=float), []

    return np.vstack(profiles), core_keys


def _summarize_profiles_iqr(profiles: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Robust summary for profiles shaped [n_profiles, n_depth_points].

    Returns (median, p25, p75) along the profile axis.
    """
    if profiles.size == 0:
        return np.array([]), np.array([]), np.array([])
    med = np.nanmedian(profiles, axis=0)
    p25 = np.nanpercentile(profiles, 25, axis=0)
    p75 = np.nanpercentile(profiles, 75, axis=0)
    return med, p25, p75


def analyze_profile_clusters(
    model: torch.nn.Module,
    dataset: REYPointDataset,
    train_indices: Sequence[int],
    test_indices: Sequence[int],
    device: torch.device,
    output_dir: str,
    max_depth_m: float = 15.0,
    num_depth_points: int = 40,
    num_clusters: int = 3,
    max_samples_per_split: int = 4000,
    batch_size: int = 256,
) -> Dict[str, object]:
    """
    Cluster vertical REY profiles and compare:
      - Original observed profiles (true REY)
      - Predicted profiles on the TRAIN set
      - Predicted profiles on the TEST set

    Procedure:
      1) Build core-level true profiles from dataset.rey_data on the model's depth-feature axis.
      2) Fit KMeans on standardized true profiles.
      3) Build predicted profiles for train/test samples at the same cores,
         assign them to the clusters learned from true profiles.
      4) Plot robust (median ± IQR) profiles for true / train-pred / test-pred.
    """
    results: Dict[str, object] = {}

    if not isinstance(dataset, REYPointDataset):
        logger.warning("ProfileClusterAnalyzer: Dataset is not REYPointDataset; skipping.")
        return results

    if not hasattr(dataset, "rey_data") or "rey" not in dataset.rey_data.columns:
        logger.warning("ProfileClusterAnalyzer: Dataset lacks 'rey' column; skipping.")
        return results


    use_age_axis = bool(getattr(dataset, "depth_feature_type", "") == "age_ka_proxy") and (
        "age_ka_proxy" in dataset.rey_data.columns
    )
    axis_col = (
        "age_ka_proxy"
        if use_age_axis
        else ("depth_physical" if "depth_physical" in dataset.rey_data.columns else "depth")
    )
    if axis_col not in dataset.rey_data.columns:
        logger.warning(f"ProfileClusterAnalyzer: Dataset lacks axis column '{axis_col}'; skipping.")
        return results

    if use_age_axis:
        axis_label = "Age proxy (ka)"
        axis_max_value = getattr(dataset, "max_age_ka_proxy", None)
        if axis_max_value is None:
            vals = dataset.rey_data["age_ka_proxy"].astype(float).to_numpy()
            vals = vals[np.isfinite(vals)]
            axis_max_value = float(np.nanpercentile(vals, 99)) if vals.size else 2580.0
        axis_max_value = float(axis_max_value)
    else:
        axis_label = "Depth (m bsf)"
        axis_max_value = float(max_depth_m)

    results["profile_axis"] = axis_col
    results["profile_axis_max"] = float(axis_max_value)

    model.eval()
    out_dir = Path(output_dir) / "profile_clusters"
    out_dir.mkdir(parents=True, exist_ok=True)


    depth_grid = np.linspace(0.0, float(axis_max_value), num_depth_points).astype(float)

    rey_df = dataset.rey_data.copy()
    rey_df = rey_df[
        np.isfinite(rey_df[axis_col])
        & np.isfinite(rey_df["rey"])
        & (rey_df[axis_col] >= 0.0)
        & (rey_df[axis_col] <= axis_max_value)
    ]
    if rey_df.empty:
        logger.warning("ProfileClusterAnalyzer: No valid REY samples in 0–max_depth window; skipping.")
        return results

    true_profiles, core_keys_true = _build_core_profiles_from_df(
        df=rey_df,
        depth_col=axis_col,
        value_col="rey",
        depth_grid=depth_grid,
        min_points_per_core=3,
    )
    if true_profiles.size == 0:
        logger.warning("ProfileClusterAnalyzer: Failed to construct any true core profiles; skipping.")
        return results

    n_cores_true = int(true_profiles.shape[0])
    results["num_cores_true_profiles"] = n_cores_true


    min_cores_per_cluster = 3
    min_cores_for_clustering = int(max(9, min_cores_per_cluster * num_clusters))

    do_clustering = bool(n_cores_true >= min_cores_for_clustering)
    scaler: Optional[StandardScaler] = None
    kmeans: Optional[KMeans] = None
    true_labels: Optional[np.ndarray] = None

    if do_clustering:
        k_eff = min(int(num_clusters), max(2, int(np.floor(n_cores_true / max(2, min_cores_per_cluster)))))
        try:
            scaler = StandardScaler()
            true_profiles_scaled = scaler.fit_transform(true_profiles)
            kmeans = KMeans(n_clusters=k_eff, random_state=42, n_init="auto")
            true_labels = kmeans.fit_predict(true_profiles_scaled)
            results["num_profile_clusters"] = int(k_eff)
        except Exception as e_km:
            logger.warning(f"ProfileClusterAnalyzer: KMeans clustering failed; using robust summary instead: {e_km}")
            do_clustering = False
            scaler = None
            kmeans = None
            true_labels = None


    def _collect_predictions(indices: Sequence[int]) -> pd.DataFrame:
        indices_list = list(indices)
        if max_samples_per_split is not None and len(indices_list) > int(max_samples_per_split):
            rng = np.random.default_rng(0)
            indices_list = rng.choice(indices_list, size=int(max_samples_per_split), replace=False).tolist()


        try:
            meta_df = dataset.rey_data.iloc[indices_list]
            axis_arr = meta_df[axis_col].to_numpy(dtype=float)
            keep = np.isfinite(axis_arr) & (axis_arr >= 0.0) & (axis_arr <= axis_max_value)
            indices_list = [idx for idx, k in zip(indices_list, keep) if k]
        except Exception:
            pass

        rows: List[Dict[str, float]] = []
        patches: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []
        metas: List[Tuple[float, float, float]] = []

        def _flush_batch() -> None:
            if not patches:
                return

            batch_x = torch.stack(patches, dim=0).to(device)
            batch_y = torch.stack(targets, dim=0).to(device).view(-1, 1)

            with torch.inference_mode(), torch.amp.autocast(
                device_type=device.type,
                enabled=torch.cuda.is_available(),
            ):
                pred_norm = model(batch_x)

            pred_norm_np = pred_norm.detach().cpu().numpy().reshape(-1)
            true_norm_np = batch_y.detach().cpu().numpy().reshape(-1)

            pred_phys_np = dataset._denormalize_rey(pred_norm_np)
            true_phys_np = dataset._denormalize_rey(true_norm_np)

            for (lon, lat, depth_m), pred_phys, true_phys in zip(
                metas, pred_phys_np, true_phys_np
            ):
                rows.append(
                    {
                        "lon": float(lon),
                        "lat": float(lat),
                        # Column name kept for backward compatibility; values represent axis_col
                        "depth_m": float(depth_m),
                        "pred_rey": float(pred_phys),
                        "true_rey": float(true_phys),
                    }
                )

            patches.clear()
            targets.clear()
            metas.clear()

        for idx in indices_list:
            try:
                combined_patch, target_rey_norm, _ = dataset[idx]
            except Exception as e_fetch:
                logger.debug(f"ProfileClusterAnalyzer: failed to fetch sample {idx}: {e_fetch}")
                continue

            row = dataset.rey_data.iloc[idx]
            lon = float(row.get("lon", row.get("longitude", np.nan)))
            lat = float(row.get("lat", row.get("latitude", np.nan)))
            depth_m = float(row.get(axis_col, np.nan))
            if not np.isfinite(lon) or not np.isfinite(lat) or not np.isfinite(depth_m):
                continue

            patches.append(combined_patch)
            targets.append(target_rey_norm.view(1))
            metas.append((lon, lat, depth_m))

            if len(patches) >= int(batch_size):
                _flush_batch()

        _flush_batch()
        return pd.DataFrame(rows)

    train_df = _collect_predictions(train_indices)
    test_df = _collect_predictions(test_indices)

    def _build_pred_profiles(pred_df: pd.DataFrame) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        if pred_df.empty:
            return np.empty((0, depth_grid.size), dtype=float), []
        pred_df = pred_df[
            np.isfinite(pred_df["depth_m"])
            & np.isfinite(pred_df["pred_rey"])
            & (pred_df["depth_m"] >= 0.0)
            & (pred_df["depth_m"] <= axis_max_value)
        ]
        if pred_df.empty:
            return np.empty((0, depth_grid.size), dtype=float), []

        pred_df = pred_df.copy()
        pred_df["lon_round"] = pred_df["lon"].round(4)
        pred_df["lat_round"] = pred_df["lat"].round(4)

        profiles: List[np.ndarray] = []
        keys: List[Tuple[float, float]] = []
        grouped = pred_df.groupby(["lon_round", "lat_round"], sort=False)
        for (lon_r, lat_r), group in grouped:
            group = group.sort_values("depth_m")
            if group.shape[0] < 3:
                continue
            depths = group["depth_m"].to_numpy(dtype=float)
            values = group["pred_rey"].to_numpy(dtype=float)
            d_min = max(depth_grid.min(), depths.min())
            d_max = min(depth_grid.max(), depths.max())
            if d_max <= d_min:
                continue
            mask_grid = (depth_grid >= d_min) & (depth_grid <= d_max)
            if not np.any(mask_grid):
                continue
            try:
                interp_vals = np.interp(depth_grid[mask_grid], depths, values)
            except Exception as e_interp:
                logger.debug(f"ProfileClusterAnalyzer: interpolation failed for predicted core {(lon_r, lat_r)}: {e_interp}")
                continue
            profile = np.full_like(depth_grid, np.nan, dtype=float)
            profile[mask_grid] = interp_vals
            finite_mask = np.isfinite(profile)
            if np.count_nonzero(finite_mask) < 0.7 * depth_grid.size:
                continue
            if not np.all(finite_mask):
                if np.any(finite_mask):
                    first_valid = np.argmax(finite_mask)
                    last_valid = len(profile) - 1 - np.argmax(finite_mask[::-1])
                    profile[:first_valid] = profile[first_valid]
                    profile[last_valid + 1 :] = profile[last_valid]
                    for i in range(first_valid + 1, last_valid + 1):
                        if not finite_mask[i]:
                            profile[i] = profile[i - 1]
                else:
                    continue
            profiles.append(profile)
            keys.append((float(lon_r), float(lat_r)))
        if not profiles:
            return np.empty((0, depth_grid.size), dtype=float), []
        return np.vstack(profiles), keys

    train_profiles, train_keys = _build_pred_profiles(train_df)
    test_profiles, test_keys = _build_pred_profiles(test_df)


    def _assign_to_clusters(pred_profiles: np.ndarray) -> Optional[np.ndarray]:
        if (not do_clustering) or pred_profiles.size == 0 or scaler is None or kmeans is None:
            return None
        try:
            pred_scaled = scaler.transform(pred_profiles)
            return kmeans.predict(pred_scaled)
        except Exception as e_assign:
            logger.warning(f"ProfileClusterAnalyzer: Failed to assign profiles to clusters: {e_assign}")
            return None

    train_labels = _assign_to_clusters(train_profiles)
    test_labels = _assign_to_clusters(test_profiles)


    try:
        with publication_style():
            if do_clustering and kmeans is not None and true_labels is not None:
                num_clusters_actual = int(kmeans.n_clusters)
                train_counts = (
                    [int(np.count_nonzero(train_labels == k)) for k in range(num_clusters_actual)]
                    if train_labels is not None
                    else [0 for _ in range(num_clusters_actual)]
                )
                test_counts = (
                    [int(np.count_nonzero(test_labels == k)) for k in range(num_clusters_actual)]
                    if test_labels is not None
                    else [0 for _ in range(num_clusters_actual)]
                )
                fig, axes = plt.subplots(
                    nrows=1,
                    ncols=num_clusters_actual,
                    figsize=(3.6 * num_clusters_actual, 5.6),
                    sharey=True,
                )
                if num_clusters_actual == 1:
                    axes = [axes]

                for k in range(num_clusters_actual):
                    ax = axes[k]
                    mask_true = true_labels == k
                    if not np.any(mask_true):
                        continue

                    n_true = int(np.count_nonzero(mask_true))
                    n_train_total = int(train_counts[k]) if k < len(train_counts) else 0
                    n_test_total = int(test_counts[k]) if k < len(test_counts) else 0
                    med_true, p25_true, p75_true = _summarize_profiles_iqr(true_profiles[mask_true])
                    ax.plot(med_true, depth_grid, label=f"True median (n={n_true})", color="black", linewidth=2.0)
                    ax.fill_betweenx(depth_grid, p25_true, p75_true, color="black", alpha=0.12, linewidth=0.0)

                    if train_labels is not None and train_profiles.size > 0:
                        mask_train = train_labels == k
                        if np.any(mask_train):
                            n_train = int(np.count_nonzero(mask_train))
                            med_train, p25_train, p75_train = _summarize_profiles_iqr(train_profiles[mask_train])
                            ax.plot(
                                med_train,
                                depth_grid,
                                label=f"Train pred median (n={n_train})",
                                color="tab:blue",
                                linestyle="--",
                            )
                            ax.fill_betweenx(depth_grid, p25_train, p75_train, color="tab:blue", alpha=0.10, linewidth=0.0)
                        else:
                            ax.text(
                                0.02,
                                0.02,
                                "No train cores",
                                transform=ax.transAxes,
                                fontsize=8,
                                color="tab:blue",
                                va="bottom",
                                ha="left",
                            )

                    if test_labels is not None and test_profiles.size > 0:
                        mask_test = test_labels == k
                        if np.any(mask_test):
                            n_test = int(np.count_nonzero(mask_test))
                            med_test, p25_test, p75_test = _summarize_profiles_iqr(test_profiles[mask_test])
                            ax.plot(
                                med_test,
                                depth_grid,
                                label=f"Test pred median (n={n_test})",
                                color="tab:orange",
                                linestyle=":",
                            )
                            ax.fill_betweenx(depth_grid, p25_test, p75_test, color="tab:orange", alpha=0.10, linewidth=0.0)
                        else:
                            ax.text(
                                0.02,
                                0.10,
                                "No test cores",
                                transform=ax.transAxes,
                                fontsize=8,
                                color="tab:orange",
                                va="bottom",
                                ha="left",
                            )

                    ax.set_ylim(float(depth_grid.min()), float(depth_grid.max()))
                    ax.invert_yaxis()
                    ax.set_xlabel("ΣREY (ppm)")
                    if k == 0:
                        ax.set_ylabel(axis_label)
                    ax.set_title(f"Cluster {k} (train n={n_train_total}, test n={n_test_total})")
                    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
                    ax.legend(frameon=False, fontsize=8)

                fig.tight_layout()
                plot_path = out_dir / "profile_clusters_true_vs_pred.png"
            else:
                fig, ax = plt.subplots(figsize=(7.2, 5.6))
                med_true, p25_true, p75_true = _summarize_profiles_iqr(true_profiles)
                ax.plot(
                    med_true,
                    depth_grid,
                    label=f"True median (n={n_cores_true})",
                    color="black",
                    linewidth=2.2,
                )
                ax.fill_betweenx(depth_grid, p25_true, p75_true, color="black", alpha=0.12, linewidth=0.0)

                if train_profiles.size > 0:
                    med_train, p25_train, p75_train = _summarize_profiles_iqr(train_profiles)
                    ax.plot(med_train, depth_grid, label="Train pred median", color="tab:blue", linestyle="--")
                    ax.fill_betweenx(depth_grid, p25_train, p75_train, color="tab:blue", alpha=0.10, linewidth=0.0)

                if test_profiles.size > 0:
                    med_test, p25_test, p75_test = _summarize_profiles_iqr(test_profiles)
                    ax.plot(med_test, depth_grid, label="Test pred median", color="tab:orange", linestyle=":")
                    ax.fill_betweenx(depth_grid, p25_test, p75_test, color="tab:orange", alpha=0.10, linewidth=0.0)

                ax.set_ylim(float(depth_grid.min()), float(depth_grid.max()))
                ax.invert_yaxis()
                ax.set_xlabel("ΣREY (ppm)")
                ax.set_ylabel(axis_label)
                ax.set_title("Profile summary (median ± IQR)")
                ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
                ax.legend(frameon=False, fontsize=9)
                fig.tight_layout()
                plot_path = out_dir / "profile_summary_true_vs_pred.png"

            save_paths = save_figure(
                fig,
                path_no_ext=plot_path.with_suffix(""),
                plot_config=None,
                formats=["png", "pdf"],
                dpi=600,
            )
            results["profile_clusters_plot"] = next(
                (p for p in save_paths if p.lower().endswith(".png")), save_paths[0]
            )
            results["profile_clusters_plot_mode"] = "clustered" if do_clustering and kmeans is not None else "summary"
            logger.info(f"ProfileClusterAnalyzer: Saved profile diagnostics plot to {save_paths}")
            plt.close(fig)
    except Exception as e_plot:
        logger.warning(f"ProfileClusterAnalyzer: Failed to generate profile diagnostics plot: {e_plot}")


    try:

        mean_true_rey = np.nanmean(true_profiles, axis=1)
        max_true_rey = np.nanmax(true_profiles, axis=1)
        if true_labels is None:
            true_labels = np.zeros(len(core_keys_true), dtype=int)

        true_assign_df = pd.DataFrame(
            {
                "lon": [lon for lon, _ in core_keys_true],
                "lat": [lat for _, lat in core_keys_true],
                "cluster_id": true_labels.astype(int),
                "mean_true_rey": mean_true_rey,
                "max_true_rey": max_true_rey,
            }
        )
        true_assign_path = out_dir / "profile_clusters_true_assignments.csv"
        true_assign_df.to_csv(true_assign_path, index=False)
        results["profile_clusters_true_csv"] = str(true_assign_path)

        if train_labels is not None and train_profiles.size > 0:
            mean_train_rey = np.nanmean(train_profiles, axis=1)
            max_train_rey = np.nanmax(train_profiles, axis=1)
            train_assign_df = pd.DataFrame(
                {
                    "lon": [lon for lon, _ in train_keys],
                    "lat": [lat for _, lat in train_keys],
                    "cluster_id": train_labels.astype(int),
                    "split": "train",
                    "mean_pred_rey": mean_train_rey,
                    "max_pred_rey": max_train_rey,
                }
            )
        else:
            train_assign_df = pd.DataFrame(columns=["lon", "lat", "cluster_id", "split", "mean_pred_rey", "max_pred_rey"])

        if test_labels is not None and test_profiles.size > 0:
            mean_test_rey = np.nanmean(test_profiles, axis=1)
            max_test_rey = np.nanmax(test_profiles, axis=1)
            test_assign_df = pd.DataFrame(
                {
                    "lon": [lon for lon, _ in test_keys],
                    "lat": [lat for _, lat in test_keys],
                    "cluster_id": test_labels.astype(int),
                    "split": "test",
                    "mean_pred_rey": mean_test_rey,
                    "max_pred_rey": max_test_rey,
                }
            )
        else:
            test_assign_df = pd.DataFrame(columns=["lon", "lat", "cluster_id", "split", "mean_pred_rey", "max_pred_rey"])

        combined_assign_df = pd.concat(
            [train_assign_df, test_assign_df], ignore_index=True
        )
        combined_path = out_dir / "profile_clusters_pred_assignments.csv"
        combined_assign_df.to_csv(combined_path, index=False)
        results["profile_clusters_pred_csv"] = str(combined_path)


        try:
            if not combined_assign_df.empty:
                counts = (
                    combined_assign_df.groupby(["cluster_id", "split"], dropna=False)
                    .size()
                    .reset_index(name="n_cores")
                )
                counts_path = out_dir / "profile_cluster_split_counts.csv"
                counts.to_csv(counts_path, index=False)
                results["profile_cluster_split_counts_csv"] = str(counts_path)
        except Exception:
            pass
    except Exception as e_csv:
        logger.warning(f"ProfileClusterAnalyzer: Failed to save cluster assignments CSV: {e_csv}")

    return results

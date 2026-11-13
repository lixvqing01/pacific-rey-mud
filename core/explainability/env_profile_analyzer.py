import os
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans

from core.datasets.rey_dataset import REYPointDataset
from core.models.cnn_mlp import MixtureOfProfilesModel
from core.utils.logger_config import logger
from core.visualization.publication_style import publication_style, save_figure


def _sample_env_features_batch(
    dataset: REYPointDataset,
    lons: np.ndarray,
    lats: np.ndarray,
    feature_names: Sequence[str],
) -> Dict[str, np.ndarray]:
    """
    Vectorized sampling of a small set of environmental variables at the nearest grid cell.

    Returns a dict mapping feature name -> array of raw values (same length as lons/lats).
    """
    values: Dict[str, np.ndarray] = {}
    if lons.size == 0 or lats.size == 0:
        return values
    if not hasattr(dataset, "env_data_np") or not hasattr(dataset, "feature_names"):
        return values

    env_lat = dataset.env_lat
    env_lon = dataset.env_lon
    lat_res = getattr(dataset, "lat_res", 0.0) or 0.0
    lon_res = getattr(dataset, "lon_res", 0.0) or 0.0

    lat_idx = np.searchsorted(env_lat, lats - lat_res / 2.0)
    lon_idx = np.searchsorted(env_lon, lons - lon_res / 2.0)
    lat_idx = np.clip(lat_idx, 0, len(env_lat) - 1).astype(int)
    lon_idx = np.clip(lon_idx, 0, len(env_lon) - 1).astype(int)

    name_to_idx = {name: i for i, name in enumerate(dataset.feature_names)}
    for feat in feature_names:
        ch_idx = name_to_idx.get(feat, None)
        if ch_idx is None:
            continue
        try:
            values[feat] = dataset.env_data_np[ch_idx, lat_idx, lon_idx].astype(float)
        except Exception as e:
            logger.debug(
                f"EnvProfileAnalyzer: failed to batch-sample '{feat}' for {lons.size} points: {e}"
            )
            values[feat] = np.full((lons.size,), np.nan, dtype=float)

    return values


def _sample_env_features(
    dataset: REYPointDataset,
    lon: float,
    lat: float,
    feature_names: Sequence[str],
) -> Dict[str, float]:
    """
    Sample a small set of environmental variables at the nearest grid cell.

    The function reuses the same search strategy as REYPointDataset.__getitem__.
    """
    values: Dict[str, float] = {}

    if not hasattr(dataset, "env_data_np") or not hasattr(dataset, "feature_names"):
        return values

    env_lat = dataset.env_lat
    env_lon = dataset.env_lon
    lat_res = getattr(dataset, "lat_res", 0.0) or 0.0
    lon_res = getattr(dataset, "lon_res", 0.0) or 0.0

    lat_idx = np.searchsorted(env_lat, lat - lat_res / 2.0)
    lon_idx = np.searchsorted(env_lon, lon - lon_res / 2.0)
    lat_idx = int(np.clip(lat_idx, 0, len(env_lat) - 1))
    lon_idx = int(np.clip(lon_idx, 0, len(env_lon) - 1))

    name_to_idx = {name: i for i, name in enumerate(dataset.feature_names)}
    for feat in feature_names:
        ch_idx = name_to_idx.get(feat, None)
        if ch_idx is None:
            continue
        try:
            values[feat] = float(dataset.env_data_np[ch_idx, lat_idx, lon_idx])
        except Exception as e:
            logger.debug(
                f"EnvProfileAnalyzer: failed to sample '{feat}' at "
                f"(lat_idx={lat_idx}, lon_idx={lon_idx}): {e}"
            )
            values[feat] = np.nan

    return values


def analyze_env_profile_relationships(
    model: torch.nn.Module,
    dataset: REYPointDataset,
    test_indices: Sequence[int],
    device: torch.device,
    output_dir: str,
    num_env_clusters: int = 4,
    max_depth_m: float = 15.0,
    max_samples: int = 3000,
    batch_size: int = 256,
) -> Dict[str, object]:
    """
    Post-hoc analysis to relate environmental "facies" to MoP profile usage
    and vertical REY behaviour.

    Steps (test set only):
      1. For each sample, obtain env_embedding and profile weights π_k.
      2. Cluster env_embeddings into a small number of environment facies.
      3. For each facies:
           - Summarise typical environment (key variables).
           - Summarise average profile weights π_k.
           - Compute an average predicted REY–depth curve.
      4. Save CSV summary and several diagnostic plots.
    """
    results: Dict[str, object] = {}

    if not isinstance(model, MixtureOfProfilesModel):
        logger.info(
            "EnvProfileAnalyzer: Model is not MixtureOfProfilesModel; "
            "skipping env–profile analysis."
        )
        return results

    if not isinstance(dataset, REYPointDataset):
        logger.info(
            "EnvProfileAnalyzer: Dataset is not REYPointDataset; "
            "skipping env–profile analysis."
        )
        return results

    if not hasattr(dataset, "rey_data"):
        logger.info(
            "EnvProfileAnalyzer: Dataset lacks rey_data; "
            "skipping env–profile analysis."
        )
        return results

    if len(test_indices) == 0:
        logger.info(
            "EnvProfileAnalyzer: Empty test_indices; nothing to analyse."
        )
        return results

    model.eval()

    out_dir = Path(output_dir) / "env_profile_relationships"
    out_dir.mkdir(parents=True, exist_ok=True)


    key_env_features: List[str] = [
        "Average sedimentation rate",
        "Water depth",
        "Bottom water dissolved oxygen",
        "Summer primary productivity",
    ]


    indices = list(test_indices)

    if max_samples is not None and len(indices) > int(max_samples):
        rng = np.random.default_rng(0)
        indices = rng.choice(indices, size=int(max_samples), replace=False).tolist()

    patches: List[torch.Tensor] = []
    meta_lon: List[float] = []
    meta_lat: List[float] = []
    meta_depth: List[float] = []

    all_pred_phys: List[np.ndarray] = []
    all_pi: List[np.ndarray] = []
    all_emb: List[np.ndarray] = []

    def _flush_batch() -> None:
        if not patches:
            return
        batch = torch.stack(patches, dim=0).to(device)
        with torch.inference_mode(), torch.amp.autocast(
            device_type=device.type,
            enabled=torch.cuda.is_available(),
        ):
            pred_norm, components = model(batch, return_components=True)

        pred_norm_np = pred_norm.detach().cpu().numpy().reshape(-1)
        pred_phys_np = dataset._denormalize_rey(pred_norm_np)
        all_pred_phys.append(np.asarray(pred_phys_np, dtype=float).reshape(-1))

        pi = components.get("pi")
        env_embedding = components.get("env_embedding")
        if pi is None or env_embedding is None:

            k_guess = 1
            d_guess = 1
            try:
                if hasattr(model, "num_profiles"):
                    k_guess = int(getattr(model, "num_profiles") or 1)
            except Exception:
                k_guess = 1
            all_pi.append(np.full((len(patches), k_guess), np.nan, dtype=float))
            all_emb.append(np.full((len(patches), d_guess), np.nan, dtype=float))
        else:
            all_pi.append(pi.detach().cpu().numpy().astype(float))
            all_emb.append(env_embedding.detach().cpu().numpy().astype(float))

        patches.clear()


    for idx in indices:
        try:
            combined_patch, _, _ = dataset[idx]
        except Exception as e_fetch:
            logger.debug(f"EnvProfileAnalyzer: failed to fetch sample {idx}: {e_fetch}")
            continue

        row_meta = dataset.rey_data.iloc[idx]
        depth_phys = float(row_meta.get("depth_physical", row_meta.get("depth", 0.0)))
        if depth_phys < 0.0 or depth_phys > max_depth_m:
            continue

        lon = float(row_meta["lon"])
        lat = float(row_meta["lat"])

        patches.append(combined_patch)
        meta_lon.append(lon)
        meta_lat.append(lat)
        meta_depth.append(depth_phys)

        if len(patches) >= int(batch_size):
            _flush_batch()

    _flush_batch()

    if not all_pred_phys:
        logger.warning("EnvProfileAnalyzer: No valid diagnostics collected on test set.")
        return results

    preds = np.concatenate(all_pred_phys, axis=0)
    pi_mat = np.concatenate(all_pi, axis=0)
    emb_mat = np.concatenate(all_emb, axis=0)

    lons = np.asarray(meta_lon, dtype=float)
    lats = np.asarray(meta_lat, dtype=float)
    depths = np.asarray(meta_depth, dtype=float)


    n = min(len(preds), len(lons), pi_mat.shape[0], emb_mat.shape[0])
    preds = preds[:n]
    pi_mat = pi_mat[:n]
    emb_mat = emb_mat[:n]
    lons = lons[:n]
    lats = lats[:n]
    depths = depths[:n]

    env_vals_batch = _sample_env_features_batch(dataset, lons=lons, lats=lats, feature_names=key_env_features)

    rows: List[Dict[str, float]] = []
    for i in range(n):
        row_dict: Dict[str, float] = {
            "lon": float(lons[i]),
            "lat": float(lats[i]),
            "depth_m": float(depths[i]),
            "pred_rey": float(preds[i]),
        }
        for j in range(pi_mat.shape[1]):
            row_dict[f"pi_{j}"] = float(pi_mat[i, j])
        for j in range(emb_mat.shape[1]):
            row_dict[f"env_emb_{j}"] = float(emb_mat[i, j])
        for feat, arr in env_vals_batch.items():
            row_dict[feat] = float(arr[i]) if i < arr.size else np.nan
        rows.append(row_dict)

    if not rows:
        logger.warning("EnvProfileAnalyzer: No valid diagnostics collected on test set.")
        return results

    df = pd.DataFrame(rows)


    emb_cols = [c for c in df.columns if c.startswith("env_emb_")]
    if len(emb_cols) == 0:
        logger.warning(
            "EnvProfileAnalyzer: No env_emb_* columns found; skipping clustering."
        )
        return results

    X_emb = df[emb_cols].to_numpy()

    if X_emb.shape[0] < num_env_clusters:
        num_env_clusters = max(1, X_emb.shape[0])

    try:
        kmeans = KMeans(n_clusters=num_env_clusters, random_state=0, n_init="auto")
        clusters = kmeans.fit_predict(X_emb)
    except Exception as e_km:
        logger.warning(f"EnvProfileAnalyzer: KMeans failed: {e_km}")
        return results

    df["env_cluster"] = clusters


    num_profiles = len([c for c in df.columns if c.startswith("pi_")])
    cluster_records: List[Dict[str, float]] = []

    depth_bins = np.array([0.0, 3.0, 6.0, 9.0, 12.0, 15.0], dtype=float)
    depth_centers = 0.5 * (depth_bins[:-1] + depth_bins[1:])


    profile_matrix = np.zeros((num_env_clusters, num_profiles), dtype=float)

    for cid in range(num_env_clusters):
        sub = df[df["env_cluster"] == cid]
        if sub.empty:
            continue

        rec: Dict[str, float] = {"env_cluster": cid, "num_samples": float(len(sub))}


        for pid in range(num_profiles):
            mean_pi = float(sub[f"pi_{pid}"].mean())
            rec[f"mean_pi_{pid}"] = mean_pi
            profile_matrix[cid, pid] = mean_pi


        for feat in key_env_features:
            if feat in sub.columns:
                rec[f"mean_{feat}"] = float(sub[feat].mean())

        cluster_records.append(rec)

    summary_df = pd.DataFrame(cluster_records)
    summary_path = out_dir / "env_profile_clusters_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    results["summary_csv"] = str(summary_path)
    logger.info(
        f"EnvProfileAnalyzer: Saved env–profile cluster summary to {summary_path}"
    )


    plot_paths: Dict[str, str] = {}


    try:
        with publication_style():
            fig, ax = plt.subplots(figsize=(6.2, 4.1))
            im = ax.imshow(profile_matrix, aspect="auto", cmap="viridis")
            ax.set_xlabel("Profile index k")
            ax.set_ylabel("Environment cluster")
            ax.set_xticks(range(num_profiles))
            ax.set_yticks(range(num_env_clusters))
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Mean π_k")
            ax.set_title("Environment clusters vs MoP profile usage")
            fig.tight_layout()

            heatmap_path = out_dir / "env_cluster_vs_profile_heatmap.png"
            save_paths = save_figure(
                fig,
                path_no_ext=heatmap_path.with_suffix(""),
                plot_config=None,
                formats=["png", "pdf"],
                dpi=600,
            )
            plot_paths["env_cluster_vs_profile_heatmap"] = next(
                (p for p in save_paths if p.lower().endswith(".png")), save_paths[0]
            )
            logger.info(
                f"EnvProfileAnalyzer: Saved env-cluster vs profile heatmap to {save_paths}"
            )
            plt.close(fig)
    except Exception as e_plot1:
        logger.warning(
            f"EnvProfileAnalyzer: Failed to generate env-cluster heatmap: {e_plot1}"
        )


    try:
        with publication_style():
            fig, ax = plt.subplots(figsize=(6.3, 5.0))

            for cid in range(num_env_clusters):
                sub = df[df["env_cluster"] == cid]
                if sub.empty:
                    continue

                depths = sub["depth_m"].to_numpy()
                preds = sub["pred_rey"].to_numpy()


                bin_indices = np.digitize(depths, depth_bins) - 1
                mean_by_bin = []
                for b in range(len(depth_centers)):
                    mask = bin_indices == b
                    if not np.any(mask):
                        mean_by_bin.append(np.nan)
                    else:
                        mean_by_bin.append(float(np.nanmean(preds[mask])))
                mean_by_bin = np.asarray(mean_by_bin, dtype=float)

                if np.all(~np.isfinite(mean_by_bin)):
                    continue

                ax.plot(
                    mean_by_bin,
                    depth_centers,
                    marker="o",
                    label=f"Cluster {cid} (n={len(sub)})",
                )

            ax.set_xlabel("Mean predicted ΣREY (ppm)")
            ax.set_ylabel("Depth (m bsf)")
            ax.invert_yaxis()
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.set_title("Average ΣREY–depth curves by environment cluster (test set)")
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()

            curves_path = out_dir / "avg_rey_depth_by_env_cluster.png"
            save_paths = save_figure(
                fig,
                path_no_ext=curves_path.with_suffix(""),
                plot_config=None,
                formats=["png", "pdf"],
                dpi=600,
            )
            plot_paths["avg_rey_depth_by_env_cluster"] = next(
                (p for p in save_paths if p.lower().endswith(".png")), save_paths[0]
            )
            logger.info(
                f"EnvProfileAnalyzer: Saved average ΣREY–depth curves to {save_paths}"
            )
            plt.close(fig)
    except Exception as e_plot2:
        logger.warning(
            f"EnvProfileAnalyzer: Failed to generate REY–depth curves plot: {e_plot2}"
        )


    if "Average sedimentation rate" in df.columns:
        try:
            sed = df["Average sedimentation rate"].to_numpy()
            pi0 = df["pi_0"].to_numpy()
            mask = np.isfinite(sed) & np.isfinite(pi0)
            if np.any(mask):
                with publication_style():
                    fig, ax = plt.subplots(figsize=(6.2, 4.1))
                    ax.scatter(
                        sed[mask],
                        pi0[mask],
                        s=18,
                        alpha=0.7,
                        edgecolors="none",
                    )
                    ax.set_xlabel("Sedimentation rate (cm/ka)")
                    ax.set_ylabel("Profile-0 weight π₀")
                    ax.grid(True, linestyle="--", alpha=0.35)
                    ax.set_title("Sedimentation rate vs profile-0 usage (test set)")
                    fig.tight_layout()

                    scatter_path = out_dir / "sed_rate_vs_pi0.png"
                    save_paths = save_figure(
                        fig,
                        path_no_ext=scatter_path.with_suffix(""),
                        plot_config=None,
                        formats=["png", "pdf"],
                        dpi=600,
                    )
                    plot_paths["sed_rate_vs_pi0"] = next(
                        (p for p in save_paths if p.lower().endswith(".png")), save_paths[0]
                    )
                    logger.info(
                        f"EnvProfileAnalyzer: Saved sed-rate vs profile weight plot to {save_paths}"
                    )
                    plt.close(fig)
        except Exception as e_plot3:
            logger.warning(
                f"EnvProfileAnalyzer: Failed to generate sed-rate vs π_0 scatter: {e_plot3}"
            )

    results["plot_paths"] = plot_paths
    return results


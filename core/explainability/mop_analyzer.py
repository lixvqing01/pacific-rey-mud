import numpy as np
import torch
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from pathlib import Path

from core.utils.logger_config import logger
from core.models.cnn_mlp import MixtureOfProfilesModel
from core.utils.config_models import TrainingArtifacts
from core.visualization.publication_style import publication_style, save_figure


def analyze_mop_depth_profiles(
    model: torch.nn.Module,
    training_artifacts: TrainingArtifacts,
    depth_transform_method: str,
    output_dir: str,
    num_depth_points: int = 100,
) -> dict:
    """
    Analyze Mixture-of-Profiles depth profiles g_k(z).

    Produces a plot of g_k(z) vs. the model's "depth feature" for each profile k,
    using the same depth transform and normalization as in training.
    """
    results: dict = {}

    if not isinstance(model, MixtureOfProfilesModel):
        return results

    try:
        norm_params = training_artifacts.norm_params
        depth_feature_type = getattr(norm_params, "depth_feature_type", None) or "depth_mbsf"
        depth_min = getattr(norm_params, "depth_feature_min", None)
        depth_max = getattr(norm_params, "depth_feature_max", None)
        if depth_min is None and depth_max is None:
            depth_min = norm_params.depth_min
            depth_max = norm_params.depth_max

        depth_min = float(depth_min) if depth_min is not None else 0.0
        depth_max = float(depth_max) if depth_max is not None else depth_min + 1.0

        if depth_max <= depth_min:
            depth_max = depth_min + 1.0

        depths_raw = np.linspace(depth_min, depth_max, num_depth_points).astype(
            np.float64
        )


        if depth_transform_method == "log1p":
            depth_transformed = np.log1p(depths_raw)
        elif depth_transform_method == "sqrt":
            depth_transformed = np.sqrt(depths_raw)
        else:
            depth_transformed = depths_raw


        channel_means = np.asarray(norm_params.channel_means, dtype=np.float64)
        channel_stds = np.asarray(norm_params.channel_stds, dtype=np.float64)
        if channel_means.size == 0 or channel_stds.size == 0:
            logger.warning(
                "MoP Analyzer: channel_means/stds empty; cannot normalize depth."
            )
            return results

        depth_mean = float(channel_means[-1])
        depth_std = float(channel_stds[-1]) if channel_stds[-1] != 0 else 1.0
        depth_norm = (depth_transformed - depth_mean) / depth_std

        device = next(model.parameters()).device
        d = torch.from_numpy(depth_norm.reshape(-1, 1)).float().to(device)

        depth_features = [d]
        for power in range(2, model.depth_feature_order + 1):
            depth_features.append(d ** power)
        x_depth = torch.cat(depth_features, dim=1)

        with torch.no_grad():
            profiles = model.depth_mlp(x_depth).cpu().numpy()

        k = profiles.shape[1]

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        with publication_style():
            fig, ax = plt.subplots(figsize=(7.6, 5.2))
            for idx_profile in range(k):
                ax.plot(
                    depths_raw,
                    profiles[:, idx_profile],
                    label=f"g{idx_profile+1}(z)",
                )

        if depth_feature_type == "age_ka_proxy":
            ax.set_xlabel("Age proxy (ka)")
        elif depth_feature_type == "legacy_time_depth":
            ax.set_xlabel("Depth/sed_rate (legacy units)")
        else:
            ax.set_xlabel("Depth (m bsf)")
            ax.set_ylabel("g_k(z) (normalized target space)")
            ax.set_title("Mixture-of-Profiles: depth profiles g_k(z)")
            ax.legend(frameon=False, ncol=2, fontsize=8)
            ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
            fig.tight_layout()

            save_paths = save_figure(
                fig,
                path_no_ext=out_dir / "mop_depth_profiles",
                plot_config=None,
                formats=["png", "pdf"],
                dpi=600,
            )
            results["mop_depth_profiles_plot_path"] = next(
                (p for p in save_paths if p.lower().endswith(".png")), save_paths[0]
            )
            logger.info(f"MoP Analyzer: Saved depth profile plot to {save_paths}")
            plt.close(fig)

    except Exception as exc:
        logger.error(
            f"MoP Analyzer: Failed to analyze depth profiles: {exc}", exc_info=True
        )

    return results


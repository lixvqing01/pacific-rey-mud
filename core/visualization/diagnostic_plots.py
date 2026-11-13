
import os
import numpy as np
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from core.utils.logger_config import logger

def plot_residuals(residuals, predictions, title_prefix, output_dir, plot_params_dict=None):
    """Plots residuals vs. predicted values."""
    if plot_params_dict is None: plot_params_dict = {}

    logger.debug(f"{title_prefix} Residual Plot: Generating residual plot...")
    residuals_np = np.asarray(residuals).flatten()
    predictions_np = np.asarray(predictions).flatten()
    valid_mask = ~np.isnan(residuals_np) & ~np.isnan(predictions_np)

    if np.sum(valid_mask) == 0:
        logger.warning(f"{title_prefix} Residual Plot: No valid residuals/predictions to plot.")
        return
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_filename_base = os.path.join(output_dir, "residual_plot")

        plt.figure(figsize=plot_params_dict.get("figsize_diag", (8, 6)))
        plt.scatter(predictions_np[valid_mask], residuals_np[valid_mask], alpha=plot_params_dict.get("scatter_alpha", 0.5))
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel("Predicted Value (Test Set)")
        plt.ylabel("Residual (True - Predicted)")
        plt.title(f"{title_prefix} Residual Plot")
        plt.grid(True)
        plt.tight_layout()
        png_path = f"{output_filename_base}.png"
        svg_path = f"{output_filename_base}.svg"
        plt.savefig(png_path, dpi=plot_params_dict.get("dpi", 300))
        plt.savefig(svg_path, dpi=plot_params_dict.get("dpi", 300))
        logger.debug(f"{title_prefix} Residual Plot: Residual plot saved to: {png_path} / {svg_path}")
        plt.close()
    except Exception as e:
        logger.error(f"{title_prefix} Residual Plot: Error generating residual plot: {e}")
        plt.close()

def plot_error_histogram(residuals, title_prefix, output_dir, plot_params_dict=None):
    """Plots a histogram of prediction errors (residuals)."""
    if plot_params_dict is None: plot_params_dict = {}

    logger.debug(f"{title_prefix} Error Histogram: Generating error histogram...")
    residuals_np = np.asarray(residuals).flatten()
    valid_mask = ~np.isnan(residuals_np)
    if np.sum(valid_mask) == 0:
        logger.warning(f"{title_prefix} Error Histogram: No valid residuals for histogram.")
        return
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_filename_base = os.path.join(output_dir, "error_histogram")

        plt.figure(figsize=plot_params_dict.get("figsize_diag", (8, 6)))
        plt.hist(residuals_np[valid_mask], bins=plot_params_dict.get("hist_bins", 30), edgecolor='black')
        plt.xlabel("Prediction Error (Residual)")
        plt.ylabel("Frequency")
        plt.title(f"{title_prefix} Prediction Error Distribution")
        plt.grid(True)
        plt.tight_layout()
        png_path = f"{output_filename_base}.png"
        svg_path = f"{output_filename_base}.svg"
        plt.savefig(png_path, dpi=plot_params_dict.get("dpi", 300))
        plt.savefig(svg_path, dpi=plot_params_dict.get("dpi", 300))
        logger.debug(f"{title_prefix} Error Histogram: Error histogram saved to: {png_path} / {svg_path}")
        plt.close()
    except Exception as e:
        logger.error(f"{title_prefix} Error Histogram: Error generating error histogram: {e}")
        plt.close()

def plot_depth_residuals(depths, residuals, title_prefix, output_dir, plot_params_dict=None):
    """Plots residuals vs. sample depths."""
    if plot_params_dict is None: plot_params_dict = {}

    logger.debug(f"{title_prefix} Depth Residual Plot: Generating depth residual plot...")
    depths_np = np.asarray(depths).flatten()
    residuals_np = np.asarray(residuals).flatten()
    valid_mask = ~np.isnan(depths_np) & ~np.isnan(residuals_np)

    if np.sum(valid_mask) == 0:
        logger.warning(f"{title_prefix} Depth Residual Plot: No valid depths/residuals to plot.")
        return
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_filename_base = os.path.join(output_dir, "depth_residual_plot")

        plt.figure(figsize=plot_params_dict.get("figsize_diag", (8, 6)))
        plt.scatter(depths_np[valid_mask], residuals_np[valid_mask], alpha=plot_params_dict.get("scatter_alpha", 0.5))
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel("Sample Depth")
        plt.ylabel("Residual (True - Predicted)")
        plt.title(f"{title_prefix} Residuals vs. Depth")
        plt.grid(True)
        plt.tight_layout()
        png_path = f"{output_filename_base}.png"
        svg_path = f"{output_filename_base}.svg"
        plt.savefig(png_path, dpi=plot_params_dict.get("dpi", 300))
        plt.savefig(svg_path, dpi=plot_params_dict.get("dpi", 300))
        logger.debug(f"{title_prefix} Depth Residual Plot: Depth residual plot saved to: {png_path} / {svg_path}")
        plt.close()
    except Exception as e:
        logger.error(f"{title_prefix} Depth Residual Plot: Error generating depth residual plot: {e}")
        plt.close()

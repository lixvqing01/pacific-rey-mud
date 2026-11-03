import logging
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from pathlib import Path
from core.utils.logger_config import logger

def plot_shap_interactions(shap_values, feature_names, output_dir, plot_config):
    """
    Plots an enhanced SHAP interaction heatmap to analyze feature coupling effects.
    """
    logger.info("Advanced Explainer Plots: Generating enhanced SHAP interaction heatmap...")
    results = {}
    plots_int_dir = Path(output_dir) / "interactions"
    plots_int_dir.mkdir(parents=True, exist_ok=True)

    try:


        if hasattr(shap_values, 'shape') and len(shap_values.shape) == 2:
            n_features_interaction = min(plot_config.get("interaction_features_to_plot", 15), len(feature_names))


            top_features_indices = list(range(int(n_features_interaction)))
            top_feature_names = [feature_names[i] for i in top_features_indices]
            top_shap_values = shap_values[:, top_features_indices]


            shap_corr = np.corrcoef(top_shap_values.T)


            mask = np.triu(np.ones_like(shap_corr, dtype=bool))

            plt.style.use('seaborn-v0_8-whitegrid')
            fig, ax = plt.subplots(figsize=(12, 10))


            sns.heatmap(
                shap_corr,
                mask=mask,
                xticklabels=top_feature_names,
                yticklabels=top_feature_names,
                annot=True,
                fmt='.2f',
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=.5,
                cbar_kws={"shrink": .5, "label": "SHAP Value Correlation"}
            )

            ax.set_title(f'Top {n_features_interaction} Feature SHAP Interaction Heatmap (Proxy)', fontsize=16, weight='bold')
            plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
            plt.yticks(rotation=0)

            plt.tight_layout(rect=[0, 0, 0.9, 1])

            output_path = plots_int_dir / "shap_interaction_correlation_heatmap.png"
            plt.savefig(output_path, dpi=plot_config.get("dpi", 300))
            plt.close(fig)

            results["shap_interaction_heatmap_path"] = str(output_path)
            logger.info(f"Advanced Explainer Plots: Saved enhanced SHAP interaction heatmap to '{output_path}'")

        else:
            logger.warning("Advanced Explainer Plots: SHAP values format not suitable for interaction analysis.")

    except Exception as e:
        logger.error("Advanced Explainer Plots: Failed to generate SHAP interaction plots.", exc_info=True)
        results["shap_interaction_heatmap_path"] = None

    return results

def plot_partial_dependence_like(shap_values, features_df, feature_names, output_dir, plot_config):
    """
    Creates plots that mimic Partial Dependence Plots (PDP) / Individual Conditional Expectation (ICE)
    using SHAP values to analyze threshold effects, with enhanced visualization.
    """
    logger.info("Advanced Explainer Plots: Generating enhanced PDP/ICE-like plots...")
    results = {}
    plots_pdp_dir = Path(output_dir) / "pdp_ice_like"
    plots_pdp_dir.mkdir(parents=True, exist_ok=True)

    try:
        n_features_to_plot = min(plot_config.get("pdp_features_to_plot", 5), len(feature_names))
        feature_indices = range(int(n_features_to_plot))

        for i in feature_indices:
            feat_name = feature_names[i]
            feat_values = features_df.iloc[:, i]
            shap_for_feat = shap_values[:, i]


            plt.style.use('seaborn-v0_8-whitegrid')
            fig, ax = plt.subplots(figsize=(10, 7))


            hb = ax.hexbin(feat_values, shap_for_feat, gridsize=50, cmap='viridis', mincnt=1, alpha=0.8)


            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess

                sorted_indices = np.argsort(feat_values)
                sorted_feat_values = feat_values.iloc[sorted_indices]
                sorted_shap_for_feat = shap_for_feat[sorted_indices]
                smoothed = lowess(sorted_shap_for_feat, sorted_feat_values, frac=0.3)
                ax.plot(smoothed[:, 0], smoothed[:, 1], color='red', lw=2, linestyle='--', label='LOESS Trend')
            except ImportError:

                z = np.polyfit(feat_values, shap_for_feat, 2)
                p = np.poly1d(z)
                sorted_feat_values = np.sort(feat_values)
                ax.plot(sorted_feat_values, p(sorted_feat_values), "r--", lw=2, alpha=0.9, label='Trend Line (Polyfit)')


            shap_threshold = np.percentile(np.abs(shap_for_feat), 95)
            high_impact_mask = np.abs(shap_for_feat) > shap_threshold
            ax.scatter(feat_values[high_impact_mask], shap_for_feat[high_impact_mask],
                       facecolors='none', edgecolors='magenta', s=80, lw=1.5,
                       label=f'Top 5% |SHAP| (>{shap_threshold:.3f})')


            cb = fig.colorbar(hb, ax=ax, label='Data Point Density')


            ax.set_xlabel(f'Feature Value: {feat_name}', fontsize=12, weight='bold')
            ax.set_ylabel(f'SHAP Value for {feat_name}', fontsize=12, weight='bold')
            ax.set_title(f'Enhanced PDP/ICE-like Plot: {feat_name}', fontsize=14, weight='bold')
            ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            ax.legend()
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)

            plt.tight_layout()

            output_path = plots_pdp_dir / f"pdp_ice_like_{feat_name.replace('/', '_')}.png"
            plt.savefig(output_path, dpi=plot_config.get("dpi", 300))
            plt.close(fig)

            logger.info(f"Advanced Explainer Plots: Saved enhanced PDP/ICE-like plot for {feat_name} to '{output_path}'")

        results["pdp_ice_like_plot_paths"] = [str(p) for p in plots_pdp_dir.glob("*.png")]
        logger.info(f"Advanced Explainer Plots: Successfully generated {len(results['pdp_ice_like_plot_paths'])} enhanced PDP/ICE-like plots.")

    except Exception as e:
        logger.error("Advanced Explainer Plots: Failed to generate enhanced PDP/ICE-like plots.", exc_info=True)
        results["pdp_ice_like_plot_paths"] = []

    return results

def plot_feature_importance_ranking(importance_df, output_dir, plot_config):
    """
    Plots an enhanced, detailed feature importance ranking.
    """
    logger.info("Advanced Explainer Plots: Generating enhanced feature importance ranking plot...")
    results = {}
    plots_imp_dir = Path(output_dir) / "feature_importance"
    plots_imp_dir.mkdir(parents=True, exist_ok=True)

    try:

        importance_df = importance_df.sort_values(by='importance', ascending=False)


        def get_color(feature_name):
            if 'depth' in feature_name.lower():
                return 'navy'
            elif 'temp' in feature_name.lower():
                return 'firebrick'
            elif 'sal' in feature_name.lower():
                return 'darkcyan'
            elif 'oxy' in feature_name.lower():
                return 'darkgreen'
            else:
                return 'darkgray'

        colors = [get_color(f) for f in importance_df['feature']]

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax1 = plt.subplots(figsize=(12, max(6, len(importance_df) * 0.4)))


        bars = ax1.barh(
            importance_df['feature'],
            importance_df['importance'],
            color=colors,
            edgecolor='black',
            height=0.7
        )
        ax1.invert_yaxis()

        ax1.set_xlabel('Mean Absolute SHAP Value (Importance)', fontsize=12, weight='bold')
        ax1.set_ylabel('Features', fontsize=12, weight='bold')
        ax1.set_title('Enhanced Feature Importance Ranking', fontsize=16, weight='bold')


        for bar in bars:
            width = bar.get_width()
            ax1.text(width * 1.01, bar.get_y() + bar.get_height() / 2,
                     f'{width:.3f}', va='center', ha='left', fontsize=9, weight='bold')

        ax1.tick_params(axis='y', labelsize=10)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)


        cumulative_importance = np.cumsum(importance_df['importance'].values)
        cumulative_percentage = 100 * cumulative_importance / cumulative_importance[-1]

        ax2 = ax1.twiny()
        ax2.plot(cumulative_percentage, importance_df['feature'],
                 color='gold', marker='o', ms=5, linestyle='--',
                 label='Cumulative Importance (%)')

        ax2.set_xlabel('Cumulative Importance (%)', fontsize=12, weight='bold', color='darkgoldenrod')
        ax2.set_xlim(0, 110)
        ax2.tick_params(axis='x', labelcolor='darkgoldenrod')
        ax2.spines['top'].set_color('darkgoldenrod')
        ax2.spines['top'].set_linewidth(1.5)


        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='navy', edgecolor='black', label='Depth'),
            Patch(facecolor='firebrick', edgecolor='black', label='Temperature'),
            Patch(facecolor='darkcyan', edgecolor='black', label='Salinity'),
            Patch(facecolor='darkgreen', edgecolor='black', label='Oxygen'),
            Patch(facecolor='darkgray', edgecolor='black', label='Other')
        ]
        ax1.legend(handles=legend_elements, title='Feature Categories', loc='lower right', frameon=True, shadow=True)

        fig.tight_layout()

        output_path = plots_imp_dir / "detailed_feature_importance_ranking.png"
        plt.savefig(output_path, dpi=plot_config.get("dpi", 300))
        plt.close(fig)

        results["detailed_feature_importance_plot_path"] = str(output_path)
        logger.info(f"Advanced Explainer Plots: Saved enhanced feature importance plot to '{output_path}'")

    except Exception as e:
        logger.error("Advanced Explainer Plots: Failed to generate detailed feature importance plot.", exc_info=True)
        results["detailed_feature_importance_plot_path"] = None

    return results



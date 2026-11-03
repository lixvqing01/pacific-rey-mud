
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from core.utils.logger_config import logger

def plot_scatter_plotly(y_true, y_pred, title_prefix, output_html_path, data_set_labels=None, plot_params_dict=None):
    """ Plots true vs predicted values using Plotly for train/test sets. """
    if plot_params_dict is None: plot_params_dict = {}
    logger.debug(f"{title_prefix} Plotly Scatter: Generating Plotly scatter plot...")
    try:
        if not isinstance(y_true, list): y_true = [y_true]
        if not isinstance(y_pred, list): y_pred = [y_pred]
        if data_set_labels is None:
            data_set_labels = [f'Set {i+1}' for i in range(len(y_true))]
        elif not isinstance(data_set_labels, list):
            data_set_labels = [data_set_labels] * len(y_true)

        if len(y_true) != len(y_pred) or len(y_true) != len(data_set_labels):
            logger.error(f"{title_prefix} Plotly Scatter: Error: Length mismatch between y_true, y_pred, and data_set_labels.")
            return

        df_list = []
        for yt, yp, label in zip(y_true, y_pred, data_set_labels):
            yt_np = np.asarray(yt).flatten()
            yp_np = np.asarray(yp).flatten()
            valid_mask = ~np.isnan(yt_np) & ~np.isnan(yp_np)
            if np.sum(valid_mask) > 0:
                n_valid = int(np.sum(valid_mask))
                label_with_n = f"{label} (n={n_valid})"
                df_list.append(
                    pd.DataFrame(
                        {
                            "True Value": yt_np[valid_mask],
                            "Predicted Value": yp_np[valid_mask],
                            "Set": label_with_n,
                        }
                    )
                )

        if not df_list:
            logger.warning(f"{title_prefix} Plotly Scatter: No valid data to plot. Skipping Plotly scatter plot.")
            return

        df = pd.concat(df_list)

        fig = px.scatter(df, x='True Value', y='Predicted Value', color='Set',
                         title=f"{title_prefix} True vs. Predicted (Plotly)",
                         labels={'True Value': 'True REY Value', 'Predicted Value': 'Predicted REY Value'},
                         hover_data={'True Value': ':.3f', 'Predicted Value': ':.3f'})

        all_values = df[['True Value', 'Predicted Value']].values.flatten()
        all_values = all_values[~np.isnan(all_values)]
        if len(all_values) > 0:
            min_val = all_values.min()
            max_val = all_values.max()
            fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                     mode='lines', name='Ideal (1:1)', line=dict(dash='dash', color='black')))
            fig.update_xaxes(range=[min_val, max_val])
            fig.update_yaxes(range=[min_val, max_val])
        else:
            logger.warning(f"{title_prefix} Plotly Scatter: Could not determine value range for 1:1 line and axes.")

        fig.update_layout(xaxis_title="True Value", yaxis_title="Predicted Value", legend_title="Data Set")

        os.makedirs(os.path.dirname(output_html_path), exist_ok=True)
        pio.write_html(fig, file=output_html_path, auto_open=False)
        logger.debug(f"{title_prefix} Plotly Scatter: Plotly scatter plot saved to: {output_html_path}")

    except Exception as e:
        logger.error(f"{title_prefix} Plotly Scatter: Error generating Plotly scatter plot: {e}")

def plot_scatter(y_true, y_pred, title_prefix, output_path_base, data_set_labels=None, plot_params_dict=None):
    """ Plots true vs predicted values using Matplotlib for train/test sets. """
    if plot_params_dict is None: plot_params_dict = {}
    logger.debug(f"{title_prefix} Matplotlib Scatter: Generating scatter plot...")
    try:

        if not isinstance(y_true, list): y_true = [y_true]
        if not isinstance(y_pred, list): y_pred = [y_pred]
        if data_set_labels is None:
            data_set_labels = [f'Set {i+1}' for i in range(len(y_true))]
        elif not isinstance(data_set_labels, list):
            data_set_labels = [data_set_labels] * len(y_true)

        if len(y_true) != len(y_pred) or len(y_true) != len(data_set_labels):
            logger.error(f"{title_prefix} Matplotlib Scatter: Error: Length mismatch between y_true, y_pred, and data_set_labels.")
            return

        plt.figure(figsize=plot_params_dict.get("figsize_scatter", (8, 6)))
        all_true_vals = []
        all_pred_vals = []
        r2_texts = []
        plotted_something = False

        for yt, yp, label in zip(y_true, y_pred, data_set_labels):
            yt_np = np.asarray(yt).flatten()
            yp_np = np.asarray(yp).flatten()

            valid_mask = ~np.isnan(yt_np) & ~np.isnan(yp_np)

            if np.sum(valid_mask) > 0:
                yt_plot = yt_np[valid_mask]
                yp_plot = yp_np[valid_mask]
                n_valid = int(np.sum(valid_mask))
                label_with_n = f"{label} (n={n_valid})"

                plt.scatter(yt_plot, yp_plot,
                            label=label_with_n,
                            alpha=plot_params_dict.get("scatter_alpha", 0.6))

                all_true_vals.extend(yt_plot)
                all_pred_vals.extend(yp_plot)
                plotted_something = True

                if len(yt_plot) > 1:

                    corr = np.corrcoef(yt_plot, yp_plot)[0, 1]
                    r2 = corr**2
                    set_name = label.replace(' Set', '')
                    r2_texts.append(f"{set_name} $R^2$: {r2:.4f}")
                else:
                    set_name = label.replace(' Set', '')
                    r2_texts.append(f"{set_name} $R^2$: N/A")
            else:
                logger.warning(f"{title_prefix} Matplotlib Scatter: No valid data points for dataset '{label}'.")

        if not plotted_something:
            logger.warning(f"{title_prefix} Matplotlib Scatter: No valid data to plot across all datasets. Skipping scatter plot.")
            plt.close()
            return

        min_val = np.nanmin(all_true_vals + all_pred_vals) if all_true_vals else np.nan
        max_val = np.nanmax(all_true_vals + all_pred_vals) if all_true_vals else np.nan

        if pd.notna(min_val) and pd.notna(max_val):
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', label="Ideal (1:1 Line)")

        if r2_texts:
            r2_display_text = "\n".join(r2_texts)
            plt.text(0.05, 0.95, r2_display_text, transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))


        plt.xlabel(plot_params_dict.get("xlabel", "True Value"))
        plt.ylabel(plot_params_dict.get("ylabel", "Predicted Value"))
        plt.title(plot_params_dict.get("title", f"{title_prefix} True vs. Predicted"))


        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend(loc=plot_params_dict.get("legend_loc", "best"))
        plt.grid(True)
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path_base), exist_ok=True)
        png_path = f"{output_path_base}.png"
        svg_path = f"{output_path_base}.svg"
        plt.savefig(png_path, dpi=plot_params_dict.get("dpi", 300))
        plt.savefig(svg_path, format='svg')
        logger.debug(f"{title_prefix} Matplotlib Scatter: Scatter plot saved to: {png_path} / {svg_path}")
        plt.close()
    except Exception as e:
        logger.error(f"{title_prefix} Matplotlib Scatter: Error generating scatter plot: {e}")
        plt.close()

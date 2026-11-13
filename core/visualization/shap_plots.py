
import os
import numpy as np
import pandas as pd
import xarray as xr
from core.utils.netcdf_io import open_dataset
import matplotlib
from collections import defaultdict
from matplotlib.ticker import FuncFormatter

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import time
import glob
import textwrap
from core.utils.logger_config import logger

import shap
from pathlib import Path

from core.visualization.publication_style import publication_style, save_figure

def save_feature_maps_to_netcdf(
    feature_maps_dict: dict,
    base_filename_prefix: str,
    output_dir: str,
    sample_spatial_info: dict,
    model_patch_size: tuple
):
    """
    Saves extracted feature maps to NetCDF files, associating them with geospatial coordinates.

    Args:
        feature_maps_dict (dict): Dictionary of feature maps (layer_name -> tensor).
                                  Tensor shape: (C, H_out, W_out) or (H_out, W_out) if single channel.
        base_filename_prefix (str): Base prefix for output NetCDF files (e.g., "sample_0_features").
        output_dir (str): Directory to save NetCDF files.
        sample_spatial_info (dict): Dictionary from get_spatial_info_for_sample.
        model_patch_size (tuple): The (height, width) of the original input patch to the CNN.
    """
    os.makedirs(output_dir, exist_ok=True)

    input_patch_lons = sample_spatial_info['patch_lon_coords']
    input_patch_lats = sample_spatial_info['patch_lat_coords']

    for layer_name, fm_tensor in feature_maps_dict.items():
        if fm_tensor.ndim == 4 and fm_tensor.shape[0] == 1:
            fm_tensor = fm_tensor.squeeze(0) 

        if fm_tensor.ndim == 2:
            fm_tensor = fm_tensor.unsqueeze(0)

        if fm_tensor.ndim != 3:
            logger.error(f"FeatureMapSave: Feature map {layer_name} has unexpected shape {fm_tensor.shape} after processing. Skipping.")
            continue

        num_channels, h_out, w_out = fm_tensor.shape

        fm_lat_coords = np.linspace(input_patch_lats[0], input_patch_lats[-1], h_out)
        fm_lon_coords = np.linspace(input_patch_lons[0], input_patch_lons[-1], w_out)

        layer_ds = xr.Dataset()

        for channel_idx in range(num_channels):
            channel_data = fm_tensor[channel_idx, :, :].cpu().numpy()

            data_array = xr.DataArray(
                channel_data,
                coords={'latitude': fm_lat_coords, 'longitude': fm_lon_coords},
                dims=('latitude', 'longitude'),
                name=f'{layer_name}_channel_{channel_idx}'
            )
            data_array.attrs['long_name'] = f'Feature map for {layer_name}, channel {channel_idx}'
            data_array.attrs['units'] = 'activation'
            data_array.attrs['sample_center_lon'] = sample_spatial_info['center_lon']
            data_array.attrs['sample_center_lat'] = sample_spatial_info['center_lat']

            layer_ds[data_array.name] = data_array

        layer_ds.attrs['layer_name'] = layer_name
        layer_ds.attrs['original_sample_info'] = str(sample_spatial_info['original_rey_point_data'])
        layer_ds.attrs['creation_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        layer_ds.attrs['source_script'] = __file__

        filename = os.path.join(output_dir, f"{base_filename_prefix}_{layer_name}.nc")
        try:
            layer_ds.to_netcdf(filename)
            logger.debug(f"FeatureMapSave: Saved feature map for {layer_name} to {filename}")
        except Exception as e:
            logger.error(f"FeatureMapSave: Error saving feature map {layer_name} to NetCDF: {e}")
def plot_feature_map_from_nc(nc_file_path, plot_output_dir, plot_config_params, max_channels_to_plot=None):
    """
    Reads a NetCDF file containing feature maps and plots them as geographic maps,
    optionally limiting the number of channels plotted per layer.

    Args:
        nc_file_path (str): Path to the NetCDF file.
        plot_output_dir (str): Directory to save the plot images.
        plot_config_params (dict): Plotting parameters from the main config (e.g., dpi, coastlines).
        max_channels_to_plot (int, optional): Maximum number of channels (variables) to plot from the file.
                                              If None or <= 0, all channels are plotted. Defaults to None.
    """
    os.makedirs(plot_output_dir, exist_ok=True)
    try:
        ds = open_dataset(nc_file_path)
        base_filename = os.path.splitext(os.path.basename(nc_file_path))[0]
        layer_name_for_log = ds.attrs.get('layer_name', 'Unknown Layer')

        all_var_names = list(ds.data_vars)
        num_total_channels = len(all_var_names)
        channels_to_plot = all_var_names

        if max_channels_to_plot is not None and max_channels_to_plot > 0 and max_channels_to_plot < num_total_channels:
            channels_to_plot = all_var_names[:max_channels_to_plot]
            logger.debug(f"FeatureMapPlot: Plotting only the first {max_channels_to_plot} out of {num_total_channels} channels for layer '{layer_name_for_log}'.")
        else:
             logger.debug(f"FeatureMapPlot: Plotting all {num_total_channels} channels for layer '{layer_name_for_log}'.")

        for var_name in channels_to_plot:
            da = ds[var_name]
            channel_data_for_stats = da.data
            stats_msg = f"Data Stats for {var_name} (Layer: {layer_name_for_log}):\n"
            stats_msg += f"  Shape: {channel_data_for_stats.shape}\n"
            if channel_data_for_stats.size > 0:
                stats_msg += f"  Min: {np.nanmin(channel_data_for_stats):.4f}\n"
                stats_msg += f"  Max: {np.nanmax(channel_data_for_stats):.4f}\n"
                stats_msg += f"  Mean: {np.nanmean(channel_data_for_stats):.4f}\n"
                stats_msg += f"  Std: {np.nanstd(channel_data_for_stats):.4f}\n"
                stats_msg += f"  Non-NaN count: {np.count_nonzero(~np.isnan(channel_data_for_stats))}\n"
            else:
                stats_msg += "  Data is empty.\n"
            logger.debug(f"FeatureMapPlotDiag: {stats_msg}")

            plt.figure(figsize=plot_config_params.get("figsize_feature_map", (10, 8)))
            ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))

            valid_data = channel_data_for_stats[~np.isnan(channel_data_for_stats)]
            if valid_data.size == 0:
                logger.warning(f"FeatureMapPlot: No valid (non-NaN) data in {var_name} from {nc_file_path}, skipping plot.")
                plt.close()
                continue

            vmin_cfg = plot_config_params.get("feature_map_vmin", None)
            vmax_cfg = plot_config_params.get("feature_map_vmax", None)

            vmin, vmax = None, None
            epsilon = 1e-6

            if vmin_cfg is not None and vmax_cfg is not None:
                try:
                    vmin = float(vmin_cfg)
                    vmax = float(vmax_cfg)
                    logger.debug(f"FeatureMapPlotVminVmax: Using user-defined vmin={vmin:.4f}, vmax={vmax:.4f} for {var_name}.")
                except ValueError:
                    logger.warning(f"FeatureMapPlotVminVmax: Invalid user-defined vmin/vmax for {var_name}. Reverting to other strategies.")
                    vmin, vmax = None, None

            if vmin is None or vmax is None:
                q_min_percent = plot_config_params.get("feature_map_plot_qmin", 2)
                q_max_percent = plot_config_params.get("feature_map_plot_qmax", 98)
                vmin_p = np.percentile(valid_data, q_min_percent)
                vmax_p = np.percentile(valid_data, q_max_percent)
                logger.debug(f"FeatureMapPlotVminVmax: Strategy 1 (Percentile): q_min={q_min_percent}%, q_max={q_max_percent}% -> vmin_p={vmin_p:.4f}, vmax_p={vmax_p:.4f}")

                if (vmax_p - vmin_p) < epsilon:
                    logger.debug(f"FeatureMapPlotVminVmax: Percentile range ({vmax_p - vmin_p:.4g}) too small for {var_name}. Trying Strategy 2 (actual min/max).")
                    vmin = valid_data.min()
                    vmax = valid_data.max()
                    logger.debug(f"FeatureMapPlotVminVmax: Strategy 2 (Actual Min/Max): vmin={vmin:.4f}, vmax={vmax:.4f}")
                else:
                    vmin, vmax = vmin_p, vmax_p

            if vmin == vmax:
                logger.debug(f"FeatureMapPlotVminVmax: vmin ({vmin:.4f}) and vmax ({vmax:.4f}) are equal for {var_name}. Adjusting slightly.")
                offset = 0.1 * abs(vmin) if vmin != 0 else 0.1
                vmin -= offset
                vmax += offset
                if vmin == vmax: 
                    vmin = -0.1
                    vmax = 0.1
                logger.debug(f"FeatureMapPlotVminVmax: Adjusted to vmin={vmin:.4f}, vmax={vmax:.4f}")

            default_cmap = 'viridis'
            cmap_to_use = plot_config_params.get("cmap", default_cmap)
            if 'feature_map_cmap' in plot_config_params :
                 cmap_to_use = plot_config_params.get("feature_map_cmap", default_cmap)

            logger.debug(f"FeatureMapPlot: Plotting {var_name} with cmap='{cmap_to_use}', vmin={vmin:.4f}, vmax={vmax:.4f}")
            da.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap_to_use,
                      add_colorbar=True, vmin=vmin, vmax=vmax)

            if plot_config_params.get("feature_map_coastlines", True):
                ax.add_feature(cfeature.COASTLINE)
            if plot_config_params.get("feature_map_borders", True):
                ax.add_feature(cfeature.BORDERS, linestyle=':')
            if plot_config_params.get("feature_map_land", True):
                ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray')
            if plot_config_params.get("feature_map_ocean", True):
                ax.add_feature(cfeature.OCEAN, facecolor='aliceblue')

            if plot_config_params.get("feature_map_gridlines", True):
                gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, dms=True, x_inline=False, y_inline=False)
                gl.top_labels = False
                gl.right_labels = False

            plot_title = da.attrs.get('long_name', var_name)
            sample_info_str = ds.attrs.get('original_sample_info', "")
            try:
                sample_dict = eval(sample_info_str)
                plot_title += f"\nSample Lon: {sample_dict.get('lon', 'N/A')}, Lat: {sample_dict.get('lat', 'N/A')}"
            except:
                pass

            plt.title(plot_title)
            plt.tight_layout()

            plot_filename = os.path.join(plot_output_dir, f"{base_filename}_{var_name}.png")
            plt.savefig(plot_filename, dpi=plot_config_params.get("dpi_feature_map", 150))
            logger.debug(f"FeatureMapPlot: Saved feature map plot to {plot_filename}")
            plt.close()

    except Exception as e:
        logger.error(f"FeatureMapPlot: Error plotting feature map from {nc_file_path}: {e}")
        if plt.get_fignums():
            plt.close('all')


def plot_shap_dependence(
    shap_values,
    feature_idx,
    feature_name,
    X_test_processed,
    feature_names,
    interaction_feature_idx=None,
    save_path=None,
    show_plot=False,
    **kwargs
):
    """
    Generates and saves a SHAP dependence plot for a given feature.

    Args:
        shap_values (np.ndarray): Array of SHAP values (usually for the test set).
                                  For multi-class, this should be SHAP values for a specific class.
        feature_idx (int or str): Index or name of the feature to plot.
        feature_name (str): Name of the feature (used for title and filename).
        X_test_processed (pd.DataFrame or np.ndarray): Processed test set features.
                                                     Must have feature names if pd.DataFrame.
        feature_names (list): List of all feature names, matching X_test_processed columns.
        interaction_feature_idx (int or str, optional): Index or name of the feature to use for interaction coloring.
                                                        Defaults to None (SHAP chooses automatically).
        save_path (str or Path, optional): Path to save the plot. If None, plot is not saved.
        show_plot (bool, optional): Whether to display the plot. Defaults to False.
        **kwargs: Additional keyword arguments to pass to shap.dependence_plot().
    """
    if not isinstance(feature_idx, (int, str)):
        logger.error(f"plot_shap_dependence: feature_idx must be an int or str, got {type(feature_idx)}")
        return

    if isinstance(feature_idx, str):
        if feature_idx not in feature_names:
            logger.error(f"plot_shap_dependence: feature_name '{feature_idx}' not found in provided feature_names.")
            return


        main_feature_for_shap_api = feature_names.index(feature_idx)
    else:
        main_feature_for_shap_api = feature_idx


    try:
        if not hasattr(X_test_processed, 'columns'):
             X_display = pd.DataFrame(X_test_processed, columns=feature_names)
        else:
             X_display = X_test_processed
    except Exception as e:
        logger.error(f"plot_shap_dependence: Error preparing X_test_processed for SHAP: {e}")

        X_display = X_test_processed


    interaction_arg = interaction_feature_idx
    if interaction_feature_idx is not None:


        if isinstance(interaction_feature_idx, int):
            if 0 <= interaction_feature_idx < len(feature_names):
                interaction_arg = feature_names[interaction_feature_idx]
            else:
                logger.warning(f"plot_shap_dependence: Invalid interaction_feature_idx {interaction_feature_idx}. SHAP will choose automatically.")
                interaction_arg = None
        elif isinstance(interaction_feature_idx, str):
            if interaction_feature_idx not in feature_names:
                logger.warning(f"plot_shap_dependence: Interaction feature '{interaction_feature_idx}' not in feature_names. SHAP will choose automatically.")
                interaction_arg = None


    logger.debug(f"Generating SHAP dependence plot for feature: '{feature_name}' (index: {main_feature_for_shap_api}). Interaction: {interaction_arg}")

    try:
        plt.figure()


        shap_kwargs = {k: v for k, v in kwargs.items() if k != 'dpi'}
        shap.dependence_plot(
            ind=main_feature_for_shap_api,
            shap_values=shap_values,
            features=X_display,
            feature_names=None,
            interaction_index=interaction_arg,
            show=False,
            **shap_kwargs
        )
        plt.title(f"SHAP Dependence Plot: {feature_name}")
        plt.tight_layout()

        if save_path:

            save_path_base = Path(save_path).with_suffix('')
            save_path_base.parent.mkdir(parents=True, exist_ok=True)

            png_path = save_path_base.with_suffix('.png')
            svg_path = save_path_base.with_suffix('.svg')


            plt.savefig(png_path, dpi=kwargs.get('dpi', 300))
            plt.savefig(svg_path, format='svg')

            logger.debug(f"SHAP dependence plot saved to: {png_path} and {svg_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

    except Exception as e:
        logger.error(f"Error generating SHAP dependence plot for '{feature_name}': {e}")
        if plt.get_fignums():
            plt.close('all')

def plot_shap_force(
    explainer_expected_value,
    shap_values_for_sample,
    feature_names,
    sample_features,
    save_path=None,
    matplotlib=True,
    show_plot=False,
    **kwargs
):
    """
    Generates and saves a SHAP force plot for a single sample.

    Args:
        explainer_expected_value (float): The base value for the SHAP plot.
        shap_values_for_sample (np.ndarray): SHAP values for the single sample.
                                            Shape should be (n_features,).
        feature_names (list): List of feature names.
        sample_features (pd.Series or np.ndarray): Feature values for the single sample.
                                                   If np.ndarray, ensure order matches feature_names.
        save_path (str or Path, optional): Path to save the plot. If None, plot is not saved.
        matplotlib (bool, optional): If True, generate a matplotlib-based plot.
                                     If False, generate an HTML/JS plot (may not save directly as image).
                                     Defaults to True.
        show_plot (bool, optional): Whether to display the plot. Defaults to False.
        **kwargs: Additional keyword arguments to pass to shap.force_plot().
    """
    logger.debug(f"Generating SHAP force plot. Matplotlib mode: {matplotlib}")


    if isinstance(sample_features, np.ndarray):
        if len(sample_features.shape) == 1:
            sample_features_display = pd.Series(sample_features, index=feature_names)
        elif len(sample_features.shape) == 2 and sample_features.shape[0] == 1:
             sample_features_display = pd.Series(sample_features[0], index=feature_names)
        else:
            logger.error(f"plot_shap_force: sample_features (np.ndarray) has unexpected shape {sample_features.shape}. Expected 1D or (1, n_features).")
            return
    elif isinstance(sample_features, pd.Series):
        sample_features_display = sample_features
    elif isinstance(sample_features, pd.DataFrame):
        if sample_features.shape[0] == 1:
            sample_features_display = sample_features.iloc[0]
        else:
            logger.error(f"plot_shap_force: sample_features (pd.DataFrame) has {sample_features.shape[0]} rows. Expected 1 row for a single sample.")
            return
    else:
        logger.error(f"plot_shap_force: Unsupported type for sample_features: {type(sample_features)}. Expected pd.Series, pd.DataFrame (1 row), or np.ndarray (1D).")
        return


    if shap_values_for_sample.ndim > 1:
        if shap_values_for_sample.shape[0] == 1:
            shap_values_for_sample = shap_values_for_sample.squeeze(0)
        else:
            logger.error(f"plot_shap_force: shap_values_for_sample has unexpected shape {shap_values_for_sample.shape}. Expected 1D or (1, n_features).")
            return

    try:
        if matplotlib:


            shap_kwargs = {k: v for k, v in kwargs.items() if k != 'dpi'}
            shap.force_plot(
                base_value=explainer_expected_value,
                shap_values=shap_values_for_sample,
                features=sample_features_display,
                feature_names=feature_names,
                matplotlib=True,
                show=False,
                **shap_kwargs
            )


            if save_path:
                save_path_base = Path(save_path).with_suffix('')
                save_path_base.parent.mkdir(parents=True, exist_ok=True)
                png_path = save_path_base.with_suffix('.png')
                svg_path = save_path_base.with_suffix('.svg')

                plt.savefig(png_path, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
                plt.savefig(svg_path, format='svg', bbox_inches='tight')
                logger.debug(f"SHAP force plot (matplotlib) saved to: {png_path} and {svg_path}")

            if show_plot:
                plt.show()
            else:
                plt.close()
        else:


            shap_kwargs = {k: v for k, v in kwargs.items() if k != 'dpi'}
            force_plot_html = shap.force_plot(
                base_value=explainer_expected_value,
                shap_values=shap_values_for_sample,
                features=sample_features_display,
                feature_names=feature_names,
                show=False,
                **shap_kwargs
            )
            if save_path:

                Path(save_path).parent.mkdir(parents=True, exist_ok=True)

                if not str(save_path).lower().endswith(".html"):
                    save_path_html = Path(str(save_path).split('.')[0] + ".html")
                    logger.warning(f"Force plot (HTML) saving to {save_path_html} as .html was not specified.")
                else:
                    save_path_html = Path(save_path)

                shap.save_html(str(save_path_html), force_plot_html)
                logger.debug(f"SHAP force plot (HTML) saved to: {save_path_html}")

            if show_plot:
                shap.initjs()


                logger.debug("To display the HTML force plot, open the saved .html file in a browser or use in a JS-enabled environment.")


    except Exception as e:
        logger.error(f"Error generating SHAP force plot: {e}")
        if matplotlib and plt.get_fignums():
            plt.close('all')

def plot_shap_waterfall(
    explainer_expected_value,
    shap_values_for_sample,
    feature_names,
    sample_features_for_display=None,
    max_display=10,
    save_path=None,
    show_plot=False,
    **kwargs
):
    """
    Generates and saves a SHAP waterfall plot for a single sample.

    Args:
        explainer_expected_value (float or np.ndarray): The base value (e.g., explainer.expected_value).
                                                       If shap_values_for_sample is an Explanation object,
                                                       its .base_values attribute might be used instead by SHAP.
        shap_values_for_sample (np.ndarray or shap.Explanation):
                                SHAP values for the single sample (1D array) or a SHAP Explanation object
                                for that sample (e.g., explainer(X_sample)[0]).
                                If an Explanation object, it should contain .values, .base_values,
                                .data (optional), and .feature_names (optional if `feature_names` arg is provided).
        feature_names (list): List of feature names. Required if shap_values_for_sample is a NumPy array,
                              or if the Explanation object doesn't have .feature_names.
        sample_features_for_display (pd.Series or np.ndarray, optional):
                                Feature values for the single sample, used for display on the plot.
                                Required if shap_values_for_sample is a NumPy array and you want values displayed.
                                If shap_values_for_sample is an Explanation object, its .data attribute is used.
        max_display (int, optional): Maximum number of features to display in the plot. Defaults to 10.
        save_path (str or Path, optional): Path to save the plot. If None, plot is not saved.
        show_plot (bool, optional): Whether to display the plot. Defaults to False.
        **kwargs: Additional keyword arguments to pass to shap.waterfall_plot().
    """
    logger.debug(f"Generating SHAP waterfall plot. Max display: {max_display}")

    try:

        if not isinstance(shap_values_for_sample, shap.Explanation):
            if sample_features_for_display is None:
                logger.warning("plot_shap_waterfall: sample_features_for_display is None. Feature values will not be shown on the waterfall plot.")


            if shap_values_for_sample.ndim > 1:
                if shap_values_for_sample.shape[0] == 1:
                    shap_values_for_sample_1d = shap_values_for_sample.squeeze(0)
                else:
                    logger.error(f"plot_shap_waterfall: shap_values_for_sample (np.ndarray) has unexpected shape {shap_values_for_sample.shape}. Expected 1D or (1, n_features).")
                    return
            else:
                shap_values_for_sample_1d = shap_values_for_sample


            sample_data_1d = None
            if sample_features_for_display is not None:
                if isinstance(sample_features_for_display, np.ndarray):
                    if len(sample_features_for_display.shape) == 1:
                        sample_data_1d = sample_features_for_display
                    elif len(sample_features_for_display.shape) == 2 and sample_features_for_display.shape[0] == 1:
                        sample_data_1d = sample_features_for_display.squeeze(0)
                    else:
                        logger.error(f"plot_shap_waterfall: sample_features_for_display (np.ndarray) has unexpected shape {sample_features_for_display.shape}. Expected 1D or (1, n_features).")

                elif isinstance(sample_features_for_display, pd.Series):
                    sample_data_1d = sample_features_for_display.values
                elif isinstance(sample_features_for_display, pd.DataFrame):
                    if sample_features_for_display.shape[0] == 1:
                        sample_data_1d = sample_features_for_display.iloc[0].values
                    else:
                        logger.error(f"plot_shap_waterfall: sample_features_for_display (pd.DataFrame) has {sample_features_for_display.shape[0]} rows. Expected 1 row.")

                else:
                    logger.warning(f"plot_shap_waterfall: Unsupported type for sample_features_for_display: {type(sample_features_for_display)}.")


            shap_explanation_obj = shap.Explanation(
                values=shap_values_for_sample_1d,
                base_values=explainer_expected_value,
                data=sample_data_1d,
                feature_names=feature_names
            )
        else:
            shap_explanation_obj = shap_values_for_sample


            if not hasattr(shap_explanation_obj, 'feature_names') or shap_explanation_obj.feature_names is None:
                 if feature_names:
                     shap_explanation_obj.feature_names = feature_names
                 else:
                     logger.error("plot_shap_waterfall: feature_names not provided and not found in SHAP Explanation object.")
                     return
            if not hasattr(shap_explanation_obj, 'base_values') or shap_explanation_obj.base_values is None:
                shap_explanation_obj.base_values = explainer_expected_value


        plt.figure()


        shap_kwargs = {k: v for k, v in kwargs.items() if k != 'dpi'}
        shap.waterfall_plot(
            shap_explanation_obj,
            max_display=max_display,
            show=False,
            **shap_kwargs
        )


        if save_path:
            save_path_base = Path(save_path).with_suffix('')
            save_path_base.parent.mkdir(parents=True, exist_ok=True)
            png_path = save_path_base.with_suffix('.png')
            svg_path = save_path_base.with_suffix('.svg')

            plt.savefig(png_path, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
            plt.savefig(svg_path, format='svg', bbox_inches='tight')
            logger.debug(f"SHAP waterfall plot saved to: {png_path} and {svg_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

    except Exception as e:
        logger.error(f"Error generating SHAP waterfall plot: {e}")
        if plt.get_fignums():
            plt.close('all')

def plot_shap_decision(
    base_value,
    shap_values,
    features,
    feature_names,
    save_path=None,
    show_plot=False,
    plot_config: dict | None = None,
    **kwargs
):
    """
    Generates and saves a SHAP decision plot.

    Args:
        base_value (float): The base value from the SHAP explainer.
        shap_values (np.ndarray): Array of SHAP values for the samples to be plotted.
        features (pd.DataFrame or np.ndarray): Feature values for the samples.
        feature_names (list): List of feature names.
        save_path (str or Path, optional): Base path to save the plot (without extension).
        show_plot (bool, optional): Whether to display the plot. Defaults to False.
        **kwargs: Additional keyword arguments to pass to shap.decision_plot().
    """
    logger.debug("Generating SHAP decision plot...")
    try:
        fig = plt.figure(figsize=kwargs.get("figsize", (8.3, 6.0)))

        shap_kwargs = {k: v for k, v in kwargs.items() if k not in ["dpi", "figsize", "plot_config"]}

        with publication_style(plot_config):
            shap.decision_plot(
                base_value=base_value,
                shap_values=shap_values,
                features=features,
                feature_names=feature_names,
                show=False,
                **shap_kwargs,
            )
            plt.title(kwargs.get("title", "SHAP decision plot"))
            plt.tight_layout()

            if save_path:
                save_path_base = Path(save_path).with_suffix("")
                save_paths = save_figure(fig, path_no_ext=save_path_base, plot_config=plot_config)
                logger.debug(f"SHAP decision plot saved to: {save_paths}")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    except Exception as e:
        logger.error(f"Error generating SHAP decision plot: {e}")
        if plt.get_fignums():
            plt.close('all')

def plot_shap_heatmap(
    shap_explanation,
    save_path=None,
    show_plot=False,
    plot_config: dict | None = None,
    **kwargs
):
    """
    Generates and saves a SHAP heatmap plot.

    Args:
        shap_explanation (shap.Explanation): A SHAP Explanation object containing values, data, and feature names.
        save_path (str or Path, optional): Base path to save the plot (without extension).
        show_plot (bool, optional): Whether to display the plot. Defaults to False.
        **kwargs: Additional keyword arguments.
    """
    logger.debug("Generating SHAP heatmap plot...")
    try:
        fig = plt.figure(figsize=kwargs.get("figsize", (8.3, 6.0)))

        with publication_style(plot_config):
            shap.plots.heatmap(shap_explanation, show=False)
            plt.title(kwargs.get("title", "SHAP heatmap"))
            plt.tight_layout()

            if save_path:
                save_path_base = Path(save_path).with_suffix("")
                save_paths = save_figure(fig, path_no_ext=save_path_base, plot_config=plot_config)
                logger.debug(f"SHAP heatmap plot saved to: {save_paths}")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    except Exception as e:
        logger.error(f"Error generating SHAP heatmap plot: {e}")
        if plt.get_fignums():
            plt.close('all')

def plot_feature_clustering(
    shap_importance,
    feature_names,
    save_path=None,
    show_plot=False,
    plot_config: dict | None = None,
    **kwargs
):
    """
    Generates and saves a feature clustering dendrogram based on SHAP importance.

    Args:
        shap_importance (np.ndarray): 1D array of global SHAP importances.
        feature_names (list): List of feature names.
        save_path (str or Path, optional): Base path to save the plot (without extension).
        show_plot (bool, optional): Whether to display the plot. Defaults to False.
        **kwargs: Additional keyword arguments.
    """
    logger.debug("Generating feature clustering plot...")
    try:
        from scipy.cluster.hierarchy import dendrogram, linkage

        fig = plt.figure(figsize=kwargs.get("figsize", (8.3, 6.0)))

        shap_arr = np.asarray(shap_importance)
        method = kwargs.get("method", None)
        metric = kwargs.get("metric", None)

        if shap_arr.ndim == 1:
            method = method or "ward"
            metric = metric or "euclidean"
            Z = linkage(shap_arr.reshape(-1, 1), method=method)
            title = kwargs.get("title", "Feature clustering (SHAP importance)")
        elif shap_arr.ndim == 2:

            n_names = len(feature_names) if feature_names is not None else None
            if n_names is not None and shap_arr.shape[0] == n_names:
                feat_mat = shap_arr
            elif n_names is not None and shap_arr.shape[1] == n_names:
                feat_mat = shap_arr.T
            else:

                feat_mat = shap_arr.T


            metric = metric or "correlation"
            method = method or ("average" if metric != "euclidean" else "ward")
            if method == "ward" and metric != "euclidean":
                method = "average"

            Z = linkage(feat_mat, method=method, metric=metric)
            title = kwargs.get("title", "Feature clustering (SHAP similarity)")
        else:
            raise ValueError(f"plot_feature_clustering: expected 1D/2D input, got {shap_arr.ndim}D")

        with publication_style(plot_config):
            dendrogram(Z, labels=feature_names, orientation=kwargs.get("orientation", "left"))
            plt.title(title)
            plt.xlabel(kwargs.get("xlabel", "Distance"))
            plt.tight_layout()

            if save_path:
                save_path_base = Path(save_path).with_suffix("")
                save_paths = save_figure(fig, path_no_ext=save_path_base, plot_config=plot_config)
                logger.debug(f"Feature clustering plot saved to: {save_paths}")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    except ImportError as ie:
        logger.error(f"Error generating feature clustering plot: SciPy is required. {ie}")
    except Exception as e:
        logger.error(f"Error generating feature clustering plot: {e}")
        if plt.get_fignums():
            plt.close('all')

def plot_multi_sample_shap_decision(
    base_value,
    shap_values,
    features,
    feature_names,
    save_path=None,
    show_plot=False,
    plot_config: dict | None = None,
    **kwargs
):
    """
    Generates and saves a SHAP decision plot for multiple samples, showing their paths.

    Args:
        base_value (float): The base value from the SHAP explainer.
        shap_values (np.ndarray): Array of SHAP values for the samples to be plotted.
        features (pd.DataFrame or np.ndarray): Feature values for the samples.
        feature_names (list): List of feature names.
        save_path (str or Path, optional): Base path to save the plot (without extension).
        show_plot (bool, optional): Whether to display the plot. Defaults to False.
        **kwargs: Additional keyword arguments to pass to shap.decision_plot().
    """
    logger.debug("Generating multi-sample SHAP decision plot...")
    try:
        fig = plt.figure(figsize=kwargs.get("figsize", (8.3, 6.0)))

        shap_kwargs = {k: v for k, v in kwargs.items() if k not in ["dpi", "figsize", "title", "plot_config"]}

        with publication_style(plot_config):
            shap.decision_plot(
                base_value=base_value,
                shap_values=shap_values,
                features=features,
                feature_names=feature_names,
                show=False,
                **shap_kwargs,
            )

            plt.title(kwargs.get("title", "SHAP decision plot"))
            plt.tight_layout()

            if save_path:
                save_path_base = Path(save_path).with_suffix("")
                save_paths = save_figure(fig, path_no_ext=save_path_base, plot_config=plot_config)
                logger.debug(f"Multi-sample SHAP decision plot saved to: {save_paths}")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    except Exception as e:
        logger.error(f"Error generating multi-sample SHAP decision plot: {e}")
        if plt.get_fignums():
            plt.close('all')

def plot_multiple_shap_force_plots(
    explainer_expected_value,
    shap_values_patch,
    shap_values_depth,
    shap_patches_np,
    shap_depths_np,
    patch_feature_names,
    depth_feature_names,
    output_dir,
    num_samples=10,
    dpi=150
):
    """
    Generates and saves SHAP force plots for multiple randomly selected samples.

    Args:
        explainer_expected_value (float): The base value for the SHAP plots.
        shap_values_patch (np.ndarray): SHAP values for the patch features.
        shap_values_depth (np.ndarray): SHAP values for the depth feature.
        shap_patches_np (np.ndarray): Actual feature values for the patches.
        shap_depths_np (np.ndarray): Actual feature values for the depth.
        patch_feature_names (list): Names of the patch features.
        depth_feature_names (list): Name(s) of the depth feature.
        output_dir (str or Path): Directory to save the plots.
        num_samples (int): The number of random samples to plot.
        dpi (int): DPI for the saved plots.

    Returns:
        list: A list of paths to the saved plot images.
    """
    logger.info(f"Generating SHAP force plots for {num_samples} random samples...")
    saved_paths = []
    total_samples = shap_values_depth.shape[0]

    if total_samples == 0:
        logger.warning("No samples available to generate force plots.")
        return saved_paths


    num_to_plot = min(num_samples, total_samples)


    random_indices = np.random.choice(total_samples, size=num_to_plot, replace=False)

    for i, sample_idx in enumerate(random_indices):
        try:

            sample_patch_shaps = shap_values_patch[sample_idx].mean(axis=(1, 2))
            sample_patch_features = shap_patches_np[sample_idx].mean(axis=(1, 2))


            sample_depth_shap = shap_values_depth[sample_idx]
            sample_depth_feature = shap_depths_np[sample_idx]

            combined_shaps = np.concatenate([sample_patch_shaps, sample_depth_shap])
            combined_features = np.concatenate([sample_patch_features, sample_depth_feature])
            combined_feature_names = patch_feature_names + depth_feature_names


            force_plot_save_path = Path(output_dir) / f"shap_force_plot_sample_{sample_idx}.png"


            plot_shap_force(
                explainer_expected_value=explainer_expected_value,
                shap_values_for_sample=combined_shaps,
                feature_names=combined_feature_names,
                sample_features=combined_features,
                save_path=force_plot_save_path,
                matplotlib=True,
                show_plot=False,
                dpi=dpi
            )

            if force_plot_save_path.exists():
                saved_paths.append(str(force_plot_save_path))
                logger.debug(f"Saved force plot for sample index {sample_idx} to {force_plot_save_path}")
            else:
                logger.warning(f"Force plot for sample index {sample_idx} was not saved, likely due to an error in plot_shap_force.")

        except Exception as e:
            logger.error(f"Failed to generate force plot for sample index {sample_idx}: {e}", exc_info=True)

    if len(saved_paths) > 0:
        logger.info(f"Successfully generated {len(saved_paths)} out of {num_to_plot} requested SHAP force plots.")
    else:
        logger.error(f"Failed to generate any SHAP force plots out of {num_to_plot} requested.")

    return saved_paths


def _smooth_lowess_like_curve(
    x: np.ndarray,
    y: np.ndarray,
    lowess_frac: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a smoothed (x, y) curve with LOWESS when available."""
    x_arr = np.asarray(x, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()
    m = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[m]
    y_arr = y_arr[m]
    if x_arr.size < 5:
        return x_arr, y_arr

    order = np.argsort(x_arr)
    x_sorted = x_arr[order]
    y_sorted = y_arr[order]
    frac = float(np.clip(lowess_frac, 0.05, 1.0))

    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess

        smoothed = lowess(y_sorted, x_sorted, frac=frac, return_sorted=True)
        return np.asarray(smoothed[:, 0], dtype=float), np.asarray(smoothed[:, 1], dtype=float)
    except Exception:

        win = max(5, int(round(frac * x_sorted.size)))
        if win % 2 == 0:
            win += 1
        y_roll = (
            pd.Series(y_sorted)
            .rolling(window=win, center=True, min_periods=max(3, win // 4))
            .mean()
            .interpolate(limit_direction="both")
            .to_numpy(dtype=float)
        )
        return x_sorted, y_roll


def _optimized_group_feature_name(feature_name: str) -> str:
    """Collapse one-hot categorical features into their base variable name."""
    s = str(feature_name)
    if "__cat_" in s:
        return s.split("__cat_", 1)[0].strip() or s
    return s


_SHAP_PUBLICATION_FEATURE_LABELS = {
    "age_ka_proxy": "Estimated sediment age",
    "Water depth": "Water depth",
    "distance_to_spreading_ridge_km": "Distance to spreading ridge",
    "distance_to_trench_km": "Distance to trench",
    "del3He": "Mantle helium tracer",
    "log1p TOC per sedimentation rate": "Organic carbon burial index",
    "Seafloor lithology": "Seafloor lithology",
    "Geomorphic Features": "Seafloor geomorphic province",
    "Bottom water dissolved oxygen": "Bottom-water oxygen",
    "Summer primary productivity": "Summer primary productivity",
    "Water depth gradient": "Bathymetric slope",
    "Average sedimentation rate": "Sedimentation rate",
    "Water depth__roughness_w9": "Bathymetric roughness",
    "Bottom current speed SD": "Bottom-current variability",
    "Macrobenthic biomass": "Macrobenthic biomass",
    "Surface phosphate": "Surface phosphate",
    "Total organic carbon": "Total organic carbon",
    "Sedimentation water depth index": "Sedimentation-depth index",
    "Bottom phosphate": "Bottom-water phosphate",
}


def _publication_shap_feature_label(feature_name: str) -> str:
    """Convert internal feature names into paper-facing axis labels."""
    base = _optimized_group_feature_name(str(feature_name)).strip()
    if base in _SHAP_PUBLICATION_FEATURE_LABELS:
        return _SHAP_PUBLICATION_FEATURE_LABELS[base]

    label = base
    replacements = {
        "distance_to_": "Distance to ",
        "_km": "",
        "__": " ",
        "_": " ",
        "log1p": "log-transformed",
    }
    for old, new in replacements.items():
        label = label.replace(old, new)
    label = " ".join(part for part in label.split() if part)
    return label[:1].upper() + label[1:] if label else base


def _wrap_axis_label(label: str, width: int = 28) -> str:
    return "\n".join(textwrap.wrap(str(label), width=width, break_long_words=False, break_on_hyphens=False))


def _publication_shap_category_label(category_name: str) -> str:
    labels = {
        "Tectono-Geomorphic Setting": "Tectono-geomorphic setting",
        "Sedimentary Burial & Stratigraphic Context": "Sedimentation and stratigraphy",
        "Biogeochemical Environment": "Biogeochemical environment",
        "Ocean Circulation & Transport Dynamics": "Ocean circulation and transport",
        "Benthic Ecological Response": "Benthic ecological response",
        "Other": "Other",
    }
    return labels.get(str(category_name), str(category_name))


def plot_shap_summary_optimized(
    X_df,
    shap_values_matrix,
    category_map: dict | None = None,
    colors_map: dict | None = None,
    r2: float | None = None,
    figsize: tuple[float, float] = (12, 9),
    save_path: str | Path = "optimized_viz_lowess.pdf",
    dpi: int = 300,
    lowess_frac: float = 0.3,
    top_n: int = 14,
    plot_config: dict | None = None,
    title: str | None = None,
):
    """
    SHAP summary with in-bar trend curve and category donut.

    Parameters
    ----------
    X_df : pd.DataFrame or np.ndarray
        Feature matrix of shape (n_samples, n_features).
    shap_values_matrix : np.ndarray or pd.DataFrame
        SHAP value matrix of shape (n_samples, n_features).
    category_map : dict
        Mapping: feature_name -> category_name.
    colors_map : dict
        Mapping: category_name -> color.
    r2 : float | None
        Optional model R² annotation shown in title.
    lowess_frac : float
        LOWESS smoothing fraction for trend curves.
    """
    results: dict = {"plot_paths": [], "primary_path": None}

    try:
        if isinstance(X_df, pd.DataFrame):
            feature_names = list(X_df.columns)
            X_arr = X_df.to_numpy(dtype=float, copy=False)
        else:
            X_arr = np.asarray(X_df, dtype=float)
            feature_names = [f"feature_{i}" for i in range(X_arr.shape[1])]

        shap_arr = np.asarray(shap_values_matrix, dtype=float)
        if shap_arr.ndim != 2 or X_arr.ndim != 2:
            raise ValueError("X_df and shap_values_matrix must be 2D.")

        n_rows = min(X_arr.shape[0], shap_arr.shape[0])
        n_cols = min(X_arr.shape[1], shap_arr.shape[1], len(feature_names))
        if n_rows <= 0 or n_cols <= 0:
            raise ValueError("No valid rows/columns for optimized SHAP summary.")

        X_arr = X_arr[:n_rows, :n_cols]
        shap_arr = shap_arr[:n_rows, :n_cols]
        feature_names = feature_names[:n_cols]

        category_map = dict(category_map or {})
        colors_map = dict(
            colors_map
            or {
                "Tectono-Geomorphic Setting": "#D6B85A",
                "Sedimentary Burial & Stratigraphic Context": "#7CBF9E",
                "Ocean Circulation & Transport Dynamics": "#7FA6D6",
                "Biogeochemical Environment": "#E38A8A",
                "Benthic Ecological Response": "#9C8DC2",
                "Other": "#C7C7C7",
            }
        )

        mean_abs = np.nanmean(np.abs(shap_arr), axis=0)

        grouped_indices: dict[str, list[int]] = {}
        for i, fn in enumerate(feature_names):
            grouped_indices.setdefault(_optimized_group_feature_name(fn), []).append(i)
        grouped_names = list(grouped_indices.keys())
        grouped_importance = np.asarray(
            [float(np.nansum(mean_abs[idxs])) for idxs in grouped_indices.values()],
            dtype=float,
        )

        order = np.argsort(grouped_importance)[::-1]
        n_show = max(1, min(int(top_n), order.size))
        order = order[:n_show]

        ordered_features = [grouped_names[i] for i in order]
        ordered_importance = np.asarray(grouped_importance[order], dtype=float)
        ordered_feature_labels = [_publication_shap_feature_label(name) for name in ordered_features]
        ordered_feature_labels_wrapped = [_wrap_axis_label(name, width=30) for name in ordered_feature_labels]


        cat_importance = defaultdict(float)
        for gname, idxs in grouped_indices.items():
            imp = float(np.nansum(mean_abs[idxs]))
            cat_name = category_map.get(gname)
            if cat_name is None:
                for idx in idxs:
                    fn = feature_names[idx]
                    if fn in category_map:
                        cat_name = category_map[fn]
                        break
            cat_name = str(cat_name or "Other")
            cat_importance[cat_name] += float(imp)
            colors_map.setdefault(cat_name, "#C7C7C7")

        cat_items = sorted(cat_importance.items(), key=lambda kv: kv[1], reverse=True)
        cat_labels = [k for k, _ in cat_items]
        cat_vals = np.asarray([v for _, v in cat_items], dtype=float)
        cat_cols = [colors_map.get(k, "#C7C7C7") for k in cat_labels]
        total_imp = float(np.nansum(cat_vals)) if cat_vals.size else 0.0

        with publication_style(plot_config):
            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(1, 2, width_ratios=[3.15, 1.15], wspace=0.08)
            ax_main = fig.add_subplot(gs[0, 0])
            ax_right = fig.add_subplot(gs[0, 1])
            ax_right.axis("off")

            y_positions = np.arange(n_show)
            for row_idx, group_idx in enumerate(order):
                feat_name = grouped_names[group_idx]
                member_indices = grouped_indices.get(feat_name, [])
                imp_val = float(ordered_importance[row_idx])
                cat_name = category_map.get(feat_name)
                if cat_name is None:
                    for idx in member_indices:
                        fn = feature_names[idx]
                        if fn in category_map:
                            cat_name = category_map[fn]
                            break
                cat_name = str(cat_name or "Other")
                bar_color = colors_map.get(cat_name, "#C7C7C7")

                ax_main.barh(
                    row_idx,
                    imp_val,
                    color=bar_color,
                    alpha=0.72,
                    height=0.78,
                    edgecolor="white",
                    linewidth=0.8,
                )

                if imp_val <= 0:
                    continue
                if len(member_indices) == 1:
                    feat_idx = member_indices[0]
                    xs, ys = _smooth_lowess_like_curve(
                        X_arr[:, feat_idx],
                        shap_arr[:, feat_idx],
                        lowess_frac=lowess_frac,
                    )
                else:


                    shap_group = np.nansum(shap_arr[:, member_indices], axis=1)
                    x_group = np.zeros(n_rows, dtype=float)
                    for local_i, feat_idx in enumerate(member_indices):
                        x_group += np.nan_to_num(X_arr[:, feat_idx], nan=0.0) * float(local_i)
                    xs, ys = _smooth_lowess_like_curve(x_group, shap_group, lowess_frac=lowess_frac)
                if xs.size < 3:
                    continue
                x_min, x_max = np.nanmin(xs), np.nanmax(xs)
                y_min, y_max = np.nanmin(ys), np.nanmax(ys)
                x_span = max(x_max - x_min, 1e-12)
                y_span = max(y_max - y_min, 1e-12)
                xs_norm = ((xs - x_min) / x_span) * imp_val
                ys_norm = row_idx + ((ys - y_min) / y_span) * 0.52 - 0.26
                ax_main.plot(xs_norm, ys_norm, color="#4A4A4A", lw=1.5, alpha=0.95)

            ax_main.set_yticks(y_positions)
            ax_main.set_yticklabels(ordered_feature_labels_wrapped, fontsize=10.5)
            ax_main.invert_yaxis()
            ax_main.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x * 1000.0:.1f}"))
            ax_main.set_xlabel(r"Mean absolute SHAP value ($10^{-3}$)")
            ax_main.grid(axis="x", alpha=0.25, linestyle="--", linewidth=0.7)
            ax_main.spines["top"].set_visible(False)
            ax_main.spines["right"].set_visible(False)

            if title:
                ax_main.set_title(str(title))
            elif r2 is not None:
                ax_main.set_title(rf"$R^2$ = {float(r2):.3f} (N = {n_rows})")
            else:
                ax_main.set_title(f"SHAP + Marginal Trend Summary (N = {n_rows})")

            if cat_vals.size > 0 and total_imp > 0:
                ax_donut = ax_right.inset_axes([0.03, 0.37, 0.94, 0.56])
                wedges, _ = ax_donut.pie(
                    cat_vals,
                    colors=cat_cols,
                    startangle=90,
                    counterclock=False,
                    wedgeprops={"width": 0.34, "edgecolor": "white"},
                )
                ax_donut.set_aspect("equal")
                ax_donut.set_xticks([])
                ax_donut.set_yticks([])

                for wi, (wedge, val) in enumerate(zip(wedges[:3], cat_vals[:3])):
                    pct = 100.0 * float(val) / total_imp
                    ang = np.deg2rad((wedge.theta1 + wedge.theta2) / 2.0)
                    ax_donut.text(
                        1.14 * np.cos(ang),
                        1.14 * np.sin(ang),
                        f"{pct:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=10,
                        weight="bold",
                    )

                ax_leg = ax_right.inset_axes([0.04, 0.02, 0.92, 0.30])
                ax_leg.axis("off")
                y0 = 0.86
                for i, (lab, col) in enumerate(zip(cat_labels, cat_cols)):
                    y = y0 - i * 0.26
                    if y < 0.02:
                        break
                    ax_leg.add_patch(
                        plt.Rectangle((0.02, y - 0.06), 0.30, 0.12, color=col, transform=ax_leg.transAxes, clip_on=False)
                    )
                    ax_leg.text(
                        0.35,
                        y,
                        _publication_shap_category_label(lab),
                        transform=ax_leg.transAxes,
                        va="center",
                        fontsize=10,
                    )

            fig.tight_layout()

            path_obj = Path(save_path)
            explicit_ext = path_obj.suffix.lower().lstrip(".")
            if explicit_ext in {"png", "pdf", "svg"}:
                save_paths = save_figure(
                    fig,
                    path_no_ext=path_obj.with_suffix(""),
                    plot_config=plot_config,
                    dpi=dpi,
                    formats=[explicit_ext],
                )
            else:
                save_paths = save_figure(
                    fig,
                    path_no_ext=path_obj.with_suffix(""),
                    plot_config=plot_config,
                    dpi=dpi,
                )
            results["plot_paths"] = save_paths
            results["primary_path"] = next((p for p in save_paths if p.lower().endswith(".png")), save_paths[0])
            logger.info(f"Optimized SHAP summary saved to: {save_paths}")
            plt.close(fig)

    except Exception as e:
        logger.error(f"Failed to generate optimized SHAP summary plot: {e}", exc_info=True)

    return results

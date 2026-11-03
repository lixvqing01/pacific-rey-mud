
import os
import numpy as np
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, ListedColormap, BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from core.utils.logger_config import logger
import xarray as xr
from typing import Optional, Tuple


def _safe_level_tag(level_value: float) -> str:
    try:
        s = f"{float(level_value):g}"
    except Exception:
        s = str(level_value)
    return s.replace(".", "p").replace("-", "neg")


def _normalize_lon_for_plot(
    lon_values: np.ndarray, data: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Normalize longitudes for stable pcolormesh plotting.

    Returns (lon_plot, data_reordered, used_0_360).
    """
    lon_vals = np.asarray(lon_values, dtype=float)
    data_vals = np.asarray(data)

    if lon_vals.ndim != 1 or data_vals.ndim != 2:
        return lon_vals, data_vals, False

    if data_vals.shape[1] != lon_vals.shape[0]:
        return lon_vals, data_vals, False

    is_monotonic = bool(np.all(np.diff(lon_vals) >= 0))
    if is_monotonic:
        return lon_vals, data_vals, False


    # Convert to 0..360 to restore monotonicity without introducing a large gap.
    lon_0360 = np.where(lon_vals < 0, lon_vals + 360.0, lon_vals)
    if np.all(np.diff(lon_0360) >= 0):
        return lon_0360, data_vals, True

    order = np.argsort(lon_0360)
    return lon_0360[order], data_vals[:, order], True


def _normalize_extent_lon(
    lon_min: float, lon_max: float, *, used_0_360: bool
) -> Tuple[float, float]:
    if lon_min is None or lon_max is None:
        return lon_min, lon_max
    if not used_0_360:
        return float(lon_min), float(lon_max)

    a = float(lon_min)
    b = float(lon_max)
    if a < 0:
        a += 360.0
    if b < 0:
        b += 360.0
    if b < a:
        b += 360.0
    return a, b


def plot_rey_map_by_level(
    rey_data_2d: xr.DataArray,
    level_value: float,
    output_dir: str,
    plot_config_params: dict,
    user_lat_min: float,
    user_lat_max: float,
    user_lon_min: float,
    user_lon_max: float,
    *,
    level_name: str,
    level_unit: str,
    level_label: Optional[str] = None,
    filename_prefix: str = "rey_map_level",
    colorbar_label: str = "Predicted REY (ppm)",
    title_prefix: str = "Predicted REY",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: Optional[str] = None,
    center_zero: bool = False,
):
    """
    Generic 2D map plotter for a vertical level (e.g., depth_mbsf or age_ka).

    This is intentionally similar to plot_rey_map_by_depth, but avoids hard-coding
    depth semantics into filenames and titles.
    """
    try:
        logger.debug(f"plot_rey_map_by_level called with {level_name}={level_value}{level_unit}")
        plt.figure(figsize=plot_config_params.get("figsize_pred_map", (12, 8)))
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))

        valid_data = rey_data_2d.data[~np.isnan(rey_data_2d.data)]
        if valid_data.size == 0:
            logger.warning(f"MapPlot: No valid data for {level_name}={level_value}{level_unit}, skipping map plot.")
            plt.close()
            return None

        if vmin is None:
            vmin = np.percentile(valid_data, plot_config_params.get("pred_map_qmin", 2)) if valid_data.size > 0 else None
        if vmax is None:
            vmax = np.percentile(valid_data, plot_config_params.get("pred_map_qmax", 98)) if valid_data.size > 0 else None
        if vmin is not None and vmax is not None and vmin == vmax:
            vmin -= 0.1 * abs(vmin) if vmin != 0 else 0.1
            vmax += 0.1 * abs(vmax) if vmax != 0 else 0.1
            if vmin == vmax:
                vmin, vmax = -0.1, 0.1

        if cmap is None:
            cmap = plot_config_params.get("pred_map_cmap", "viridis")

        if 'longitude' in rey_data_2d.coords:
            lon = rey_data_2d.coords['longitude']
        elif 'lon' in rey_data_2d.coords:
            lon = rey_data_2d.coords['lon']
        else:
            raise KeyError("Could not find a longitude coordinate named 'longitude' or 'lon'.")

        if 'latitude' in rey_data_2d.coords:
            lat = rey_data_2d.coords['latitude']
        elif 'lat' in rey_data_2d.coords:
            lat = rey_data_2d.coords['lat']
        else:
            raise KeyError("Could not find a latitude coordinate named 'latitude' or 'lat'.")

        data = rey_data_2d.data
        if np.any(data < 0):
            logger.debug(f"Original data has min value {np.nanmin(data)}. Masking negative values.")
            data = np.where(data < 0, np.nan, data)

        lon_plot, data_plot, used_0_360 = _normalize_lon_for_plot(np.asarray(lon), data)
        norm = None
        if center_zero and vmin is not None and vmax is not None:
            norm = TwoSlopeNorm(vmin=float(vmin), vcenter=0.0, vmax=float(vmax))
        mesh = ax.pcolormesh(
            lon_plot,
            np.asarray(lat),
            data_plot,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=None if norm is not None else vmin,
            vmax=None if norm is not None else vmax,
            norm=norm,
            shading="auto",
        )
        plt.colorbar(mesh, ax=ax, label=colorbar_label, shrink=0.7)

        if plot_config_params.get("pred_map_coastlines", True):
            ax.add_feature(cfeature.COASTLINE)
        if plot_config_params.get("pred_map_borders", True):
            ax.add_feature(cfeature.BORDERS, linestyle=':')
        if plot_config_params.get("pred_map_land", True):
            ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray')
        if plot_config_params.get("pred_map_ocean", True):
            ax.add_feature(cfeature.OCEAN, facecolor='aliceblue')

        if plot_config_params.get("pred_map_gridlines", True):
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False

        if level_name == "Summary":
            plt.title(f"{title_prefix}")
        else:
            label = level_label if level_label is not None else f"{level_value:g} {level_unit}"
            plt.title(f"{title_prefix} at {level_name} = {label}")

        plt.tight_layout()

        if user_lon_min is not None and user_lon_max is not None and user_lat_min is not None and user_lat_max is not None:
            e_lon_min, e_lon_max = _normalize_extent_lon(user_lon_min, user_lon_max, used_0_360=used_0_360)
            extent_to_set = [e_lon_min, e_lon_max, user_lat_min, user_lat_max]
        else:
            valid_data_indices = np.where(
                ~np.isnan(data_plot) & (data_plot > plot_config_params.get("pred_map_valid_threshold", -1))
            )
            if valid_data_indices[0].size > 0:
                min_lat_idx, max_lat_idx = valid_data_indices[0].min(), valid_data_indices[0].max()
                min_lon_idx, max_lon_idx = valid_data_indices[1].min(), valid_data_indices[1].max()

                auto_lat_min = float(np.asarray(lat)[min_lat_idx])
                auto_lat_max = float(np.asarray(lat)[max_lat_idx])
                auto_lon_min = float(np.asarray(lon_plot)[min_lon_idx])
                auto_lon_max = float(np.asarray(lon_plot)[max_lon_idx])

                lat_pad = (auto_lat_max - auto_lat_min) * 0.1
                lon_pad = (auto_lon_max - auto_lon_min) * 0.1

                extent_to_set = [
                    auto_lon_min - lon_pad,
                    auto_lon_max + lon_pad,
                    max(auto_lat_min - lat_pad, float(np.asarray(lat).min())),
                    min(auto_lat_max + lat_pad, float(np.asarray(lat).max())),
                ]
            else:
                logger.warning("No valid data points found to automatically determine extent. Defaulting to global.")
                extent_to_set = None

        if extent_to_set:
            ax.set_extent(extent_to_set, crs=ccrs.PlateCarree())

        filename = f"{filename_prefix}_{_safe_level_tag(level_value)}.png"
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=plot_config_params.get("dpi_pred_map", 150))
        logger.debug(f"MapPlot: Saved map plot to {output_path}")
        plt.close()
        return output_path

    except Exception as e:
        logger.error(f"MapPlot: Error plotting map for {level_name}={level_value}{level_unit}: {e}")
        if plt.get_fignums():
            plt.close('all')
        return None


def plot_rey_map_2d(
    rey_data_2d: xr.DataArray,
    *,
    output_dir: str,
    plot_config_params: dict,
    user_lat_min: float,
    user_lat_max: float,
    user_lon_min: float,
    user_lon_max: float,
    filename_prefix: str,
    colorbar_label: str,
    title: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: Optional[str] = None,
    center_zero: bool = False,
):
    """
    2D map plotter for summary products (no vertical dimension).
    """
    try:
        plt.figure(figsize=plot_config_params.get("figsize_pred_map", (12, 8)))
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))

        data = np.asarray(rey_data_2d.data)
        if np.any(data < 0):
            data = np.where(data < 0, np.nan, data)

        if "longitude" in rey_data_2d.coords:
            lon = rey_data_2d.coords["longitude"]
        elif "lon" in rey_data_2d.coords:
            lon = rey_data_2d.coords["lon"]
        else:
            raise KeyError("Could not find a longitude coordinate named 'longitude' or 'lon'.")

        if "latitude" in rey_data_2d.coords:
            lat = rey_data_2d.coords["latitude"]
        elif "lat" in rey_data_2d.coords:
            lat = rey_data_2d.coords["lat"]
        else:
            raise KeyError("Could not find a latitude coordinate named 'latitude' or 'lat'.")

        valid_data = data[~np.isnan(data)]
        if valid_data.size == 0:
            logger.warning("SummaryMapPlot: No valid data, skipping map plot.")
            plt.close()
            return None

        if vmin is None:
            vmin = float(np.percentile(valid_data, plot_config_params.get("pred_map_qmin", 2)))
        if vmax is None:
            vmax = float(np.percentile(valid_data, plot_config_params.get("pred_map_qmax", 98)))
        if vmin == vmax:
            vmin -= 0.1 * abs(vmin) if vmin != 0 else 0.1
            vmax += 0.1 * abs(vmax) if vmax != 0 else 0.1
            if vmin == vmax:
                vmin, vmax = -0.1, 0.1

        if cmap is None:
            cmap = plot_config_params.get("pred_map_cmap", "viridis")

        lon_plot, data_plot, used_0_360 = _normalize_lon_for_plot(np.asarray(lon), data)
        norm = None
        if center_zero:
            norm = TwoSlopeNorm(vmin=float(vmin), vcenter=0.0, vmax=float(vmax))

        mesh = ax.pcolormesh(
            lon_plot,
            np.asarray(lat),
            data_plot,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=None if norm is not None else vmin,
            vmax=None if norm is not None else vmax,
            norm=norm,
            shading="auto",
        )
        plt.colorbar(mesh, ax=ax, label=colorbar_label, shrink=0.7)

        if plot_config_params.get("pred_map_coastlines", True):
            ax.add_feature(cfeature.COASTLINE)
        if plot_config_params.get("pred_map_borders", True):
            ax.add_feature(cfeature.BORDERS, linestyle=":")
        if plot_config_params.get("pred_map_land", True):
            ax.add_feature(cfeature.LAND, edgecolor="black", facecolor="lightgray")
        if plot_config_params.get("pred_map_ocean", True):
            ax.add_feature(cfeature.OCEAN, facecolor="aliceblue")

        if plot_config_params.get("pred_map_gridlines", True):
            gl = ax.gridlines(
                crs=ccrs.PlateCarree(),
                draw_labels=True,
                linewidth=1,
                color="gray",
                alpha=0.5,
                linestyle="--",
            )
            gl.top_labels = False
            gl.right_labels = False

        plt.title(title)
        plt.tight_layout()

        if user_lon_min is not None and user_lon_max is not None and user_lat_min is not None and user_lat_max is not None:
            e_lon_min, e_lon_max = _normalize_extent_lon(user_lon_min, user_lon_max, used_0_360=used_0_360)
            ax.set_extent([e_lon_min, e_lon_max, user_lat_min, user_lat_max], crs=ccrs.PlateCarree())

        filename = f"{filename_prefix}.png"
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=plot_config_params.get("dpi_pred_map", 150))
        plt.close()
        return output_path
    except Exception as e:
        logger.error(f"SummaryMapPlot: Error plotting summary map '{filename_prefix}': {e}")
        if plt.get_fignums():
            plt.close("all")
        return None


def plot_categorical_map_2d(
    categories_2d: xr.DataArray,
    *,
    categories: list[str],
    output_dir: str,
    plot_config_params: dict,
    user_lat_min: float,
    user_lat_max: float,
    user_lon_min: float,
    user_lon_max: float,
    filename_prefix: str,
    title: str,
    colors: Optional[list[str]] = None,
):
    """
    2D categorical map plotter (integer-coded classes) with discrete legend.
    """
    try:
        plt.figure(figsize=plot_config_params.get("figsize_pred_map", (12, 8)))
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))

        data = np.asarray(categories_2d.data, dtype=float)

        data = np.where(data < 0, np.nan, data)
        valid = np.isfinite(data)
        if not np.any(valid):
            logger.warning("CategoricalMapPlot: No valid data, skipping.")
            plt.close()
            return None

        if "longitude" in categories_2d.coords:
            lon = categories_2d.coords["longitude"]
        elif "lon" in categories_2d.coords:
            lon = categories_2d.coords["lon"]
        else:
            raise KeyError("Could not find a longitude coordinate named 'longitude' or 'lon'.")

        if "latitude" in categories_2d.coords:
            lat = categories_2d.coords["latitude"]
        elif "lat" in categories_2d.coords:
            lat = categories_2d.coords["lat"]
        else:
            raise KeyError("Could not find a latitude coordinate named 'latitude' or 'lat'.")

        n = int(len(categories))
        if colors is None:

            colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]
        cmap = ListedColormap(colors[:n], name="categorical_classes")
        cmap.set_bad("white")
        bounds = np.arange(-0.5, n + 0.5, 1.0)
        norm = BoundaryNorm(bounds, ncolors=n, clip=True)

        lon_plot, data_plot, used_0_360 = _normalize_lon_for_plot(np.asarray(lon), data)
        data_plot = np.ma.array(data_plot, mask=~np.isfinite(data_plot))
        mesh = ax.pcolormesh(
            lon_plot,
            np.asarray(lat),
            data_plot,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=norm,
            shading="auto",
        )
        cbar = plt.colorbar(mesh, ax=ax, shrink=0.7, ticks=np.arange(n))
        cbar.set_label("Class")
        cbar.ax.set_yticklabels(categories)

        if plot_config_params.get("pred_map_coastlines", True):
            ax.add_feature(cfeature.COASTLINE)
        if plot_config_params.get("pred_map_borders", True):
            ax.add_feature(cfeature.BORDERS, linestyle=":")
        if plot_config_params.get("pred_map_land", True):
            ax.add_feature(cfeature.LAND, edgecolor="black", facecolor="lightgray")
        if plot_config_params.get("pred_map_ocean", True):
            ax.add_feature(cfeature.OCEAN, facecolor="aliceblue")

        if plot_config_params.get("pred_map_gridlines", True):
            gl = ax.gridlines(
                crs=ccrs.PlateCarree(),
                draw_labels=True,
                linewidth=1,
                color="gray",
                alpha=0.5,
                linestyle="--",
            )
            gl.top_labels = False
            gl.right_labels = False

        plt.title(title)
        plt.tight_layout()

        if user_lon_min is not None and user_lon_max is not None and user_lat_min is not None and user_lat_max is not None:
            e_lon_min, e_lon_max = _normalize_extent_lon(user_lon_min, user_lon_max, used_0_360=used_0_360)
            ax.set_extent([e_lon_min, e_lon_max, user_lat_min, user_lat_max], crs=ccrs.PlateCarree())

        filename = f"{filename_prefix}.png"
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=plot_config_params.get("dpi_pred_map", 150))
        plt.close()
        return output_path
    except Exception as e:
        logger.error(f"CategoricalMapPlot: Error plotting '{filename_prefix}': {e}")
        if plt.get_fignums():
            plt.close("all")
        return None

def plot_rey_map_by_depth(
    rey_data_2d: xr.DataArray,
    depth_level: float,
    output_dir: str,
    plot_config_params: dict,
    user_lat_min: float,
    user_lat_max: float,
    user_lon_min: float,
    user_lon_max: float,
    colorbar_label: str = "Predicted REY (ppm)",
    title_prefix: str = "Predicted REY",
    depth_label: str = None
):
    """
    Plots a 2D REY concentration map for a specific depth level.

    Args:
        rey_data_2d (xr.DataArray): 2D DataArray of REY predictions for the depth level.
                                   Must have 'latitude' and 'longitude' coordinates.
        depth_level (float): The depth value for this map.
        output_dir (str): Directory to save the plot image.
        plot_config_params (dict): Plotting parameters (e.g., dpi, cmap, coastlines, gridlines).
        user_lat_min (float): User-defined minimum latitude.
        user_lat_max (float): User-defined maximum latitude.
        user_lon_min (float): User-defined minimum longitude.
        user_lon_max (float): User-defined maximum longitude.
    """
    try:
        logger.debug(f"plot_rey_map_by_depth called with depth_level={depth_level}")
        plt.figure(figsize=plot_config_params.get("figsize_pred_map", (12, 8)))

        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))

        valid_data = rey_data_2d.data[~np.isnan(rey_data_2d.data)]
        if valid_data.size == 0:
            logger.warning(f"DepthMapPlot: No valid data for depth {depth_level}m, skipping map plot.")
            plt.close()
            return None

        vmin = np.percentile(valid_data, plot_config_params.get("pred_map_qmin", 2)) if valid_data.size > 0 else None
        vmax = np.percentile(valid_data, plot_config_params.get("pred_map_qmax", 98)) if valid_data.size > 0 else None
        if vmin is not None and vmax is not None and vmin == vmax:
            vmin -= 0.1 * abs(vmin) if vmin != 0 else 0.1
            vmax += 0.1 * abs(vmax) if vmax != 0 else 0.1
            if vmin == vmax: vmin, vmax = -0.1, 0.1

        cmap = plot_config_params.get("pred_map_cmap", "viridis")


        if 'longitude' in rey_data_2d.coords:
            lon = rey_data_2d.coords['longitude']
        elif 'lon' in rey_data_2d.coords:
            lon = rey_data_2d.coords['lon']
        else:
            raise KeyError("Could not find a longitude coordinate named 'longitude' or 'lon'.")

        if 'latitude' in rey_data_2d.coords:
            lat = rey_data_2d.coords['latitude']
        elif 'lat' in rey_data_2d.coords:
            lat = rey_data_2d.coords['lat']
        else:
            raise KeyError("Could not find a latitude coordinate named 'latitude' or 'lat'.")
        data = rey_data_2d.data


        if np.any(data < 0):
            logger.debug(f"Original data has min value {np.nanmin(data)}. Masking negative values.")
            data = np.where(data < 0, np.nan, data)


        data_min = np.nanmin(data)
        data_max = np.nanmax(data)
        data_mean = np.nanmean(data)
        has_nan = np.isnan(data).any()
        nan_count = np.isnan(data).sum()
        total_count = data.size

        logger.debug(f"Data for plotting (after masking negatives): min={data_min}, max={data_max}, mean={data_mean}, has_nan={has_nan}, nan_count={nan_count}/{total_count}")

        logger.debug(f"Calling imshow. lon.shape={lon.shape}, lat.shape={lat.shape}, data.shape={data.shape}")
        extent = [lon.min(), lon.max(), lat.min(), lat.max()]
        logger.debug(f"Calculated extent for imshow: {extent}")
        logger.debug(f"Calculated vmin: {vmin}, vmax: {vmax}")


        lon_2d, lat_2d = np.meshgrid(lon, lat)

        mesh = ax.pcolormesh(lon_2d, lat_2d, data, transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(mesh, ax=ax, label=colorbar_label, shrink=0.7)


        logger.debug(f"plot_rey_map_by_depth received user bounds: lon_min={user_lon_min}, lon_max={user_lon_max}, lat_min={user_lat_min}, lat_max={user_lat_max}")


        if plot_config_params.get("pred_map_coastlines", True):
            ax.add_feature(cfeature.COASTLINE)
        if plot_config_params.get("pred_map_borders", True):
            ax.add_feature(cfeature.BORDERS, linestyle=':')
        if plot_config_params.get("pred_map_land", True):
            ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray')
        if plot_config_params.get("pred_map_ocean", True):
            ax.add_feature(cfeature.OCEAN, facecolor='aliceblue')

        if plot_config_params.get("pred_map_gridlines", True):
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False

        depth_label_final = depth_label if depth_label is not None else f"{depth_level}m Depth"
        plt.title(f"{title_prefix} at {depth_label_final}")


        plt.tight_layout()


        if user_lon_min is not None and user_lon_max is not None and \
           user_lat_min is not None and user_lat_max is not None:
            extent_to_set = [user_lon_min, user_lon_max, user_lat_min, user_lat_max]
            logger.debug(f"Using user-defined extent: {extent_to_set}")
        else:

            valid_data_indices = np.where(~np.isnan(data) & (data > plot_config_params.get("pred_map_valid_threshold", -1)))
            if valid_data_indices[0].size > 0:
                min_lat_idx, max_lat_idx = valid_data_indices[0].min(), valid_data_indices[0].max()
                min_lon_idx, max_lon_idx = valid_data_indices[1].min(), valid_data_indices[1].max()

                auto_lat_min = lat[min_lat_idx]
                auto_lat_max = lat[max_lat_idx]
                auto_lon_min = lon[min_lon_idx]
                auto_lon_max = lon[max_lon_idx]


                lat_pad = (auto_lat_max - auto_lat_min) * 0.1
                lon_pad = (auto_lon_max - auto_lon_min) * 0.1

                extent_to_set = [
                    max(auto_lon_min - lon_pad, lon.min()),
                    min(auto_lon_max + lon_pad, lon.max()),
                    max(auto_lat_min - lat_pad, lat.min()),
                    min(auto_lat_max + lat_pad, lat.max())
                ]
                logger.debug(f"Automatically calculated extent with padding: {extent_to_set}")
            else:
                logger.warning("No valid data points found to automatically determine extent. Defaulting to global.")
                extent_to_set = None


        if extent_to_set:
            logger.debug(f"Setting final extent to: {extent_to_set}")
            ax.set_extent(extent_to_set, crs=ccrs.Geodetic())
        else:
            logger.debug("No extent to set, map will be global.")

        filename = f"rey_map_depth_{depth_level}m.png"
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=plot_config_params.get("dpi_pred_map", 150))
        logger.debug(f"DepthMapPlot: Saved REY prediction map to {output_path}")
        plt.close()
        return output_path

    except Exception as e:
        logger.error(f"DepthMapPlot: Error plotting REY map for depth {depth_level}m: {e}")
        if plt.get_fignums(): plt.close('all')
        return None

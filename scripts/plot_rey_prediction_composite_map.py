from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from core.utils.netcdf_io import open_dataset
from matplotlib.colors import LightSource, LinearSegmentedColormap, ListedColormap, LogNorm, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.ndimage import binary_closing, binary_fill_holes, binary_opening, distance_transform_edt, gaussian_filter, label

try:
    from global_land_mask import globe

    HAS_GLOBAL_LAND_MASK = True
except Exception:
    globe = None
    HAS_GLOBAL_LAND_MASK = False

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

    HAS_CARTOPY = True
except Exception:
    ccrs = None
    cfeature = None
    LatitudeFormatter = None
    LongitudeFormatter = None
    HAS_CARTOPY = False


DEFAULT_PREDICTION_DIR = Path("products")
DEFAULT_ENV_NC = Path("data/env_gridded.nc")
DEFAULT_POINTS_PARQUET = Path("data/geochem_class_table.parquet")
DEFAULT_BOUNDARIES_GMT = Path("data/boundaries.gmt")
DEFAULT_TOPOGRAPHY_NC = Path("data/env_gridded.nc")
DEFAULT_DISTANCE_MAX_KM = 5000.0
DEFAULT_MIN_WATER_DEPTH_M = 3500.0
DEFAULT_APPEARANCE_PROB_THRESHOLD = 0.7
MECHANISM_NAMES = [
    "carbonate_background",
    "class1_enriched",
    "class2_slow_sedimentation",
]
MECHANISM_COLORS = {
    0: "#8d8d8d",
    1: "#b94743",
    2: "#2f75b5",
}
CLASS_P1000_BLUE_CMAP = LinearSegmentedColormap.from_list(
    "class1_p1000_blue",
    ["#d9ecff", "#8fc3ee", "#3b84c6", "#084a91"],
)
CLASS_P1000_RED_CMAP = LinearSegmentedColormap.from_list(
    "class2_p1000_red",
    ["#ffe0dc", "#f5a199", "#d7554f", "#8f1515"],
)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _coord_names(ds: xr.Dataset) -> tuple[str, str]:
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    if lat_name not in ds.coords or lon_name not in ds.coords:
        raise KeyError(f"Could not find latitude/longitude coordinates. Coords={list(ds.coords)}")
    return lat_name, lon_name


def _as_2d(ds: xr.Dataset, values: np.ndarray) -> np.ndarray:
    lat_name, lon_name = _coord_names(ds)
    if values.ndim == 2:
        return values
    n_lat = int(ds.sizes[lat_name])
    n_lon = int(ds.sizes[lon_name])
    return values.reshape(n_lat, n_lon)


def _extent(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray, list[float]]:
    lat_name, lon_name = _coord_names(ds)
    lats = np.asarray(ds[lat_name].values, dtype=float)
    lons = np.asarray(ds[lon_name].values, dtype=float)
    return lats, lons, [float(np.nanmin(lons)), float(np.nanmax(lons)), float(np.nanmin(lats)), float(np.nanmax(lats))]


def _load_bathymetry(env_nc: Path, target: xr.Dataset, variable: str) -> np.ndarray | None:
    if not env_nc.exists():
        return None
    with open_dataset(env_nc) as env:
        if variable not in env.data_vars:
            return None
        env_lat, env_lon = _coord_names(env)
        target_lat, target_lon = _coord_names(target)
        interp = env[variable].interp(
            {env_lat: target[target_lat], env_lon: target[target_lon]},
            kwargs={"fill_value": "extrapolate"},
        )
        return np.asarray(interp.values, dtype=float)


def _load_bathymetry_native(env_nc: Path, extent: list[float], variable: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]] | None:
    if not env_nc.exists():
        return None
    with open_dataset(env_nc) as env:
        if variable not in env.data_vars:
            return None
        lat_name, lon_name = _coord_names(env)
        lat = env[lat_name]
        lon = env[lon_name]
        subset = env[variable].where(
            (lat >= extent[2])
            & (lat <= extent[3])
            & (lon >= extent[0])
            & (lon <= extent[1]),
            drop=True,
        )
        if subset.size == 0:
            return None
        lats = np.asarray(subset[lat_name].values, dtype=float)
        lons = np.asarray(subset[lon_name].values, dtype=float)
        values = np.asarray(subset.values, dtype=float)
        native_extent = [
            float(np.nanmin(lons)),
            float(np.nanmax(lons)),
            float(np.nanmin(lats)),
            float(np.nanmax(lats)),
        ]
        return lats, lons, values, native_extent


def _load_topography_native(
    topo_nc: Path,
    extent: list[float],
    variable: str = "water_depth",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]] | None:
    if topo_nc is None or not Path(topo_nc).exists():
        return None
    with open_dataset(topo_nc) as topo:
        if variable not in topo.data_vars:
            if len(topo.data_vars) != 1:
                return None
            variable = next(iter(topo.data_vars))
        lat_name, lon_name = _coord_names(topo)
        lats_all = np.asarray(topo[lat_name].values, dtype=float)
        lons_all = np.asarray(topo[lon_name].values, dtype=float) % 360.0
        values_all = np.asarray(topo[variable].values, dtype=float)
        if values_all.ndim != 2:
            return None

        lon_order = np.argsort(lons_all)
        lons_sorted = lons_all[lon_order]
        values_sorted = values_all[:, lon_order]
        rounded_lons = np.round(lons_sorted, decimals=8)
        _, unique_idx = np.unique(rounded_lons, return_index=True)
        unique_idx = np.sort(unique_idx)
        lons_sorted = lons_sorted[unique_idx]
        values_sorted = values_sorted[:, unique_idx]

        lat_mask = (lats_all >= float(extent[2])) & (lats_all <= float(extent[3]))
        lon_mask = (lons_sorted >= float(extent[0])) & (lons_sorted <= float(extent[1]))
        if not np.any(lat_mask) or not np.any(lon_mask):
            return None
        lats = lats_all[lat_mask]
        lons = lons_sorted[lon_mask]
        values = values_sorted[np.ix_(lat_mask, lon_mask)]
        native_extent = [
            float(np.nanmin(lons)),
            float(np.nanmax(lons)),
            float(np.nanmin(lats)),
            float(np.nanmax(lats)),
        ]
        return lats, lons, values, native_extent


def _load_core_support(points_parquet: Path, extent: list[float]) -> pd.DataFrame:
    if not points_parquet.exists():
        return pd.DataFrame()
    points = pd.read_parquet(points_parquet)
    rename = {}
    if "lon" in points.columns and "longitude" not in points.columns:
        rename["lon"] = "longitude"
    if "lat" in points.columns and "latitude" not in points.columns:
        rename["lat"] = "latitude"
    if "rey" in points.columns and "rey_ppm" not in points.columns:
        rename["rey"] = "rey_ppm"
    points = points.rename(columns=rename)
    required = {"longitude", "latitude", "rey_ppm", "core_id"}
    missing = required - set(points.columns)
    if missing:
        raise KeyError(f"Point table is missing required columns for core support map: {sorted(missing)}")
    points = points.copy()
    points["longitude"] = points["longitude"].astype(float) % 360.0
    points["latitude"] = points["latitude"].astype(float)
    points["rey_ppm"] = pd.to_numeric(points["rey_ppm"], errors="coerce")
    points = points[
        (points["longitude"] >= extent[0])
        & (points["longitude"] <= extent[1])
        & (points["latitude"] >= extent[2])
        & (points["latitude"] <= extent[3])
        & points["rey_ppm"].notna()
    ].copy()
    if points.empty:
        return pd.DataFrame()
    agg = (
        points.groupby("core_id", dropna=False)
        .agg(
            longitude=("longitude", "mean"),
            latitude=("latitude", "mean"),
            n_samples=("rey_ppm", "size"),
            max_rey_ppm=("rey_ppm", "max"),
            p90_rey_ppm=("rey_ppm", lambda s: float(np.nanpercentile(s, 90))),
            n_ge1000=("rey_ppm", lambda s: int((s >= 1000.0).sum())),
            n_ge1500=("rey_ppm", lambda s: int((s >= 1500.0).sum())),
        )
        .reset_index()
    )
    agg["support_tier"] = np.where(agg["n_ge1500"] > 0, 2, np.where(agg["n_ge1000"] > 0, 1, 0)).astype(int)
    return agg


def _load_gmt_lines(path: Path | None) -> list[np.ndarray]:
    if path is None or not Path(path).exists():
        return []
    segments: list[np.ndarray] = []
    current: list[tuple[float, float]] = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(">"):
                if len(current) >= 2:
                    segments.extend(_split_wrapped_lon_segment(np.asarray(current, dtype=float)))
                current = []
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                lon = float(parts[0]) % 360.0
                lat = float(parts[1])
            except ValueError:
                continue
            current.append((lon, lat))
    if len(current) >= 2:
        segments.extend(_split_wrapped_lon_segment(np.asarray(current, dtype=float)))
    return segments


def _split_wrapped_lon_segment(segment: np.ndarray, max_jump: float = 180.0) -> list[np.ndarray]:
    if segment.shape[0] < 2:
        return []
    split_idx = np.where(np.abs(np.diff(segment[:, 0])) > float(max_jump))[0] + 1
    if split_idx.size == 0:
        return [segment]
    pieces = np.split(segment, split_idx)
    return [piece for piece in pieces if piece.shape[0] >= 2]


def _plot_gmt_lines(
    ax: plt.Axes,
    segments: list[np.ndarray],
    *,
    color: str = "#585858",
    linewidth: float = 0.42,
    alpha: float = 0.58,
    linestyle: str = "solid",
    zorder: int = 18,
) -> None:
    if not segments:
        return
    kwargs: dict[str, Any] = {"transform": ccrs.PlateCarree()} if _is_geoaxes(ax) else {}
    for segment in segments:
        ax.plot(
            segment[:, 0],
            segment[:, 1],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            linestyle=linestyle,
            solid_capstyle="round",
            zorder=zorder,
            **kwargs,
        )


def _nearest_core_distance_km(lons: np.ndarray, lats: np.ndarray, cores: pd.DataFrame, chunk_size: int = 200_000) -> np.ndarray:
    lon2, lat2 = np.meshgrid(np.asarray(lons, dtype=float), np.asarray(lats, dtype=float))
    flat_lon = lon2.ravel()
    flat_lat = lat2.ravel()
    out = np.full(flat_lon.shape, np.nan, dtype=np.float32)
    if cores.empty:
        return out.reshape(lat2.shape)

    core_lon = np.deg2rad(np.asarray(cores["longitude"].values, dtype=float))
    core_lat = np.deg2rad(np.asarray(cores["latitude"].values, dtype=float))
    earth_radius_km = 6371.0088
    for start in range(0, flat_lon.size, int(chunk_size)):
        stop = min(start + int(chunk_size), flat_lon.size)
        qlon = np.deg2rad(flat_lon[start:stop])[:, None]
        qlat = np.deg2rad(flat_lat[start:stop])[:, None]
        dlon = np.abs(qlon - core_lon[None, :])
        dlon = np.minimum(dlon, 2.0 * np.pi - dlon)
        dlat = qlat - core_lat[None, :]
        a = np.sin(dlat / 2.0) ** 2 + np.cos(qlat) * np.cos(core_lat[None, :]) * np.sin(dlon / 2.0) ** 2
        dist = 2.0 * earth_radius_km * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
        out[start:stop] = np.nanmin(dist, axis=1).astype(np.float32)
    return out.reshape(lat2.shape)


def _distance_to_land_km(lons: np.ndarray, lats: np.ndarray) -> np.ndarray | None:
    if not HAS_GLOBAL_LAND_MASK or globe is None:
        return None
    lon2, lat2 = np.meshgrid(np.asarray(lons, dtype=float), np.asarray(lats, dtype=float))
    lon_wrapped = np.where(lon2 > 180.0, lon2 - 360.0, lon2)
    try:
        land_mask = np.asarray(globe.is_land(lat2, lon_wrapped), dtype=bool)
    except Exception:
        return None
    if land_mask.size == 0 or not bool(np.any(land_mask)):
        return None
    lat_step = float(np.nanmedian(np.abs(np.diff(lats)))) if len(lats) > 1 else 0.1
    lon_step = float(np.nanmedian(np.abs(np.diff(lons)))) if len(lons) > 1 else 0.1
    mean_lat = float(np.nanmean(lats)) if len(lats) else 0.0
    lat_km = 111.32 * max(lat_step, 1e-6)
    lon_km = 111.32 * max(0.05, float(np.cos(np.deg2rad(mean_lat)))) * max(lon_step, 1e-6)
    return distance_transform_edt(~land_mask, sampling=(lat_km, lon_km)).astype(np.float32)


def _land_mask(lons: np.ndarray, lats: np.ndarray) -> np.ndarray | None:
    if not HAS_GLOBAL_LAND_MASK or globe is None:
        return None
    lon2, lat2 = np.meshgrid(np.asarray(lons, dtype=float), np.asarray(lats, dtype=float))
    lon_wrapped = np.where(lon2 > 180.0, lon2 - 360.0, lon2)
    try:
        return np.asarray(globe.is_land(lat2, lon_wrapped), dtype=bool)
    except Exception:
        return None


def _mechanism_layers(ds2: xr.Dataset, ds3: xr.Dataset | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if (
        "dominant_mechanism_class_id_0_2580ka" in ds2
        and "dominant_mechanism_confidence_0_2580ka" in ds2
        and "prob_class1_enriched_mean_0_2580ka" in ds2
        and "prob_class2_slow_sedimentation_mean_0_2580ka" in ds2
    ):
        dominant = np.asarray(ds2["dominant_mechanism_class_id_0_2580ka"].values, dtype=np.int16)
        confidence = np.asarray(ds2["dominant_mechanism_confidence_0_2580ka"].values, dtype=float)
        class1 = np.asarray(ds2["prob_class1_enriched_mean_0_2580ka"].values, dtype=float)
        class2 = np.asarray(ds2["prob_class2_slow_sedimentation_mean_0_2580ka"].values, dtype=float)
        carbonate = np.clip(1.0 - class1 - class2, 0.0, 1.0)
        probs = np.stack([carbonate, class1, class2], axis=0)
        return dominant, confidence, probs

    if ds3 is None:
        raise ValueError("Dominant mechanism fields are missing from the 2D summary and 3D prediction file was not provided.")

    required = [f"prob_{name}" for name in MECHANISM_NAMES]
    missing = [name for name in required if name not in ds3.data_vars]
    if missing:
        raise KeyError(f"Missing mechanism probability variables in 3D prediction file: {missing}")
    probs = np.stack([np.nanmean(ds3[f"prob_{name}"].values, axis=0) for name in MECHANISM_NAMES], axis=0)
    dominant = np.nanargmax(probs, axis=0).astype(np.int16)
    confidence = np.nanmax(probs, axis=0).astype(float)
    return dominant, confidence, probs


def _rgba_mechanism(dominant: np.ndarray, confidence: np.ndarray, min_prob: float) -> np.ndarray:
    rgba = np.zeros((*dominant.shape, 4), dtype=float)
    for class_id, color in MECHANISM_COLORS.items():
        rgba[dominant == class_id, :3] = to_rgba(color)[:3]
    alpha = np.clip((confidence - min_prob) / max(1.0 - min_prob, 1e-6), 0.0, 1.0)
    rgba[..., 3] = 0.15 + 0.55 * alpha
    rgba[~np.isfinite(confidence) | (confidence < min_prob), 3] = 0.0
    return rgba


def _robust_background(values: np.ndarray | None, fallback_shape: tuple[int, int]) -> np.ndarray:
    if values is None:
        return np.ones(fallback_shape, dtype=float) * 0.82
    bg = np.asarray(values, dtype=float)
    if bg.ndim != 2:
        bg = bg.reshape(fallback_shape)
    finite = np.isfinite(bg)
    if not np.any(finite):
        return np.ones(fallback_shape, dtype=float) * 0.82
    lo, hi = np.nanpercentile(bg[finite], [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(bg[finite])), float(np.nanmax(bg[finite]))
    scaled = np.clip((bg - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return 0.18 + 0.72 * scaled


def _smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    filled = np.where(finite, arr, 0.0)
    weights = gaussian_filter(finite.astype(float), sigma=sigma, mode="nearest")
    smoothed = gaussian_filter(filled, sigma=sigma, mode="nearest")
    return np.divide(smoothed, weights, out=np.full_like(smoothed, np.nan), where=weights > 1e-6)


def _finite_values(values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if mask is not None:
        finite &= np.asarray(mask, dtype=bool)
    out = arr[finite]
    if out.size == 0 and mask is not None:
        out = arr[np.isfinite(arr)]
    return out


def _clean_mask(mask: np.ndarray, min_cells: int) -> np.ndarray:
    cleaned = np.asarray(mask, dtype=bool)
    if cleaned.ndim != 2 or not np.any(cleaned):
        return cleaned
    cleaned = binary_closing(cleaned, iterations=1)
    cleaned = binary_opening(cleaned, iterations=1)
    labels, n_labels = label(cleaned)
    if n_labels == 0:
        return cleaned
    keep = np.zeros_like(cleaned, dtype=bool)
    for idx in range(1, n_labels + 1):
        region = labels == idx
        if int(region.sum()) >= int(min_cells):
            keep |= region
    return keep


def _clean_display_mask(mask: np.ndarray, min_cells: int, *, closing_iterations: int = 2) -> np.ndarray:
    cleaned = np.asarray(mask, dtype=bool)
    if cleaned.ndim != 2 or not np.any(cleaned):
        return cleaned
    cleaned = binary_closing(cleaned, iterations=max(int(closing_iterations), 1))
    cleaned = binary_fill_holes(cleaned)
    labels, n_labels = label(cleaned)
    if n_labels == 0:
        return cleaned
    keep = np.zeros_like(cleaned, dtype=bool)
    for idx in range(1, n_labels + 1):
        region = labels == idx
        if int(region.sum()) >= int(min_cells):
            keep |= region
    return keep


def _contour_mask(
    ax: plt.Axes,
    lons: np.ndarray,
    lats: np.ndarray,
    mask: np.ndarray,
    color: str,
    linewidth: float,
    linestyle: str = "solid",
) -> None:
    if np.any(mask):
        kwargs = {"transform": ccrs.PlateCarree()} if _is_geoaxes(ax) else {}
        ax.contour(
            lons,
            lats,
            mask.astype(float),
            levels=[0.5],
            colors=[color],
            linewidths=linewidth,
            linestyles=linestyle,
            **kwargs,
        )


def _is_geoaxes(ax: plt.Axes) -> bool:
    return bool(HAS_CARTOPY and hasattr(ax, "projection"))


def _map_axes(title: str, extent: list[float], *, figsize: tuple[float, float] = (8.0, 5.6), dpi: int = 220) -> tuple[plt.Figure, plt.Axes]:
    if HAS_CARTOPY:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=180))
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        xticks = np.arange(120, 291, 40)
        yticks = np.arange(-60, 61, 20)
        ax.set_xticks(xticks, crs=ccrs.PlateCarree())
        ax.set_yticks(yticks, crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
        ax.yaxis.set_major_formatter(LatitudeFormatter())
    else:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    return fig, ax


def _imshow_map(ax: plt.Axes, values: np.ndarray, extent: list[float], **kwargs: Any) -> Any:
    if _is_geoaxes(ax):
        kwargs.setdefault("transform", ccrs.PlateCarree())
    return ax.imshow(values, origin="lower", extent=extent, aspect="auto", **kwargs)


def _draw_land_terrain_and_masked_water(
    ax: plt.Axes,
    lons: np.ndarray,
    lats: np.ndarray,
    extent: list[float],
    domain_mask: np.ndarray,
    topography_nc: Path | None = DEFAULT_TOPOGRAPHY_NC,
) -> None:
    rendered_topography = False
    topo_native = _load_topography_native(Path(topography_nc), extent) if topography_nc is not None else None
    if topo_native is not None:
        _, _, topo_values, topo_extent = topo_native
        land = np.isfinite(topo_values) & (topo_values > 0.0)
        if np.any(land):
            elev = np.where(land, topo_values, 0.0)
            finite_elev = elev[land]
            lo, hi = np.nanpercentile(finite_elev, [1.0, 99.5])
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = float(np.nanmin(finite_elev)), float(np.nanmax(finite_elev))
            scaled = np.clip((elev - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
            land_cmap = LinearSegmentedColormap.from_list(
                "paper_land_topography",
                [
                    (0.00, "#dceccf"),
                    (0.18, "#b7d58f"),
                    (0.38, "#d7cf9e"),
                    (0.62, "#b79572"),
                    (0.82, "#d8cbb6"),
                    (1.00, "#f2f1ec"),
                ],
            )
            base = land_cmap(scaled)
            shade = LightSource(azdeg=315.0, altdeg=40.0).hillshade(elev, vert_exag=0.055, dx=1.0, dy=1.0)
            rgba = np.zeros((*elev.shape, 4), dtype=float)
            rgba[..., :3] = np.clip(base[..., :3] * (0.82 + 0.18 * shade[..., None]), 0.0, 1.0)
            rgba[..., 3] = np.where(land, 0.82, 0.0)
            _imshow_map(ax, rgba, topo_extent, interpolation="bilinear", zorder=0)
            rendered_topography = True

    if _is_geoaxes(ax) and not rendered_topography:
        ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#cfe4bf", edgecolor="none", zorder=0)

    land = _land_mask(lons, lats)
    if land is not None and not _is_geoaxes(ax):
        lat2 = np.meshgrid(np.asarray(lons, dtype=float), np.asarray(lats, dtype=float))[1]
        relief = 0.52 + 0.24 * np.sin(np.deg2rad(lat2 + 18.0)) + 0.14 * np.cos(np.deg2rad(lat2 * 3.0))
        relief = np.clip(relief, 0.0, 1.0)
        land_rgba = LinearSegmentedColormap.from_list("muted_land", ["#d8efc3", "#9fcb72", "#d5c27a", "#b07a4f", "#f4f2ea"])(relief)
        land_rgba[..., 3] = np.where(land, 0.74, 0.0)
        _imshow_map(ax, land_rgba, extent, interpolation="bilinear", zorder=0)

    masked_water = ~np.asarray(domain_mask, dtype=bool)
    if land is not None:
        masked_water &= ~land
    water_rgba = np.zeros((*masked_water.shape, 4), dtype=float)
    water_rgba[masked_water] = to_rgba("white", alpha=1.0)
    _imshow_map(ax, water_rgba, extent, interpolation="nearest", zorder=1)


def _contourf_map(ax: plt.Axes, lons: np.ndarray, lats: np.ndarray, values: np.ndarray, **kwargs: Any) -> Any:
    if _is_geoaxes(ax):
        kwargs.setdefault("transform", ccrs.PlateCarree())
    return ax.contourf(lons, lats, values, **kwargs)


def _hatch_mask(
    ax: plt.Axes,
    lons: np.ndarray,
    lats: np.ndarray,
    mask: np.ndarray,
    *,
    hatch: str,
    color: str,
    linewidth: float,
    zorder: int,
) -> Any:
    if not np.any(mask):
        return None
    cs = _contourf_map(
        ax,
        lons,
        lats,
        mask.astype(float),
        levels=[0.5, 1.5],
        colors=["none"],
        hatches=[hatch],
        zorder=zorder,
    )
    collections = getattr(cs, "collections", None)
    if collections is None:
        cs.set_edgecolor(color)
        cs.set_facecolor("none")
        cs.set_linewidth(linewidth)
        cs.set_alpha(0.95)
    else:
        for collection in collections:
            collection.set_edgecolor(color)
            collection.set_facecolor("none")
            collection.set_linewidth(linewidth)
            collection.set_alpha(0.95)
    return cs


def _scatter_map(ax: plt.Axes, x: Any, y: Any, **kwargs: Any) -> Any:
    if _is_geoaxes(ax):
        kwargs.setdefault("transform", ccrs.PlateCarree())
    return ax.scatter(x, y, **kwargs)


def _scatter_mask_cells(
    ax: plt.Axes,
    lons: np.ndarray,
    lats: np.ndarray,
    mask: np.ndarray,
    *,
    color: str,
    size: float,
    alpha: float,
    zorder: int,
) -> None:
    if not np.any(mask):
        return
    yy, xx = np.where(mask)
    _scatter_map(
        ax,
        lons[xx],
        lats[yy],
        s=size,
        c=color,
        marker="o",
        linewidths=0.0,
        alpha=alpha,
        zorder=zorder,
    )


def _finish_map(
    ax: plt.Axes,
    *,
    land_facecolor: str | None = "white",
    coastline_resolution: str = "110m",
    coastline_linewidth: float = 0.45,
) -> None:
    if _is_geoaxes(ax):
        ax.gridlines(color="white", alpha=0.08, linewidth=0.24, draw_labels=False)
        if land_facecolor is not None:
            ax.add_feature(cfeature.LAND.with_scale(coastline_resolution), facecolor=land_facecolor, edgecolor="none", zorder=20)
        ax.coastlines(resolution=coastline_resolution, color="#303030", linewidth=coastline_linewidth, zorder=21)
    else:
        ax.grid(color="white", alpha=0.07, linewidth=0.22)


def _legend_upper_right(ax: plt.Axes, *args: Any, **kwargs: Any) -> Any:
    fig = ax.figure
    if not args and "handles" not in kwargs:
        handles, labels = ax.get_legend_handles_labels()
        kwargs["handles"] = handles
        kwargs["labels"] = labels
    bbox = ax.get_position()
    kwargs.setdefault("loc", "upper right")
    kwargs.setdefault("bbox_to_anchor", (bbox.x1 - 0.01, bbox.y1 - 0.01))
    kwargs.setdefault("bbox_transform", fig.transFigure)
    kwargs.setdefault("borderaxespad", 0.2)
    kwargs.setdefault("frameon", True)
    kwargs.setdefault("framealpha", 0.9)
    legend = fig.legend(*args, **kwargs)
    legend.set_zorder(1000)
    return legend


def _compact_map_legend(
    ax: plt.Axes,
    domain_handles: list[Any],
    screening_handles: list[Any],
) -> Any:
    legend = ax.legend(
        handles=[*domain_handles, *screening_handles],
        loc="upper right",
        bbox_to_anchor=(0.986, 0.985),
        ncol=1,
        fontsize=7.2,
        frameon=True,
        framealpha=0.88,
        facecolor="white",
        edgecolor="#c8c8c8",
        borderpad=0.40,
        labelspacing=0.30,
        handlelength=1.34,
        handleheight=0.64,
        handletextpad=0.50,
        borderaxespad=0.18,
    )
    legend.set_zorder(1000)
    for text in legend.get_texts():
        text.set_fontweight("regular")
    return legend


def _contour_or_scatter(
    ax: plt.Axes,
    lons: np.ndarray,
    lats: np.ndarray,
    mask: np.ndarray,
    *,
    color: str,
    linewidth: float,
    marker_size: float,
    label: str,
) -> None:
    if not np.any(mask):
        return
    if mask.ndim == 2 and mask.shape[0] > 1 and mask.shape[1] > 1:
        ax.contour(lons, lats, mask.astype(float), levels=[0.5], colors=[color], linewidths=linewidth)
        return
    yy, xx = np.where(mask)
    ax.scatter(lons[xx], lats[yy], s=marker_size, c=color, marker="o", linewidths=0, alpha=0.78, label=label)


def _top_percent_cut(values: np.ndarray, mask: np.ndarray, top_percent: float) -> float:
    vals = _finite_values(values, mask)
    if vals.size == 0:
        return float("nan")
    pct = 100.0 - float(top_percent)
    return float(np.nanpercentile(vals, np.clip(pct, 0.0, 100.0)))


def _plot_classwise_p1000_map(
    *,
    output_dir: Path,
    lons: np.ndarray,
    lats: np.ndarray,
    extent: list[float],
    dominant: np.ndarray,
    p1000: np.ndarray,
    deep_water_mask: np.ndarray,
    min_water_depth_m: float,
    hotspot_min_cells: int,
    boundary_segments: list[np.ndarray] | None = None,
    apply_water_depth_mask: bool = True,
    distance_to_land_km: np.ndarray | None = None,
    min_distance_to_land_km: float | None = None,
    top_percent: float = 10.0,
    class1_prob: np.ndarray | None = None,
    class2_prob: np.ndarray | None = None,
    class1_threshold_label: str = "P1000",
    class2_threshold_label: str = "P1000",
    title: str = "Facies-separated P1000 screening domains",
    output_stem: str = "paper_classwise_p1000_top10_map",
    artifact_prefix: str = "classwise_p1000_top10",
    stats_prefix: str = "classwise_p1000",
) -> tuple[dict[str, str], dict[str, Any]]:
    class1_prob = p1000 if class1_prob is None else np.asarray(class1_prob, dtype=float)
    class2_prob = p1000 if class2_prob is None else np.asarray(class2_prob, dtype=float)
    domain_mask = deep_water_mask if apply_water_depth_mask else np.isfinite(p1000)
    if min_distance_to_land_km is not None and distance_to_land_km is not None:
        domain_mask = domain_mask & np.isfinite(distance_to_land_km) & (distance_to_land_km >= float(min_distance_to_land_km))
    class0_mask = domain_mask & (dominant == 0) & np.isfinite(p1000)
    class1_mask = domain_mask & (dominant == 1) & np.isfinite(class1_prob)
    class2_mask = domain_mask & (dominant == 2) & np.isfinite(class2_prob)

    class1_cut = _top_percent_cut(class1_prob, class1_mask, top_percent)
    class2_cut = _top_percent_cut(class2_prob, class2_mask, top_percent)
    class1_top = class1_mask & np.isfinite(class1_cut) & (class1_prob >= class1_cut)
    class2_top = class2_mask & np.isfinite(class2_cut) & (class2_prob >= class2_cut)
    class1_top_display = class1_top
    class2_top_display = class2_top

    class1_values = np.where(class1_mask, class1_prob, np.nan)
    class2_values = np.where(class2_mask, class2_prob, np.nan)

    class1_values = class1_values.copy()
    class2_values = class2_values.copy()
    class1_display_boost = class1_top_display & class1_mask & np.isfinite(class1_cut)
    class2_display_boost = class2_top_display & class2_mask & np.isfinite(class2_cut)
    class1_values[class1_display_boost] = np.maximum(class1_values[class1_display_boost], class1_cut)
    class2_values[class2_display_boost] = np.maximum(class2_values[class2_display_boost], class2_cut)
    finite_class1 = _finite_values(class1_prob, class1_mask)
    finite_class2 = _finite_values(class2_prob, class2_mask)
    class1_vmax = float(max(0.1, np.nanpercentile(finite_class1, 98.0))) if finite_class1.size else 1.0
    class2_vmax = float(max(0.1, np.nanpercentile(finite_class2, 98.0))) if finite_class2.size else 1.0
    class1_vmax = min(max(class1_vmax, 0.1), 1.0)
    class2_vmax = min(max(class2_vmax, 0.1), 1.0)

    fig, ax = _map_axes(
        title,
        extent,
        figsize=(10.8, 6.2),
        dpi=260,
    )
    class0_values = np.where(class0_mask, 1.0, np.nan)
    _imshow_map(
        ax,
        class0_values,
        extent,
        cmap=ListedColormap(["#d9d9d9"]),
        vmin=0.0,
        vmax=1.0,
        alpha=0.72,
        interpolation="nearest",
        zorder=3,
    )
    im1 = _imshow_map(
        ax,
        class1_values,
        extent,
        cmap=CLASS_P1000_RED_CMAP,
        vmin=0.0,
        vmax=class1_vmax,
        alpha=0.94,
        interpolation="nearest",
        zorder=4,
    )
    im2 = _imshow_map(
        ax,
        class2_values,
        extent,
        cmap=CLASS_P1000_BLUE_CMAP,
        vmin=0.0,
        vmax=class2_vmax,
        alpha=0.94,
        interpolation="nearest",
        zorder=5,
    )
    _contour_mask(ax, lons, lats, class1_mask, "#bd6a64", 0.45)
    _contour_mask(ax, lons, lats, class2_mask, "#4d7fb5", 0.45)
    _plot_gmt_lines(
        ax,
        boundary_segments or [],
        color="#4f4f4f",
        linewidth=0.38,
        alpha=0.52,
        linestyle=(0, (3.0, 2.2)),
        zorder=9,
    )
    _hatch_mask(ax, lons, lats, class1_top_display, hatch="\\\\\\\\", color="#7f0000", linewidth=0.70, zorder=10)
    _hatch_mask(ax, lons, lats, class2_top_display, hatch="////", color="#063b74", linewidth=0.70, zorder=11)
    _contour_mask(ax, lons, lats, class1_top_display, "white", 4.10)
    _contour_mask(ax, lons, lats, class1_top_display, "#7f0000", 2.25)
    _contour_mask(ax, lons, lats, class2_top_display, "white", 4.10)
    _contour_mask(ax, lons, lats, class2_top_display, "#063b74", 2.25)
    _finish_map(ax)

    fig.subplots_adjust(left=0.07, right=0.80, top=0.91, bottom=0.10)
    cax1 = fig.add_axes([0.835, 0.22, 0.020, 0.55])
    cax2 = fig.add_axes([0.925, 0.22, 0.020, 0.55])
    cbar1 = fig.colorbar(im1, cax=cax1, orientation="vertical")
    cbar2 = fig.colorbar(im2, cax=cax2, orientation="vertical")
    cbar1.set_label(f"Facies II {class1_threshold_label}", fontsize=8.0, labelpad=6)
    cbar2.set_label(f"Facies III {class2_threshold_label}", fontsize=8.0, labelpad=6)
    cbar1.ax.yaxis.set_ticks_position("right")
    cbar1.ax.yaxis.set_label_position("right")
    cbar2.ax.yaxis.set_ticks_position("right")
    cbar2.ax.yaxis.set_label_position("right")
    cbar1.ax.tick_params(labelsize=7.6, length=2.5, pad=2)
    cbar2.ax.tick_params(labelsize=7.6, length=2.5, pad=2)
    handles = [
        Patch(facecolor="#d9d9d9", edgecolor="#a8a8a8", label="Facies I domain"),
        Patch(facecolor="#ef8a82", edgecolor="#bd6a64", label="Facies II domain"),
        Patch(facecolor="#6aa9df", edgecolor="#4d7fb5", label="Facies III domain"),
        Patch(facecolor="none", edgecolor="#7f0000", hatch="\\\\\\\\", linewidth=1.0, label=f"Facies II top {top_percent:g}% {class1_threshold_label} >= {class1_cut:.2f}"),
        Patch(facecolor="none", edgecolor="#063b74", hatch="////", linewidth=1.0, label=f"Facies III top {top_percent:g}% {class2_threshold_label} >= {class2_cut:.2f}"),
    ]
    if boundary_segments:
        handles.append(
            Line2D([0], [0], color="#4f4f4f", linewidth=0.7, alpha=0.7, linestyle=(0, (3.0, 2.2)), label="Plate boundary")
        )
    _legend_upper_right(ax, handles=handles, fontsize=8)
    png = output_dir / f"{output_stem}.png"
    svg = output_dir / f"{output_stem}.svg"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    boundary_png = output_dir / f"{output_stem.replace('_map', '')}_plate_boundary_map.png"
    boundary_svg = output_dir / f"{output_stem.replace('_map', '')}_plate_boundary_map.svg"
    if boundary_segments:
        fig.savefig(boundary_png, bbox_inches="tight")
        fig.savefig(boundary_svg, bbox_inches="tight")
    plt.close(fig)

    artifacts = {
        f"{artifact_prefix}_png": str(png),
        f"{artifact_prefix}_svg": str(svg),
    }
    if boundary_segments:
        artifacts.update(
            {
                f"{artifact_prefix}_plate_boundary_png": str(boundary_png),
                f"{artifact_prefix}_plate_boundary_svg": str(boundary_svg),
            }
        )
    stats = {
        f"{stats_prefix}_top_percent": float(top_percent),
        f"{stats_prefix}_class1_metric": str(class1_threshold_label),
        f"{stats_prefix}_class2_metric": str(class2_threshold_label),
        f"{stats_prefix}_class1_colorbar_vmax": float(class1_vmax),
        f"{stats_prefix}_class2_colorbar_vmax": float(class2_vmax),
        f"{stats_prefix}_class0_domain_cells": int(np.nansum(class0_mask)),
        f"{stats_prefix}_class1_domain_cells": int(np.nansum(class1_mask)),
        f"{stats_prefix}_class2_domain_cells": int(np.nansum(class2_mask)),
        f"{stats_prefix}_class1_top10_cutoff": class1_cut,
        f"{stats_prefix}_class2_top10_cutoff": class2_cut,
        f"{stats_prefix}_class1_top10_cells_raw": int(np.nansum(class1_top)),
        f"{stats_prefix}_class2_top10_cells_raw": int(np.nansum(class2_top)),
        f"{stats_prefix}_class1_top10_cells_display": int(np.nansum(class1_top_display)),
        f"{stats_prefix}_class2_top10_cells_display": int(np.nansum(class2_top_display)),
        f"{stats_prefix}_water_depth_mask_applied": bool(apply_water_depth_mask),
        f"{stats_prefix}_min_water_depth_m": float(min_water_depth_m) if apply_water_depth_mask else None,
        f"{stats_prefix}_min_distance_to_land_km": (
            float(min_distance_to_land_km) if min_distance_to_land_km is not None and distance_to_land_km is not None else None
        ),
        f"{stats_prefix}_distance_to_land_available": bool(distance_to_land_km is not None),
    }
    if stats_prefix == "classwise_p1000":
        stats.update(
            {
                "classwise_p1000_top_percent": float(top_percent),
                "class1_p1000_colorbar_vmax": float(class1_vmax),
                "class2_p1000_colorbar_vmax": float(class2_vmax),
                "class0_domain_cells": int(np.nansum(class0_mask)),
                "class1_domain_cells": int(np.nansum(class1_mask)),
                "class2_domain_cells": int(np.nansum(class2_mask)),
                "class1_p1000_top10_cutoff": class1_cut,
                "class2_p1000_top10_cutoff": class2_cut,
                "class1_p1000_top10_cells_raw": int(np.nansum(class1_top)),
                "class2_p1000_top10_cells_raw": int(np.nansum(class2_top)),
                "class1_p1000_top10_cells_display": int(np.nansum(class1_top_display)),
                "class2_p1000_top10_cells_display": int(np.nansum(class2_top_display)),
                "classwise_p1000_water_depth_mask_applied": bool(apply_water_depth_mask),
                "classwise_p1000_min_water_depth_m": float(min_water_depth_m) if apply_water_depth_mask else None,
                "classwise_p1000_min_distance_to_land_km": (
                    float(min_distance_to_land_km)
                    if min_distance_to_land_km is not None and distance_to_land_km is not None
                    else None
                ),
                "classwise_p1000_distance_to_land_available": bool(distance_to_land_km is not None),
            }
        )
    return artifacts, stats


def _plot_classwise_hotspot_design_variants(
    *,
    output_dir: Path,
    lons: np.ndarray,
    lats: np.ndarray,
    extent: list[float],
    dominant: np.ndarray,
    p1000: np.ndarray,
    deep_water_mask: np.ndarray,
    min_water_depth_m: float,
    boundary_segments: list[np.ndarray] | None = None,
    distance_to_land_km: np.ndarray | None = None,
    min_distance_to_land_km: float | None = None,
    top_percent: float = 5.0,
    class1_prob: np.ndarray | None = None,
    class2_prob: np.ndarray | None = None,
    class1_threshold_label: str = "P1000",
    class2_threshold_label: str = "P800",
    output_suffix: str = "",
    title_suffix: str = "",
    artifact_prefix: str = "classwise_p1000_p800_top5",
    stats_prefix: str = "classwise_p1000_p800_top5_variant",
) -> tuple[dict[str, str], dict[str, Any]]:
    class1_prob = p1000 if class1_prob is None else np.asarray(class1_prob, dtype=float)
    class2_prob = p1000 if class2_prob is None else np.asarray(class2_prob, dtype=float)
    domain_mask = deep_water_mask & np.isfinite(p1000)
    if min_distance_to_land_km is not None and distance_to_land_km is not None:
        domain_mask = domain_mask & np.isfinite(distance_to_land_km) & (distance_to_land_km >= float(min_distance_to_land_km))

    class0_mask = domain_mask & (dominant == 0) & np.isfinite(p1000)
    class1_mask = domain_mask & (dominant == 1) & np.isfinite(class1_prob)
    class2_mask = domain_mask & (dominant == 2) & np.isfinite(class2_prob)
    class1_cut = _top_percent_cut(class1_prob, class1_mask, top_percent)
    class2_cut = _top_percent_cut(class2_prob, class2_mask, top_percent)
    class1_top = class1_mask & np.isfinite(class1_cut) & (class1_prob >= class1_cut)
    class2_top = class2_mask & np.isfinite(class2_cut) & (class2_prob >= class2_cut)

    def _draw_hotspots(ax: plt.Axes, *, hatch: bool = True) -> None:
        if hatch:
            _hatch_mask(ax, lons, lats, class1_top, hatch="\\\\\\\\", color="#7f0000", linewidth=0.62, zorder=12)
            _hatch_mask(ax, lons, lats, class2_top, hatch="////", color="#063b74", linewidth=0.62, zorder=13)
        _contour_mask(ax, lons, lats, class1_top, "white", 3.60)
        _contour_mask(ax, lons, lats, class1_top, "#7f0000", 1.90)
        _contour_mask(ax, lons, lats, class2_top, "white", 3.60)
        _contour_mask(ax, lons, lats, class2_top, "#063b74", 1.90)

    hotspot_handles = [
        Patch(
            facecolor="none",
            edgecolor="#7f0000",
            hatch="\\\\\\\\",
            linewidth=1.0,
            label=f"Facies II top {top_percent:g}% {class1_threshold_label} >= {class1_cut:.2f}",
        ),
        Patch(
            facecolor="none",
            edgecolor="#063b74",
            hatch="////",
            linewidth=1.0,
            label=f"Facies III top {top_percent:g}% {class2_threshold_label} >= {class2_cut:.2f}",
        ),
    ]
    if boundary_segments:
        hotspot_handles.append(
            Line2D([0], [0], color="#4f4f4f", linewidth=0.7, alpha=0.7, linestyle=(0, (3.0, 2.2)), label="Plate boundary")
        )

    fig, ax = _map_axes(
        f"Facies-specific screening cores only{title_suffix}",
        extent,
        figsize=(10.8, 6.2),
        dpi=260,
    )
    _plot_gmt_lines(
        ax,
        boundary_segments or [],
        color="#4f4f4f",
        linewidth=0.38,
        alpha=0.52,
        linestyle=(0, (3.0, 2.2)),
        zorder=9,
    )
    _draw_hotspots(ax, hatch=False)
    _finish_map(ax)
    outline_handles = [
        Patch(
            facecolor="none",
            edgecolor="#7f0000",
            linewidth=1.2,
            label=f"Facies II top {top_percent:g}% {class1_threshold_label} >= {class1_cut:.2f}",
        ),
        Patch(
            facecolor="none",
            edgecolor="#063b74",
            linewidth=1.2,
            label=f"Facies III top {top_percent:g}% {class2_threshold_label} >= {class2_cut:.2f}",
        ),
    ]
    if boundary_segments:
        outline_handles.append(
            Line2D([0], [0], color="#4f4f4f", linewidth=0.7, alpha=0.7, linestyle=(0, (3.0, 2.2)), label="Plate boundary")
        )
    _legend_upper_right(ax, handles=outline_handles, fontsize=8)
    hotspot_only_png = output_dir / f"paper_classwise_p1000_p800_top5_hotspots_only{output_suffix}_map.png"
    hotspot_only_svg = output_dir / f"paper_classwise_p1000_p800_top5_hotspots_only{output_suffix}_map.svg"
    fig.savefig(hotspot_only_png, bbox_inches="tight")
    fig.savefig(hotspot_only_svg, bbox_inches="tight")
    plt.close(fig)

    mechanism_rgba = np.zeros((*dominant.shape, 4), dtype=float)
    mechanism_rgba[class0_mask] = to_rgba("#d8d8d8", alpha=0.84)
    mechanism_rgba[class1_mask] = to_rgba("#df7f78", alpha=0.78)
    mechanism_rgba[class2_mask] = to_rgba("#8fbddd", alpha=0.78)

    fig, ax = _map_axes(
        f"Mechanism domains and screening cores{title_suffix}",
        extent,
        figsize=(9.8, 7.0),
        dpi=260,
    )
    _draw_land_terrain_and_masked_water(ax, lons, lats, extent, domain_mask)
    _imshow_map(
        ax,
        mechanism_rgba,
        extent,
        interpolation="nearest",
        zorder=3,
    )
    _contour_mask(ax, lons, lats, class1_mask, "#bd6a64", 0.28)
    _contour_mask(ax, lons, lats, class2_mask, "#4d7fb5", 0.28)
    _plot_gmt_lines(
        ax,
        boundary_segments or [],
        color="#4f4f4f",
        linewidth=0.38,
        alpha=0.52,
        linestyle=(0, (3.0, 2.2)),
        zorder=9,
    )
    _draw_hotspots(ax, hatch=True)
    _finish_map(ax, land_facecolor=None, coastline_resolution="10m", coastline_linewidth=0.24)
    domain_handles = [
        Patch(facecolor="white", edgecolor="#bdbdbd", label=f"Water depth < {min_water_depth_m:g} m"),
        Patch(facecolor="#d8d8d8", edgecolor="#a8a8a8", label="Facies I"),
        Patch(facecolor="#df7f78", edgecolor="#bd6a64", label="Facies II"),
        Patch(facecolor="#8fbddd", edgecolor="#4d7fb5", label="Facies III"),
    ]
    screening_handles = [
        Patch(
            facecolor="none",
            edgecolor="#7f0000",
            hatch="\\\\\\\\",
            linewidth=1.0,
            label=f"Facies II core ({class1_threshold_label} >= {class1_cut:.2f})",
        ),
        Patch(
            facecolor="none",
            edgecolor="#063b74",
            hatch="////",
            linewidth=1.0,
            label=f"Facies III core ({class2_threshold_label} >= {class2_cut:.2f})",
        ),
    ]
    if boundary_segments:
        screening_handles.append(
            Line2D([0], [0], color="#4f4f4f", linewidth=0.7, alpha=0.7, linestyle=(0, (3.0, 2.2)), label="Plate boundary")
        )
    _compact_map_legend(ax, domain_handles, screening_handles)
    mechanism_png = output_dir / f"paper_classwise_p1000_p800_top5_mechanism_solid{output_suffix}_map.png"
    mechanism_svg = output_dir / f"paper_classwise_p1000_p800_top5_mechanism_solid{output_suffix}_map.svg"
    fig.savefig(mechanism_png, bbox_inches="tight")
    fig.savefig(mechanism_svg, bbox_inches="tight")
    plt.close(fig)

    artifacts = {
        f"{artifact_prefix}_hotspots_only_png": str(hotspot_only_png),
        f"{artifact_prefix}_hotspots_only_svg": str(hotspot_only_svg),
        f"{artifact_prefix}_mechanism_solid_png": str(mechanism_png),
        f"{artifact_prefix}_mechanism_solid_svg": str(mechanism_svg),
    }
    stats = {
        f"{stats_prefix}_top_percent": float(top_percent),
        f"{stats_prefix}_class1_metric": str(class1_threshold_label),
        f"{stats_prefix}_class2_metric": str(class2_threshold_label),
        f"{stats_prefix}_class0_domain_cells": int(np.nansum(class0_mask)),
        f"{stats_prefix}_class1_domain_cells": int(np.nansum(class1_mask)),
        f"{stats_prefix}_class2_domain_cells": int(np.nansum(class2_mask)),
        f"{stats_prefix}_class1_cutoff": float(class1_cut),
        f"{stats_prefix}_class2_cutoff": float(class2_cut),
        f"{stats_prefix}_class1_cells": int(np.nansum(class1_top)),
        f"{stats_prefix}_class2_cells": int(np.nansum(class2_top)),
        f"{stats_prefix}_min_water_depth_m": float(min_water_depth_m),
        f"{stats_prefix}_min_distance_to_land_km": (
            float(min_distance_to_land_km) if min_distance_to_land_km is not None and distance_to_land_km is not None else None
        ),
    }
    return artifacts, stats


def _age_layer_token(age_ka: float) -> str:
    if np.isfinite(age_ka) and abs(age_ka - round(age_ka)) < 1e-6:
        return f"{int(round(age_ka)):04d}ka"
    return f"{age_ka:g}ka".replace(".", "p").replace("-", "m")


def _parse_age_layers(value: str | None) -> list[float] | None:
    if value is None or not str(value).strip():
        return None
    out: list[float] = []
    for raw in str(value).split(","):
        item = raw.strip()
        if not item:
            continue
        out.append(float(item))
    return out or None


def _plot_layerwise_classwise_mechanism_solid_map(
    *,
    output_dir: Path,
    lons: np.ndarray,
    lats: np.ndarray,
    extent: list[float],
    dominant: np.ndarray,
    p1000: np.ndarray,
    p800: np.ndarray,
    deep_water_mask: np.ndarray,
    min_water_depth_m: float,
    age_ka: float,
    top_percent: float = 5.0,
    boundary_segments: list[np.ndarray] | None = None,
    write_svg: bool = False,
) -> tuple[dict[str, str], dict[str, Any]]:
    domain_mask = deep_water_mask & np.isfinite(p1000)
    class0_mask = domain_mask & (dominant == 0)
    class1_mask = domain_mask & (dominant == 1) & np.isfinite(p1000)
    class2_mask = domain_mask & (dominant == 2) & np.isfinite(p800)
    class1_cut = _top_percent_cut(p1000, class1_mask, top_percent)
    class2_cut = _top_percent_cut(p800, class2_mask, top_percent)
    class1_top = class1_mask & np.isfinite(class1_cut) & (p1000 >= class1_cut)
    class2_top = class2_mask & np.isfinite(class2_cut) & (p800 >= class2_cut)

    def _draw_hotspots(ax: plt.Axes) -> None:
        _hatch_mask(ax, lons, lats, class1_top, hatch="\\\\\\\\", color="#7f0000", linewidth=0.62, zorder=12)
        _hatch_mask(ax, lons, lats, class2_top, hatch="////", color="#063b74", linewidth=0.62, zorder=13)
        _contour_mask(ax, lons, lats, class1_top, "white", 3.60)
        _contour_mask(ax, lons, lats, class1_top, "#7f0000", 1.90)
        _contour_mask(ax, lons, lats, class2_top, "white", 3.60)
        _contour_mask(ax, lons, lats, class2_top, "#063b74", 1.90)

    mechanism_rgba = np.zeros((*dominant.shape, 4), dtype=float)
    mechanism_rgba[class0_mask] = to_rgba("#d8d8d8", alpha=0.84)
    mechanism_rgba[class1_mask] = to_rgba("#df7f78", alpha=0.78)
    mechanism_rgba[class2_mask] = to_rgba("#8fbddd", alpha=0.78)

    fig, ax = _map_axes(
        f"Mechanism domains and screening cores ({age_ka:g} ka)",
        extent,
        figsize=(9.8, 7.0),
        dpi=260,
    )
    _draw_land_terrain_and_masked_water(ax, lons, lats, extent, domain_mask)
    _imshow_map(
        ax,
        mechanism_rgba,
        extent,
        interpolation="nearest",
        zorder=3,
    )
    _contour_mask(ax, lons, lats, class1_mask, "#bd6a64", 0.28)
    _contour_mask(ax, lons, lats, class2_mask, "#4d7fb5", 0.28)
    _plot_gmt_lines(
        ax,
        boundary_segments or [],
        color="#4f4f4f",
        linewidth=0.38,
        alpha=0.52,
        linestyle=(0, (3.0, 2.2)),
        zorder=9,
    )
    _draw_hotspots(ax)
    _finish_map(ax, land_facecolor=None, coastline_resolution="10m", coastline_linewidth=0.24)
    domain_handles = [
        Patch(facecolor="white", edgecolor="#bdbdbd", label=f"Water depth < {min_water_depth_m:g} m"),
        Patch(facecolor="#d8d8d8", edgecolor="#a8a8a8", label="Facies I"),
        Patch(facecolor="#df7f78", edgecolor="#bd6a64", label="Facies II"),
        Patch(facecolor="#8fbddd", edgecolor="#4d7fb5", label="Facies III"),
    ]
    screening_handles = [
        Patch(
            facecolor="none",
            edgecolor="#7f0000",
            hatch="\\\\\\\\",
            linewidth=1.0,
            label=f"Facies II core (P1000 >= {class1_cut:.2f})",
        ),
        Patch(
            facecolor="none",
            edgecolor="#063b74",
            hatch="////",
            linewidth=1.0,
            label=f"Facies III core (P800 >= {class2_cut:.2f})",
        ),
    ]
    if boundary_segments:
        screening_handles.append(
            Line2D([0], [0], color="#4f4f4f", linewidth=0.7, alpha=0.7, linestyle=(0, (3.0, 2.2)), label="Plate boundary")
        )
    _compact_map_legend(ax, domain_handles, screening_handles)

    token = _age_layer_token(age_ka)
    png = output_dir / f"paper_classwise_p1000_p800_top5_mechanism_solid_{token}_map.png"
    fig.savefig(png, bbox_inches="tight")
    artifacts = {"png": str(png)}
    if write_svg:
        svg = output_dir / f"paper_classwise_p1000_p800_top5_mechanism_solid_{token}_map.svg"
        fig.savefig(svg, bbox_inches="tight")
        artifacts["svg"] = str(svg)
    plt.close(fig)

    stats = {
        "age_ka": float(age_ka),
        "top_percent": float(top_percent),
        "min_water_depth_m": float(min_water_depth_m),
        "class0_domain_cells": int(np.nansum(class0_mask)),
        "class1_domain_cells": int(np.nansum(class1_mask)),
        "class2_domain_cells": int(np.nansum(class2_mask)),
        "class1_p1000_cutoff": float(class1_cut),
        "class2_p800_cutoff": float(class2_cut),
        "class1_top_cells": int(np.nansum(class1_top)),
        "class2_top_cells": int(np.nansum(class2_top)),
        **artifacts,
    }
    return artifacts, stats


def plot_layerwise_classwise_p1000_p800_maps(
    *,
    prediction_dir: Path,
    env_nc: Path,
    output_dir: Path | None = None,
    bathymetry_var: str = "Water depth",
    min_water_depth_m: float = 3500.0,
    top_percent: float = 5.0,
    boundary_gmt: Path | None = DEFAULT_BOUNDARIES_GMT,
    age_layers_ka: list[float] | None = None,
    write_svg: bool = False,
) -> dict[str, Any]:
    prediction_dir = Path(prediction_dir)
    pred3_nc = prediction_dir / "continuous_rey_mechanism_threshold_predictions_3d.nc"
    if not pred3_nc.exists():
        raise FileNotFoundError(f"Missing 3D prediction NetCDF: {pred3_nc}")
    output_dir = Path(output_dir) if output_dir else prediction_dir / "maps" / "layerwise_classwise_p1000_p800_top5_mechanism_solid"
    output_dir.mkdir(parents=True, exist_ok=True)

    boundary_segments = _load_gmt_lines(boundary_gmt)
    rows: list[dict[str, Any]] = []
    artifacts: dict[str, str] = {}
    with open_dataset(pred3_nc) as ds3:
        required = [
            "prob_rey_ge_800ppm",
            "prob_rey_ge_1000ppm",
            "prob_carbonate_background",
            "prob_class1_enriched",
            "prob_class2_slow_sedimentation",
        ]
        missing = [name for name in required if name not in ds3.data_vars]
        if missing:
            raise KeyError(f"Missing required variables in 3D prediction NetCDF: {missing}")
        lats, lons, extent = _extent(ds3)
        bathy = _load_bathymetry(env_nc, ds3, bathymetry_var)
        if bathy is None:
            deep_water_mask = np.ones((len(lats), len(lons)), dtype=bool)
        else:
            bathy = _as_2d(ds3, np.asarray(bathy, dtype=float))
            deep_water_mask = np.isfinite(bathy) & (bathy <= -float(min_water_depth_m))

        all_ages = np.asarray(ds3["age_ka"].values, dtype=float)
        if age_layers_ka is None:
            selected_indices = list(range(len(all_ages)))
        else:
            selected_indices = []
            for requested_age in age_layers_ka:
                idx = int(np.nanargmin(np.abs(all_ages - float(requested_age))))
                if idx not in selected_indices:
                    selected_indices.append(idx)

        for idx in selected_indices:
            age_ka = float(all_ages[idx])
            layer = ds3.isel(age_ka=idx)
            p1000 = np.asarray(layer["prob_rey_ge_1000ppm"].values, dtype=float)
            p800 = np.asarray(layer["prob_rey_ge_800ppm"].values, dtype=float)
            if "pred_mechanism_class_id" in layer:
                dominant = np.asarray(layer["pred_mechanism_class_id"].values, dtype=np.int16)
            else:
                probs = np.stack(
                    [
                        np.asarray(layer["prob_carbonate_background"].values, dtype=float),
                        np.asarray(layer["prob_class1_enriched"].values, dtype=float),
                        np.asarray(layer["prob_class2_slow_sedimentation"].values, dtype=float),
                    ],
                    axis=0,
                )
                dominant = np.nanargmax(probs, axis=0).astype(np.int16)
            layer_artifacts, layer_stats = _plot_layerwise_classwise_mechanism_solid_map(
                output_dir=output_dir,
                lons=lons,
                lats=lats,
                extent=extent,
                dominant=dominant,
                p1000=p1000,
                p800=p800,
                deep_water_mask=deep_water_mask,
                min_water_depth_m=float(min_water_depth_m),
                age_ka=age_ka,
                top_percent=float(top_percent),
                boundary_segments=boundary_segments,
                write_svg=write_svg,
            )
            rows.append(layer_stats)
            artifacts[_age_layer_token(age_ka)] = layer_artifacts["png"]

    table_path = output_dir / "layerwise_classwise_p1000_p800_top5_stats.csv"
    pd.DataFrame(rows).to_csv(table_path, index=False, encoding="utf-8-sig")
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "prediction_dir": str(prediction_dir),
        "prediction_3d_nc": str(pred3_nc),
        "env_nc": str(env_nc),
        "output_dir": str(output_dir),
        "bathymetry_var": bathymetry_var,
        "min_water_depth_m": float(min_water_depth_m),
        "top_percent": float(top_percent),
        "boundary_gmt": str(boundary_gmt) if boundary_gmt is not None else None,
        "n_boundary_segments": int(len(boundary_segments)),
        "n_layers": int(len(rows)),
        "stats_csv": str(table_path),
        "map_pngs": artifacts,
    }
    summary_path = output_dir / "layerwise_classwise_p1000_p800_top5_summary.json"
    _write_json(summary_path, summary)
    return summary


def _plot_p1000_probability_screening_map(
    *,
    output_dir: Path,
    lons: np.ndarray,
    lats: np.ndarray,
    extent: list[float],
    p1000: np.ndarray,
    domain_mask: np.ndarray,
    boundary_segments: list[np.ndarray] | None = None,
    min_water_depth_m: float | None = None,
    min_distance_to_land_km: float | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    p1000_domain = np.where(domain_mask & np.isfinite(p1000), p1000, np.nan)
    finite = _finite_values(p1000_domain)
    vmax = float(min(1.0, max(0.25, np.nanpercentile(finite, 99.0)))) if finite.size else 1.0

    fig, ax = _map_axes(
        "P(REY >= 1000 ppm) screening probability",
        extent,
        figsize=(10.4, 6.2),
        dpi=260,
    )
    im = _imshow_map(
        ax,
        p1000_domain,
        extent,
        cmap="YlOrRd",
        vmin=0.0,
        vmax=vmax,
        alpha=0.92,
        interpolation="bilinear",
        zorder=4,
    )
    _plot_gmt_lines(
        ax,
        boundary_segments or [],
        color="#5f5f5f",
        linewidth=0.36,
        alpha=0.48,
        linestyle=(0, (3.0, 2.2)),
        zorder=9,
    )
    _finish_map(ax)
    fig.subplots_adjust(left=0.07, right=0.86, top=0.91, bottom=0.10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.022, shrink=0.78)
    cbar.ax.set_ylabel("P(REY >= 1000 ppm)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    handles: list[Any] = []
    if boundary_segments:
        handles.append(
            Line2D([0], [0], color="#5f5f5f", linewidth=0.7, alpha=0.65, linestyle=(0, (3.0, 2.2)), label="Plate boundary")
        )
    if handles:
        _legend_upper_right(ax, handles=handles, fontsize=8)

    png = output_dir / "paper_p1000_probability_screening_map.png"
    svg = output_dir / "paper_p1000_probability_screening_map.svg"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)

    return (
        {
            "p1000_probability_screening_png": str(png),
            "p1000_probability_screening_svg": str(svg),
        },
        {
            "p1000_probability_screening_domain_cells": int(np.nansum(domain_mask & np.isfinite(p1000))),
            "p1000_probability_screening_colorbar_vmax": vmax,
            "p1000_probability_screening_min_water_depth_m": min_water_depth_m,
            "p1000_probability_screening_min_distance_to_land_km": min_distance_to_land_km,
        },
    )


def _plot_facies_p1000_hotspot_only_map(
    *,
    output_dir: Path,
    lons: np.ndarray,
    lats: np.ndarray,
    extent: list[float],
    dominant: np.ndarray,
    p1000: np.ndarray,
    domain_mask: np.ndarray,
    boundary_segments: list[np.ndarray] | None = None,
    min_water_depth_m: float | None = None,
    min_distance_to_land_km: float | None = None,
    top_percent: float = 10.0,
) -> tuple[dict[str, str], dict[str, Any]]:
    facies_ii = domain_mask & (dominant == 1) & np.isfinite(p1000)
    facies_iii = domain_mask & (dominant == 2) & np.isfinite(p1000)
    facies_ii_cut = _top_percent_cut(p1000, facies_ii, top_percent)
    facies_iii_cut = _top_percent_cut(p1000, facies_iii, top_percent)
    facies_ii_hot = facies_ii & np.isfinite(facies_ii_cut) & (p1000 >= facies_ii_cut)
    facies_iii_hot = facies_iii & np.isfinite(facies_iii_cut) & (p1000 >= facies_iii_cut)

    domain_rgba = np.zeros((*p1000.shape, 4), dtype=float)
    domain_rgba[domain_mask & np.isfinite(p1000), :] = to_rgba("#f4f4f4", 0.68)

    hotspot_rgba = np.zeros((*p1000.shape, 4), dtype=float)
    hotspot_rgba[facies_ii_hot, :] = to_rgba("#a4141c", 0.92)
    hotspot_rgba[facies_iii_hot, :] = to_rgba("#064f93", 0.92)

    fig, ax = _map_axes(
        "Facies-specific P1000 hotspot domains",
        extent,
        figsize=(10.4, 6.2),
        dpi=260,
    )
    _imshow_map(ax, domain_rgba, extent, interpolation="nearest", zorder=3)
    _imshow_map(ax, hotspot_rgba, extent, interpolation="nearest", zorder=6)
    _plot_gmt_lines(
        ax,
        boundary_segments or [],
        color="#5f5f5f",
        linewidth=0.36,
        alpha=0.48,
        linestyle=(0, (3.0, 2.2)),
        zorder=9,
    )
    _finish_map(ax)

    handles = [
        Patch(facecolor="#a4141c", edgecolor="#7f0000", label=f"Facies II top {top_percent:g}% P1000 >= {facies_ii_cut:.2f}"),
        Patch(facecolor="#064f93", edgecolor="#063b74", label=f"Facies III top {top_percent:g}% P1000 >= {facies_iii_cut:.2f}"),
    ]
    if boundary_segments:
        handles.append(
            Line2D([0], [0], color="#5f5f5f", linewidth=0.7, alpha=0.65, linestyle=(0, (3.0, 2.2)), label="Plate boundary")
        )
    _legend_upper_right(ax, handles=handles, fontsize=8)

    png = output_dir / "paper_facies_p1000_hotspot_only_map.png"
    svg = output_dir / "paper_facies_p1000_hotspot_only_map.svg"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)

    return (
        {
            "facies_p1000_hotspot_only_png": str(png),
            "facies_p1000_hotspot_only_svg": str(svg),
        },
        {
            "facies_p1000_hotspot_only_top_percent": float(top_percent),
            "facies_p1000_hotspot_only_min_water_depth_m": min_water_depth_m,
            "facies_p1000_hotspot_only_min_distance_to_land_km": min_distance_to_land_km,
            "facies_ii_p1000_hotspot_domain_cells": int(np.nansum(facies_ii)),
            "facies_iii_p1000_hotspot_domain_cells": int(np.nansum(facies_iii)),
            "facies_ii_p1000_hotspot_cutoff": facies_ii_cut,
            "facies_iii_p1000_hotspot_cutoff": facies_iii_cut,
            "facies_ii_p1000_hotspot_cells": int(np.nansum(facies_ii_hot)),
            "facies_iii_p1000_hotspot_cells": int(np.nansum(facies_iii_hot)),
        },
    )


def _facies_percentile_thresholds(values: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    vals = _finite_values(values, mask)
    if vals.size == 0:
        return {"p50": float("nan"), "p75": float("nan"), "p90": float("nan"), "p95": float("nan")}
    p50, p75, p90, p95 = np.nanpercentile(vals, [50.0, 75.0, 90.0, 95.0])
    return {"p50": float(p50), "p75": float(p75), "p90": float(p90), "p95": float(p95)}


def _plot_facies_p1000_percentile_tier_map(
    *,
    output_dir: Path,
    lons: np.ndarray,
    lats: np.ndarray,
    extent: list[float],
    dominant: np.ndarray,
    p1000: np.ndarray,
    deep_water_mask: np.ndarray,
    min_water_depth_m: float | None,
    hotspot_min_cells: int,
    title: str | None = None,
    output_stem: str = "paper_facies_p1000_percentile_tiers_map",
    artifact_prefix: str = "facies_p1000_percentile_tiers",
    stats_prefix: str = "facies",
) -> tuple[dict[str, str], dict[str, Any]]:
    facies_i = deep_water_mask & (dominant == 0) & np.isfinite(p1000)
    facies_ii = deep_water_mask & (dominant == 1) & np.isfinite(p1000)
    facies_iii = deep_water_mask & (dominant == 2) & np.isfinite(p1000)
    facies_ii_q = _facies_percentile_thresholds(p1000, facies_ii)
    facies_iii_q = _facies_percentile_thresholds(p1000, facies_iii)

    colors = {
        "facies_i": "#d7d7d7",
        "ii_low": "#fde8e3",
        "ii_50_75": "#f6b1a8",
        "ii_75_90": "#df675f",
        "ii_top10": "#9d1c20",
        "iii_low": "#e0efff",
        "iii_50_75": "#9ec9ee",
        "iii_75_90": "#4c90cf",
        "iii_top10": "#0b4d91",
    }
    rgba = np.ones((*p1000.shape, 4), dtype=float)
    rgba[..., 3] = 0.0

    def fill(mask: np.ndarray, color: str, alpha: float) -> None:
        rgba[mask, :3] = to_rgba(color)[:3]
        rgba[mask, 3] = alpha

    fill(facies_i, colors["facies_i"], 0.86)

    ii_low = facies_ii & (p1000 < facies_ii_q["p50"])
    ii_50_75 = facies_ii & (p1000 >= facies_ii_q["p50"]) & (p1000 < facies_ii_q["p75"])
    ii_75_90 = facies_ii & (p1000 >= facies_ii_q["p75"]) & (p1000 < facies_ii_q["p90"])
    ii_top10 = facies_ii & (p1000 >= facies_ii_q["p90"])
    iii_low = facies_iii & (p1000 < facies_iii_q["p50"])
    iii_50_75 = facies_iii & (p1000 >= facies_iii_q["p50"]) & (p1000 < facies_iii_q["p75"])
    iii_75_90 = facies_iii & (p1000 >= facies_iii_q["p75"]) & (p1000 < facies_iii_q["p90"])
    iii_top10 = facies_iii & (p1000 >= facies_iii_q["p90"])

    fill(ii_low, colors["ii_low"], 0.80)
    fill(ii_50_75, colors["ii_50_75"], 0.88)
    fill(ii_75_90, colors["ii_75_90"], 0.94)
    fill(ii_top10, colors["ii_top10"], 0.98)
    fill(iii_low, colors["iii_low"], 0.80)
    fill(iii_50_75, colors["iii_50_75"], 0.88)
    fill(iii_75_90, colors["iii_75_90"], 0.94)
    fill(iii_top10, colors["iii_top10"], 0.98)

    ii_top10_display = _clean_display_mask(ii_top10, max(1, int(hotspot_min_cells)), closing_iterations=1)
    iii_top10_display = _clean_display_mask(iii_top10, max(1, int(hotspot_min_cells)), closing_iterations=1)

    map_title = title or f"Facies-specific P1000 percentile tiers (>={min_water_depth_m:g} m water depth)"
    fig, ax = _map_axes(map_title, extent, figsize=(11.4, 6.9), dpi=260)
    ax.set_title(map_title, fontsize=16, pad=12)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.tick_params(labelsize=10)

    _imshow_map(ax, rgba, extent, interpolation="nearest", zorder=4)
    _contour_mask(ax, lons, lats, facies_ii, "#c77a73", 0.55)
    _contour_mask(ax, lons, lats, facies_iii, "#5c91c7", 0.55)
    _contour_mask(ax, lons, lats, ii_top10_display, "white", 4.25)
    _contour_mask(ax, lons, lats, ii_top10_display, colors["ii_top10"], 2.35)
    _contour_mask(ax, lons, lats, iii_top10_display, "white", 4.25)
    _contour_mask(ax, lons, lats, iii_top10_display, colors["iii_top10"], 2.35)
    _finish_map(ax)

    handles = [
        Patch(facecolor=colors["facies_i"], edgecolor="#9a9a9a", label="Facies I"),
        Patch(facecolor=colors["ii_low"], edgecolor="#c77a73", label="Facies II: <P50"),
        Patch(facecolor=colors["ii_50_75"], edgecolor="#c77a73", label="Facies II: P50-P75"),
        Patch(facecolor=colors["ii_75_90"], edgecolor="#c77a73", label="Facies II: P75-P90"),
        Patch(facecolor=colors["ii_top10"], edgecolor=colors["ii_top10"], label=f"Facies II: top 10% (P1000 >= {facies_ii_q['p90']:.2f})"),
        Patch(facecolor=colors["iii_low"], edgecolor="#5c91c7", label="Facies III: <P50"),
        Patch(facecolor=colors["iii_50_75"], edgecolor="#5c91c7", label="Facies III: P50-P75"),
        Patch(facecolor=colors["iii_75_90"], edgecolor="#5c91c7", label="Facies III: P75-P90"),
        Patch(facecolor=colors["iii_top10"], edgecolor=colors["iii_top10"], label=f"Facies III: top 10% (P1000 >= {facies_iii_q['p90']:.2f})"),
    ]
    legend = ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=3,
        fontsize=10,
        frameon=True,
        framealpha=0.94,
        handlelength=1.6,
        columnspacing=1.2,
    )
    legend.set_zorder(1000)

    png = output_dir / f"{output_stem}.png"
    svg = output_dir / f"{output_stem}.svg"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)

    artifacts = {
        f"{artifact_prefix}_png": str(png),
        f"{artifact_prefix}_svg": str(svg),
    }
    raw_stats = {
        "p1000_tier_map_water_depth_mask_applied": min_water_depth_m is not None,
        "p1000_tier_map_min_water_depth_m": float(min_water_depth_m) if min_water_depth_m is not None else None,
        "ii_p1000_p50": facies_ii_q["p50"],
        "ii_p1000_p75": facies_ii_q["p75"],
        "ii_p1000_p90_top10_cutoff": facies_ii_q["p90"],
        "iii_p1000_p50": facies_iii_q["p50"],
        "iii_p1000_p75": facies_iii_q["p75"],
        "iii_p1000_p90_top10_cutoff": facies_iii_q["p90"],
        "i_domain_cells": int(np.nansum(facies_i)),
        "ii_domain_cells": int(np.nansum(facies_ii)),
        "iii_domain_cells": int(np.nansum(facies_iii)),
        "ii_p1000_low_cells": int(np.nansum(ii_low)),
        "ii_p1000_p50_p75_cells": int(np.nansum(ii_50_75)),
        "ii_p1000_p75_p90_cells": int(np.nansum(ii_75_90)),
        "ii_p1000_top10_cells_raw": int(np.nansum(ii_top10)),
        "ii_p1000_top10_cells_display": int(np.nansum(ii_top10_display)),
        "iii_p1000_low_cells": int(np.nansum(iii_low)),
        "iii_p1000_p50_p75_cells": int(np.nansum(iii_50_75)),
        "iii_p1000_p75_p90_cells": int(np.nansum(iii_75_90)),
        "iii_p1000_top10_cells_raw": int(np.nansum(iii_top10)),
        "iii_p1000_top10_cells_display": int(np.nansum(iii_top10_display)),
    }
    stats = {f"{stats_prefix}_{key}": value for key, value in raw_stats.items()}
    return artifacts, stats


def _plot_facies_p1000_clean_screening_map(
    *,
    output_dir: Path,
    lons: np.ndarray,
    lats: np.ndarray,
    extent: list[float],
    dominant: np.ndarray,
    p1000: np.ndarray,
    domain_mask: np.ndarray,
    hotspot_min_cells: int,
    include_mid_tier: bool,
    title: str,
    output_stem: str,
    artifact_prefix: str,
    stats_prefix: str,
    hotspot_percentile: float = 95.0,
) -> tuple[dict[str, str], dict[str, Any]]:
    facies_i = domain_mask & (dominant == 0) & np.isfinite(p1000)
    facies_ii = domain_mask & (dominant == 1) & np.isfinite(p1000)
    facies_iii = domain_mask & (dominant == 2) & np.isfinite(p1000)
    facies_ii_q = _facies_percentile_thresholds(p1000, facies_ii)
    facies_iii_q = _facies_percentile_thresholds(p1000, facies_iii)
    hotspot_key = f"p{int(hotspot_percentile):d}"
    if hotspot_key not in facies_ii_q or hotspot_key not in facies_iii_q:
        raise ValueError(f"Unsupported hotspot_percentile={hotspot_percentile:g}")
    top_label = f"Top {100.0 - float(hotspot_percentile):g}%"
    high_label = f"P75-P{int(hotspot_percentile):d}"

    ii_mid = facies_ii & (p1000 >= facies_ii_q["p50"]) & (p1000 < facies_ii_q["p75"])
    ii_high = facies_ii & (p1000 >= facies_ii_q["p75"]) & (p1000 < facies_ii_q[hotspot_key])
    ii_top = facies_ii & (p1000 >= facies_ii_q[hotspot_key])
    iii_mid = facies_iii & (p1000 >= facies_iii_q["p50"]) & (p1000 < facies_iii_q["p75"])
    iii_high = facies_iii & (p1000 >= facies_iii_q["p75"]) & (p1000 < facies_iii_q[hotspot_key])
    iii_top = facies_iii & (p1000 >= facies_iii_q[hotspot_key])

    ii_top_display = _clean_display_mask(ii_top, max(1, int(hotspot_min_cells)), closing_iterations=1)
    iii_top_display = _clean_display_mask(iii_top, max(1, int(hotspot_min_cells)), closing_iterations=1)

    colors = {
        "facies_i": "#eeeeee",
        "ii_mid": "#f9d2cc",
        "ii_high": "#df6b63",
        "ii_top": "#981b1e",
        "iii_mid": "#d6eafd",
        "iii_high": "#4f93cf",
        "iii_top": "#084d8d",
    }
    rgba = np.ones((*p1000.shape, 4), dtype=float)
    rgba[..., 3] = 0.0

    def fill(mask: np.ndarray, color: str, alpha: float) -> None:
        rgba[mask, :3] = to_rgba(color)[:3]
        rgba[mask, 3] = alpha

    fill(facies_i, colors["facies_i"], 0.48)
    if include_mid_tier:
        fill(ii_mid, colors["ii_mid"], 0.46)
        fill(iii_mid, colors["iii_mid"], 0.46)
    fill(ii_high, colors["ii_high"], 0.72)
    fill(iii_high, colors["iii_high"], 0.72)
    fill(ii_top, colors["ii_top"], 0.96)
    fill(iii_top, colors["iii_top"], 0.96)

    fig, ax = _map_axes(title, extent, figsize=(10.8, 6.55), dpi=270)
    ax.set_title(title, fontsize=15, pad=10)
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.tick_params(labelsize=9.5)

    _imshow_map(ax, rgba, extent, interpolation="nearest", zorder=4)
    _contour_mask(ax, lons, lats, ii_top_display, "white", 4.0)
    _contour_mask(ax, lons, lats, ii_top_display, colors["ii_top"], 2.15)
    _contour_mask(ax, lons, lats, iii_top_display, "white", 4.0)
    _contour_mask(ax, lons, lats, iii_top_display, colors["iii_top"], 2.15)
    _finish_map(ax)

    handles = [
        Patch(facecolor=colors["facies_i"], edgecolor="#b8b8b8", label="Facies I"),
    ]
    if include_mid_tier:
        handles.extend(
            [
            Patch(facecolor=colors["ii_mid"], edgecolor="#d08a82", label="Facies II P50-P75"),
            Patch(facecolor=colors["iii_mid"], edgecolor="#82afd9", label="Facies III P50-P75"),
            ]
        )
    handles.extend(
        [
            Patch(facecolor=colors["ii_high"], edgecolor="#b9524d", label=f"Facies II {high_label}"),
            Patch(facecolor=colors["ii_top"], edgecolor=colors["ii_top"], label=f"Facies II {top_label}"),
            Patch(facecolor=colors["iii_high"], edgecolor="#3f7fb7", label=f"Facies III {high_label}"),
            Patch(facecolor=colors["iii_top"], edgecolor=colors["iii_top"], label=f"Facies III {top_label}"),
        ]
    )
    legend = ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=4,
        fontsize=9,
        frameon=True,
        framealpha=0.94,
        handlelength=1.55,
        columnspacing=1.0,
    )
    legend.set_zorder(1000)

    note = (
        "Within-facies P1000 thresholds\n"
        f"Facies II: P75={facies_ii_q['p75']:.2f}, P{int(hotspot_percentile):d}={facies_ii_q[hotspot_key]:.2f}\n"
        f"Facies III: P75={facies_iii_q['p75']:.2f}, P{int(hotspot_percentile):d}={facies_iii_q[hotspot_key]:.2f}"
    )
    ax.text(
        0.012,
        0.018,
        note,
        transform=ax.transAxes,
        fontsize=8.5,
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#c8c8c8", "alpha": 0.86, "boxstyle": "round,pad=0.32"},
        zorder=1001,
    )

    png = output_dir / f"{output_stem}.png"
    svg = output_dir / f"{output_stem}.svg"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)

    artifacts = {
        f"{artifact_prefix}_png": str(png),
        f"{artifact_prefix}_svg": str(svg),
    }
    raw_stats = {
        "include_mid_tier": bool(include_mid_tier),
        "hotspot_percentile": float(hotspot_percentile),
        "ii_p1000_p75": facies_ii_q["p75"],
        "ii_p1000_p90": facies_ii_q["p90"],
        "ii_p1000_p95": facies_ii_q["p95"],
        f"ii_p1000_p{int(hotspot_percentile):d}_hotspot_cutoff": facies_ii_q[hotspot_key],
        "iii_p1000_p75": facies_iii_q["p75"],
        "iii_p1000_p90": facies_iii_q["p90"],
        "iii_p1000_p95": facies_iii_q["p95"],
        f"iii_p1000_p{int(hotspot_percentile):d}_hotspot_cutoff": facies_iii_q[hotspot_key],
        "i_domain_cells": int(np.nansum(facies_i)),
        "ii_domain_cells": int(np.nansum(facies_ii)),
        "iii_domain_cells": int(np.nansum(facies_iii)),
        "ii_p1000_p50_p75_cells": int(np.nansum(ii_mid)),
        f"ii_p1000_p75_p{int(hotspot_percentile):d}_cells": int(np.nansum(ii_high)),
        "ii_p1000_hotspot_cells_raw": int(np.nansum(ii_top)),
        "ii_p1000_hotspot_cells_display": int(np.nansum(ii_top_display)),
        "iii_p1000_p50_p75_cells": int(np.nansum(iii_mid)),
        f"iii_p1000_p75_p{int(hotspot_percentile):d}_cells": int(np.nansum(iii_high)),
        "iii_p1000_hotspot_cells_raw": int(np.nansum(iii_top)),
        "iii_p1000_hotspot_cells_display": int(np.nansum(iii_top_display)),
    }
    stats = {f"{stats_prefix}_{key}": value for key, value in raw_stats.items()}
    return artifacts, stats


def _plot_facies_specific_threshold_screening_map(
    *,
    output_dir: Path,
    lons: np.ndarray,
    lats: np.ndarray,
    extent: list[float],
    dominant: np.ndarray,
    p1000: np.ndarray,
    p1500: np.ndarray,
    domain_mask: np.ndarray,
    boundary_segments: list[np.ndarray],
    hotspot_min_cells: int,
    hotspot_percentile: float = 95.0,
) -> tuple[dict[str, str], dict[str, Any]]:
    facies_i = domain_mask & (dominant == 0) & (np.isfinite(p1000) | np.isfinite(p1500))
    facies_ii = domain_mask & (dominant == 1) & np.isfinite(p1500)
    facies_iii = domain_mask & (dominant == 2) & np.isfinite(p1000)
    facies_ii_q = _facies_percentile_thresholds(p1500, facies_ii)
    facies_iii_q = _facies_percentile_thresholds(p1000, facies_iii)
    hotspot_key = f"p{int(hotspot_percentile):d}"
    if hotspot_key not in facies_ii_q or hotspot_key not in facies_iii_q:
        raise ValueError(f"Unsupported hotspot_percentile={hotspot_percentile:g}")
    top_label = f"Top {100.0 - float(hotspot_percentile):g}%"
    high_label = f"P75-P{int(hotspot_percentile):d}"

    ii_high = facies_ii & (p1500 >= facies_ii_q["p75"]) & (p1500 < facies_ii_q[hotspot_key])
    ii_top = facies_ii & (p1500 >= facies_ii_q[hotspot_key])
    iii_high = facies_iii & (p1000 >= facies_iii_q["p75"]) & (p1000 < facies_iii_q[hotspot_key])
    iii_top = facies_iii & (p1000 >= facies_iii_q[hotspot_key])
    ii_top_display = _clean_display_mask(ii_top, max(1, int(hotspot_min_cells)), closing_iterations=1)
    iii_top_display = _clean_display_mask(iii_top, max(1, int(hotspot_min_cells)), closing_iterations=1)

    colors = {
        "facies_i": "#eeeeee",
        "ii_high": "#df6b63",
        "ii_top": "#981b1e",
        "iii_high": "#4f93cf",
        "iii_top": "#084d8d",
    }
    rgba = np.ones((*p1000.shape, 4), dtype=float)
    rgba[..., 3] = 0.0

    def fill(mask: np.ndarray, color: str, alpha: float) -> None:
        rgba[mask, :3] = to_rgba(color)[:3]
        rgba[mask, 3] = alpha

    fill(facies_i, colors["facies_i"], 0.48)
    fill(ii_high, colors["ii_high"], 0.72)
    fill(iii_high, colors["iii_high"], 0.72)
    fill(ii_top, colors["ii_top"], 0.96)
    fill(iii_top, colors["iii_top"], 0.96)

    title = "Facies-specific REY threshold screening domains"
    fig, ax = _map_axes(title, extent, figsize=(10.8, 6.55), dpi=270)
    ax.set_title(title, fontsize=15, pad=10)
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.tick_params(labelsize=9.5)

    _imshow_map(ax, rgba, extent, interpolation="nearest", zorder=4)
    _plot_gmt_lines(ax, boundary_segments)
    _contour_mask(ax, lons, lats, ii_top_display, "white", 4.0)
    _contour_mask(ax, lons, lats, ii_top_display, colors["ii_top"], 2.15)
    _contour_mask(ax, lons, lats, iii_top_display, "white", 4.0)
    _contour_mask(ax, lons, lats, iii_top_display, colors["iii_top"], 2.15)
    _finish_map(ax)

    handles = [
        Patch(facecolor=colors["facies_i"], edgecolor="#b8b8b8", label="Facies I"),
        Patch(facecolor=colors["ii_high"], edgecolor="#b9524d", label=f"Facies II P1500 {high_label}"),
        Patch(facecolor=colors["ii_top"], edgecolor=colors["ii_top"], label=f"Facies II P1500 {top_label}"),
        Patch(facecolor=colors["iii_high"], edgecolor="#3f7fb7", label=f"Facies III P1000 {high_label}"),
        Patch(facecolor=colors["iii_top"], edgecolor=colors["iii_top"], label=f"Facies III P1000 {top_label}"),
    ]
    if boundary_segments:
        handles.append(Line2D([0], [0], color="#585858", linewidth=0.7, alpha=0.75, label="Plate boundary"))
    legend = ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        fontsize=9,
        frameon=True,
        framealpha=0.94,
        handlelength=1.55,
        columnspacing=1.0,
    )
    legend.set_zorder(1000)

    note = (
        "Within-facies screening thresholds\n"
        f"Facies II: P1500 P75={facies_ii_q['p75']:.2f}, P{int(hotspot_percentile):d}={facies_ii_q[hotspot_key]:.2f}\n"
        f"Facies III: P1000 P75={facies_iii_q['p75']:.2f}, P{int(hotspot_percentile):d}={facies_iii_q[hotspot_key]:.2f}"
    )
    ax.text(
        0.012,
        0.018,
        note,
        transform=ax.transAxes,
        fontsize=8.5,
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#c8c8c8", "alpha": 0.86, "boxstyle": "round,pad=0.32"},
        zorder=1001,
    )

    png = output_dir / "paper_facies_specific_threshold_screening_clean_map.png"
    svg = output_dir / "paper_facies_specific_threshold_screening_clean_map.svg"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)

    artifacts = {
        "facies_specific_threshold_screening_clean_png": str(png),
        "facies_specific_threshold_screening_clean_svg": str(svg),
    }
    stats = {
        "facies_specific_threshold_hotspot_percentile": float(hotspot_percentile),
        "facies_specific_threshold_facies_ii_metric": "max_prob_rey_ge_1500ppm",
        "facies_specific_threshold_facies_iii_metric": "max_prob_rey_ge_1000ppm",
        "facies_specific_threshold_ii_p1500_p75": facies_ii_q["p75"],
        "facies_specific_threshold_ii_p1500_p90": facies_ii_q["p90"],
        "facies_specific_threshold_ii_p1500_p95": facies_ii_q["p95"],
        "facies_specific_threshold_iii_p1000_p75": facies_iii_q["p75"],
        "facies_specific_threshold_iii_p1000_p90": facies_iii_q["p90"],
        "facies_specific_threshold_iii_p1000_p95": facies_iii_q["p95"],
        "facies_specific_threshold_i_domain_cells": int(np.nansum(facies_i)),
        "facies_specific_threshold_ii_domain_cells": int(np.nansum(facies_ii)),
        "facies_specific_threshold_iii_domain_cells": int(np.nansum(facies_iii)),
        "facies_specific_threshold_ii_p1500_p75_p95_cells": int(np.nansum(ii_high)),
        "facies_specific_threshold_ii_p1500_top5_cells_raw": int(np.nansum(ii_top)),
        "facies_specific_threshold_ii_p1500_top5_cells_display": int(np.nansum(ii_top_display)),
        "facies_specific_threshold_iii_p1000_p75_p95_cells": int(np.nansum(iii_high)),
        "facies_specific_threshold_iii_p1000_top5_cells_raw": int(np.nansum(iii_top)),
        "facies_specific_threshold_iii_p1000_top5_cells_display": int(np.nansum(iii_top_display)),
    }
    return artifacts, stats


def _smooth_display_field(mask: np.ndarray, *, sigma: float = 1.15) -> np.ndarray:
    values = gaussian_filter(np.asarray(mask, dtype=float), sigma=float(sigma), mode="nearest")
    values[values < 0.08] = np.nan
    return values


def _plot_facies_specific_threshold_publication_map(
    *,
    output_dir: Path,
    lons: np.ndarray,
    lats: np.ndarray,
    extent: list[float],
    dominant: np.ndarray,
    p1000: np.ndarray,
    p1500: np.ndarray,
    domain_mask: np.ndarray,
    boundary_segments: list[np.ndarray],
    hotspot_min_cells: int,
    hotspot_percentile: float = 95.0,
    smooth_sigma: float = 1.15,
) -> tuple[dict[str, str], dict[str, Any]]:
    facies_ii = domain_mask & (dominant == 1) & np.isfinite(p1500)
    facies_iii = domain_mask & (dominant == 2) & np.isfinite(p1000)
    facies_ii_q = _facies_percentile_thresholds(p1500, facies_ii)
    facies_iii_q = _facies_percentile_thresholds(p1000, facies_iii)
    hotspot_key = f"p{int(hotspot_percentile):d}"

    ii_halo = facies_ii & (p1500 >= facies_ii_q["p75"])
    ii_core = facies_ii & (p1500 >= facies_ii_q[hotspot_key])
    iii_halo = facies_iii & (p1000 >= facies_iii_q["p75"])
    iii_core = facies_iii & (p1000 >= facies_iii_q[hotspot_key])

    ii_halo_display = _clean_display_mask(ii_halo, max(1, int(hotspot_min_cells)), closing_iterations=1)
    iii_halo_display = _clean_display_mask(iii_halo, max(1, int(hotspot_min_cells)), closing_iterations=1)
    ii_core_display = _clean_display_mask(ii_core, max(1, int(hotspot_min_cells)), closing_iterations=1)
    iii_core_display = _clean_display_mask(iii_core, max(1, int(hotspot_min_cells)), closing_iterations=1)

    ii_halo_field = _smooth_display_field(ii_halo_display, sigma=smooth_sigma)
    iii_halo_field = _smooth_display_field(iii_halo_display, sigma=smooth_sigma)
    ii_core_field = _smooth_display_field(ii_core_display, sigma=smooth_sigma)
    iii_core_field = _smooth_display_field(iii_core_display, sigma=smooth_sigma)

    colors = {
        "ii_halo": "#e88b83",
        "ii_core": "#981b1e",
        "iii_halo": "#6da7d8",
        "iii_core": "#084d8d",
    }

    title = "Facies-specific REY threshold screening"
    fig, ax = _map_axes(title, extent, figsize=(10.6, 6.55), dpi=300)
    ax.set_title(title, fontsize=14.5, pad=9)
    ax.set_xlabel("Longitude", fontsize=10.5)
    ax.set_ylabel("Latitude", fontsize=10.5)
    ax.tick_params(labelsize=9.2)


    valid_domain = np.where(domain_mask & np.isfinite(p1000), 1.0, np.nan)
    _imshow_map(
        ax,
        valid_domain,
        extent,
        cmap=ListedColormap(["#f3f3f3"]),
        vmin=0.0,
        vmax=1.0,
        alpha=0.24,
        interpolation="nearest",
        zorder=1,
    )
    _contourf_map(ax, lons, lats, ii_halo_field, levels=[0.18, 1.01], colors=[colors["ii_halo"]], alpha=0.36, zorder=5, antialiased=True)
    _contourf_map(ax, lons, lats, iii_halo_field, levels=[0.18, 1.01], colors=[colors["iii_halo"]], alpha=0.34, zorder=6, antialiased=True)
    _plot_gmt_lines(ax, boundary_segments, color="#5e5e5e", linewidth=0.34, alpha=0.42, linestyle=(0, (3.0, 2.5)), zorder=9)
    _contourf_map(ax, lons, lats, ii_core_field, levels=[0.28, 1.01], colors=[colors["ii_core"]], alpha=0.92, zorder=12, antialiased=True)
    _contourf_map(ax, lons, lats, iii_core_field, levels=[0.28, 1.01], colors=[colors["iii_core"]], alpha=0.92, zorder=13, antialiased=True)
    _contour_mask(ax, lons, lats, ii_core_display, "white", 3.0)
    _contour_mask(ax, lons, lats, ii_core_display, colors["ii_core"], 1.45)
    _contour_mask(ax, lons, lats, iii_core_display, "white", 3.0)
    _contour_mask(ax, lons, lats, iii_core_display, colors["iii_core"], 1.45)
    _finish_map(ax)

    handles = [
        Patch(facecolor=colors["ii_halo"], edgecolor="none", alpha=0.42, label="Facies II P1500 P75-P95"),
        Patch(facecolor=colors["ii_core"], edgecolor=colors["ii_core"], label="Facies II P1500 Top 5%"),
        Patch(facecolor=colors["iii_halo"], edgecolor="none", alpha=0.42, label="Facies III P1000 P75-P95"),
        Patch(facecolor=colors["iii_core"], edgecolor=colors["iii_core"], label="Facies III P1000 Top 5%"),
        Line2D([0], [0], color="#5e5e5e", linewidth=0.65, alpha=0.65, linestyle=(0, (3.0, 2.5)), label="Plate boundary"),
    ]
    legend = ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.205),
        ncol=3,
        fontsize=8.8,
        frameon=True,
        framealpha=0.94,
        handlelength=1.45,
        columnspacing=0.9,
    )
    legend.set_zorder(1000)

    note = (
        f"Within-facies percentiles: II P1500 P95={facies_ii_q[hotspot_key]:.2f}; "
        f"III P1000 P95={facies_iii_q[hotspot_key]:.2f}"
    )
    ax.text(
        0.012,
        0.018,
        note,
        transform=ax.transAxes,
        fontsize=8.0,
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#d0d0d0", "alpha": 0.82, "boxstyle": "round,pad=0.28"},
        zorder=1001,
    )

    fig.subplots_adjust(left=0.07, right=0.985, top=0.91, bottom=0.22)

    png = output_dir / "paper_facies_specific_threshold_screening_publication_map.png"
    svg = output_dir / "paper_facies_specific_threshold_screening_publication_map.svg"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)

    artifacts = {
        "facies_specific_threshold_screening_publication_png": str(png),
        "facies_specific_threshold_screening_publication_svg": str(svg),
    }
    stats = {
        "facies_specific_threshold_publication_smooth_sigma": float(smooth_sigma),
        "facies_specific_threshold_publication_ii_p1500_p75": facies_ii_q["p75"],
        "facies_specific_threshold_publication_ii_p1500_p95": facies_ii_q["p95"],
        "facies_specific_threshold_publication_iii_p1000_p75": facies_iii_q["p75"],
        "facies_specific_threshold_publication_iii_p1000_p95": facies_iii_q["p95"],
        "facies_specific_threshold_publication_ii_core_cells_display": int(np.nansum(ii_core_display)),
        "facies_specific_threshold_publication_iii_core_cells_display": int(np.nansum(iii_core_display)),
    }
    return artifacts, stats


def plot_composite(
    prediction_dir: Path,
    env_nc: Path,
    points_parquet: Path,
    output_dir: Path,
    *,
    bathymetry_var: str,
    mechanism_min_prob: float,
    hotspot_prob1000: float,
    hotspot_prob1500: float,
    hotspot_score_percentile: float,
    smoothing_sigma: float,
    mechanism_domain_min_cells: int,
    hotspot_min_cells: int,
    distance_max_km: float,
    min_water_depth_m: float,
    appearance_prob_threshold: float,
    boundary_gmt: Path | None = DEFAULT_BOUNDARIES_GMT,
) -> dict[str, Any]:
    summary_nc = prediction_dir / "continuous_rey_paper_summary_2d.nc"
    pred3_nc = prediction_dir / "continuous_rey_mechanism_threshold_predictions_3d.nc"
    if not summary_nc.exists():
        raise FileNotFoundError(f"Missing 2D summary NetCDF: {summary_nc}")
    if not pred3_nc.exists():
        pred3_nc = None

    output_dir.mkdir(parents=True, exist_ok=True)
    with open_dataset(summary_nc) as ds2_open:
        ds2 = ds2_open.load()
    ds3 = None
    if pred3_nc is not None:
        with open_dataset(pred3_nc) as ds3_open:
            ds3 = ds3_open.load()

    lats, lons, extent = _extent(ds2)
    boundary_segments = _load_gmt_lines(boundary_gmt)
    distance_to_land_km = _distance_to_land_km(lons, lats)
    core_support = _load_core_support(points_parquet, extent)
    distance_to_core_km = _nearest_core_distance_km(lons, lats, core_support)
    dominant, mechanism_confidence, mech_probs = _mechanism_layers(ds2, ds3)
    dominant = _as_2d(ds2, dominant)
    mechanism_confidence = _as_2d(ds2, mechanism_confidence)
    mech_probs = np.stack([_as_2d(ds2, mech_probs[i]) for i in range(mech_probs.shape[0])], axis=0)

    max_rey = _as_2d(ds2, np.asarray(ds2["max_pred_rey_ppm_0_2580ka"].values, dtype=float))
    p1000 = _as_2d(ds2, np.asarray(ds2["max_prob_rey_ge_1000ppm_0_2580ka"].values, dtype=float))
    p1500 = _as_2d(ds2, np.asarray(ds2["max_prob_rey_ge_1500ppm_0_2580ka"].values, dtype=float))
    if ds3 is not None and "prob_rey_ge_800ppm" in ds3:
        p800 = _as_2d(ds2, np.nanmax(np.asarray(ds3["prob_rey_ge_800ppm"].values, dtype=float), axis=0))
    else:
        p800 = None
    score1000 = _as_2d(ds2, np.asarray(ds2["screening_score_ge1000"].values, dtype=float))
    score1500 = _as_2d(ds2, np.asarray(ds2["screening_score_ge1500"].values, dtype=float))
    confidence = _as_2d(ds2, np.asarray(ds2["prediction_confidence_mean_0_2580ka"].values, dtype=float))
    if ds3 is not None and "prob_class1_enriched" in ds3 and "prob_class2_slow_sedimentation" in ds3:
        class1_appearance_prob = _as_2d(ds2, np.nanmax(np.asarray(ds3["prob_class1_enriched"].values, dtype=float), axis=0))
        class2_appearance_prob = _as_2d(ds2, np.nanmax(np.asarray(ds3["prob_class2_slow_sedimentation"].values, dtype=float), axis=0))
    else:
        class1_appearance_prob = mech_probs[1].copy()
        class2_appearance_prob = mech_probs[2].copy()
    class1_appearance = class1_appearance_prob >= float(appearance_prob_threshold)
    class2_appearance = class2_appearance_prob >= float(appearance_prob_threshold)
    enriched_appearance_mask = class1_appearance | class2_appearance
    appearance_class = np.zeros(dominant.shape, dtype=np.int16)
    appearance_class[class2_appearance] = 2
    appearance_class[class1_appearance & (class1_appearance_prob >= class2_appearance_prob)] = 1
    bathy = _load_bathymetry(env_nc, ds2, bathymetry_var)
    bathy_native = _load_bathymetry_native(env_nc, extent, bathymetry_var)
    background = _smooth(_robust_background(bathy, dominant.shape), sigma=max(float(smoothing_sigma), 0.1))
    if bathy is None:
        water_depth_abs = np.full(dominant.shape, np.nan, dtype=float)
        deep_water_mask = np.ones(dominant.shape, dtype=bool)
    else:
        bathy = _as_2d(ds2, np.asarray(bathy, dtype=float))
        water_depth_abs = np.where(np.isfinite(bathy) & (bathy < 0.0), np.abs(bathy), np.nan)
        deep_water_mask = np.isfinite(bathy) & (bathy <= -float(min_water_depth_m))
    classwise_min_water_depth_m = 3500.0
    classwise_4000_min_water_depth_m = 4000.0
    classwise_min_distance_to_land_km = None
    if bathy is None:
        classwise_deep_water_mask = np.ones(dominant.shape, dtype=bool)
        classwise_4000_deep_water_mask = np.ones(dominant.shape, dtype=bool)
    else:
        classwise_deep_water_mask = np.isfinite(bathy) & (bathy <= -classwise_min_water_depth_m)
        classwise_4000_deep_water_mask = np.isfinite(bathy) & (bathy <= -classwise_4000_min_water_depth_m)

    class1_smooth = _smooth(mech_probs[1], sigma=float(smoothing_sigma))
    class2_smooth = _smooth(mech_probs[2], sigma=float(smoothing_sigma))
    p1000_smooth = _smooth(p1000, sigma=float(smoothing_sigma))
    p1500_smooth = _smooth(p1500, sigma=float(smoothing_sigma))
    score1000_smooth = _smooth(score1000, sigma=float(smoothing_sigma))
    score1500_smooth = _smooth(score1500, sigma=float(smoothing_sigma))
    confidence_smooth = _smooth(confidence, sigma=float(smoothing_sigma))
    class1_cut = float(max(0.38, np.nanpercentile(_finite_values(class1_smooth, deep_water_mask), 82.0)))
    class2_cut = float(max(0.72, np.nanpercentile(_finite_values(class2_smooth, deep_water_mask), 72.0)))
    class1_domain_all = _clean_mask(class1_smooth >= class1_cut, int(mechanism_domain_min_cells))
    class2_domain_all = _clean_mask((class2_smooth >= class2_cut) & ~class1_domain_all, int(mechanism_domain_min_cells))
    class0_domain_all = ~(class1_domain_all | class2_domain_all)
    mechanism_domain_mask = class1_domain_all | class2_domain_all
    class1_domain = class1_domain_all & deep_water_mask
    class2_domain = class2_domain_all & deep_water_mask

    score_cut = float(np.nanpercentile(_finite_values(score1500_smooth, deep_water_mask), hotspot_score_percentile))
    hotspot1500_raw_all = ((p1500_smooth >= hotspot_prob1500) | (score1500_smooth >= score_cut)) & (confidence_smooth >= 0.35)
    hotspot1500_all = _clean_mask(hotspot1500_raw_all, int(hotspot_min_cells))
    hotspot1500_deep_all = hotspot1500_all & deep_water_mask
    dominant_noncarbonate_mask = dominant != 0
    mechanism_consistency_mask = dominant_noncarbonate_mask
    class0_hotspot = hotspot1500_deep_all & ~dominant_noncarbonate_mask
    outside_mechanism_domain_hotspot = hotspot1500_deep_all & ~mechanism_domain_mask
    mechanism_inconsistent_hotspot = class0_hotspot
    mechanism_consistent_hotspot = hotspot1500_deep_all & mechanism_consistency_mask
    hotspot1500 = mechanism_consistent_hotspot
    candidate1000_raw = (
        (p1000_smooth >= hotspot_prob1000)
        & ~hotspot1500_deep_all
        & (confidence_smooth >= 0.35)
        & deep_water_mask
        & mechanism_consistency_mask
    )
    candidate1000 = _clean_mask(candidate1000_raw, int(hotspot_min_cells))
    high_rey_peak = _clean_mask((max_rey >= 1500.0) & ~hotspot1500_deep_all & deep_water_mask & mechanism_consistency_mask, int(hotspot_min_cells))
    score1000_deep = np.where(deep_water_mask, score1000_smooth, np.nan)
    score1500_deep = np.where(deep_water_mask, score1500_smooth, np.nan)
    score1000_screen = np.where(deep_water_mask & mechanism_consistency_mask, score1000_smooth, np.nan)
    score1500_screen = np.where(deep_water_mask & mechanism_consistency_mask, score1500_smooth, np.nan)
    score1000_cut = float(np.nanpercentile(_finite_values(score1000_screen), 90.0))
    score1500_cut_95 = float(np.nanpercentile(_finite_values(score1500_screen), 95.0))
    score1000_plot_vmax = float(max(score1000_cut * 1.15, np.nanpercentile(_finite_values(score1000_deep), 98.0)))
    score1000_display_top10 = _clean_display_mask(score1000_deep >= score1000_cut, int(hotspot_min_cells) * 2)
    score1500_deep_cut_95 = float(np.nanpercentile(_finite_values(score1500_deep), 95.0))
    score1500_plot_vmax = float(max(score1500_deep_cut_95 * 1.15, np.nanpercentile(_finite_values(score1500_deep), 98.0)))
    p1000_appearance_cut = float(np.nanpercentile(_finite_values(p1000_smooth, deep_water_mask & enriched_appearance_mask), 85.0))
    p1500_appearance_cut = float(np.nanpercentile(_finite_values(p1500_smooth, deep_water_mask & enriched_appearance_mask), 95.0))
    p1500_dominant_cut = float(np.nanpercentile(_finite_values(p1500_smooth, deep_water_mask & mechanism_consistency_mask), 95.0))
    p1500_probability_hotspot = _clean_mask(
        (p1500_smooth >= p1500_dominant_cut)
        & deep_water_mask
        & mechanism_consistency_mask
        & (confidence_smooth >= 0.35),
        int(hotspot_min_cells),
    )
    appearance_p1500_tier = _clean_mask(
        (p1500_smooth >= p1500_appearance_cut)
        & deep_water_mask
        & enriched_appearance_mask
        & (confidence_smooth >= 0.35),
        int(hotspot_min_cells),
    )
    appearance_p1000_tier = _clean_mask(
        (p1000_smooth >= p1000_appearance_cut)
        & deep_water_mask
        & enriched_appearance_mask
        & (confidence_smooth >= 0.35),
        int(hotspot_min_cells),
    )
    appearance_hotspot1500 = appearance_p1500_tier
    appearance_hotspot_tier = np.zeros(dominant.shape, dtype=np.int16)
    appearance_hotspot_tier[appearance_p1000_tier] = 1
    appearance_hotspot_tier[appearance_p1500_tier] = 2
    if bathy_native is not None:
        native_lats, native_lons, native_bathy, native_extent = bathy_native
        native_deep_water_mask = np.isfinite(native_bathy) & (native_bathy <= -float(min_water_depth_m))
        native_background = _robust_background(native_bathy, native_bathy.shape)
        score1000_da = xr.DataArray(score1000_deep, coords={"latitude": lats, "longitude": lons}, dims=("latitude", "longitude"))
        score1500_da = xr.DataArray(score1500_deep, coords={"latitude": lats, "longitude": lons}, dims=("latitude", "longitude"))
        score1000_screen_da = xr.DataArray(score1000_screen, coords={"latitude": lats, "longitude": lons}, dims=("latitude", "longitude"))
        score1500_screen_da = xr.DataArray(score1500_screen, coords={"latitude": lats, "longitude": lons}, dims=("latitude", "longitude"))
        score1000_native = np.asarray(
            score1000_da.interp(latitude=native_lats, longitude=native_lons, kwargs={"fill_value": np.nan}).values,
            dtype=float,
        )
        score1500_native = np.asarray(
            score1500_da.interp(latitude=native_lats, longitude=native_lons, kwargs={"fill_value": np.nan}).values,
            dtype=float,
        )
        score1000_native_screen = np.asarray(
            score1000_screen_da.interp(latitude=native_lats, longitude=native_lons, kwargs={"fill_value": np.nan}).values,
            dtype=float,
        )
        score1500_native_screen = np.asarray(
            score1500_screen_da.interp(latitude=native_lats, longitude=native_lons, kwargs={"fill_value": np.nan}).values,
            dtype=float,
        )
        score1000_native_deep = np.where(native_deep_water_mask, score1000_native, np.nan)
        score1500_native_deep = np.where(native_deep_water_mask, score1500_native, np.nan)
        score1000_native_screen = np.where(native_deep_water_mask, score1000_native_screen, np.nan)
        score1500_native_screen = np.where(native_deep_water_mask, score1500_native_screen, np.nan)
    else:
        native_lats = lats
        native_lons = lons
        native_extent = extent
        native_background = background
        native_deep_water_mask = deep_water_mask
        score1000_native_deep = score1000_deep
        score1500_native_deep = score1500_deep
        score1000_native_screen = score1000_screen
        score1500_native_screen = score1500_screen

    fig, ax = plt.subplots(figsize=(11.5, 6.1), dpi=220)
    ax.imshow(background, origin="lower", extent=extent, cmap="Greys", aspect="auto", alpha=0.42)
    ax.contourf(lons, lats, class2_domain.astype(float), levels=[0.5, 1.5], colors=[MECHANISM_COLORS[2]], alpha=0.20)
    ax.contourf(lons, lats, class1_domain.astype(float), levels=[0.5, 1.5], colors=[MECHANISM_COLORS[1]], alpha=0.24)
    _contour_mask(ax, lons, lats, class2_domain, MECHANISM_COLORS[2], 0.85)
    _contour_mask(ax, lons, lats, class1_domain, MECHANISM_COLORS[1], 0.9)
    ax.contourf(lons, lats, hotspot1500.astype(float), levels=[0.5, 1.5], colors=["#8f0000"], alpha=0.20)
    _contour_mask(ax, lons, lats, hotspot1500, "#6e0000", 1.8)
    ax.set_title("REY enrichment mechanisms and high-priority screening domains")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.grid(color="white", alpha=0.16, linewidth=0.45)
    mechanism_handles = [
        Patch(facecolor=MECHANISM_COLORS[0], edgecolor=MECHANISM_COLORS[0], alpha=0.35, label="class0 dominant"),
        Patch(facecolor=MECHANISM_COLORS[1], edgecolor=MECHANISM_COLORS[1], alpha=0.35, label="class1 dominant"),
        Patch(facecolor=MECHANISM_COLORS[2], edgecolor=MECHANISM_COLORS[2], alpha=0.30, label="class2 dominant"),
    ]
    hotspot_handles = [
        Patch(facecolor="#8f0000", edgecolor="#6e0000", alpha=0.32, label=f"non-carbonate P1500 priority within >={min_water_depth_m:g} m"),
    ]
    leg1 = ax.legend(handles=mechanism_handles, loc="lower left", title="Mechanism domains", frameon=True, framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=hotspot_handles, loc="lower right", title="Quantitative hotspot screen", frameon=True, framealpha=0.9)
    fig.tight_layout()
    composite_png = output_dir / "composite_mechanism_hotspots.png"
    composite_svg = output_dir / "composite_mechanism_hotspots.svg"
    fig.savefig(composite_png, bbox_inches="tight")
    fig.savefig(composite_svg, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.7), dpi=220, constrained_layout=True, sharex=True, sharey=True)
    for ax in axes:
        ax.imshow(background, origin="lower", extent=extent, cmap="Greys", aspect="auto", alpha=0.36)
        ax.set_xlabel("Longitude")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.grid(color="white", alpha=0.14, linewidth=0.45)
    axes[0].set_ylabel("Latitude")
    axes[0].set_title("A. Mechanism domains")
    axes[0].contourf(lons, lats, class2_domain.astype(float), levels=[0.5, 1.5], colors=[MECHANISM_COLORS[2]], alpha=0.22)
    axes[0].contourf(lons, lats, class1_domain.astype(float), levels=[0.5, 1.5], colors=[MECHANISM_COLORS[1]], alpha=0.26)
    _contour_mask(axes[0], lons, lats, class2_domain, MECHANISM_COLORS[2], 0.9)
    _contour_mask(axes[0], lons, lats, class1_domain, MECHANISM_COLORS[1], 0.95)
    axes[0].legend(handles=mechanism_handles, loc="lower left", frameon=True, framealpha=0.9)

    axes[1].set_title("B. High-priority REY screening")
    score_plot = np.where(hotspot1500, score1500_smooth, np.nan)
    im = axes[1].imshow(score_plot, origin="lower", extent=extent, cmap="Reds", aspect="auto", alpha=0.62)
    _contour_mask(axes[1], lons, lats, hotspot1500, "#6e0000", 1.8)
    axes[1].legend(handles=hotspot_handles, loc="lower right", frameon=True, framealpha=0.9)
    cbar = fig.colorbar(im, ax=axes[1], shrink=0.78)
    cbar.ax.set_ylabel("P1500 screening score")
    two_panel_png = output_dir / "paper_two_panel_mechanism_hotspots.png"
    two_panel_svg = output_dir / "paper_two_panel_mechanism_hotspots.svg"
    fig.savefig(two_panel_png, bbox_inches="tight")
    fig.savefig(two_panel_svg, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.3), dpi=220, constrained_layout=True)
    im0 = axes[0].imshow(score1500_deep, origin="lower", extent=extent, cmap="magma", aspect="auto")
    axes[0].contour(lons, lats, hotspot1500.astype(float), levels=[0.5], colors=["white"], linewidths=1.0)
    axes[0].set_title("Reliability-weighted P1500 screening score")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    fig.colorbar(im0, ax=axes[0], shrink=0.86, label="screening_score_ge1500 in deep-water domain")
    tier = np.where(hotspot1500, 2, np.where(candidate1000, 1, 0))
    cmap = ListedColormap(["#d8d8d8", "#f0a22e", "#7f0000"])
    im1 = axes[1].imshow(tier, origin="lower", extent=extent, cmap=cmap, vmin=0, vmax=2, aspect="auto")
    axes[1].set_title("Hotspot priority tier")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    cbar = fig.colorbar(im1, ax=axes[1], shrink=0.86, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["background", "P1000 candidate", "P1500 priority"])
    quantitative_png = output_dir / "composite_hotspot_quantitative_panels.png"
    fig.savefig(quantitative_png, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.4), dpi=220, constrained_layout=True, sharex=True, sharey=True)
    for ax in axes:
        ax.imshow(background, origin="lower", extent=extent, cmap="Greys", aspect="auto", alpha=0.34)
        ax.set_xlabel("Longitude")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.grid(color="white", alpha=0.14, linewidth=0.42)
    axes[0].set_ylabel("Latitude")

    axes[0].set_title("A. Observed core-scale REY support")
    if not core_support.empty:
        low = core_support[core_support["support_tier"] == 0]
        mid = core_support[core_support["support_tier"] == 1]
        high = core_support[core_support["support_tier"] == 2]
        if not low.empty:
            axes[0].scatter(
                low["longitude"],
                low["latitude"],
                s=np.clip(np.sqrt(low["n_samples"]) * 7.0, 10, 50),
                c="#8f8f8f",
                alpha=0.42,
                linewidths=0,
                label="<1000 ppm cores",
            )
        if not mid.empty:
            axes[0].scatter(
                mid["longitude"],
                mid["latitude"],
                s=np.clip(np.sqrt(mid["n_samples"]) * 8.5, 18, 70),
                c="#f0a22e",
                edgecolors="#8a5a00",
                linewidths=0.45,
                alpha=0.82,
                label="core has REY >=1000",
            )
        if not high.empty:
            axes[0].scatter(
                high["longitude"],
                high["latitude"],
                s=np.clip(np.sqrt(high["n_samples"]) * 10.0, 26, 95),
                c="#8f0000",
                edgecolors="white",
                linewidths=0.5,
                alpha=0.9,
                label="core has REY >=1500",
            )
        axes[0].legend(loc="lower left", frameon=True, framealpha=0.88, fontsize=8)

    for image in list(axes[1].images):
        image.remove()
    axes[1].set_facecolor("#f8f8f6")
    axes[1].set_title("B. Mechanism domains")
    axes[1].contourf(lons, lats, class0_domain_all.astype(float), levels=[0.5, 1.5], colors=[MECHANISM_COLORS[0]], alpha=0.18)
    axes[1].contourf(lons, lats, class2_domain_all.astype(float), levels=[0.5, 1.5], colors=[MECHANISM_COLORS[2]], alpha=0.22)
    axes[1].contourf(lons, lats, class1_domain_all.astype(float), levels=[0.5, 1.5], colors=[MECHANISM_COLORS[1]], alpha=0.26)
    _contour_mask(axes[1], lons, lats, class0_domain_all, MECHANISM_COLORS[0], 0.55)
    _contour_mask(axes[1], lons, lats, class2_domain_all, MECHANISM_COLORS[2], 0.8)
    _contour_mask(axes[1], lons, lats, class1_domain_all, MECHANISM_COLORS[1], 0.85)
    axes[1].legend(handles=mechanism_handles, loc="lower left", frameon=True, framealpha=0.88, fontsize=8)

    axes[2].set_title(f"C. P1000 screening intensity (>={min_water_depth_m:g} m)")
    im = axes[2].imshow(score1000_deep, origin="lower", extent=extent, cmap="YlOrRd", aspect="auto", alpha=0.74, vmin=0.0, vmax=score1000_plot_vmax)
    _contour_mask(axes[2], lons, lats, score1000_display_top10, "white", 2.15)
    _contour_mask(axes[2], lons, lats, score1000_display_top10, "#7a0000", 1.15)
    screen_handles = [
        Line2D([0], [0], color="#7a0000", linewidth=1.25, label=f"top 10% P1000 score >= {score1000_cut:.2f}"),
    ]
    axes[2].legend(handles=screen_handles, loc="lower right", frameon=True, framealpha=0.88, fontsize=8)
    cbar = fig.colorbar(im, ax=axes[2], shrink=0.78)
    cbar.ax.set_ylabel("P1000 screening score")
    three_panel_png = output_dir / "paper_three_panel_evidence_screening.png"
    three_panel_svg = output_dir / "paper_three_panel_evidence_screening.svg"
    fig.savefig(three_panel_png, bbox_inches="tight")
    fig.savefig(three_panel_svg, bbox_inches="tight")
    plt.close(fig)

    distance_plot = np.clip(distance_to_core_km, 0.0, float(distance_max_km))
    fig, ax = _map_axes("Distance to nearest observed REY core", extent)
    im_dist = _imshow_map(
        ax,
        distance_plot,
        extent,
        cmap="viridis_r",
        vmin=0.0,
        vmax=float(distance_max_km),
        alpha=0.92,
    )
    _finish_map(ax)
    if not core_support.empty:
        low = core_support[core_support["support_tier"] == 0]
        mid = core_support[core_support["support_tier"] == 1]
        high = core_support[core_support["support_tier"] == 2]
        if not low.empty:
            _scatter_map(
                ax,
                low["longitude"],
                low["latitude"],
                s=np.clip(np.sqrt(low["n_samples"]) * 8.0, 12, 58),
                c="#8f8f8f",
                alpha=0.45,
                linewidths=0,
                zorder=30,
                label="<1000 ppm cores",
            )
        if not mid.empty:
            _scatter_map(
                ax,
                mid["longitude"],
                mid["latitude"],
                s=np.clip(np.sqrt(mid["n_samples"]) * 9.5, 20, 80),
                c="#f0a22e",
                edgecolors="#8a5a00",
                linewidths=0.45,
                alpha=0.84,
                zorder=31,
                label="core has REY >=1000",
            )
        if not high.empty:
            _scatter_map(
                ax,
                high["longitude"],
                high["latitude"],
                s=np.clip(np.sqrt(high["n_samples"]) * 10.5, 30, 105),
                c="#8f0000",
                edgecolors="white",
                linewidths=0.5,
                alpha=0.92,
                zorder=32,
                label="core has REY >=1500",
            )
    cbar = fig.colorbar(im_dist, ax=ax, shrink=0.82)
    cbar.ax.set_ylabel("Distance to nearest observed REY core (km)")
    if not core_support.empty:
        _legend_upper_right(ax, fontsize=8)
    observed_png = output_dir / "paper_observed_core_support_map.png"
    observed_svg = output_dir / "paper_observed_core_support_map.svg"
    distance_support_png = output_dir / "paper_core_distance_support_map.png"
    distance_support_svg = output_dir / "paper_core_distance_support_map.svg"
    fig.savefig(observed_png, bbox_inches="tight")
    fig.savefig(observed_svg, bbox_inches="tight")
    fig.savefig(distance_support_png, bbox_inches="tight")
    fig.savefig(distance_support_svg, bbox_inches="tight")
    plt.close(fig)

    fig, ax = _map_axes(f"Dominant mechanism summary (>={min_water_depth_m:g} m)", extent)
    dominant_values = np.where(deep_water_mask & np.isfinite(mechanism_confidence), dominant.astype(float), np.nan)
    paper_mechanism_colors = {
        0: "#d9d9d9",
        1: "#eec7c4",
        2: "#c7d9ea",
    }
    paper_mechanism_handles = [
        Patch(facecolor=paper_mechanism_colors[0], edgecolor="#bdbdbd", alpha=0.88, label="class0"),
        Patch(facecolor=paper_mechanism_colors[1], edgecolor="#d79b98", alpha=0.88, label="class1"),
        Patch(facecolor=paper_mechanism_colors[2], edgecolor="#8fb5d4", alpha=0.88, label="class2"),
    ]
    dominant_cmap = ListedColormap([paper_mechanism_colors[0], paper_mechanism_colors[1], paper_mechanism_colors[2]])
    _imshow_map(
        ax,
        dominant_values,
        extent,
        cmap=dominant_cmap,
        vmin=-0.5,
        vmax=2.5,
        alpha=0.95,
        interpolation="nearest",
        zorder=4,
    )
    _finish_map(ax)
    _legend_upper_right(ax, handles=paper_mechanism_handles, fontsize=8)
    mechanism_png = output_dir / "paper_mechanism_domains_map.png"
    mechanism_svg = output_dir / "paper_mechanism_domains_map.svg"
    fig.savefig(mechanism_png, bbox_inches="tight")
    fig.savefig(mechanism_svg, bbox_inches="tight")
    plt.close(fig)

    fig, ax = _map_axes(f"Dominant mechanism and P1500 >= {p1500_dominant_cut:.2f} zones (>={min_water_depth_m:g} m)", extent)
    _imshow_map(
        ax,
        dominant_values,
        extent,
        cmap=dominant_cmap,
        vmin=-0.5,
        vmax=2.5,
        alpha=0.95,
        interpolation="nearest",
        zorder=4,
    )
    _hatch_mask(ax, lons, lats, p1500_probability_hotspot, hatch="xxxx", color="#4a0000", linewidth=0.5, zorder=9)
    _contour_mask(ax, lons, lats, p1500_probability_hotspot, "#2a0000", 1.45)
    _finish_map(ax)
    overlay_handles = [
        *paper_mechanism_handles,
        Patch(facecolor="none", edgecolor="#4a0000", hatch="xxxx", linewidth=0.8, label=f"P1500 >= {p1500_dominant_cut:.2f}"),
    ]
    _legend_upper_right(ax, handles=overlay_handles, fontsize=8)
    overlay_png = output_dir / "paper_mechanism_hotspot_overlay_map.png"
    overlay_svg = output_dir / "paper_mechanism_hotspot_overlay_map.svg"
    fig.savefig(overlay_png, bbox_inches="tight")
    fig.savefig(overlay_svg, bbox_inches="tight")
    plt.close(fig)

    fig, ax = _map_axes(f"Class1/2 mechanism appearance and REY probability tiers (>={min_water_depth_m:g} m)", extent)
    appearance_values = np.where(deep_water_mask & np.isfinite(mechanism_confidence), appearance_class.astype(float), np.nan)
    appearance_cmap = ListedColormap(["#e1e1e1", "#eec7c4", "#c7d9ea"])
    _imshow_map(
        ax,
        appearance_values,
        extent,
        cmap=appearance_cmap,
        vmin=-0.5,
        vmax=2.5,
        alpha=0.92,
        interpolation="nearest",
        zorder=4,
    )
    _hatch_mask(ax, lons, lats, appearance_p1000_tier, hatch="////", color="#c78318", linewidth=0.35, zorder=8)
    _hatch_mask(ax, lons, lats, appearance_p1500_tier, hatch="xxxx", color="#4a0000", linewidth=0.5, zorder=9)
    _contour_mask(ax, lons, lats, appearance_p1000_tier, "#d79225", 0.65)
    _contour_mask(ax, lons, lats, appearance_p1500_tier, "#2a0000", 1.25)
    _finish_map(ax)
    appearance_handles = [
        Patch(facecolor="#d9d9d9", edgecolor="#bdbdbd", alpha=0.42, label="no class1/2 appearance"),
        Patch(facecolor=MECHANISM_COLORS[1], edgecolor=MECHANISM_COLORS[1], alpha=0.35, label="class1 appearance"),
        Patch(facecolor=MECHANISM_COLORS[2], edgecolor=MECHANISM_COLORS[2], alpha=0.30, label="class2 appearance"),
        Patch(facecolor="none", edgecolor="#c78318", hatch="////", linewidth=0.7, label=f"P1000 >= {p1000_appearance_cut:.2f}"),
        Patch(facecolor="none", edgecolor="#4a0000", hatch="xxxx", linewidth=0.9, label=f"P1500 >= {p1500_appearance_cut:.2f}"),
    ]
    _legend_upper_right(ax, handles=appearance_handles, fontsize=8)
    appearance_overlay_png = output_dir / "paper_mechanism_appearance_hotspot_overlay_map.png"
    appearance_overlay_svg = output_dir / "paper_mechanism_appearance_hotspot_overlay_map.svg"
    fig.savefig(appearance_overlay_png, bbox_inches="tight")
    fig.savefig(appearance_overlay_svg, bbox_inches="tight")
    plt.close(fig)

    rey_domain_mask = deep_water_mask & np.isfinite(max_rey) & (max_rey > 0.0)
    rey_values = np.where(rey_domain_mask, max_rey, np.nan)
    rey_finite = _finite_values(max_rey, rey_domain_mask)
    if rey_finite.size:
        rey_vmin = float(max(50.0, np.nanpercentile(rey_finite, 5.0)))
        rey_vmax = float(max(rey_vmin * 1.1, np.nanpercentile(rey_finite, 98.0)))
        rey_hotspot_cut = float(np.nanpercentile(rey_finite, 95.0))
    else:
        rey_vmin = 50.0
        rey_vmax = 2000.0
        rey_hotspot_cut = 1500.0
    rey_hotspot95 = _clean_mask(rey_domain_mask & (max_rey >= rey_hotspot_cut), int(hotspot_min_cells))

    fig, ax = _map_axes(f"Predicted peak REY and top 5% REY hotspots (>={min_water_depth_m:g} m)", extent)
    im_rey = _imshow_map(
        ax,
        rey_values,
        extent,
        cmap="YlGnBu",
        norm=LogNorm(vmin=rey_vmin, vmax=rey_vmax),
        alpha=0.94,
        interpolation="nearest",
        zorder=4,
    )
    _contour_mask(ax, lons, lats, rey_hotspot95, "white", 2.8)
    _contour_mask(ax, lons, lats, rey_hotspot95, "#7a0000", 1.65)
    _finish_map(ax)
    cbar = fig.colorbar(im_rey, ax=ax, shrink=0.82)
    cbar.ax.set_ylabel("Predicted peak REY (ppm)")
    rey_handles = [
        Line2D([0], [0], color="#7a0000", linewidth=1.8, label=f"top 5% REY >= {rey_hotspot_cut:.0f} ppm"),
    ]
    _legend_upper_right(ax, handles=rey_handles, fontsize=8)
    rey_continuous_png = output_dir / "paper_rey_continuous_hotspot95_map.png"
    rey_continuous_svg = output_dir / "paper_rey_continuous_hotspot95_map.svg"
    rey_continuous_legacy_png = output_dir / "paper_rey_continuous_appearance_hotspot_map.png"
    rey_continuous_legacy_svg = output_dir / "paper_rey_continuous_appearance_hotspot_map.svg"
    fig.savefig(rey_continuous_png, bbox_inches="tight")
    fig.savefig(rey_continuous_svg, bbox_inches="tight")
    fig.savefig(rey_continuous_legacy_png, bbox_inches="tight")
    fig.savefig(rey_continuous_legacy_svg, bbox_inches="tight")
    plt.close(fig)

    rey_rank_percentile = np.full(max_rey.shape, np.nan, dtype=float)
    if rey_finite.size:
        ranks = pd.Series(rey_finite).rank(method="average", pct=True).to_numpy(dtype=float) * 100.0
        rey_rank_percentile[rey_domain_mask] = ranks
    rey_rank_top10 = _clean_mask(rey_domain_mask & (rey_rank_percentile >= 90.0), int(hotspot_min_cells))
    rey_rank_top5 = _clean_mask(rey_domain_mask & (rey_rank_percentile >= 95.0), int(hotspot_min_cells))

    fig, ax = _map_axes(f"Model-ranked REY enrichment percentile (>={min_water_depth_m:g} m)", extent)
    im_rank = _imshow_map(
        ax,
        rey_rank_percentile,
        extent,
        cmap="viridis",
        vmin=0.0,
        vmax=100.0,
        alpha=0.94,
        interpolation="nearest",
        zorder=4,
    )
    _contour_mask(ax, lons, lats, rey_rank_top5, "white", 3.0)
    _contour_mask(ax, lons, lats, rey_rank_top5, "#7a0000", 1.85)
    _finish_map(ax)
    cbar = fig.colorbar(im_rank, ax=ax, shrink=0.82)
    cbar.ax.set_ylabel("Predicted REY rank percentile")
    rank_handles = [
        Line2D([0], [0], color="#7a0000", linewidth=1.9, label="top 5% ranked REY"),
    ]
    _legend_upper_right(ax, handles=rank_handles, fontsize=8)
    rey_rank_png = output_dir / "paper_rey_rank_percentile_hotspot_map.png"
    rey_rank_svg = output_dir / "paper_rey_rank_percentile_hotspot_map.svg"
    fig.savefig(rey_rank_png, bbox_inches="tight")
    fig.savefig(rey_rank_svg, bbox_inches="tight")
    plt.close(fig)

    fig, ax = _map_axes(f"P1000 screening intensity (>={min_water_depth_m:g} m water depth)", extent)
    _imshow_map(ax, native_background, native_extent, cmap="Greys", alpha=0.16, interpolation="bilinear")
    im = _imshow_map(
        ax,
        score1000_native_deep,
        native_extent,
        cmap="YlOrRd",
        vmin=0.0,
        vmax=score1000_plot_vmax,
        alpha=0.78,
        interpolation="bilinear",
    )
    _contour_mask(ax, lons, lats, score1000_display_top10, "white", 2.25)
    _contour_mask(ax, lons, lats, score1000_display_top10, "#7a0000", 1.25)
    _finish_map(ax)
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.ax.set_ylabel("P1000 screening score")
    _legend_upper_right(ax, handles=screen_handles, fontsize=8)
    screening_png = output_dir / "paper_model_screening_intensity_map.png"
    screening_svg = output_dir / "paper_model_screening_intensity_map.svg"
    fig.savefig(screening_png, bbox_inches="tight")
    fig.savefig(screening_svg, bbox_inches="tight")
    plt.close(fig)

    split_map_min_water_depth_m = 3500.0
    if bathy is None:
        split_map_deep_water_mask = np.ones(dominant.shape, dtype=bool)
    else:
        split_map_deep_water_mask = np.isfinite(bathy) & (bathy <= -split_map_min_water_depth_m)
    split_map_domain = split_map_deep_water_mask & np.isfinite(p1000)

    if classwise_min_distance_to_land_km is not None and distance_to_land_km is not None:
        classwise_distance_mask = np.isfinite(distance_to_land_km) & (
            distance_to_land_km >= float(classwise_min_distance_to_land_km)
        )
    else:
        classwise_distance_mask = np.ones(dominant.shape, dtype=bool)
    classwise_map_domain = classwise_deep_water_mask & classwise_distance_mask & np.isfinite(p1000)
    p1000_probability_artifacts, p1000_probability_stats = _plot_p1000_probability_screening_map(
        output_dir=output_dir,
        lons=lons,
        lats=lats,
        extent=extent,
        p1000=p1000,
        domain_mask=split_map_domain,
        boundary_segments=boundary_segments,
        min_water_depth_m=float(split_map_min_water_depth_m),
        min_distance_to_land_km=None,
    )
    facies_hotspot_only_artifacts, facies_hotspot_only_stats = _plot_facies_p1000_hotspot_only_map(
        output_dir=output_dir,
        lons=lons,
        lats=lats,
        extent=extent,
        dominant=dominant,
        p1000=p1000,
        domain_mask=split_map_domain,
        boundary_segments=boundary_segments,
        min_water_depth_m=float(split_map_min_water_depth_m),
        min_distance_to_land_km=None,
        top_percent=10.0,
    )
    classwise_artifacts, classwise_stats = _plot_classwise_p1000_map(
        output_dir=output_dir,
        lons=lons,
        lats=lats,
        extent=extent,
        dominant=dominant,
        p1000=p1000,
        deep_water_mask=classwise_deep_water_mask,
        min_water_depth_m=float(classwise_min_water_depth_m),
        hotspot_min_cells=int(hotspot_min_cells),
        boundary_segments=boundary_segments,
        apply_water_depth_mask=True,
        distance_to_land_km=distance_to_land_km,
        min_distance_to_land_km=classwise_min_distance_to_land_km,
        top_percent=10.0,
    )
    if p800 is not None:
        classwise_p800_artifacts, classwise_p800_stats = _plot_classwise_p1000_map(
            output_dir=output_dir,
            lons=lons,
            lats=lats,
            extent=extent,
            dominant=dominant,
            p1000=p1000,
            deep_water_mask=classwise_deep_water_mask,
            min_water_depth_m=float(classwise_min_water_depth_m),
            hotspot_min_cells=int(hotspot_min_cells),
            boundary_segments=boundary_segments,
            apply_water_depth_mask=True,
            distance_to_land_km=distance_to_land_km,
            min_distance_to_land_km=classwise_min_distance_to_land_km,
            top_percent=10.0,
            class1_prob=p1000,
            class2_prob=p800,
            class1_threshold_label="P1000",
            class2_threshold_label="P800",
            title="Facies-separated P1000/P800 screening domains",
            output_stem="paper_classwise_p1000_p800_top10_map",
            artifact_prefix="classwise_p1000_p800_top10",
            stats_prefix="classwise_p1000_p800",
        )
        classwise_p800_top5_artifacts, classwise_p800_top5_stats = _plot_classwise_p1000_map(
            output_dir=output_dir,
            lons=lons,
            lats=lats,
            extent=extent,
            dominant=dominant,
            p1000=p1000,
            deep_water_mask=classwise_deep_water_mask,
            min_water_depth_m=float(classwise_min_water_depth_m),
            hotspot_min_cells=int(hotspot_min_cells),
            boundary_segments=boundary_segments,
            apply_water_depth_mask=True,
            distance_to_land_km=distance_to_land_km,
            min_distance_to_land_km=classwise_min_distance_to_land_km,
            top_percent=5.0,
            class1_prob=p1000,
            class2_prob=p800,
            class1_threshold_label="P1000",
            class2_threshold_label="P800",
            title="Facies-separated P1000/P800 screening cores",
            output_stem="paper_classwise_p1000_p800_top5_map",
            artifact_prefix="classwise_p1000_p800_top5",
            stats_prefix="classwise_p1000_p800_top5",
        )
        classwise_p800_top5_variant_artifacts, classwise_p800_top5_variant_stats = _plot_classwise_hotspot_design_variants(
            output_dir=output_dir,
            lons=lons,
            lats=lats,
            extent=extent,
            dominant=dominant,
            p1000=p1000,
            deep_water_mask=classwise_deep_water_mask,
            min_water_depth_m=float(classwise_min_water_depth_m),
            boundary_segments=boundary_segments,
            distance_to_land_km=distance_to_land_km,
            min_distance_to_land_km=classwise_min_distance_to_land_km,
            top_percent=5.0,
            class1_prob=p1000,
            class2_prob=p800,
            class1_threshold_label="P1000",
            class2_threshold_label="P800",
        )
        classwise_p800_top5_4000m_artifacts, classwise_p800_top5_4000m_stats = _plot_classwise_hotspot_design_variants(
            output_dir=output_dir,
            lons=lons,
            lats=lats,
            extent=extent,
            dominant=dominant,
            p1000=p1000,
            deep_water_mask=classwise_4000_deep_water_mask,
            min_water_depth_m=float(classwise_4000_min_water_depth_m),
            boundary_segments=boundary_segments,
            distance_to_land_km=distance_to_land_km,
            min_distance_to_land_km=classwise_min_distance_to_land_km,
            top_percent=5.0,
            class1_prob=p1000,
            class2_prob=p800,
            class1_threshold_label="P1000",
            class2_threshold_label="P800",
            output_suffix="_4000m",
            title_suffix=" (>=4000 m)",
            artifact_prefix="classwise_p1000_p800_top5_4000m",
            stats_prefix="classwise_p1000_p800_top5_4000m",
        )
    else:
        classwise_p800_artifacts = {}
        classwise_p800_stats = {
            "classwise_p1000_p800_available": False,
            "classwise_p1000_p800_missing_reason": "prob_rey_ge_800ppm not found in 3D prediction NetCDF",
        }
        classwise_p800_top5_artifacts = {}
        classwise_p800_top5_stats = {
            "classwise_p1000_p800_top5_available": False,
            "classwise_p1000_p800_top5_missing_reason": "prob_rey_ge_800ppm not found in 3D prediction NetCDF",
        }
        classwise_p800_top5_variant_artifacts = {}
        classwise_p800_top5_variant_stats = {
            "classwise_p1000_p800_top5_variant_available": False,
            "classwise_p1000_p800_top5_variant_missing_reason": "prob_rey_ge_800ppm not found in 3D prediction NetCDF",
        }
        classwise_p800_top5_4000m_artifacts = {}
        classwise_p800_top5_4000m_stats = {
            "classwise_p1000_p800_top5_4000m_available": False,
            "classwise_p1000_p800_top5_4000m_missing_reason": "prob_rey_ge_800ppm not found in 3D prediction NetCDF",
        }
    facies_tier_artifacts, facies_tier_stats = _plot_facies_p1000_percentile_tier_map(
        output_dir=output_dir,
        lons=lons,
        lats=lats,
        extent=extent,
        dominant=dominant,
        p1000=p1000,
        deep_water_mask=deep_water_mask,
        min_water_depth_m=float(min_water_depth_m),
        hotspot_min_cells=int(hotspot_min_cells),
    )
    unmasked_facies_tier_artifacts, unmasked_facies_tier_stats = _plot_facies_p1000_percentile_tier_map(
        output_dir=output_dir,
        lons=lons,
        lats=lats,
        extent=extent,
        dominant=dominant,
        p1000=p1000,
        deep_water_mask=np.isfinite(p1000),
        min_water_depth_m=None,
        hotspot_min_cells=int(hotspot_min_cells),
        title="Facies-specific P1000 percentile tiers (no water-depth mask)",
        output_stem="paper_facies_p1000_percentile_tiers_unmasked_map",
        artifact_prefix="facies_p1000_percentile_tiers_unmasked",
        stats_prefix="facies_unmasked",
    )
    clean_facies_tier_artifacts, clean_facies_tier_stats = _plot_facies_p1000_clean_screening_map(
        output_dir=output_dir,
        lons=lons,
        lats=lats,
        extent=extent,
        dominant=dominant,
        p1000=p1000,
        domain_mask=np.isfinite(p1000),
        hotspot_min_cells=int(hotspot_min_cells),
        include_mid_tier=True,
        title="Facies-specific high-value REY screening tiers",
        output_stem="paper_facies_p1000_percentile_tiers_clean_map",
        artifact_prefix="facies_p1000_percentile_tiers_clean",
        stats_prefix="facies_clean",
    )
    clean_top_quartile_artifacts, clean_top_quartile_stats = _plot_facies_p1000_clean_screening_map(
        output_dir=output_dir,
        lons=lons,
        lats=lats,
        extent=extent,
        dominant=dominant,
        p1000=p1000,
        domain_mask=np.isfinite(p1000),
        hotspot_min_cells=int(hotspot_min_cells),
        include_mid_tier=False,
        title="Facies-specific high-value REY screening domains",
        output_stem="paper_facies_p1000_top_quartile_clean_map",
        artifact_prefix="facies_p1000_top_quartile_clean",
        stats_prefix="facies_top_quartile_clean",
    )
    facies_specific_threshold_artifacts, facies_specific_threshold_stats = _plot_facies_specific_threshold_screening_map(
        output_dir=output_dir,
        lons=lons,
        lats=lats,
        extent=extent,
        dominant=dominant,
        p1000=p1000,
        p1500=p1500,
        domain_mask=np.isfinite(p1000) | np.isfinite(p1500),
        boundary_segments=boundary_segments,
        hotspot_min_cells=int(hotspot_min_cells),
        hotspot_percentile=95.0,
    )
    facies_specific_publication_artifacts, facies_specific_publication_stats = _plot_facies_specific_threshold_publication_map(
        output_dir=output_dir,
        lons=lons,
        lats=lats,
        extent=extent,
        dominant=dominant,
        p1000=p1000,
        p1500=p1500,
        domain_mask=np.isfinite(p1000) | np.isfinite(p1500),
        boundary_segments=boundary_segments,
        hotspot_min_cells=int(hotspot_min_cells),
        hotspot_percentile=95.0,
        smooth_sigma=1.15,
    )

    lon2, lat2 = np.meshgrid(lons, lats)
    table = pd.DataFrame(
        {
            "longitude": lon2.ravel(),
            "latitude": lat2.ravel(),
            "dominant_mechanism_class_id": dominant.ravel(),
            "dominant_mechanism_name": [MECHANISM_NAMES[int(x)] if np.isfinite(x) and 0 <= int(x) < len(MECHANISM_NAMES) else "" for x in dominant.ravel()],
            "dominant_mechanism_confidence": mechanism_confidence.ravel(),
            "prob_carbonate_background_mean": mech_probs[0].ravel(),
            "prob_class1_enriched_mean": mech_probs[1].ravel(),
            "prob_class2_slow_sedimentation_mean": mech_probs[2].ravel(),
            "class1_appearance_prob_max": class1_appearance_prob.ravel(),
            "class2_appearance_prob_max": class2_appearance_prob.ravel(),
            "appearance_class_id": appearance_class.ravel(),
            "has_class1_or_class2_appearance": enriched_appearance_mask.ravel().astype(int),
            "bathymetry_value_m": bathy.ravel() if bathy is not None else np.full(lon2.size, np.nan),
            "water_depth_abs_m": water_depth_abs.ravel(),
            "is_deep_water_domain": deep_water_mask.ravel().astype(int),
            "distance_to_land_km": distance_to_land_km.ravel() if distance_to_land_km is not None else np.full(lon2.size, np.nan),
            "is_classwise_p1000_display_domain": classwise_map_domain.ravel().astype(int),
            "max_pred_rey_ppm": max_rey.ravel(),
            "max_prob_rey_ge_800ppm": p800.ravel() if p800 is not None else np.full(lon2.size, np.nan),
            "max_prob_rey_ge_1000ppm": p1000.ravel(),
            "max_prob_rey_ge_1500ppm": p1500.ravel(),
            "screening_score_ge1000": score1000.ravel(),
            "screening_score_ge1500": score1500.ravel(),
            "prediction_confidence_mean": confidence.ravel(),
            "hotspot_tier": tier.ravel(),
            "is_p1500_priority_all_depths": hotspot1500_all.ravel().astype(int),
            "is_p1500_priority_deep_all_mechanisms": hotspot1500_deep_all.ravel().astype(int),
            "is_p1500_priority": hotspot1500.ravel().astype(int),
            "is_p1500_priority_deep": hotspot1500.ravel().astype(int),
            "is_p1500_probability_tier": p1500_probability_hotspot.ravel().astype(int),
            "is_mechanism_consistent_p1500": mechanism_consistent_hotspot.ravel().astype(int),
            "is_mechanism_inconsistent_p1500": mechanism_inconsistent_hotspot.ravel().astype(int),
            "is_outside_mechanism_domain_p1500": outside_mechanism_domain_hotspot.ravel().astype(int),
            "is_appearance_consistent_p1500": appearance_hotspot1500.ravel().astype(int),
            "is_appearance_p1000_tier": appearance_p1000_tier.ravel().astype(int),
            "is_appearance_p1500_tier": appearance_p1500_tier.ravel().astype(int),
            "appearance_hotspot_tier": appearance_hotspot_tier.ravel(),
            "is_p1000_candidate": candidate1000.ravel().astype(int),
        }
    )
    table = table.sort_values(
        ["hotspot_tier", "screening_score_ge1500", "max_prob_rey_ge_1500ppm", "max_pred_rey_ppm"],
        ascending=[False, False, False, False],
    )
    table_path = output_dir / "composite_hotspot_table.csv"
    table.to_csv(table_path, index=False, encoding="utf-8-sig")

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "prediction_dir": str(prediction_dir),
        "summary_nc": str(summary_nc),
        "prediction_3d_nc": str(pred3_nc) if pred3_nc is not None else None,
        "env_nc": str(env_nc),
        "points_parquet": str(points_parquet),
        "bathymetry_var": bathymetry_var,
        "min_water_depth_m": float(min_water_depth_m),
        "mechanism_min_prob": float(mechanism_min_prob),
        "hotspot_prob1000": float(hotspot_prob1000),
        "hotspot_prob1500": float(hotspot_prob1500),
        "hotspot_score_percentile": float(hotspot_score_percentile),
        "smoothing_sigma": float(smoothing_sigma),
        "mechanism_domain_min_cells": int(mechanism_domain_min_cells),
        "hotspot_min_cells": int(hotspot_min_cells),
        "distance_max_km": float(distance_max_km),
        "appearance_prob_threshold": float(appearance_prob_threshold),
        "class1_domain_cutoff": class1_cut,
        "class2_domain_cutoff": class2_cut,
        "screening_score_ge1000_cutoff": score1000_cut,
        "screening_score_ge1500_cutoff": score_cut,
        "screening_score_ge1500_top5_cutoff": score1500_cut_95,
        "screening_score_ge1000_plot_vmax": score1000_plot_vmax,
        "screening_score_ge1500_deep_top5_cutoff": score1500_deep_cut_95,
        "screening_score_ge1500_plot_vmax": score1500_plot_vmax,
        "appearance_p1000_probability_cutoff": p1000_appearance_cut,
        "appearance_p1500_probability_cutoff": p1500_appearance_cut,
        "dominant_p1500_probability_cutoff": p1500_dominant_cut,
        "rey_continuous_map_vmin_ppm": rey_vmin,
        "rey_continuous_map_vmax_ppm": rey_vmax,
        "rey_continuous_hotspot95_cutoff_ppm": rey_hotspot_cut,
        "rey_rank_top10_percentile_cutoff": 90.0,
        "rey_rank_top5_percentile_cutoff": 95.0,
        **p1000_probability_stats,
        **facies_hotspot_only_stats,
        **classwise_stats,
        **classwise_p800_stats,
        **classwise_p800_top5_stats,
        **classwise_p800_top5_variant_stats,
        **classwise_p800_top5_4000m_stats,
        **facies_tier_stats,
        **unmasked_facies_tier_stats,
        **clean_facies_tier_stats,
        **clean_top_quartile_stats,
        **facies_specific_threshold_stats,
        **facies_specific_publication_stats,
        "boundary_gmt": str(boundary_gmt) if boundary_gmt is not None else None,
        "n_boundary_segments": int(len(boundary_segments)),
        "distance_to_nearest_core_km_min": float(np.nanmin(distance_to_core_km)) if np.isfinite(distance_to_core_km).any() else None,
        "distance_to_nearest_core_km_median": float(np.nanmedian(distance_to_core_km)) if np.isfinite(distance_to_core_km).any() else None,
        "distance_to_nearest_core_km_max": float(np.nanmax(distance_to_core_km)) if np.isfinite(distance_to_core_km).any() else None,
        "n_cells": int(table.shape[0]),
        "n_deep_water_cells": int(np.nansum(deep_water_mask)),
        "n_class1_domain_cells": int(np.nansum(class1_domain)),
        "n_class2_domain_cells": int(np.nansum(class2_domain)),
        "n_class1_appearance_cells": int(np.nansum(appearance_class == 1)),
        "n_class2_appearance_cells": int(np.nansum(appearance_class == 2)),
        "n_no_class1_or_class2_appearance_cells": int(np.nansum(appearance_class == 0)),
        "n_p1500_priority_cells_all_depths": int(np.nansum(hotspot1500_all)),
        "n_p1500_priority_cells_deep_all_mechanisms": int(np.nansum(hotspot1500_deep_all)),
        "n_p1500_priority_cells": int(np.nansum(hotspot1500)),
        "n_p1500_priority_cells_deep": int(np.nansum(hotspot1500)),
        "n_p1500_probability_tier_cells": int(np.nansum(p1500_probability_hotspot)),
        "n_p1500_priority_cells_deep_class0": int(np.nansum(mechanism_inconsistent_hotspot)),
        "n_p1500_priority_cells_deep_nonclass0": int(np.nansum(hotspot1500_deep_all & dominant_noncarbonate_mask)),
        "n_p1500_priority_cells_deep_mechanism_domain": int(np.nansum(mechanism_consistent_hotspot)),
        "n_p1500_priority_cells_appearance_consistent": int(np.nansum(appearance_hotspot1500)),
        "n_appearance_p1000_tier_cells": int(np.nansum(appearance_p1000_tier)),
        "n_appearance_p1000_only_tier_cells": int(np.nansum(appearance_hotspot_tier == 1)),
        "n_appearance_p1500_tier_cells": int(np.nansum(appearance_hotspot_tier == 2)),
        "n_appearance_p1500_top_tier_cells": int(np.nansum(appearance_hotspot_tier == 2)),
        "n_appearance_p1500_priority_tier_cells": int(np.nansum(appearance_hotspot_tier == 3)),
        "n_p1500_priority_cells_outside_mechanism_domain": int(np.nansum(outside_mechanism_domain_hotspot)),
        "n_p1500_priority_cells_masked_class0": int(np.nansum(mechanism_inconsistent_hotspot)),
        "n_p1000_candidate_cells": int(np.nansum(candidate1000)),
        "n_p1000_display_top10_cells": int(np.nansum(score1000_display_top10)),
        "n_peak_rey_ge1500_only_cells": int(np.nansum(high_rey_peak)),
        "n_rey_continuous_hotspot95_cells": int(np.nansum(rey_hotspot95)),
        "n_rey_rank_top10_cells": int(np.nansum(rey_rank_top10)),
        "n_rey_rank_top5_cells": int(np.nansum(rey_rank_top5)),
        "p1500_priority_deep_class_counts": {
            MECHANISM_NAMES[class_id]: int(np.nansum(hotspot1500 & (dominant == class_id)))
            for class_id in range(len(MECHANISM_NAMES))
        },
        "p1500_priority_deep_all_mechanisms_class_counts": {
            MECHANISM_NAMES[class_id]: int(np.nansum(hotspot1500_deep_all & (dominant == class_id)))
            for class_id in range(len(MECHANISM_NAMES))
        },
        "p1500_priority_all_depths_class_counts": {
            MECHANISM_NAMES[class_id]: int(np.nansum(hotspot1500_all & (dominant == class_id)))
            for class_id in range(len(MECHANISM_NAMES))
        },
        "p1500_priority_appearance_class_counts": {
            "no_class1_or_class2_appearance": int(np.nansum(appearance_hotspot1500 & (appearance_class == 0))),
            "class1_appearance": int(np.nansum(appearance_hotspot1500 & (appearance_class == 1))),
            "class2_appearance": int(np.nansum(appearance_hotspot1500 & (appearance_class == 2))),
        },
        "n_observed_cores": int(core_support.shape[0]) if not core_support.empty else 0,
        "n_observed_cores_ge1000": int((core_support["n_ge1000"] > 0).sum()) if not core_support.empty else 0,
        "n_observed_cores_ge1500": int((core_support["n_ge1500"] > 0).sum()) if not core_support.empty else 0,
        "artifacts": {
            "composite_png": str(composite_png),
            "composite_svg": str(composite_svg),
            "two_panel_png": str(two_panel_png),
            "two_panel_svg": str(two_panel_svg),
            "three_panel_png": str(three_panel_png),
            "three_panel_svg": str(three_panel_svg),
            "observed_png": str(observed_png),
            "observed_svg": str(observed_svg),
            "distance_support_png": str(distance_support_png),
            "distance_support_svg": str(distance_support_svg),
            "mechanism_png": str(mechanism_png),
            "mechanism_svg": str(mechanism_svg),
            "mechanism_hotspot_overlay_png": str(overlay_png),
            "mechanism_hotspot_overlay_svg": str(overlay_svg),
            "mechanism_appearance_hotspot_overlay_png": str(appearance_overlay_png),
            "mechanism_appearance_hotspot_overlay_svg": str(appearance_overlay_svg),
            "rey_continuous_hotspot95_png": str(rey_continuous_png),
            "rey_continuous_hotspot95_svg": str(rey_continuous_svg),
            "rey_continuous_appearance_hotspot_png": str(rey_continuous_legacy_png),
            "rey_continuous_appearance_hotspot_svg": str(rey_continuous_legacy_svg),
            "rey_rank_percentile_hotspot_png": str(rey_rank_png),
            "rey_rank_percentile_hotspot_svg": str(rey_rank_svg),
            "screening_png": str(screening_png),
            "screening_svg": str(screening_svg),
            **p1000_probability_artifacts,
            **facies_hotspot_only_artifacts,
            **classwise_artifacts,
            **classwise_p800_artifacts,
            **classwise_p800_top5_artifacts,
            **classwise_p800_top5_variant_artifacts,
            **classwise_p800_top5_4000m_artifacts,
            **facies_tier_artifacts,
            **unmasked_facies_tier_artifacts,
            **clean_facies_tier_artifacts,
            **clean_top_quartile_artifacts,
            **facies_specific_threshold_artifacts,
            **facies_specific_publication_artifacts,
            "quantitative_png": str(quantitative_png),
            "hotspot_table_csv": str(table_path),
        },
    }
    summary_path = output_dir / "composite_map_summary.json"
    _write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot paper-style REY mechanism distribution and hotspot screening maps.")
    parser.add_argument("--prediction-dir", type=Path, default=DEFAULT_PREDICTION_DIR)
    parser.add_argument("--env-nc", type=Path, default=DEFAULT_ENV_NC)
    parser.add_argument("--points-parquet", type=Path, default=DEFAULT_POINTS_PARQUET)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--boundary-gmt", type=Path, default=DEFAULT_BOUNDARIES_GMT)
    parser.add_argument("--bathymetry-var", default="Water depth")
    parser.add_argument("--mechanism-min-prob", type=float, default=0.55)
    parser.add_argument("--hotspot-prob1000", type=float, default=0.50)
    parser.add_argument("--hotspot-prob1500", type=float, default=0.65)
    parser.add_argument("--hotspot-score-percentile", type=float, default=98.0)
    parser.add_argument("--smoothing-sigma", type=float, default=1.2)
    parser.add_argument("--mechanism-domain-min-cells", type=int, default=24)
    parser.add_argument("--hotspot-min-cells", type=int, default=14)
    parser.add_argument("--distance-max-km", type=float, default=DEFAULT_DISTANCE_MAX_KM)
    parser.add_argument("--min-water-depth-m", type=float, default=DEFAULT_MIN_WATER_DEPTH_M)
    parser.add_argument("--appearance-prob-threshold", type=float, default=DEFAULT_APPEARANCE_PROB_THRESHOLD)
    parser.add_argument(
        "--layerwise-classwise-only",
        action="store_true",
        help="Only plot per-age-layer classwise P1000/P800 Top-N mechanism-solid maps from the 3D prediction NetCDF.",
    )
    parser.add_argument(
        "--layerwise-output-dir",
        type=Path,
        default=None,
        help="Output directory for per-age-layer maps. Defaults to prediction_dir/maps/layerwise_classwise_p1000_p800_top5_mechanism_solid.",
    )
    parser.add_argument(
        "--layerwise-age-layers-ka",
        default=None,
        help="Optional comma-separated age layers to render; nearest available 3D NetCDF layers are used.",
    )
    parser.add_argument("--layerwise-top-percent", type=float, default=5.0)
    parser.add_argument("--layerwise-write-svg", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir or (args.prediction_dir / "maps")
    if args.layerwise_classwise_only:
        summary = plot_layerwise_classwise_p1000_p800_maps(
            prediction_dir=args.prediction_dir,
            env_nc=args.env_nc,
            output_dir=args.layerwise_output_dir or output_dir / "layerwise_classwise_p1000_p800_top5_mechanism_solid",
            bathymetry_var=args.bathymetry_var,
            min_water_depth_m=float(args.min_water_depth_m),
            top_percent=float(args.layerwise_top_percent),
            boundary_gmt=args.boundary_gmt,
            age_layers_ka=_parse_age_layers(args.layerwise_age_layers_ka),
            write_svg=bool(args.layerwise_write_svg),
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    summary = plot_composite(
        prediction_dir=args.prediction_dir,
        env_nc=args.env_nc,
        points_parquet=args.points_parquet,
        output_dir=output_dir,
        bathymetry_var=args.bathymetry_var,
        mechanism_min_prob=float(args.mechanism_min_prob),
        hotspot_prob1000=float(args.hotspot_prob1000),
        hotspot_prob1500=float(args.hotspot_prob1500),
        hotspot_score_percentile=float(args.hotspot_score_percentile),
        smoothing_sigma=float(args.smoothing_sigma),
        mechanism_domain_min_cells=int(args.mechanism_domain_min_cells),
        hotspot_min_cells=int(args.hotspot_min_cells),
        distance_max_km=float(args.distance_max_km),
        min_water_depth_m=float(args.min_water_depth_m),
        appearance_prob_threshold=float(args.appearance_prob_threshold),
        boundary_gmt=args.boundary_gmt,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from core.utils.netcdf_io import open_dataset
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FixedLocator, MaxNLocator, NullLocator

try:
    from scipy.stats import kruskal
except Exception:  # pragma: no cover - optional statistics dependency
    kruskal = None

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except Exception:  # pragma: no cover - optional plotting dependency
    ccrs = None
    cfeature = None

from core.utils.serialization_utils import NumpyJSONEncoder

logger = logging.getLogger(__name__)


DEFAULT_CLASS_TABLE = Path(
    "data/geochem_class_table.parquet"
)
DEFAULT_RAW_ENV_NC = Path(
    "data/env_gridded.nc"
)
DEFAULT_BATHYMETRY_NC = Path("data/env_gridded.nc")
DEFAULT_PLATE_BOUNDARIES = Path("data/boundaries.gmt")

FACIES_LABELS = {
    0: "Facies I",
    1: "Facies II",
    2: "Facies III",
}
FACIES_NAMES = {
    0: "carbonate_background",
    1: "class1_enriched",
    2: "class2_slow_sedimentation",
}
FACIES_COLORS = {
    0: "#8F969E",
    1: "#D62728",
    2: "#2C7FB8",
}
REY_WARM_CMAP = LinearSegmentedColormap.from_list(
    "rey_warm_screening",
    ["#FFF4C6", "#F4C76B", "#EC873E", "#C83D3D", "#68182B"],
)
BATHYMETRY_COLORS = ["#9ED4D8", "#5EA3B5", "#2F7392", "#123F5C"]
BATHYMETRY_CMAP = ListedColormap(BATHYMETRY_COLORS, name="satellite_bathymetry_depth_bands")
BATHYMETRY_DEPTH_LEVELS_M = np.asarray([0.0, 2000.0, 4000.0, 6000.0, 11000.0])
LAND_RELIEF_COLORS = ["#66835F", "#8F9865", "#B09A6F", "#C7B797", "#E2DCCF"]
LAND_RELIEF_CMAP = ListedColormap(LAND_RELIEF_COLORS, name="satellite_land_relief")
LAND_RELIEF_LEVELS_M = np.asarray([0.0, 250.0, 750.0, 1500.0, 3000.0, 9000.0])
DEFAULT_PROFILE_CORES = [
    "319*",
    "XTGC082B",
    "U1367",
    "160*",
    "1220A",
    "162*",
]
BASE_ENV_FEATURES = [
    ("Water depth", "Water depth"),
    ("Water depth gradient", "Depth gradient"),
    ("Sediment thickness", "Sediment thickness"),
    ("Average sedimentation rate", "Sedimentation rate"),
    ("Bottom water dissolved oxygen", "Bottom oxygen"),
    ("Bottom phosphate", "Bottom phosphate"),
    ("Surface phosphate", "Surface phosphate"),
    ("Total organic carbon", "TOC"),
    ("Summer primary productivity", "Primary productivity"),
    ("Bottom current speed SD", "Bottom-current SD"),
    ("Macrobenthic biomass", "Macrobenthic biomass"),
    ("Gravity anomaly", "Gravity anomaly"),
    ("Vertical gravity gradient", "Vertical gravity gradient"),
]
REE_COLUMNS = ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Y"]
PAAS_VALUES = {
    "La": 38.2,
    "Ce": 79.6,
    "Pr": 8.83,
    "Nd": 33.9,
    "Sm": 5.55,
    "Eu": 1.08,
    "Gd": 4.66,
    "Tb": 0.774,
    "Dy": 4.68,
    "Ho": 0.991,
    "Er": 2.85,
    "Tm": 0.405,
    "Yb": 2.82,
    "Lu": 0.433,
    "Y": 27.0,
}
HEATMAP_FEATURES = [
    ("CaO", "log1p", "CaO"),
    ("TiO2", "log1p", "TiO2"),
    ("Al2O3", "log1p", "Al2O3"),
    ("P2O5", "log1p", "P2O5"),
    ("MnO", "log1p", "MnO"),
    ("Fe2O3", "log1p", "Fe2O3"),
    ("Mo", "log1p", "Mo"),
    ("V", "log1p", "V"),
    ("Zn", "log1p", "Zn"),
    ("proxy_ce_anom", "raw", "Ce anomaly"),
    ("proxy_y_ho", "raw", "Y/Ho"),
    ("proxy_la_yb", "raw", "La/Yb"),
    ("proxy_dy_yb", "raw", "Dy/Yb"),
    ("proxy_metal_index", "raw", "Metal index"),
    ("rey", "log1p", "REY"),
]
CLUSTER_FEATURE_LABELS = {
    "log_ratio_cao_al": "log(CaO/Al2O3)",
    "log_ratio_tio2_al": "log(TiO2/Al2O3)",
    "log_ratio_p_al": "log(P2O5/Al2O3)",
    "log_ratio_mn_al": "log(MnO/Al2O3)",
    "log_ratio_fe_al": "log(Fe2O3/Al2O3)",
    "log_ratio_mo_al": "log(Mo/Al2O3)",
    "log_ratio_v_al": "log(V/Al2O3)",
    "log_ratio_zn_al": "log(Zn/Al2O3)",
    "proxy_ce_anom": "Ce anomaly",
    "proxy_y_ho": "Y/Ho",
    "proxy_la_yb": "La/Yb",
    "proxy_dy_yb": "Dy/Yb",
    "proxy_metal_index": "Metal index",
}
BOXPLOT_GROUPS = [
    {
        "key": "major_oxides_rey",
        "stem": "geochem_facies_boxplots_major_oxides_rey",
        "title": "Major geochemical components by facies",
        "features": [
            ("rey", "Total REY", "ppm", "log"),
            ("CaO", "CaO", "wt.%", "log"),
            ("Al2O3", "Al$_2$O$_3$", "wt.%", "log"),
            ("TiO2", "TiO$_2$", "wt.%", "log"),
            ("P2O5", "P$_2$O$_5$", "wt.%", "log"),
            ("MnO", "MnO", "wt.%", "log"),
            ("Fe2O3", "Fe$_2$O$_3$", "wt.%", "log"),
        ],
    },
    {
        "key": "trace_metals",
        "stem": "geochem_facies_boxplots_trace_metals",
        "title": "Trace metals by facies",
        "features": [
            ("Mo", "Mo", "ppm", "log"),
            ("V", "V", "ppm", "log"),
            ("Zn", "Zn", "ppm", "log"),
            ("proxy_metal_index", "Metal index", "index", "log"),
        ],
    },
    {
        "key": "ree_y",
        "stem": "geochem_facies_boxplots_ree_y",
        "title": "Rare earth elements and Y by facies",
        "features": [(col, col, "ppm", "log") for col in REE_COLUMNS],
    },
    {
        "key": "geochemical_indices",
        "stem": "geochem_facies_boxplots_geochemical_indices",
        "title": "Geochemical indices by facies",
        "features": [
            ("proxy_ce_anom", "Ce anomaly", "index", "linear"),
            ("proxy_y_ho", "Y/Ho", "ratio", "linear"),
            ("proxy_la_yb", "La/Yb", "ratio", "log"),
            ("proxy_dy_yb", "Dy/Yb", "ratio", "linear"),
        ],
    },
]


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, cls=NumpyJSONEncoder, indent=2, ensure_ascii=False)


def _save_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    *,
    tight_layout: bool = True,
    bbox_inches: str | None = "tight",
    dpi: int = 240,
    pad_inches: float = 0.1,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}.png"
    svg = output_dir / f"{stem}.svg"
    if tight_layout:
        fig.tight_layout()
    fig.savefig(png, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
    fig.savefig(svg, bbox_inches=bbox_inches, pad_inches=pad_inches)
    plt.close(fig)
    return {"png": str(png), "svg": str(svg)}


def _load_table(path: Path, label_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing geochemical class table: {path}")
    df = pd.read_parquet(path)
    required = {"lon", "lat", label_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Class table is missing required columns: {missing}")
    out = df.copy()
    out[label_col] = pd.to_numeric(out[label_col], errors="coerce").astype("Int64")
    out = out[out[label_col].isin([0, 1, 2])].copy()
    if out.empty:
        raise ValueError(f"No valid 3-facies rows found in {path} using {label_col}.")
    out["facies_id"] = out[label_col].astype(int)
    out["facies_label"] = out["facies_id"].map(FACIES_LABELS)
    out["facies_name"] = out["facies_id"].map(FACIES_NAMES)
    return out


def _to_0360(lons: pd.Series) -> np.ndarray:
    values = pd.to_numeric(lons, errors="coerce").to_numpy(dtype=float)
    return np.mod(values, 360.0)


def _read_gmt_line_segments(path: Path) -> list[np.ndarray]:
    if not path.exists():
        return []
    segments: list[np.ndarray] = []
    current: list[tuple[float, float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(">"):
                if len(current) >= 2:
                    segments.append(np.asarray(current, dtype=float))
                current = []
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                lon = float(parts[0])
                lat = float(parts[1])
            except ValueError:
                continue
            current.append((lon % 360.0, lat))
    if len(current) >= 2:
        segments.append(np.asarray(current, dtype=float))

    split_segments: list[np.ndarray] = []
    for segment in segments:
        start = 0
        jumps = np.where(np.abs(np.diff(segment[:, 0])) > 180.0)[0]
        for jump in jumps:
            part = segment[start : jump + 1]
            if len(part) >= 2:
                split_segments.append(part)
            start = jump + 1
        part = segment[start:]
        if len(part) >= 2:
            split_segments.append(part)
    return split_segments


def _plot_plate_boundaries(
    ax: plt.Axes,
    boundaries_gmt: Path | None,
    *,
    transform: Any = None,
    color: str = "#8C8C8C",
    linewidth: float = 0.55,
    alpha: float = 0.62,
    zorder: int = 2,
) -> bool:
    if boundaries_gmt is None:
        return False
    segments = _read_gmt_line_segments(boundaries_gmt)
    if not segments:
        return False
    for segment in segments:
        kwargs = dict(color=color, linewidth=linewidth, linestyle=(0, (3.0, 2.4)), alpha=alpha, zorder=zorder)
        if transform is not None:
            ax.plot(segment[:, 0], segment[:, 1], transform=transform, **kwargs)
        else:
            ax.plot(segment[:, 0], segment[:, 1], **kwargs)
    return True


def _plot_subtle_bathymetry(
    ax: plt.Axes,
    env_nc: Path | None,
    *,
    transform: Any = None,
    zorder: float = 0.4,
) -> bool:
    if env_nc is None or not Path(env_nc).exists():
        return False
    try:
        with open_dataset(env_nc) as ds:
            variable = next((name for name in ["water_depth", "Water depth"] if name in ds.data_vars), None)
            if variable is None:
                return False
            lat_name = "latitude" if "latitude" in ds.coords else "lat"
            lon_name = "longitude" if "longitude" in ds.coords else "lon"
            source = ds[variable]
            source_lons = np.mod(np.asarray(source[lon_name].to_numpy(), dtype=float), 360.0)
            source = source.assign_coords({lon_name: source_lons}).sortby(lon_name)
            sorted_lons = np.asarray(source[lon_name].to_numpy(), dtype=float)
            _, unique_lon_indices = np.unique(sorted_lons, return_index=True)
            if len(unique_lon_indices) != len(sorted_lons):
                source = source.isel({lon_name: np.sort(unique_lon_indices)})
            subset = source.where(
                (source[lat_name] >= -55.0)
                & (source[lat_name] <= 50.0)
                & (source[lon_name] >= 118.0)
                & (source[lon_name] <= 292.0),
                drop=True,
            )
            if subset.size == 0:
                return False
            subset = subset.isel({lat_name: slice(None, None, 2), lon_name: slice(None, None, 2)})
            depth = subset.to_numpy()
            lats = np.asarray(subset[lat_name].to_numpy(), dtype=float)
            lons = np.asarray(subset[lon_name].to_numpy(), dtype=float)
        ocean_depth = np.where(np.isfinite(depth) & (depth < 0.0), np.clip(-depth, 0.0, 11000.0), np.nan)
        land_elevation = np.where(np.isfinite(depth) & (depth >= 0.0), np.clip(depth, 0.0, 9000.0), np.nan)
        if not np.isfinite(ocean_depth).any():
            return False
        depth_levels = BATHYMETRY_DEPTH_LEVELS_M
        norm = BoundaryNorm(depth_levels, BATHYMETRY_CMAP.N, clip=True)
        ocean_kwargs = dict(
            levels=depth_levels,
            cmap=BATHYMETRY_CMAP,
            norm=norm,
            alpha=1.0,
            antialiased=True,
            extend="max",
            zorder=zorder,
        )
        land_norm = BoundaryNorm(LAND_RELIEF_LEVELS_M, LAND_RELIEF_CMAP.N, clip=True)
        land_kwargs = dict(
            levels=LAND_RELIEF_LEVELS_M,
            cmap=LAND_RELIEF_CMAP,
            norm=land_norm,
            alpha=1.0,
            antialiased=True,
            extend="max",
            zorder=zorder + 0.3,
        )
        if transform is not None:
            ocean_contours = ax.contourf(lons, lats, ocean_depth, transform=transform, **ocean_kwargs)
            land_contours = ax.contourf(lons, lats, land_elevation, transform=transform, **land_kwargs)
            ax.contour(
                lons,
                lats,
                ocean_depth,
                levels=[2000.0, 4000.0, 6000.0],
                colors="#A7CED5",
                linewidths=0.24,
                alpha=0.48,
                transform=transform,
                zorder=zorder + 0.1,
            )
        else:
            ocean_contours = ax.contourf(lons, lats, ocean_depth, **ocean_kwargs)
            land_contours = ax.contourf(lons, lats, land_elevation, **land_kwargs)
            ax.contour(
                lons,
                lats,
                ocean_depth,
                levels=[2000.0, 4000.0, 6000.0],
                colors="#A7CED5",
                linewidths=0.24,
                alpha=0.48,
                zorder=zorder + 0.1,
            )
        for contour_set in [ocean_contours, land_contours]:
            for collection in getattr(contour_set, "collections", []):
                collection.set_rasterized(True)
        return True
    except Exception as exc:
        logger.warning("Failed to draw subtle bathymetry background: %s", exc)
        return False


def _plot_facies_map(df: pd.DataFrame, output_dir: Path, boundaries_gmt: Path | None) -> dict[str, str]:
    plt.rcParams.update({"font.size": 16, "axes.titlesize": 21, "axes.labelsize": 17, "legend.fontsize": 14})
    if ccrs is not None and cfeature is not None:
        projection = ccrs.PlateCarree(central_longitude=180)
        fig = plt.figure(figsize=(12.8, 8.3), dpi=220)
        ax = plt.axes(projection=projection)
        ax.set_extent([120, 290, -60, 60], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#F1F0EA", edgecolor="none", zorder=0)
        ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="#FAFAF7", edgecolor="none", zorder=0)
        ax.coastlines(resolution="110m", color="#9C9C9C", linewidth=0.75, zorder=3)
        gl = ax.gridlines(draw_labels=True, linewidth=0.35, color="#D9D9D9", alpha=0.65, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 13}
        gl.ylabel_style = {"size": 13}
        transform = ccrs.PlateCarree()
    else:
        fig, ax = plt.subplots(figsize=(12.8, 8.3), dpi=220)
        ax.set_xlim(120, 290)
        ax.set_ylim(-60, 60)
        ax.set_facecolor("#FAFAF7")
        ax.grid(True, color="#D9D9D9", linestyle="--", linewidth=0.5)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        transform = None

    has_boundaries = _plot_plate_boundaries(ax, boundaries_gmt, transform=transform)
    lons = _to_0360(df["lon"])
    lats = pd.to_numeric(df["lat"], errors="coerce").to_numpy(dtype=float)
    for facies_id in [0, 1, 2]:
        mask = df["facies_id"].to_numpy(dtype=int) == facies_id
        kwargs = dict(
            s=120,
            color=FACIES_COLORS[facies_id],
            label=f"{FACIES_LABELS[facies_id]} (n={int(mask.sum())})",
            edgecolors="white",
            linewidths=0.82,
            alpha=0.9,
            zorder=5,
        )
        if transform is not None:
            ax.scatter(lons[mask], lats[mask], transform=transform, **kwargs)
        else:
            ax.scatter(lons[mask], lats[mask], **kwargs)
    ax.set_title("Geochemical Facies Distribution")
    handles, labels = ax.get_legend_handles_labels()
    if has_boundaries:
        handles.append(Line2D([0], [0], color="#8C8C8C", linestyle=(0, (3.0, 2.4)), linewidth=1.0))
        labels.append("Plate boundary")
    ax.legend(
        handles,
        labels,
        loc="lower left",
        frameon=True,
        framealpha=0.96,
        facecolor="white",
        edgecolor="#CCCCCC",
        markerscale=1.18,
    )
    return _save_figure(fig, output_dir, "geochem_facies_map")


def _plot_rey_sample_location_map(
    df: pd.DataFrame,
    output_dir: Path,
    boundaries_gmt: Path | None,
    env_nc: Path | None = None,
    *,
    profile_cores: list[str] | None = None,
    highlight_profile_cores: bool = False,
    stem: str = "geochem_rey_sample_location_map",
) -> tuple[dict[str, str], pd.DataFrame]:
    if "rey" not in df.columns:
        raise KeyError("Continuous REY sample-location map requires a `rey` column.")
    group_col = "core_id" if "core_id" in df.columns else None
    work = df.copy()
    work["lon360"] = _to_0360(work["lon"])
    work["lat_num"] = pd.to_numeric(work["lat"], errors="coerce")
    work["rey_num"] = pd.to_numeric(work["rey"], errors="coerce")
    work = work[np.isfinite(work["lon360"]) & np.isfinite(work["lat_num"]) & np.isfinite(work["rey_num"])]
    work = work[work["rey_num"] > 0.0]
    if work.empty:
        raise ValueError("No valid positive REY samples found for the sample-location map.")

    site_bin_deg = 0.1
    work["site_lon_bin"] = np.round(work["lon360"] / site_bin_deg) * site_bin_deg
    work["site_lat_bin"] = np.round(work["lat_num"] / site_bin_deg) * site_bin_deg
    group_keys = ["site_lon_bin", "site_lat_bin"]
    if group_col:
        loc = (
            work.groupby(group_keys, dropna=False)
            .agg(
                lon=("lon360", "mean"),
                lat=("lat_num", "mean"),
                n_samples=("rey_num", "size"),
                n_cores=(group_col, "nunique"),
                core_ids=(group_col, lambda s: "; ".join(sorted({str(v) for v in s.dropna()}))),
                rey_median_ppm=("rey_num", "median"),
                rey_max_ppm=("rey_num", "max"),
                rey_p90_ppm=("rey_num", lambda s: float(np.nanpercentile(s, 90))),
            )
            .reset_index()
        )
        loc.insert(0, "location_id", [f"site_{idx + 1:03d}" for idx in range(len(loc))])
    else:
        loc = work.rename(columns={"lon360": "lon", "lat_num": "lat", "rey_num": "rey_max_ppm"}).copy()
        loc["location_id"] = np.arange(len(loc), dtype=int)
        loc["n_samples"] = 1
        loc["n_cores"] = 1
        loc["core_ids"] = loc["location_id"].astype(str)
        loc["rey_median_ppm"] = loc["rey_max_ppm"]
        loc["rey_p90_ppm"] = loc["rey_max_ppm"]
        loc = loc[
            [
                "location_id",
                "lon",
                "lat",
                "n_samples",
                "n_cores",
                "core_ids",
                "rey_median_ppm",
                "rey_max_ppm",
                "rey_p90_ppm",
            ]
        ]
    loc["site_bin_deg"] = site_bin_deg

    loc.to_csv(output_dir / "geochem_rey_sample_location_summary.csv", index=False)
    bathymetry_nc = DEFAULT_BATHYMETRY_NC if DEFAULT_BATHYMETRY_NC.exists() else env_nc

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.0,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
        }
    )
    map_extent = [118, 292, -53, 48]
    if ccrs is not None and cfeature is not None:
        projection = ccrs.PlateCarree(central_longitude=180)
        fig = plt.figure(figsize=(7.2, 4.3), dpi=300, facecolor="white")
        ax = fig.add_axes([0.035, 0.095, 0.85, 0.82], projection=projection)
        ax.set_extent(map_extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#123F5C", edgecolor="none", zorder=0)
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#758A65", edgecolor="none", zorder=0.2)
        has_bathymetry = _plot_subtle_bathymetry(
            ax,
            bathymetry_nc,
            transform=ccrs.PlateCarree(),
            zorder=1,
        )
        ax.coastlines(resolution="50m", color="#27383A", linewidth=0.52, zorder=5)
        ax.spines["geo"].set_edgecolor("#35474A")
        ax.spines["geo"].set_linewidth(0.55)
        gl = ax.gridlines(draw_labels=True, linewidth=0.22, color="#DCE7E8", alpha=0.34, linestyle=(0, (1.2, 2.2)))
        gl.top_labels = False
        gl.right_labels = False
        gl.xlocator = FixedLocator([-150, -120, -90, 120, 150, 180])
        gl.ylocator = FixedLocator([-40, -20, 0, 20, 40])
        gl.xlabel_style = {"size": 7.0, "color": "#364649"}
        gl.ylabel_style = {"size": 7.0, "color": "#364649"}
        gl.rotate_labels = False
        transform = ccrs.PlateCarree()
    else:
        fig = plt.figure(figsize=(7.2, 4.3), dpi=300, facecolor="white")
        ax = fig.add_axes([0.065, 0.095, 0.81, 0.82])
        ax.set_xlim(map_extent[0], map_extent[1])
        ax.set_ylim(map_extent[2], map_extent[3])
        ax.set_facecolor("#123F5C")
        has_bathymetry = _plot_subtle_bathymetry(ax, bathymetry_nc, zorder=1)
        ax.grid(True, color="#71858C", linestyle=(0, (1.2, 2.2)), linewidth=0.24, alpha=0.35)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        transform = None

    has_boundaries = _plot_plate_boundaries(
        ax,
        boundaries_gmt,
        transform=transform,
        color="#DCE7E8",
        linewidth=0.42,
        alpha=0.66,
        zorder=4,
    )
    rey_values = loc["rey_max_ppm"].to_numpy(dtype=float)
    vmin = 30.0
    vmax = 10000.0
    if float(np.nanmax(rey_values)) > vmax:
        vmax = float(10.0 ** np.ceil(np.log10(np.nanmax(rey_values))))
    norm = LogNorm(vmin=vmin, vmax=vmax)
    sizes = 17.0 + 9.5 * np.sqrt(np.clip(loc["n_samples"].to_numpy(dtype=float), 1.0, None))
    sizes = np.clip(sizes, 27.0, 116.0)
    plot_order = np.argsort(rey_values, kind="stable")
    plot_lons = loc["lon"].to_numpy(dtype=float)[plot_order]
    plot_lats = loc["lat"].to_numpy(dtype=float)[plot_order]
    plot_sizes = sizes[plot_order]
    plot_rey = rey_values[plot_order]

    scatter_kwargs = dict(
        s=plot_sizes,
        c=plot_rey,
        cmap=REY_WARM_CMAP,
        norm=norm,
        edgecolors="#4B3936",
        linewidths=0.28,
        alpha=0.97,
        zorder=8,
    )
    if transform is not None:
        ax.scatter(
            plot_lons,
            plot_lats,
            transform=transform,
            s=plot_sizes + 15.0,
            color="#FFFFFF",
            linewidths=0,
            alpha=0.94,
            zorder=7,
        )
        sc = ax.scatter(plot_lons, plot_lats, transform=transform, **scatter_kwargs)
    else:
        ax.scatter(
            plot_lons,
            plot_lats,
            s=plot_sizes + 15.0,
            color="#FFFFFF",
            linewidths=0,
            alpha=0.94,
            zorder=7,
        )
        sc = ax.scatter(plot_lons, plot_lats, **scatter_kwargs)

    profile_summary: pd.DataFrame | None = None
    if highlight_profile_cores and "core_id" in df.columns:
        try:
            _, profile_summary = _select_profile_cores(df, profile_cores, n_profiles=6)
        except Exception as exc:
            profile_summary = None
            logger.warning("Failed to select profile cores for sample-location map: %s", exc)
    if profile_summary is not None and not profile_summary.empty:
        profile_summary = profile_summary.copy()
        profile_summary["label"] = profile_summary["display_order"].astype(int).astype(str)
        profile_summary["lon"] = pd.to_numeric(profile_summary["lon_mean"], errors="coerce")
        profile_summary["lat"] = pd.to_numeric(profile_summary["lat_mean"], errors="coerce")
        profile_summary = profile_summary.dropna(subset=["lon", "lat"])
        profile_summary.to_csv(output_dir / "geochem_rey_profile_core_location_summary.csv", index=False)
        if not profile_summary.empty:
            marker_kwargs = dict(
                s=265,
                marker="*",
                facecolor="#151515",
                edgecolor="white",
                linewidth=1.25,
                alpha=0.98,
                zorder=12,
                label="Profile cores",
            )
            if transform is not None:
                ax.scatter(profile_summary["lon"], profile_summary["lat"], transform=transform, **marker_kwargs)
            else:
                ax.scatter(profile_summary["lon"], profile_summary["lat"], **marker_kwargs)
            offsets = [
                (1.5, 1.7),
                (1.6, -1.8),
                (1.4, -2.4),
                (1.4, 1.6),
                (-5.8, -1.9),
                (-5.6, 1.8),
            ]
            for idx, row in enumerate(profile_summary.itertuples(index=False)):
                dx, dy = offsets[idx % len(offsets)]
                label = f"{int(row.display_order)}  {row.core_id}"
                text_kwargs = dict(
                    ha="left" if dx >= 0 else "right",
                    va="center",
                    fontsize=9.2,
                    color="#111111",
                    bbox=dict(facecolor="white", edgecolor="#4A4A4A", boxstyle="round,pad=0.22", alpha=0.93),
                    zorder=13,
                )
                if transform is not None:
                    ax.text(float(row.lon) + dx, float(row.lat) + dy, label, transform=transform, **text_kwargs)
                else:
                    ax.text(float(row.lon) + dx, float(row.lat) + dy, label, **text_kwargs)

    map_position = ax.get_position()
    cax = fig.add_axes([0.895, map_position.y0, 0.018, map_position.height])
    cbar = fig.colorbar(sc, cax=cax, orientation="vertical")
    cbar.set_label("Maximum REY at each mapped location (ppm)", labelpad=7.0, fontsize=7.2)
    tick_values = [value for value in [30, 100, 300, 1000, 3000, 10000] if value <= vmax * 1.01]
    if not tick_values:
        tick_values = [vmin, vmax]
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels([f"{int(value):,d}" for value in tick_values])
    cbar.ax.yaxis.set_minor_locator(NullLocator())
    cbar.ax.tick_params(which="major", labelsize=6.5, width=0.55, length=3.0, pad=2.2, colors="#343A3B")
    cbar.outline.set_linewidth(0.50)
    cbar.outline.set_edgecolor("#4D5354")

    size_refs = [1, 20, 60]
    size_handles = [
        plt.scatter(
            [],
            [],
            s=float(np.clip(17.0 + 9.5 * np.sqrt(n), 27.0, 116.0)),
            facecolor="#D8DDDC",
            edgecolor="#4E5657",
            linewidth=0.45,
            label=f"{n}",
        )
        for n in size_refs
    ]
    for handle, n in zip(size_handles, size_refs):
        handle.set_label(f"{n} sample" if n == 1 else f"{n} samples")
    handles: list[Any] = size_handles
    labels = [h.get_label() for h in handles]
    if has_boundaries:
        handles.append(Line2D([0], [0], color="#8C8C8C", linestyle=(0, (3.0, 2.4)), linewidth=0.9))
        labels.append("Plate boundary")
    if profile_summary is not None and not profile_summary.empty:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="*",
                color="none",
                markerfacecolor="#151515",
                markeredgecolor="white",
                markeredgewidth=1.0,
                markersize=12,
            )
        )
        labels.append("Profile cores")
    legend = ax.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0.014, 0.018),
        ncol=1,
        frameon=True,
        fancybox=False,
        framealpha=0.94,
        facecolor="white",
        edgecolor="#737B7D",
        borderpad=0.58,
        labelspacing=0.42,
        columnspacing=1.15,
        handlelength=1.55,
        handletextpad=0.52,
        borderaxespad=0.0,
        fontsize=6.15,
    )
    legend.get_frame().set_linewidth(0.55)
    return _save_figure(fig, output_dir, stem, tight_layout=False, dpi=300, pad_inches=0.0), loc


def _plot_ree_profiles(df: pd.DataFrame, output_dir: Path) -> tuple[dict[str, str], pd.DataFrame]:
    available = [col for col in REE_COLUMNS if col in df.columns and col in PAAS_VALUES]
    if not available:
        raise KeyError("No REE columns with PAAS reference values found for profile plotting.")
    rows = []
    for facies_id in [0, 1, 2]:
        sub = df[df["facies_id"] == facies_id]
        for col in available:
            paas = float(PAAS_VALUES[col])
            vals = pd.to_numeric(sub[col], errors="coerce") / paas
            vals = vals.where(vals > 0)
            rows.append(
                {
                    "facies_id": facies_id,
                    "facies_label": FACIES_LABELS[facies_id],
                    "feature": col,
                    "paas_ppm": paas,
                    "median": float(vals.median()),
                    "p25": float(vals.quantile(0.25)),
                    "p75": float(vals.quantile(0.75)),
                    "n": int(vals.notna().sum()),
                }
            )
    profile = pd.DataFrame(rows)
    profile.to_csv(output_dir / "geochem_facies_ree_profiles.csv", index=False)

    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.labelsize": 22,
            "legend.fontsize": 17,
            "xtick.labelsize": 16,
            "ytick.labelsize": 17,
        }
    )
    fig, ax = plt.subplots(figsize=(10.2, 8.7), dpi=240)
    x = np.arange(len(available))
    for facies_id in [0, 1, 2]:
        sub = profile[profile["facies_id"] == facies_id].set_index("feature").loc[available]
        med = sub["median"].to_numpy(dtype=float)
        p25 = sub["p25"].to_numpy(dtype=float)
        p75 = sub["p75"].to_numpy(dtype=float)
        ax.plot(
            x,
            med,
            color=FACIES_COLORS[facies_id],
            marker="o",
            markersize=7.6,
            linewidth=3.0,
            label=f"{FACIES_LABELS[facies_id]} (n={int((df['facies_id'] == facies_id).sum())})",
        )
        ax.fill_between(x, p25, p75, color=FACIES_COLORS[facies_id], alpha=0.12, linewidth=0)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(available, rotation=0, fontsize=16)
    ax.set_ylabel("Sample / PAAS", labelpad=11)
    ax.grid(True, which="major", axis="y", linestyle="--", linewidth=0.8, alpha=0.52)
    ax.grid(True, which="minor", axis="y", linestyle=":", linewidth=0.5, alpha=0.28)
    ax.legend(frameon=False, ncol=1, loc="lower right", handlelength=2.8, borderaxespad=0.8)
    ax.tick_params(axis="both", which="major", length=4.5, width=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.13, right=0.975, top=0.975, bottom=0.13)
    return _save_figure(
        fig,
        output_dir,
        "geochem_facies_ree_profiles",
        tight_layout=False,
        bbox_inches=None,
    ), profile


def _robust_z(values: pd.Series) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce").astype(float)
    median = float(arr.median())
    q25 = float(arr.quantile(0.25))
    q75 = float(arr.quantile(0.75))
    scale = (q75 - q25) / 1.349
    if not np.isfinite(scale) or scale < 1e-9:
        scale = float(arr.std(ddof=0))
    if not np.isfinite(scale) or scale < 1e-9:
        scale = 1.0
    return (arr - median) / scale


def _plot_geochem_heatmap(df: pd.DataFrame, output_dir: Path) -> tuple[dict[str, str], pd.DataFrame]:
    transformed = pd.DataFrame(index=df.index)
    labels = []
    for col, transform, label in HEATMAP_FEATURES:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").astype(float)
        if transform == "log1p":
            vals = np.log1p(np.clip(vals, 0.0, None))
        transformed[label] = _robust_z(pd.Series(vals, index=df.index))
        labels.append(label)
    if not labels:
        raise KeyError("No geochemical columns found for heatmap plotting.")

    rows = []
    for facies_id in [0, 1, 2]:
        sub = transformed.loc[df["facies_id"] == facies_id, labels]
        row = sub.median(axis=0, skipna=True)
        rows.append(row)
    heat = pd.DataFrame(rows, index=[FACIES_LABELS[i] for i in [0, 1, 2]], columns=labels)
    heat.to_csv(output_dir / "geochem_facies_feature_heatmap_values.csv")

    plt.rcParams.update({"font.size": 15, "axes.titlesize": 21, "axes.labelsize": 17})
    fig, ax = plt.subplots(figsize=(14.2, 5.6), dpi=220)
    vmax = float(np.nanmax(np.abs(heat.to_numpy(dtype=float))))
    vmax = max(1.0, min(3.0, vmax))
    image = ax.imshow(heat.to_numpy(dtype=float), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(3))
    ax.set_yticklabels([FACIES_LABELS[i] for i in [0, 1, 2]], fontsize=16)
    ax.set_title("Facies Geochemical Fingerprints")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Mechanism facies")
    for y in range(3):
        for x in range(len(labels)):
            value = heat.iloc[y, x]
            if np.isfinite(value):
                color = "white" if abs(value) > vmax * 0.58 else "#222222"
                ax.text(x, y, f"{value:.1f}", ha="center", va="center", fontsize=9.5, color=color)
    cbar = fig.colorbar(image, ax=ax, shrink=0.82, pad=0.015)
    cbar.ax.set_ylabel("Robust z-score of transformed median", rotation=90, labelpad=14)
    return _save_figure(fig, output_dir, "geochem_facies_feature_heatmap"), heat


def _coord_name(ds: xr.Dataset, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in ds.coords or name in ds.dims:
            return name
    raise KeyError(f"None of the coordinate names are present: {candidates}")


def _point_longitudes_for_dataset(lons: pd.Series, lon_coord: xr.DataArray) -> np.ndarray:
    values = pd.to_numeric(lons, errors="coerce").to_numpy(dtype=float)
    lon_values = np.asarray(lon_coord.to_numpy(), dtype=float)
    lon_min = float(np.nanmin(lon_values))
    lon_max = float(np.nanmax(lon_values))
    if lon_min >= 0.0 or lon_max > 180.0:
        return np.mod(values, 360.0)
    return ((values + 180.0) % 360.0) - 180.0


def _sample_environment_features(
    df: pd.DataFrame,
    env_nc: Path,
    features: list[tuple[str, str]],
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    if not env_nc.exists():
        raise FileNotFoundError(f"Missing raw environment grid for facies heatmap: {env_nc}")
    with open_dataset(env_nc) as ds:
        lon_name = _coord_name(ds, ("longitude", "lon", "x"))
        lat_name = _coord_name(ds, ("latitude", "lat", "y"))
        available = [(var, label) for var, label in features if var in ds.data_vars]
        if not available:
            raise KeyError(f"No requested base continuous environment variables were found in {env_nc}.")

        lons = _point_longitudes_for_dataset(df["lon"], ds[lon_name])
        lats = pd.to_numeric(df["lat"], errors="coerce").to_numpy(dtype=float)
        points = {
            lon_name: xr.DataArray(lons, dims="points"),
            lat_name: xr.DataArray(lats, dims="points"),
        }
        selected = ds[[var for var, _ in available]]
        linear = selected.interp(points, method="linear")
        nearest = selected.interp(points, method="nearest")

        sampled = pd.DataFrame(index=df.index)
        for var, _ in available:
            vals = np.asarray(linear[var].to_numpy(), dtype=float)
            fallback = np.asarray(nearest[var].to_numpy(), dtype=float)
            vals = np.where(np.isfinite(vals), vals, fallback)
            sampled[var] = vals
    return sampled, available


def _plot_environment_heatmap(
    df: pd.DataFrame,
    output_dir: Path,
    env_nc: Path | None,
) -> tuple[dict[str, str] | None, pd.DataFrame | None, pd.DataFrame | None]:
    if env_nc is None:
        return None, None, None
    sampled, available = _sample_environment_features(df, env_nc, BASE_ENV_FEATURES)
    labels = [label for _, label in available]
    raw_medians = []
    z_data = pd.DataFrame(index=df.index)
    for var, label in available:
        z_data[label] = _robust_z(sampled[var])

    heat_rows = []
    for facies_id in [0, 1, 2]:
        facies_mask = df["facies_id"] == facies_id
        z_row = z_data.loc[facies_mask, labels].median(axis=0, skipna=True)
        heat_rows.append(z_row)
        raw_row = {"facies_id": facies_id, "facies_label": FACIES_LABELS[facies_id]}
        for var, label in available:
            raw_row[var] = float(pd.to_numeric(sampled.loc[facies_mask, var], errors="coerce").median())
            raw_row[f"{var}__label"] = label
        raw_medians.append(raw_row)

    heat = pd.DataFrame(heat_rows, index=[FACIES_LABELS[i] for i in [0, 1, 2]], columns=labels)
    raw = pd.DataFrame(raw_medians)
    heat.to_csv(output_dir / "geochem_facies_environment_heatmap_values.csv")
    raw.to_csv(output_dir / "geochem_facies_environment_raw_medians.csv", index=False)

    plt.rcParams.update({"font.size": 15, "axes.titlesize": 21, "axes.labelsize": 17})
    fig, ax = plt.subplots(figsize=(14.8, 5.8), dpi=220)
    arr = heat.to_numpy(dtype=float)
    vmax = float(np.nanmax(np.abs(arr))) if np.isfinite(arr).any() else 1.0
    vmax = max(1.0, min(3.0, vmax))
    image = ax.imshow(arr, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=38, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(3))
    ax.set_yticklabels([FACIES_LABELS[i] for i in [0, 1, 2]], fontsize=16)
    ax.set_title("Raw Environmental Background by Facies")
    ax.set_xlabel("Original continuous environmental variable")
    ax.set_ylabel("Mechanism facies")
    for y in range(3):
        for x in range(len(labels)):
            value = heat.iloc[y, x]
            if np.isfinite(value):
                color = "white" if abs(value) > vmax * 0.56 else "#1F1F1F"
                ax.text(x, y, f"{value:.1f}", ha="center", va="center", fontsize=9.2, color=color)
    cbar = fig.colorbar(image, ax=ax, shrink=0.82, pad=0.012)
    cbar.ax.set_title("Robust z", fontsize=12, pad=8)
    fig.subplots_adjust(left=0.09, right=0.95, top=0.87, bottom=0.28)
    return _save_figure(fig, output_dir, "geochem_facies_environment_heatmap", tight_layout=False), heat, raw


def _format_lon_lat(lon: float, lat: float) -> str:
    lon360 = float(lon) % 360.0
    if lon360 > 180.0:
        lon_text = f"{360.0 - lon360:.1f}°W"
    else:
        lon_text = f"{lon360:.1f}°E"
    lat_text = f"{abs(float(lat)):.1f}°{'S' if float(lat) < 0 else 'N'}"
    return f"{lat_text}, {lon_text}"


def _depth_bin_edges(depths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(depths, dtype=float)
    if values.size == 1:
        half = 0.35
        return values - half, values + half
    mids = (values[:-1] + values[1:]) / 2.0
    first_gap = max(float(values[1] - values[0]), 0.08)
    last_gap = max(float(values[-1] - values[-2]), 0.08)
    tops = np.r_[values[0] - first_gap / 2.0, mids]
    bottoms = np.r_[mids, values[-1] + last_gap / 2.0]
    return tops, bottoms


def _score_core_profiles(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for core_id, group in df.groupby("core_id", dropna=True):
        g = group.sort_values("depth")
        if len(g) == 0:
            continue
        counts = g["facies_id"].value_counts().to_dict()
        depth_range = float(g["depth"].max() - g["depth"].min())
        n_ge1000 = int((pd.to_numeric(g["rey"], errors="coerce") >= 1000.0).sum()) if "rey" in g else 0
        n_ge1500 = int((pd.to_numeric(g["rey"], errors="coerce") >= 1500.0).sum()) if "rey" in g else 0
        rey_max = float(pd.to_numeric(g["rey"], errors="coerce").max()) if "rey" in g else np.nan
        n_facies = int(g["facies_id"].nunique())
        score = (
            min(len(g), 80) / 80.0 * 2.0
            + min(depth_range, 50.0) / 50.0 * 1.35
            + n_facies * 1.05
            + min(n_ge1000, 20) / 20.0 * 2.0
            + (min(rey_max, 3000.0) / 3000.0 * 1.45 if np.isfinite(rey_max) else 0.0)
        )
        rows.append(
            {
                "core_id": str(core_id),
                "n_samples": int(len(g)),
                "depth_min_m": float(g["depth"].min()),
                "depth_max_m": float(g["depth"].max()),
                "depth_range_m": depth_range,
                "rey_median_ppm": float(pd.to_numeric(g["rey"], errors="coerce").median()) if "rey" in g else np.nan,
                "rey_max_ppm": rey_max,
                "n_rey_ge_1000": n_ge1000,
                "n_rey_ge_1500": n_ge1500,
                "n_facies": n_facies,
                "facies_counts": "; ".join(f"{FACIES_LABELS[int(k)]}: {int(v)}" for k, v in sorted(counts.items())),
                "lon_mean": float((_to_0360(g["lon"])).mean()) if "lon" in g else np.nan,
                "lat_mean": float(pd.to_numeric(g["lat"], errors="coerce").mean()) if "lat" in g else np.nan,
                "selection_score": float(score),
            }
        )
    return pd.DataFrame(rows).sort_values("selection_score", ascending=False)


def _select_profile_cores(
    df: pd.DataFrame,
    requested_cores: list[str] | None,
    *,
    n_profiles: int = 6,
) -> tuple[list[str], pd.DataFrame]:
    if "core_id" not in df.columns or "depth" not in df.columns or "rey" not in df.columns:
        raise KeyError("Core profile plotting requires `core_id`, `depth`, and `rey` columns.")
    scores = _score_core_profiles(df)
    available = set(scores["core_id"].astype(str))
    requested = requested_cores or DEFAULT_PROFILE_CORES
    selected: list[str] = []
    for core_id in requested:
        core_text = str(core_id).strip()
        if core_text and core_text in available and core_text not in selected:
            selected.append(core_text)
    binary_sets = {"0,1", "0,2"}
    allowed_core_ids: set[str] = set()
    for core_id, group in df.groupby("core_id", dropna=True):
        facies_key = ",".join(str(int(v)) for v in sorted(group["facies_id"].dropna().astype(int).unique()))
        if facies_key in binary_sets:
            allowed_core_ids.add(str(core_id))
    for core_id in scores["core_id"].astype(str):
        if len(selected) >= n_profiles:
            break
        if (
            core_id not in selected
            and core_id in allowed_core_ids
            and int(scores.loc[scores["core_id"] == core_id, "n_samples"].iloc[0]) >= 20
        ):
            selected.append(core_id)
    selected = selected[:n_profiles]
    summary = scores[scores["core_id"].isin(selected)].copy()
    summary["display_order"] = summary["core_id"].map({core_id: idx + 1 for idx, core_id in enumerate(selected)})
    summary = summary.sort_values("display_order")
    return selected, summary


def _plot_selected_core_profiles(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    requested_cores: list[str] | None = None,
) -> tuple[dict[str, str], pd.DataFrame]:
    selected, summary = _select_profile_cores(df, requested_cores, n_profiles=6)
    if not selected:
        raise ValueError("No valid cores found for vertical profile plotting.")
    summary.to_csv(output_dir / "geochem_facies_selected_core_profile_summary.csv", index=False)

    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 12,
            "axes.linewidth": 0.8,
        }
    )
    ncols = 3
    nrows = int(np.ceil(len(selected) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.2, 9.65), dpi=240, squeeze=False)
    x_min, x_max = 25.0, 9000.0

    for ax, core_id in zip(axes.ravel(), selected):
        core = df[df["core_id"].astype(str) == core_id].copy()
        core["depth"] = pd.to_numeric(core["depth"], errors="coerce")
        core["rey"] = pd.to_numeric(core["rey"], errors="coerce")
        core = core[np.isfinite(core["depth"]) & np.isfinite(core["rey"])].sort_values("depth")
        if core.empty:
            ax.axis("off")
            continue
        depths = core["depth"].to_numpy(dtype=float)
        rey = np.clip(core["rey"].to_numpy(dtype=float), 1e-3, None)
        facies = core["facies_id"].to_numpy(dtype=int)
        tops, bottoms = _depth_bin_edges(depths)
        y_top = min(0.0, float(np.nanmin(tops)))
        y_bottom = float(np.nanmax(bottoms))

        ax.plot(rey, depths, color="#1E1E1E", linewidth=0.9, alpha=0.78, zorder=2)
        for facies_id in [0, 1, 2]:
            mask = facies == facies_id
            if np.any(mask):
                ax.scatter(
                    rey[mask],
                    depths[mask],
                    s=30,
                    color=FACIES_COLORS[facies_id],
                    edgecolors="white",
                    linewidths=0.55,
                    alpha=0.96,
                    zorder=3,
                )
        ax.set_xscale("log")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_bottom, y_top)
        ax.set_xticks([100, 300, 1000, 3000])
        ax.set_xticklabels(["100", "300", "1000", "3000"])
        ax.grid(True, axis="x", color="#D8D8D8", linestyle="--", linewidth=0.55, alpha=0.6)
        ax.grid(True, axis="y", color="#E5E5E5", linestyle="-", linewidth=0.45, alpha=0.55)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Total REY (ppm)")
        ax.set_ylabel("Core depth (m)")

        lon = float((_to_0360(core["lon"])).mean()) if "lon" in core else np.nan
        lat = float(pd.to_numeric(core["lat"], errors="coerce").mean()) if "lat" in core else np.nan
        title = f"Core {core_id}"
        if np.isfinite(lon) and np.isfinite(lat):
            title += f"\n{_format_lon_lat(lon, lat)}"
        ax.set_title(title, pad=8)

        bar_ax = ax.inset_axes([0.018, 0.08, 0.030, 0.82])
        for top, bottom, facies_id in zip(tops, bottoms, facies):
            bar_ax.add_patch(
                Rectangle(
                    (0.0, float(top)),
                    1.0,
                    float(bottom - top),
                    facecolor=FACIES_COLORS[int(facies_id)],
                    edgecolor="none",
                    alpha=0.95,
                )
            )
        bar_ax.set_xlim(0, 1)
        bar_ax.set_ylim(y_bottom, y_top)
        bar_ax.set_xticks([])
        bar_ax.set_yticks([])
        for spine in bar_ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.48)
            spine.set_edgecolor("#4A4A4A")
        bar_ax.set_facecolor("white")
        ax.text(
            0.054,
            0.965,
            "Facies",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.0,
            color="#333333",
        )

    for ax in axes.ravel()[len(selected):]:
        ax.axis("off")

    handles = [
        Patch(facecolor=FACIES_COLORS[i], edgecolor="none", label=FACIES_LABELS[i])
        for i in [0, 1, 2]
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.014))
    fig.tight_layout(rect=[0.0, 0.06, 1.0, 0.985])
    return _save_figure(fig, output_dir, "geochem_facies_selected_core_profiles", tight_layout=False), summary


def _clean_box_values(values: pd.Series, scale: str) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if scale == "log":
        arr = arr[arr > 0.0]
    return arr


def _feature_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return slug or "feature"


def _all_boxplot_features() -> list[tuple[str, str, str, str, str]]:
    seen: set[str] = set()
    features: list[tuple[str, str, str, str, str]] = []
    for group in BOXPLOT_GROUPS:
        for col, label, unit, scale in group["features"]:
            if col in seen:
                continue
            seen.add(col)
            features.append((str(group["key"]), col, label, unit, scale))
    return features


def _boxplot_group_stats(
    df: pd.DataFrame,
    features: list[tuple[str, str, str, str]],
    group_key: str,
    group_title: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for col, label, unit, scale in features:
        if col not in df.columns:
            continue
        groups = []
        for facies_id in [0, 1, 2]:
            vals = _clean_box_values(df.loc[df["facies_id"] == facies_id, col], scale)
            groups.append(vals)
        valid_groups = [vals for vals in groups if len(vals) > 0]
        p_value = np.nan
        if kruskal is not None and len(valid_groups) >= 2:
            try:
                p_value = float(kruskal(*valid_groups, nan_policy="omit").pvalue)
            except Exception:
                p_value = np.nan
        for facies_id, vals in zip([0, 1, 2], groups):
            rows.append(
                {
                    "group": group_key,
                    "group_title": group_title,
                    "feature": col,
                    "label": label,
                    "unit": unit,
                    "scale": scale,
                    "facies_id": facies_id,
                    "facies_label": FACIES_LABELS[facies_id],
                    "n": int(len(vals)),
                    "median": float(np.nanmedian(vals)) if len(vals) else np.nan,
                    "q25": float(np.nanpercentile(vals, 25)) if len(vals) else np.nan,
                    "q75": float(np.nanpercentile(vals, 75)) if len(vals) else np.nan,
                    "p05": float(np.nanpercentile(vals, 5)) if len(vals) else np.nan,
                    "p95": float(np.nanpercentile(vals, 95)) if len(vals) else np.nan,
                    "mean": float(np.nanmean(vals)) if len(vals) else np.nan,
                    "kruskal_p": p_value,
                }
            )
    return rows


def _plot_single_boxplot_group(
    df: pd.DataFrame,
    output_dir: Path,
    group: dict[str, Any],
) -> dict[str, str]:
    features = [item for item in group["features"] if item[0] in df.columns]
    if not features:
        raise KeyError(f"No columns found for boxplot group {group['key']}.")

    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.titlesize": 17,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 14,
            "axes.linewidth": 0.8,
        }
    )
    n_features = len(features)
    ncols = 4 if n_features > 4 else n_features
    nrows = int(np.ceil(n_features / ncols))
    width = 4.0 * ncols
    height = 3.55 * nrows + 1.15
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), dpi=220, squeeze=False)
    rng = np.random.default_rng(20260611)
    positions = [1, 2, 3]

    for ax, (col, label, unit, scale) in zip(axes.ravel(), features):
        data = [_clean_box_values(df.loc[df["facies_id"] == facies_id, col], scale) for facies_id in [0, 1, 2]]
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=0.58,
            patch_artist=True,
            showfliers=False,
            whis=(5, 95),
            medianprops={"color": "#111111", "linewidth": 1.55},
            boxprops={"linewidth": 1.05},
            whiskerprops={"linewidth": 0.95, "color": "#555555"},
            capprops={"linewidth": 0.95, "color": "#555555"},
        )
        for patch, facies_id in zip(bp["boxes"], [0, 1, 2]):
            patch.set_facecolor(FACIES_COLORS[facies_id])
            patch.set_edgecolor("#333333")
            patch.set_alpha(0.58)

        for idx, (facies_id, vals) in enumerate(zip([0, 1, 2], data), start=1):
            if len(vals) == 0:
                continue
            if len(vals) > 420:
                vals = rng.choice(vals, size=420, replace=False)
            x = rng.normal(idx, 0.055, size=len(vals))
            ax.scatter(
                x,
                vals,
                s=6,
                color=FACIES_COLORS[facies_id],
                alpha=0.16,
                linewidths=0,
                rasterized=True,
                zorder=1,
            )

        if scale == "log":
            ax.set_yscale("log")
        ax.set_title(label, pad=8)
        ax.set_xticks(positions)
        ax.set_xticklabels(["I", "II", "III"])
        ax.set_xlim(0.45, 3.55)
        ax.set_ylabel(unit)
        ax.grid(True, axis="y", color="#D0D0D0", linewidth=0.55, alpha=0.55)
        ax.grid(False, axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#4A4A4A")
        ax.spines["bottom"].set_color("#4A4A4A")

    for ax in axes.ravel()[len(features):]:
        ax.axis("off")

    handles = [
        Patch(facecolor=FACIES_COLORS[i], edgecolor="#333333", alpha=0.58, label=FACIES_LABELS[i])
        for i in [0, 1, 2]
    ]
    fig.suptitle(group["title"], fontsize=23, y=0.985)
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.015))
    fig.tight_layout(rect=[0.0, 0.065, 1.0, 0.94])
    return _save_figure(fig, output_dir, group["stem"], tight_layout=False)


def _plot_single_feature_distribution(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    col: str,
    label: str,
    unit: str,
    kind: str,
    scale: str,
) -> dict[str, str]:
    data = [_clean_box_values(df.loc[df["facies_id"] == facies_id, col], scale) for facies_id in [0, 1, 2]]
    if not any(len(vals) for vals in data):
        raise ValueError(f"No valid values for {col}.")

    plt.rcParams.update(
        {
            "font.size": 17,
            "axes.titlesize": 24,
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 15,
            "axes.linewidth": 0.9,
        }
    )
    fig, ax = plt.subplots(figsize=(5.2, 5.6), dpi=240)
    positions = [1, 2, 3]
    rng = np.random.default_rng(20260611)

    if kind == "box":
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=0.58,
            patch_artist=True,
            showfliers=False,
            whis=(5, 95),
            medianprops={"color": "#111111", "linewidth": 1.8},
            boxprops={"linewidth": 1.2},
            whiskerprops={"linewidth": 1.0, "color": "#555555"},
            capprops={"linewidth": 1.0, "color": "#555555"},
        )
        for patch, facies_id in zip(bp["boxes"], [0, 1, 2]):
            patch.set_facecolor(FACIES_COLORS[facies_id])
            patch.set_edgecolor("#333333")
            patch.set_alpha(0.62)
    elif kind == "violin":
        non_empty_positions = [pos for pos, vals in zip(positions, data) if len(vals)]
        non_empty_data = [vals for vals in data if len(vals)]
        vp = ax.violinplot(
            non_empty_data,
            positions=non_empty_positions,
            widths=0.76,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body, pos in zip(vp["bodies"], non_empty_positions):
            facies_id = positions.index(pos)
            body.set_facecolor(FACIES_COLORS[facies_id])
            body.set_edgecolor("#333333")
            body.set_alpha(0.46)
            body.set_linewidth(0.9)
        for pos, vals in zip(positions, data):
            if len(vals) == 0:
                continue
            q25, med, q75 = np.nanpercentile(vals, [25, 50, 75])
            ax.vlines(pos, q25, q75, color="#222222", linewidth=2.0, zorder=5)
            ax.hlines(med, pos - 0.22, pos + 0.22, color="#111111", linewidth=2.2, zorder=6)
    else:
        raise ValueError(f"Unknown distribution kind: {kind}")

    for idx, (facies_id, vals) in enumerate(zip([0, 1, 2], data), start=1):
        if len(vals) == 0:
            continue
        plot_vals = vals
        if len(plot_vals) > 520:
            plot_vals = rng.choice(plot_vals, size=520, replace=False)
        x = rng.normal(idx, 0.052, size=len(plot_vals))
        ax.scatter(
            x,
            plot_vals,
            s=7,
            color=FACIES_COLORS[facies_id],
            alpha=0.14 if kind == "box" else 0.11,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )

    if scale == "log":
        ax.set_yscale("log")
        scale_label = "log scale"
    else:
        scale_label = "linear scale"
    ax.set_title(label, pad=12)
    ax.set_xticks(positions)
    ax.set_xticklabels(["Facies I", "Facies II", "Facies III"])
    ax.set_xlim(0.45, 3.55)
    ax.set_ylabel(f"{unit} ({scale_label})" if unit else scale_label)
    ax.grid(True, axis="y", color="#D2D2D2", linewidth=0.58, alpha=0.62)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#4A4A4A")
    ax.spines["bottom"].set_color("#4A4A4A")
    fig.tight_layout()
    stem = f"geochem_facies_{kind}_{_feature_slug(col)}_{scale}"
    return _save_figure(fig, output_dir, stem, tight_layout=False)


def _plot_individual_element_distributions(df: pd.DataFrame, output_dir: Path) -> dict[str, dict[str, dict[str, str]]]:
    artifacts: dict[str, dict[str, dict[str, str]]] = {
        "box_default_scale": {},
        "box_linear": {},
        "violin_default_scale": {},
        "violin_linear": {},
    }
    for _, col, label, unit, default_scale in _all_boxplot_features():
        if col not in df.columns:
            continue
        log_scale = "log" if default_scale == "log" else "linear"
        artifacts["box_default_scale"][col] = _plot_single_feature_distribution(
            df,
            output_dir / "boxplots_individual_default_scale",
            col=col,
            label=label,
            unit=unit,
            kind="box",
            scale=log_scale,
        )
        artifacts["box_linear"][col] = _plot_single_feature_distribution(
            df,
            output_dir / "boxplots_individual_linear",
            col=col,
            label=label,
            unit=unit,
            kind="box",
            scale="linear",
        )
        artifacts["violin_default_scale"][col] = _plot_single_feature_distribution(
            df,
            output_dir / "violinplots_individual_default_scale",
            col=col,
            label=label,
            unit=unit,
            kind="violin",
            scale=log_scale,
        )
        artifacts["violin_linear"][col] = _plot_single_feature_distribution(
            df,
            output_dir / "violinplots_individual_linear",
            col=col,
            label=label,
            unit=unit,
            kind="violin",
            scale="linear",
        )
    return artifacts


def _plot_element_boxplots(df: pd.DataFrame, output_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    artifacts: dict[str, Any] = {}
    stats_rows: list[dict[str, Any]] = []
    for group in BOXPLOT_GROUPS:
        artifacts[group["key"]] = _plot_single_boxplot_group(df, output_dir, group)
        stats_rows.extend(_boxplot_group_stats(df, group["features"], group["key"], group["title"]))
    artifacts["individual"] = _plot_individual_element_distributions(df, output_dir)
    stats = pd.DataFrame(stats_rows)
    stats.to_csv(output_dir / "geochem_facies_element_boxplot_stats.csv", index=False)
    return artifacts, stats


def _load_classification_meta(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _default_classification_json(class_table: Path) -> Path | None:
    candidate = class_table.parent / "geochem_classification.json"
    return candidate if candidate.exists() else None


def _stage2_feature_cols(meta: dict[str, Any], df: pd.DataFrame) -> list[str]:
    stage2 = meta.get("stage2") if isinstance(meta.get("stage2"), dict) else {}
    cols = stage2.get("feature_cols") if isinstance(stage2, dict) else None
    if not cols:
        mechanism = meta.get("mechanism_artifacts") if isinstance(meta.get("mechanism_artifacts"), dict) else {}
        result = mechanism.get("result") if isinstance(mechanism.get("result"), dict) else {}
        cols = result.get("feature_cols")
    if not cols:
        cols = [
            "log_ratio_p_al",
            "log_ratio_mn_al",
            "log_ratio_fe_al",
            "log_ratio_mo_al",
            "log_ratio_v_al",
            "log_ratio_zn_al",
            "proxy_ce_anom",
            "proxy_y_ho",
            "proxy_la_yb",
            "proxy_dy_yb",
            "proxy_metal_index",
        ]
    return [str(c) for c in cols if str(c) in df.columns]


def _stage1_feature_cols(meta: dict[str, Any], df: pd.DataFrame) -> list[str]:
    stage1 = meta.get("stage1") if isinstance(meta.get("stage1"), dict) else {}
    cols = stage1.get("feature_cols") if isinstance(stage1, dict) else None
    if not cols:
        cols = ["log_ratio_cao_al", "log_ratio_tio2_al"]
    return [str(c) for c in cols if str(c) in df.columns]


def _feature_label(col: str) -> str:
    return CLUSTER_FEATURE_LABELS.get(col, col)


def _cluster_design_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, pd.DataFrame]:
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import RobustScaler

    if not feature_cols:
        raise ValueError("No clustering feature columns are available.")
    x = df.loc[:, feature_cols].copy()
    for col in feature_cols:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    valid = x.notna().any(axis=1)
    x = x.loc[valid].copy()
    if x.empty:
        raise ValueError("No valid rows are available for clustering diagnostics.")
    arr = x.to_numpy(dtype=float)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    arr = SimpleImputer(strategy="median").fit_transform(arr)
    arr = RobustScaler(with_centering=True, with_scaling=True).fit_transform(arr)
    return arr, x


def _compute_kmeans_metrics(
    x: np.ndarray,
    *,
    k_values: range = range(2, 7),
    random_state: int = 42,
    n_init: int = 20,
    max_iter: int = 300,
) -> pd.DataFrame:
    from sklearn.cluster import KMeans
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

    rows: list[dict[str, float | int]] = []
    for k in [int(v) for v in k_values if 2 <= int(v) < x.shape[0]]:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init, max_iter=max_iter, init="k-means++")
        labels = km.fit_predict(x)
        row: dict[str, float | int] = {
            "k": k,
            "inertia": float(getattr(km, "inertia_", np.nan)),
            "silhouette": np.nan,
            "davies_bouldin": np.nan,
            "calinski_harabasz": np.nan,
        }
        try:
            row["silhouette"] = float(silhouette_score(x, labels))
        except Exception:
            pass
        try:
            row["davies_bouldin"] = float(davies_bouldin_score(x, labels))
        except Exception:
            pass
        try:
            row["calinski_harabasz"] = float(calinski_harabasz_score(x, labels))
        except Exception:
            pass
        rows.append(row)
    if not rows:
        raise ValueError("No valid KMeans metrics could be computed.")
    return pd.DataFrame(rows)


def _plot_stage2_k_selection(
    metrics: pd.DataFrame,
    output_dir: Path,
    *,
    selected_k: int = 2,
) -> dict[str, str]:
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13})
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.1), dpi=240)
    metric_specs = [
        ("inertia", "Within-cluster SSE", "#5C6670", "lower"),
        ("silhouette", "Silhouette score", "#2C7FB8", "higher"),
        ("davies_bouldin", "Davies-Bouldin index", "#C95F3F", "lower"),
        ("calinski_harabasz", "Calinski-Harabasz score", "#4D8B57", "higher"),
    ]
    for ax, (col, label, color, direction) in zip(axes.ravel(), metric_specs):
        y = pd.to_numeric(metrics[col], errors="coerce")
        ax.plot(metrics["k"], y, color=color, marker="o", linewidth=2.2, markersize=6)
        if selected_k in set(metrics["k"].astype(int).tolist()):
            ax.axvline(selected_k, color="#1F1F1F", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.set_xlabel("Candidate k")
        ax.set_ylabel(label)
        ax.set_title(f"{label} ({direction} is better)")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, color="#E2E2E2", linewidth=0.7, alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.text(
        0.52,
        0.988,
        "Stage 2 only: non-carbonate split into Facies II and Facies III",
        ha="center",
        va="top",
        fontsize=17,
    )
    fig.text(
        0.52,
        0.952,
        "KMeans cluster-number diagnostics after Stage 1 removes the carbonate endmember",
        ha="center",
        va="top",
        fontsize=11.2,
        color="#4A4A4A",
    )
    fig.text(
        0.52,
        0.035,
        "Dashed line marks the adopted k=2 setting for the Stage-2 non-carbonate mechanism split.",
        ha="center",
        va="bottom",
        fontsize=10.5,
        color="#4A4A4A",
    )
    fig.tight_layout(rect=[0.02, 0.07, 0.98, 0.91])
    return _save_figure(fig, output_dir, "geochem_facies_stage2_k_selection", tight_layout=False)


def _plot_two_stage_k_selection(
    metrics: pd.DataFrame,
    output_dir: Path,
    *,
    selected_k: int = 2,
) -> dict[str, str] | None:
    if metrics.empty or not {"stage", "k", "relative_inertia"}.issubset(metrics.columns):
        return None

    stage_styles = {
        "Stage 1": {
            "label": "Stage 1: carbonate vs non-carbonate",
            "color": "#5C6670",
            "marker": "o",
        },
        "Stage 2": {
            "label": "Stage 2: Facies II vs Facies III",
            "color": "#2C7FB8",
            "marker": "s",
        },
    }
    metric_specs = [
        ("relative_inertia", "Relative within-cluster SSE", "lower"),
        ("silhouette", "Silhouette score", "higher"),
        ("davies_bouldin", "Davies-Bouldin index", "lower"),
        ("calinski_harabasz", "Calinski-Harabasz score", "higher"),
    ]

    plt.rcParams.update({"font.size": 13, "axes.titlesize": 14.5, "axes.labelsize": 13})
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.25), dpi=240)
    for ax, (col, label, direction) in zip(axes.ravel(), metric_specs):
        for stage, style in stage_styles.items():
            sub = metrics[metrics["stage"] == stage].sort_values("k")
            if sub.empty or col not in sub.columns:
                continue
            y = pd.to_numeric(sub[col], errors="coerce")
            ax.plot(
                sub["k"],
                y,
                color=style["color"],
                marker=style["marker"],
                linewidth=2.1,
                markersize=5.6,
                label=style["label"],
                alpha=0.95,
            )
        if selected_k in set(metrics["k"].astype(int).tolist()):
            ax.axvline(selected_k, color="#1F1F1F", linewidth=0.95, linestyle="--", alpha=0.62)
        ax.set_xlabel("Candidate k")
        ax.set_ylabel(label)
        ax.set_title(f"{label} ({direction} is better)")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, color="#E4E4E4", linewidth=0.7, alpha=0.76)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.905),
        ncol=2,
        frameon=True,
        framealpha=0.94,
        handlelength=2.0,
        columnspacing=1.6,
    )
    fig.text(
        0.52,
        0.988,
        "Two-stage KMeans cluster-number diagnostics",
        ha="center",
        va="top",
        fontsize=17,
    )
    fig.text(
        0.52,
        0.952,
        "Stage 1 fits all samples; Stage 2 fits only the non-carbonate domain",
        ha="center",
        va="top",
        fontsize=11.2,
        color="#4A4A4A",
    )
    fig.text(
        0.52,
        0.032,
        "Dashed line marks the adopted k=2 setting; SSE is normalized to each stage's k=2 value.",
        ha="center",
        va="bottom",
        fontsize=10.4,
        color="#4A4A4A",
    )
    fig.tight_layout(rect=[0.02, 0.07, 0.98, 0.86])
    return _save_figure(fig, output_dir, "geochem_facies_two_stage_k_selection", tight_layout=False)


def _plot_stage1_endmember_scatter(
    df: pd.DataFrame,
    output_dir: Path,
    feature_cols: list[str],
) -> dict[str, str] | None:
    if len(feature_cols) < 2 or "mech_endmember_id" not in df.columns:
        return None
    x_col, y_col = feature_cols[:2]
    plot_df = df[[x_col, y_col, "mech_endmember_id"]].copy()
    plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors="coerce")
    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")
    plot_df["mech_endmember_id"] = pd.to_numeric(plot_df["mech_endmember_id"], errors="coerce")
    plot_df = plot_df.dropna()
    if plot_df.empty:
        return None

    plt.rcParams.update({"font.size": 13, "axes.titlesize": 16, "axes.labelsize": 14})
    fig, ax = plt.subplots(figsize=(6.9, 5.8), dpi=240)
    colors = {0: "#8F969E", 1: "#3C6E8F"}
    labels = {0: "carbonate endmember", 1: "non-carbonate domain"}
    for class_id in [0, 1]:
        sub = plot_df[plot_df["mech_endmember_id"].astype(int) == class_id]
        if sub.empty:
            continue
        ax.scatter(
            sub[x_col],
            sub[y_col],
            s=18,
            color=colors[class_id],
            edgecolors="white",
            linewidths=0.25,
            alpha=0.72,
            label=f"{labels[class_id]}",
        )
    ax.set_xlabel(_feature_label(x_col))
    ax.set_ylabel(_feature_label(y_col))
    ax.set_title("Stage 1: carbonate endmember vs non-carbonate domain", pad=16)
    ax.text(
        0.02,
        1.015,
        "Input: Ca-Al-Ti ratios; output: Facies I candidate vs samples passed to Stage 2",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        color="#4A4A4A",
    )
    ax.grid(True, color="#E5E5E5", linewidth=0.75, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", frameon=True, framealpha=0.92)
    return _save_figure(fig, output_dir, "geochem_facies_stage1_endmember_scatter")


def _plot_two_stage_joint_diagnostics(df: pd.DataFrame, output_dir: Path) -> dict[str, str] | None:
    required = {
        "log_ratio_cao_al",
        "log_ratio_tio2_al",
        "log_ratio_fe_al",
        "log_ratio_p_al",
        "mech_endmember_id",
        "facies_id",
    }
    if not required.issubset(df.columns):
        return None
    plot_df = df[list(required)].copy()
    for col in required:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    plot_df = plot_df.dropna()
    if plot_df.empty:
        return None

    plt.rcParams.update({"font.size": 13, "axes.titlesize": 16, "axes.labelsize": 14})
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.3), dpi=240)

    ax = axes[0]
    stage1_colors = {0: FACIES_COLORS[0], 1: "#3C6E8F"}
    stage1_labels = {0: "Facies I candidate", 1: "passed to Stage 2"}
    for class_id in [0, 1]:
        sub = plot_df[plot_df["mech_endmember_id"].astype(int) == class_id]
        ax.scatter(
            sub["log_ratio_cao_al"],
            sub["log_ratio_tio2_al"],
            s=14,
            color=stage1_colors[class_id],
            edgecolors="white",
            linewidths=0.18,
            alpha=0.68,
            label=stage1_labels[class_id],
        )
    ax.set_title("Stage 1: carbonate endmember screen", pad=10)
    ax.set_xlabel("log(CaO/Al$_2$O$_3$)")
    ax.set_ylabel("log(TiO$_2$/Al$_2$O$_3$)")
    ax.legend(loc="best", frameon=True, framealpha=0.92, fontsize=11)

    ax = axes[1]
    stage2_df = plot_df[plot_df["mech_endmember_id"].astype(int) == 1]
    for facies_id in [1, 2]:
        sub = stage2_df[stage2_df["facies_id"].astype(int) == facies_id]
        ax.scatter(
            sub["log_ratio_fe_al"],
            sub["log_ratio_p_al"],
            s=14,
            color=FACIES_COLORS[facies_id],
            edgecolors="white",
            linewidths=0.18,
            alpha=0.68,
            label=FACIES_LABELS[facies_id],
        )
    ax.set_title("Stage 2: non-carbonate mechanism split", pad=10)
    ax.set_xlabel("log(Fe$_2$O$_3$/Al$_2$O$_3$)")
    ax.set_ylabel("log(P$_2$O$_5$/Al$_2$O$_3$)")
    ax.legend(loc="best", frameon=True, framealpha=0.92, fontsize=11)

    for ax in axes:
        ax.grid(True, color="#E5E5E5", linewidth=0.75, alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Two-stage geochemical facies classification diagnostics", fontsize=19, y=0.995)
    fig.text(
        0.5,
        0.02,
        "Stage 1 separates carbonate background; Stage 2 separates the remaining non-carbonate samples into Facies II and Facies III.",
        ha="center",
        va="bottom",
        fontsize=11,
        color="#4A4A4A",
    )
    fig.tight_layout(rect=[0.02, 0.065, 0.98, 0.93])
    return _save_figure(fig, output_dir, "geochem_facies_two_stage_joint_diagnostics", tight_layout=False)


def _plot_pca_3d_scatter(
    df: pd.DataFrame,
    output_dir: Path,
    feature_cols: list[str],
    *,
    label_col: str,
    stem: str,
    title: str,
    color_map: dict[int, str],
    label_map: dict[int, str],
) -> tuple[dict[str, str] | None, pd.DataFrame | None]:
    from sklearn.decomposition import PCA

    if label_col not in df.columns or not feature_cols:
        return None, None
    labels = pd.to_numeric(df[label_col], errors="coerce")
    valid = labels.notna()
    x, used = _cluster_design_matrix(df.loc[valid].copy(), feature_cols)
    label_values = labels.loc[used.index].astype(int).to_numpy()
    if x.shape[0] < 5 or len(set(label_values.tolist())) < 2:
        return None, None

    pca = PCA(n_components=3, random_state=0)
    coords = pca.fit_transform(x)
    pca_df = pd.DataFrame(
        {
            "pc1": coords[:, 0],
            "pc2": coords[:, 1],
            "pc3": coords[:, 2],
            label_col: label_values,
        },
        index=used.index,
    )
    csv_path = output_dir / f"{stem}_coordinates.csv"
    pca_df.to_csv(csv_path, index=False)

    plt.rcParams.update({"font.size": 12, "axes.titlesize": 15, "axes.labelsize": 12})
    fig = plt.figure(figsize=(8.2, 6.8), dpi=240)
    ax = fig.add_subplot(111, projection="3d")
    for class_id in sorted(set(label_values.tolist())):
        sub = pca_df[pca_df[label_col] == class_id]
        ax.scatter(
            sub["pc1"],
            sub["pc2"],
            sub["pc3"],
            s=15,
            color=color_map.get(int(class_id), "#999999"),
            edgecolors="white",
            linewidths=0.18,
            alpha=0.64,
            depthshade=False,
            label=label_map.get(int(class_id), f"class {class_id}"),
        )
    explained = pca.explained_variance_ratio_ * 100.0
    ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)", labelpad=8)
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)", labelpad=8)
    ax.set_zlabel(f"PC3 ({explained[2]:.1f}%)", labelpad=8)
    ax.set_title(title, pad=18)
    ax.view_init(elev=22, azim=-48)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_facecolor((0.98, 0.98, 0.98, 1.0))
        axis.pane.set_edgecolor((0.86, 0.86, 0.86, 1.0))
    ax.grid(True, color="#E6E6E6")
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.98), frameon=True, framealpha=0.92)
    paths = _save_figure(fig, output_dir, stem, tight_layout=False)
    return paths, pca_df


def _plot_real_geochem_3d_scatter(
    df: pd.DataFrame,
    output_dir: Path,
) -> tuple[dict[str, str] | None, pd.DataFrame | None]:
    feature_cols = ["log_ratio_cao_al", "log_ratio_fe_al", "log_ratio_p_al"]
    if any(col not in df.columns for col in feature_cols) or "facies_id" not in df.columns:
        return None, None

    plot_df = df[[*feature_cols, "facies_id"]].copy()
    for col in feature_cols:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    plot_df["facies_id"] = pd.to_numeric(plot_df["facies_id"], errors="coerce")
    plot_df = plot_df.dropna()
    if plot_df.empty:
        return None, None

    csv_path = output_dir / "geochem_facies_real_geochem_3d_scatter_coordinates.csv"
    plot_df.rename(
        columns={
            "log_ratio_cao_al": "log_cao_al",
            "log_ratio_fe_al": "log_fe_al",
            "log_ratio_p_al": "log_p_al",
        }
    ).to_csv(csv_path, index=False)

    plt.rcParams.update({"font.size": 12, "axes.titlesize": 16, "axes.labelsize": 12})
    fig = plt.figure(figsize=(8.2, 6.8), dpi=240)
    ax = fig.add_subplot(111, projection="3d")
    for facies_id in [0, 1, 2]:
        sub = plot_df[plot_df["facies_id"].astype(int) == facies_id]
        if sub.empty:
            continue
        ax.scatter(
            sub["log_ratio_cao_al"],
            sub["log_ratio_fe_al"],
            sub["log_ratio_p_al"],
            s=15,
            color=FACIES_COLORS[facies_id],
            edgecolors="white",
            linewidths=0.18,
            alpha=0.64,
            depthshade=False,
            label=FACIES_LABELS[facies_id],
        )

    ax.set_xlabel("log(CaO/Al$_2$O$_3$)", labelpad=9)
    ax.set_ylabel("log(Fe$_2$O$_3$/Al$_2$O$_3$)", labelpad=9)
    ax.set_zlabel("")
    ax.set_title("Final two-stage facies labels in geochemical-ratio space", pad=18)
    ax.text2D(
        0.50,
        0.935,
        "Stage 1 defines Facies I; Stage 2 separates the non-carbonate samples into Facies II and III",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10.5,
        color="#4A4A4A",
    )
    ax.view_init(elev=22, azim=-42)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_facecolor((0.98, 0.98, 0.98, 1.0))
        axis.pane.set_edgecolor((0.86, 0.86, 0.86, 1.0))
    ax.grid(True, color="#E6E6E6")
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.98), frameon=True, framealpha=0.92)
    ax.text2D(
        1.075,
        0.52,
        "log(P$_2$O$_5$/Al$_2$O$_3$)",
        transform=ax.transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=12.8,
        color="#111111",
    )
    fig.subplots_adjust(left=0.02, right=0.80, top=0.92, bottom=0.02)
    paths = _save_figure(fig, output_dir, "geochem_facies_real_geochem_3d_scatter", tight_layout=False)
    return paths, plot_df


def _transformed_axis_values(series: pd.Series, transform: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if transform == "log1p":
        return np.log1p(values.clip(lower=0.0))
    return values


def _plot_real_variable_3d_scatter(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    axes_spec: tuple[tuple[str, str, str], tuple[str, str, str], tuple[str, str, str]],
    stem: str,
    title: str,
    subtitle: str,
) -> tuple[dict[str, str] | None, pd.DataFrame | None]:
    cols = [spec[0] for spec in axes_spec]
    if any(col not in df.columns for col in cols) or "facies_id" not in df.columns:
        return None, None

    plot_df = pd.DataFrame(index=df.index)
    for col, _, transform in axes_spec:
        plot_df[col] = _transformed_axis_values(df[col], transform)
    plot_df["facies_id"] = pd.to_numeric(df["facies_id"], errors="coerce")
    plot_df = plot_df.dropna()
    if plot_df.empty:
        return None, None
    plot_df.to_csv(output_dir / f"{stem}_coordinates.csv", index=False)

    plt.rcParams.update({"font.size": 12, "axes.titlesize": 15, "axes.labelsize": 12})
    fig = plt.figure(figsize=(8.2, 6.8), dpi=240)
    ax = fig.add_subplot(111, projection="3d")
    x_col, x_label, _ = axes_spec[0]
    y_col, y_label, _ = axes_spec[1]
    z_col, z_label, _ = axes_spec[2]
    for facies_id in [0, 1, 2]:
        sub = plot_df[plot_df["facies_id"].astype(int) == facies_id]
        if sub.empty:
            continue
        ax.scatter(
            sub[x_col],
            sub[y_col],
            sub[z_col],
            s=15,
            color=FACIES_COLORS[facies_id],
            edgecolors="white",
            linewidths=0.18,
            alpha=0.64,
            depthshade=False,
            label=FACIES_LABELS[facies_id],
        )
    ax.set_xlabel(x_label, labelpad=9)
    ax.set_ylabel(y_label, labelpad=9)
    ax.set_zlabel("")
    ax.set_title(title, pad=18)
    ax.view_init(elev=22, azim=-42)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_facecolor((0.98, 0.98, 0.98, 1.0))
        axis.pane.set_edgecolor((0.86, 0.86, 0.86, 1.0))
    ax.grid(True, color="#E6E6E6")
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.98), frameon=True, framealpha=0.92)
    ax.text2D(
        1.075,
        0.52,
        z_label,
        transform=ax.transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=12.8,
        color="#111111",
    )
    fig.subplots_adjust(left=0.02, right=0.80, top=0.92, bottom=0.02)
    paths = _save_figure(fig, output_dir, stem, tight_layout=False)
    return paths, plot_df


def _plot_clustering_diagnostics(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    classification_json: Path | None,
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    meta = _load_classification_meta(classification_json)
    stage1_cols = _stage1_feature_cols(meta, df)
    stage2_cols = _stage2_feature_cols(meta, df)
    artifacts: dict[str, Any] = {
        "classification_json": str(classification_json) if classification_json else None,
        "stage1_features": stage1_cols,
        "stage2_features": stage2_cols,
    }

    stage1_paths = _plot_stage1_endmember_scatter(df, output_dir, stage1_cols)
    if stage1_paths:
        artifacts["stage1_endmember_scatter"] = stage1_paths
    joint_paths = _plot_two_stage_joint_diagnostics(df, output_dir)
    if joint_paths:
        artifacts["two_stage_joint_diagnostics"] = joint_paths

    stage1_metrics_df: pd.DataFrame | None = None
    if stage1_cols:
        try:
            x_stage1, _ = _cluster_design_matrix(df, stage1_cols)
            stage1_metrics_df = _compute_kmeans_metrics(x_stage1, k_values=range(2, 7))
            stage1_metrics_df.insert(0, "stage", "Stage 1")
        except Exception as exc:
            logger.warning("Failed to compute Stage-1 KMeans diagnostics: %s", exc)

    if "mech_endmember_id" in df.columns:
        stage2_df = df[pd.to_numeric(df["mech_endmember_id"], errors="coerce") == 1].copy()
    else:
        stage2_df = df[df["facies_id"].isin([1, 2])].copy()
    metrics_df: pd.DataFrame | None = None
    if not stage2_df.empty and stage2_cols:
        x_stage2, used_stage2 = _cluster_design_matrix(stage2_df, stage2_cols)
        metrics_df = _compute_kmeans_metrics(x_stage2, k_values=range(2, 7))
        metrics_df.to_csv(output_dir / "geochem_facies_stage2_k_selection_metrics.csv", index=False)
        selected_k = int((meta.get("stage2") or {}).get("n_clusters", 2)) if isinstance(meta.get("stage2"), dict) else 2
        artifacts["stage2_k_selection"] = _plot_stage2_k_selection(metrics_df, output_dir, selected_k=selected_k)
        if stage1_metrics_df is not None:
            stage2_metrics_df = metrics_df.copy()
            stage2_metrics_df.insert(0, "stage", "Stage 2")
            combined_metrics = pd.concat([stage1_metrics_df, stage2_metrics_df], ignore_index=True)
            combined_metrics["relative_inertia"] = combined_metrics.groupby("stage")["inertia"].transform(
                lambda values: values / values.iloc[0] if np.isfinite(values.iloc[0]) and values.iloc[0] else np.nan
            )
            combined_metrics_path = output_dir / "geochem_facies_two_stage_k_selection_metrics.csv"
            combined_metrics.to_csv(combined_metrics_path, index=False)
            combined_paths = _plot_two_stage_k_selection(combined_metrics, output_dir, selected_k=selected_k)
            if combined_paths:
                artifacts["two_stage_k_selection"] = combined_paths
                artifacts["two_stage_k_selection_metrics_csv"] = str(combined_metrics_path)

        stage2_label_col = "mech_subclass_id" if "mech_subclass_id" in stage2_df.columns else "facies_id"
        stage2_colors = {0: FACIES_COLORS[1], 1: FACIES_COLORS[2], 2: FACIES_COLORS[2]}
        stage2_labels = {0: "Facies II domain", 1: "Facies III domain", 2: "Facies III domain"}
        paths, coords = _plot_pca_3d_scatter(
            stage2_df.loc[used_stage2.index].copy(),
            output_dir,
            stage2_cols,
            label_col=stage2_label_col,
            stem="geochem_facies_stage2_pca_3d_scatter",
            title="Stage 2 only: non-carbonate samples in mechanism-feature PCA space",
            color_map=stage2_colors,
            label_map=stage2_labels,
        )
        if paths:
            artifacts["stage2_pca_3d_scatter"] = paths
            artifacts["stage2_pca_3d_coordinates_csv"] = str(output_dir / "geochem_facies_stage2_pca_3d_scatter_coordinates.csv")

    final_cols = [c for c in [*stage1_cols, *stage2_cols] if c in df.columns]
    final_cols = list(dict.fromkeys(final_cols))
    if final_cols:
        paths, coords = _plot_real_geochem_3d_scatter(df, output_dir)
        if paths:
            artifacts["real_geochem_3d_scatter"] = paths
            artifacts["real_geochem_3d_coordinates_csv"] = str(output_dir / "geochem_facies_real_geochem_3d_scatter_coordinates.csv")

        variable_3d_specs = [
            (
                "major_oxides_3d_scatter",
                "geochem_facies_major_oxides_3d_scatter",
                (
                    ("CaO", "log1p(CaO)", "log1p"),
                    ("Fe2O3", "log1p(Fe$_2$O$_3$)", "log1p"),
                    ("Al2O3", "log1p(Al$_2$O$_3$)", "log1p"),
                ),
                "Final facies labels in major-oxide space",
                "Axes use observed major-oxide concentrations with log1p scaling.",
            ),
            (
                "p_fe_rey_3d_scatter",
                "geochem_facies_p_fe_rey_3d_scatter",
                (
                    ("P2O5", "log1p(P$_2$O$_5$)", "log1p"),
                    ("Fe2O3", "log1p(Fe$_2$O$_3$)", "log1p"),
                    ("rey", "log1p(total REY)", "log1p"),
                ),
                "Final facies labels in P-Fe-REY space",
                "This view emphasizes phosphate, Fe-bearing carriers, and observed REY grade.",
            ),
            (
                "cao_p_rey_3d_scatter",
                "geochem_facies_cao_p_rey_3d_scatter",
                (
                    ("CaO", "log1p(CaO)", "log1p"),
                    ("P2O5", "log1p(P$_2$O$_5$)", "log1p"),
                    ("rey", "log1p(total REY)", "log1p"),
                ),
                "Final facies labels in Ca-P-REY space",
                "This view contrasts carbonate dilution against phosphate-associated REY enrichment.",
            ),
        ]
        variable_artifacts: dict[str, Any] = {}
        for key, stem, axes_spec, title, subtitle in variable_3d_specs:
            paths, coords = _plot_real_variable_3d_scatter(
                df,
                output_dir,
                axes_spec=axes_spec,
                stem=stem,
                title=title,
                subtitle=subtitle,
            )
            if paths:
                variable_artifacts[key] = paths
                variable_artifacts[f"{key}_coordinates_csv"] = str(output_dir / f"{stem}_coordinates.csv")
        if variable_artifacts:
            artifacts["real_variable_3d_scatters"] = variable_artifacts

        paths, coords = _plot_pca_3d_scatter(
            df,
            output_dir,
            final_cols,
            label_col="facies_id",
            stem="geochem_facies_mechanism_pca_3d_scatter",
            title="Final two-stage facies labels in mechanism-feature PCA space",
            color_map=FACIES_COLORS,
            label_map=FACIES_LABELS,
        )
        if paths:
            artifacts["mechanism_pca_3d_scatter"] = paths
            artifacts["mechanism_pca_3d_coordinates_csv"] = str(output_dir / "geochem_facies_mechanism_pca_3d_scatter_coordinates.csv")

    return artifacts, metrics_df


def _write_summary(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    grouped = df.groupby(["facies_id", "facies_label", "facies_name"], dropna=False)
    summary = grouped.size().rename("n_samples").reset_index()
    if "core_id" in df.columns:
        core_counts = grouped["core_id"].nunique(dropna=True).rename("n_cores").reset_index()
        summary = summary.merge(core_counts, on=["facies_id", "facies_label", "facies_name"], how="left")
    else:
        summary["n_cores"] = 0
    if "rey" in df.columns:
        rey_stats = (
            grouped["rey"]
            .agg(
                median_rey="median",
                p90_rey=lambda s: float(pd.to_numeric(s, errors="coerce").quantile(0.90)),
                ge1000=lambda s: int((pd.to_numeric(s, errors="coerce") >= 1000.0).sum()),
                ge1500=lambda s: int((pd.to_numeric(s, errors="coerce") >= 1500.0).sum()),
            )
            .reset_index()
        )
        summary = summary.merge(rey_stats, on=["facies_id", "facies_label", "facies_name"], how="left")
    else:
        summary["median_rey"] = np.nan
        summary["p90_rey"] = np.nan
        summary["ge1000"] = 0
        summary["ge1500"] = 0
    summary.to_csv(output_dir / "geochem_facies_summary.csv", index=False)
    return summary


def export_geochem_facies_figures(
    *,
    class_table: Path,
    output_dir: Path,
    label_col: str = "mech_class_id",
    env_nc: Path | None = DEFAULT_RAW_ENV_NC,
    boundaries_gmt: Path | None = DEFAULT_PLATE_BOUNDARIES,
    profile_cores: list[str] | None = None,
    classification_json: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if classification_json is None:
        classification_json = _default_classification_json(class_table)
    df = _load_table(class_table, label_col)
    summary_df = _write_summary(df, output_dir)
    map_paths = _plot_facies_map(df, output_dir, boundaries_gmt)
    rey_location_map_paths, rey_location_summary = _plot_rey_sample_location_map(
        df,
        output_dir,
        boundaries_gmt,
        env_nc=env_nc,
    )
    ree_paths, _ = _plot_ree_profiles(df, output_dir)
    heatmap_paths, _ = _plot_geochem_heatmap(df, output_dir)
    env_heatmap_paths, _, _ = _plot_environment_heatmap(df, output_dir, env_nc)
    boxplot_paths, _ = _plot_element_boxplots(df, output_dir)
    core_profile_paths, core_profile_summary = _plot_selected_core_profiles(
        df,
        output_dir,
        requested_cores=profile_cores,
    )
    clustering_paths, clustering_metrics = _plot_clustering_diagnostics(
        df,
        output_dir,
        classification_json=classification_json,
    )
    artifacts = {
        "facies_map": map_paths,
        "rey_sample_location_map": rey_location_map_paths,
        "ree_profiles": ree_paths,
        "feature_heatmap": heatmap_paths,
        "environment_heatmap": env_heatmap_paths,
        "selected_core_profiles": core_profile_paths,
        "clustering_diagnostics": clustering_paths,
        "element_boxplots": boxplot_paths,
        "facies_summary_csv": str(output_dir / "geochem_facies_summary.csv"),
        "rey_sample_location_summary_csv": str(output_dir / "geochem_rey_sample_location_summary.csv"),
        "ree_profiles_csv": str(output_dir / "geochem_facies_ree_profiles.csv"),
        "feature_heatmap_values_csv": str(output_dir / "geochem_facies_feature_heatmap_values.csv"),
        "environment_heatmap_values_csv": str(output_dir / "geochem_facies_environment_heatmap_values.csv")
        if env_heatmap_paths
        else None,
        "environment_raw_medians_csv": str(output_dir / "geochem_facies_environment_raw_medians.csv")
        if env_heatmap_paths
        else None,
        "element_boxplot_stats_csv": str(output_dir / "geochem_facies_element_boxplot_stats.csv"),
        "selected_core_profile_summary_csv": str(output_dir / "geochem_facies_selected_core_profile_summary.csv"),
        "stage2_k_selection_metrics_csv": str(output_dir / "geochem_facies_stage2_k_selection_metrics.csv")
        if clustering_metrics is not None
        else None,
        "two_stage_k_selection_metrics_csv": (
            clustering_paths.get("two_stage_k_selection_metrics_csv")
            if isinstance(clustering_paths, dict)
            else None
        ),
    }
    result = {
        "status": "completed",
        "class_table": str(class_table),
        "classification_json": str(classification_json) if classification_json else None,
        "env_nc": str(env_nc) if env_nc else None,
        "boundaries_gmt": str(boundaries_gmt) if boundaries_gmt else None,
        "output_dir": str(output_dir),
        "label_col": label_col,
        "facies_labels": FACIES_LABELS,
        "facies_names": FACIES_NAMES,
        "paas_values_ppm": PAAS_VALUES,
        "n_samples": int(len(df)),
        "facies_summary": summary_df.to_dict(orient="records"),
        "rey_sample_location_summary": rey_location_summary.to_dict(orient="records"),
        "selected_core_profile_summary": core_profile_summary.to_dict(orient="records"),
        "stage2_k_selection_metrics": clustering_metrics.to_dict(orient="records")
        if clustering_metrics is not None
        else None,
        "artifacts": artifacts,
    }
    _write_json(output_dir / "geochem_facies_figure_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Export geochemical facies figures for paper-facing preprocessing diagnostics.")
    parser.add_argument("--class-table", type=Path, default=DEFAULT_CLASS_TABLE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label-col", default="mech_class_id")
    parser.add_argument("--env-nc", type=Path, default=DEFAULT_RAW_ENV_NC)
    parser.add_argument("--boundaries-gmt", type=Path, default=DEFAULT_PLATE_BOUNDARIES)
    parser.add_argument(
        "--classification-json",
        type=Path,
        default=None,
        help="Optional geochem_classification.json for clustering feature lists and saved KMeans metadata.",
    )
    parser.add_argument(
        "--profile-cores",
        default="",
        help="Optional comma-separated core_id list for the selected-core vertical profile figure.",
    )
    args = parser.parse_args()
    profile_cores = [item.strip() for item in str(args.profile_cores).split(",") if item.strip()] or None
    summary = export_geochem_facies_figures(
        class_table=args.class_table,
        output_dir=args.output_dir,
        label_col=str(args.label_col),
        env_nc=args.env_nc,
        boundaries_gmt=args.boundaries_gmt,
        profile_cores=profile_cores,
        classification_json=args.classification_json,
    )
    print(json.dumps(summary, cls=NumpyJSONEncoder, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

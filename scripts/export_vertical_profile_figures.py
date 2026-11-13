from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from core.utils.netcdf_io import open_dataset
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from core.utils.serialization_utils import NumpyJSONEncoder


DEFAULT_CLASS_TABLE = Path(
    "data/geochem_class_table.parquet"
)
DEFAULT_PREDICTION_DIR = Path("products")
DEFAULT_ENV_NC = Path("data/env_gridded.nc")
DEFAULT_SED_RATE_NC = Path("data/Average sedimentation rate.nc")
DEFAULT_OUTPUT_DIR = Path("outputs/vertical_profile_figures")

PREDICTION_3D_NC = "continuous_rey_mechanism_threshold_predictions_3d.nc"
PREDICTION_2D_NC = "continuous_rey_paper_summary_2d.nc"
AGE_MIN_KA = 0.0
AGE_MAX_KA = 2580.0
FACIES_IDS = [0, 1, 2]
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
    1: "#D65F5F",
    2: "#4C8FC4",
}
FACIES_LIGHT = {
    0: "#C4C8CC",
    1: "#F0B3AF",
    2: "#A9CDE7",
}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, cls=NumpyJSONEncoder, indent=2, ensure_ascii=False)


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}.png"
    svg = output_dir / f"{stem}.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png), "svg": str(svg)}


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12.0,
            "axes.titlesize": 14.5,
            "axes.labelsize": 12.5,
            "xtick.labelsize": 10.8,
            "ytick.labelsize": 10.8,
            "legend.fontsize": 10.0,
            "axes.linewidth": 0.9,
            "axes.edgecolor": "#333333",
            "axes.titlepad": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.dpi": 300,
        }
    )


def _style_axis(ax: plt.Axes, *, grid_axis: str = "both") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.tick_params(axis="both", colors="#222222", length=3.4, width=0.8)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color="#D8D8D8", linewidth=0.55, alpha=0.58)
    ax.set_axisbelow(True)


def _add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=15.0,
        fontweight="bold",
        color="#111111",
    )


def _coord_names(ds: xr.Dataset) -> tuple[str, str]:
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    if lat_name not in ds.coords or lon_name not in ds.coords:
        raise KeyError(f"Could not find latitude/longitude coordinates in {list(ds.coords)}")
    return lat_name, lon_name


def _vertical_dim(ds: xr.Dataset) -> str:
    for name in ("age_ka", "depth_mbsf", "depth"):
        if name in ds.coords or name in ds.dims:
            return name
    raise KeyError(f"Could not find a vertical coordinate in {list(ds.coords)}")


def _load_class_table(path: Path, label_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing geochemical class table: {path}")
    df = pd.read_parquet(path)
    required = {"lon", "lat", "depth", "rey", label_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Class table is missing required columns: {missing}")
    out = df.copy()
    out[label_col] = pd.to_numeric(out[label_col], errors="coerce").astype("Int64")
    out = out[out[label_col].isin(FACIES_IDS)].copy()
    out["facies_id"] = out[label_col].astype(int)
    out["facies_label"] = out["facies_id"].map(FACIES_LABELS)
    out["facies_name"] = out["facies_id"].map(FACIES_NAMES)
    out["rey"] = pd.to_numeric(out["rey"], errors="coerce")
    out["depth"] = pd.to_numeric(out["depth"], errors="coerce")
    out = out[out["rey"].notna() & out["depth"].notna()].copy()
    if out.empty:
        raise ValueError(f"No valid rows found in {path} using label column {label_col}.")
    return out


def _compute_age_proxy(
    df: pd.DataFrame,
    *,
    sed_rate_nc: Path | None,
    sed_rate_var: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    for col in ("age_ka_proxy", "age_ka", "apparent_age_ka"):
        if col in out.columns:
            out["age_ka_proxy"] = pd.to_numeric(out[col], errors="coerce")
            return out, {
                "age_source": col,
                "sedimentation_rate_nc": None,
                "sedimentation_rate_var": None,
            }

    if sed_rate_nc is None or not sed_rate_nc.exists():
        raise FileNotFoundError(
            "No age field was found in the class table and the sedimentation-rate grid is unavailable: "
            f"{sed_rate_nc}"
        )

    with open_dataset(sed_rate_nc) as ds:
        if sed_rate_var not in ds.data_vars:
            raise KeyError(f"{sed_rate_nc} does not contain sedimentation-rate variable '{sed_rate_var}'.")
        sed = ds[sed_rate_var]
        lon_name = "lon" if "lon" in sed.coords else "longitude"
        lat_name = "lat" if "lat" in sed.coords else "latitude"
        lon_values = pd.to_numeric(out["lon"], errors="coerce").to_numpy(dtype=float)
        lat_values = pd.to_numeric(out["lat"], errors="coerce").to_numpy(dtype=float)
        interp_lon = ((lon_values + 180.0) % 360.0) - 180.0
        linear = sed.interp(
            {lon_name: xr.DataArray(interp_lon, dims="points"), lat_name: xr.DataArray(lat_values, dims="points")},
            method="linear",
        ).to_numpy()
        nearest = sed.interp(
            {lon_name: xr.DataArray(interp_lon, dims="points"), lat_name: xr.DataArray(lat_values, dims="points")},
            method="nearest",
        ).to_numpy()
    sed_values = np.asarray(linear, dtype=float)
    sed_values = np.where(np.isfinite(sed_values), sed_values, np.asarray(nearest, dtype=float))
    sed_values = np.clip(sed_values, 1e-3, None)
    depth_m = np.clip(out["depth"].to_numpy(dtype=float), 0.0, None)
    out["sed_rate_raw_cm_per_ka"] = sed_values
    out["age_ka_proxy"] = depth_m * 100.0 / sed_values
    return out, {
        "age_source": "computed_from_depth_and_sedimentation_rate",
        "sedimentation_rate_nc": str(sed_rate_nc),
        "sedimentation_rate_var": sed_rate_var,
        "sed_rate_min_cm_per_ka": float(np.nanmin(sed_values)),
        "sed_rate_median_cm_per_ka": float(np.nanmedian(sed_values)),
        "sed_rate_max_cm_per_ka": float(np.nanmax(sed_values)),
    }


def _binned_facies_stats(
    df: pd.DataFrame,
    *,
    bin_width_ka: float,
    age_max_ka: float,
) -> pd.DataFrame:
    bins = np.arange(0.0, age_max_ka + bin_width_ka, bin_width_ka, dtype=float)
    if bins[-1] < age_max_ka:
        bins = np.append(bins, age_max_ka)
    labels = (bins[:-1] + bins[1:]) / 2.0
    work = df.copy()
    work["age_bin_mid_ka"] = pd.cut(
        work["age_ka_proxy"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )
    rows: list[dict[str, Any]] = []
    for facies_id in FACIES_IDS:
        sub_facies = work[work["facies_id"] == facies_id]
        for mid in labels:
            sub = sub_facies[sub_facies["age_bin_mid_ka"].astype(float) == float(mid)]
            rey = pd.to_numeric(sub["rey"], errors="coerce").dropna()
            row = {
                "facies_id": facies_id,
                "facies_label": FACIES_LABELS[facies_id],
                "age_bin_mid_ka": float(mid),
                "age_bin_start_ka": float(mid - bin_width_ka / 2.0),
                "age_bin_end_ka": float(mid + bin_width_ka / 2.0),
                "n": int(len(rey)),
            }
            if len(rey):
                row.update(
                    {
                        "rey_median": float(rey.median()),
                        "rey_mean": float(rey.mean()),
                        "rey_q25": float(rey.quantile(0.25)),
                        "rey_q75": float(rey.quantile(0.75)),
                        "frac_ge_800": float((rey >= 800.0).mean()),
                        "frac_ge_1000": float((rey >= 1000.0).mean()),
                        "frac_ge_1500": float((rey >= 1500.0).mean()),
                    }
                )
            else:
                row.update(
                    {
                        "rey_median": np.nan,
                        "rey_mean": np.nan,
                        "rey_q25": np.nan,
                        "rey_q75": np.nan,
                        "frac_ge_800": np.nan,
                        "frac_ge_1000": np.nan,
                        "frac_ge_1500": np.nan,
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _sample_core_profiles(df: pd.DataFrame, max_cores: int, seed: int) -> list[pd.DataFrame]:
    if "core_id" not in df.columns:
        return []
    core_sizes = df.groupby("core_id").size()
    eligible = core_sizes[core_sizes >= 3].index.to_numpy()
    if len(eligible) == 0:
        return []
    rng = np.random.default_rng(seed)
    if len(eligible) > max_cores:
        eligible = rng.choice(eligible, size=max_cores, replace=False)
    profiles = []
    for core_id in eligible:
        sub = df[df["core_id"] == core_id].sort_values("age_ka_proxy")
        profiles.append(sub)
    return profiles


def _plot_observed_vertical_profiles(
    df: pd.DataFrame,
    binned: pd.DataFrame,
    output_dir: Path,
    *,
    max_core_lines: int,
    seed: int,
) -> dict[str, str]:
    _set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 10.2), dpi=190, constrained_layout=True)
    ax = axes[0, 0]
    for profile in _sample_core_profiles(df, max_core_lines, seed):
        ax.plot(
            profile["rey"].to_numpy(dtype=float),
            profile["age_ka_proxy"].to_numpy(dtype=float),
            color="#B8B8B8",
            linewidth=0.55,
            alpha=0.18,
            zorder=1,
        )
    for facies_id in FACIES_IDS:
        sub = df[df["facies_id"] == facies_id]
        ax.scatter(
            sub["rey"],
            sub["age_ka_proxy"],
            s=11,
            color=FACIES_COLORS[facies_id],
            alpha=0.58,
            edgecolors="none",
            rasterized=True,
            label=f"{FACIES_LABELS[facies_id]} (n={len(sub)})",
            zorder=3,
        )
    ax.set_xscale("log")
    ax.set_ylim(AGE_MAX_KA, AGE_MIN_KA)
    ax.set_xlabel("Observed REY (ppm, log scale)")
    ax.set_ylabel("Apparent age proxy (ka)")
    ax.set_title("Observed samples and core-wise profiles")
    ax.legend(frameon=False, loc="lower right", handlelength=1.2)
    _style_axis(ax)
    _add_panel_label(ax, "a")

    ax = axes[0, 1]
    for facies_id in FACIES_IDS:
        sub = binned[binned["facies_id"] == facies_id].copy()
        ok = sub["n"].to_numpy(dtype=int) > 0
        if not ok.any():
            continue
        age = sub.loc[ok, "age_bin_mid_ka"].to_numpy(dtype=float)
        med = sub.loc[ok, "rey_median"].to_numpy(dtype=float)
        q25 = sub.loc[ok, "rey_q25"].to_numpy(dtype=float)
        q75 = sub.loc[ok, "rey_q75"].to_numpy(dtype=float)
        mean = sub.loc[ok, "rey_mean"].to_numpy(dtype=float)
        ax.fill_betweenx(age, q25, q75, color=FACIES_COLORS[facies_id], alpha=0.13, linewidth=0)
        ax.plot(med, age, color=FACIES_COLORS[facies_id], linewidth=2.2, label=f"{FACIES_LABELS[facies_id]} median")
        ax.plot(mean, age, color=FACIES_COLORS[facies_id], linewidth=1.15, linestyle="--", alpha=0.78)
    ax.set_xscale("log")
    ax.set_ylim(AGE_MAX_KA, AGE_MIN_KA)
    ax.set_xlabel("Observed REY (ppm, log scale)")
    ax.set_ylabel("Apparent age proxy (ka)")
    ax.set_title("Binned facies median and mean profiles")
    ax.legend(frameon=False, loc="upper left", handlelength=1.7)
    _style_axis(ax)
    _add_panel_label(ax, "b")

    ax = axes[1, 0]
    panel_specs = [
        (1, "frac_ge_1000", "Facies II observed REY >= 1000 ppm", FACIES_COLORS[1]),
        (2, "frac_ge_800", "Facies III observed REY >= 800 ppm", FACIES_COLORS[2]),
        (1, "frac_ge_1500", "Facies II observed REY >= 1500 ppm", "#8F1D1B"),
        (2, "frac_ge_1000", "Facies III observed REY >= 1000 ppm", "#134E7C"),
    ]
    for facies_id, col, label, color in panel_specs:
        sub = binned[(binned["facies_id"] == facies_id) & (binned["n"] > 0)]
        ax.plot(
            sub[col].to_numpy(dtype=float),
            sub["age_bin_mid_ka"].to_numpy(dtype=float),
            color=color,
            linewidth=2.0 if col in {"frac_ge_1000", "frac_ge_800"} else 1.25,
            linestyle="-" if col in {"frac_ge_1000", "frac_ge_800"} else "--",
            label=label,
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(AGE_MAX_KA, AGE_MIN_KA)
    ax.set_xlabel("Observed threshold support fraction")
    ax.set_ylabel("Apparent age proxy (ka)")
    ax.set_title("High-value support by apparent-age bin")
    ax.legend(
        frameon=True,
        framealpha=0.94,
        facecolor="white",
        edgecolor="#CCCCCC",
        loc="upper right",
        handlelength=1.8,
    )
    _style_axis(ax)
    _add_panel_label(ax, "c")

    ax = axes[1, 1]
    piv = binned.pivot(index="age_bin_mid_ka", columns="facies_id", values="n").fillna(0.0)
    for facies_id in FACIES_IDS:
        if facies_id not in piv.columns:
            piv[facies_id] = 0.0
    piv = piv.sort_index()
    y = piv.index.to_numpy(dtype=float)
    left = np.zeros_like(y)
    for facies_id in FACIES_IDS:
        vals = piv[facies_id].to_numpy(dtype=float)
        ax.barh(
            y,
            vals,
            height=135.0,
            left=left,
            color=FACIES_COLORS[facies_id],
            alpha=0.75,
            edgecolor="white",
            linewidth=0.25,
            label=FACIES_LABELS[facies_id],
        )
        left = left + vals
    ax.set_ylim(AGE_MAX_KA, AGE_MIN_KA)
    ax.set_xlabel("Observed sample count")
    ax.set_ylabel("Apparent age proxy (ka)")
    ax.set_title("Vertical sample support by facies")
    ax.legend(frameon=False, loc="lower right", ncol=1)
    _style_axis(ax, grid_axis="x")
    _add_panel_label(ax, "d")
    return _save_figure(fig, output_dir, "observed_vertical_facies_profiles")


def _dataset_var(ds: xr.Dataset, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in ds.data_vars:
            return name
    raise KeyError(f"None of these variables are present: {candidates}")


def _load_bathymetry(env_nc: Path, target: xr.Dataset, variable: str) -> np.ndarray:
    if not env_nc.exists():
        raise FileNotFoundError(f"Missing environment NetCDF for water-depth mask: {env_nc}")
    with open_dataset(env_nc) as env:
        if variable not in env.data_vars:
            raise KeyError(f"{env_nc} does not contain water-depth variable '{variable}'.")
        env_lat, env_lon = _coord_names(env)
        target_lat, target_lon = _coord_names(target)
        interp = env[variable].interp(
            {env_lat: target[target_lat], env_lon: target[target_lon]},
            kwargs={"fill_value": "extrapolate"},
        )
        return np.asarray(interp.to_numpy(), dtype=float)


def _deep_water_mask(bathy: np.ndarray, min_water_depth_m: float) -> np.ndarray:
    values = np.asarray(bathy, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(values.shape, dtype=bool)
    median = float(np.nanmedian(values[finite]))
    if median < 0.0:
        return finite & (values <= -float(min_water_depth_m))
    return finite & (np.abs(values) >= float(min_water_depth_m))


def _finite_values(values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if mask is not None:
        finite &= np.asarray(mask, dtype=bool)
    return arr[finite]


def _top_percent_cut(values: np.ndarray, mask: np.ndarray, top_percent: float) -> float:
    vals = _finite_values(values, mask)
    if vals.size == 0:
        return float("nan")
    return float(np.nanpercentile(vals, np.clip(100.0 - float(top_percent), 0.0, 100.0)))


def _profile_quantiles(values: np.ndarray) -> pd.DataFrame:
    arr = np.asarray(values, dtype=float)
    return pd.DataFrame(
        {
            "q25": np.nanpercentile(arr, 25, axis=1),
            "median": np.nanpercentile(arr, 50, axis=1),
            "q75": np.nanpercentile(arr, 75, axis=1),
            "mean": np.nanmean(arr, axis=1),
        }
    )


def _masked_profiles(da: xr.DataArray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(da.to_numpy(), dtype=float)
    if values.ndim != 3:
        raise ValueError(f"Expected a 3D array for {da.name}, got shape {values.shape}.")
    return values[:, np.asarray(mask, dtype=bool)]


def _peak_age_from_profiles(profiles: np.ndarray, ages: np.ndarray) -> np.ndarray:
    arr = np.asarray(profiles, dtype=float)
    safe = np.where(np.isfinite(arr), arr, -np.inf)
    idx = np.argmax(safe, axis=0)
    return np.asarray(ages, dtype=float)[idx]


def _sample_profile_columns(profiles: np.ndarray, max_lines: int, seed: int) -> np.ndarray:
    arr = np.asarray(profiles, dtype=float)
    n = arr.shape[1]
    if n <= max_lines:
        return arr
    rng = np.random.default_rng(seed)
    idx = rng.choice(np.arange(n), size=int(max_lines), replace=False)
    return arr[:, idx]


def _plot_profile_ribbon(
    ax: plt.Axes,
    ages: np.ndarray,
    profiles: np.ndarray,
    *,
    color: str,
    light_color: str,
    xlabel: str,
    title: str,
    label: str,
    max_lines: int,
    seed: int,
) -> pd.DataFrame:
    sampled = _sample_profile_columns(profiles, max_lines=max_lines, seed=seed)
    for idx in range(sampled.shape[1]):
        ax.plot(sampled[:, idx], ages, color=light_color, alpha=0.07, linewidth=0.55, zorder=1)
    stats = _profile_quantiles(profiles)
    ax.fill_betweenx(ages, stats["q25"], stats["q75"], color=color, alpha=0.18, linewidth=0, zorder=2)
    ax.plot(stats["median"], ages, color=color, linewidth=2.4, label=f"{label} median", zorder=3)
    ax.plot(stats["mean"], ages, color=color, linewidth=1.15, linestyle="--", alpha=0.78, label=f"{label} mean", zorder=3)
    ax.set_ylim(float(np.nanmax(ages)), float(np.nanmin(ages)))
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Apparent age layer (ka)")
    ax.set_title(title)
    ax.legend(frameon=False, loc="lower right", handlelength=1.7)
    _style_axis(ax)
    return stats


def _load_prediction_context(
    *,
    prediction_dir: Path,
    env_nc: Path,
    bathymetry_var: str,
    min_water_depth_m: float,
    top_percent: float,
) -> dict[str, Any]:
    pred3_path = prediction_dir / PREDICTION_3D_NC
    pred2_path = prediction_dir / PREDICTION_2D_NC
    if not pred3_path.exists():
        raise FileNotFoundError(f"Missing 3D prediction NetCDF: {pred3_path}")
    if not pred2_path.exists():
        raise FileNotFoundError(f"Missing 2D prediction NetCDF: {pred2_path}")

    ds3 = open_dataset(pred3_path)
    ds2 = open_dataset(pred2_path)
    try:
        vert = _vertical_dim(ds3)
        ages = np.asarray(ds3[vert].to_numpy(), dtype=float)
        dominant_name = _dataset_var(ds2, ("dominant_mechanism_class_id_0_2580ka",))
        p1000_2d_name = _dataset_var(ds2, ("max_prob_rey_ge_1000ppm_0_2580ka",))
        p800_3d_name = _dataset_var(ds3, ("prob_rey_ge_800ppm", "prob_rey_ge_800"))
        p1000_3d_name = _dataset_var(ds3, ("prob_rey_ge_1000ppm", "prob_rey_ge_1000"))
        rey_3d_name = _dataset_var(ds3, ("pred_rey_ppm", "predicted_rey_ppm", "pred_rey"))

        dominant = np.asarray(ds2[dominant_name].to_numpy(), dtype=np.int16)
        p1000_max = np.asarray(ds2[p1000_2d_name].to_numpy(), dtype=float)
        p800_max = np.asarray(ds3[p800_3d_name].max(dim=vert, skipna=True).to_numpy(), dtype=float)
        bathy = _load_bathymetry(env_nc, ds2, bathymetry_var)
        deep_mask = _deep_water_mask(bathy, min_water_depth_m)
        domain = deep_mask & np.isfinite(p1000_max)
        facies2_domain = domain & (dominant == 1) & np.isfinite(p1000_max)
        facies3_domain = domain & (dominant == 2) & np.isfinite(p800_max)
        facies2_cut = _top_percent_cut(p1000_max, facies2_domain, top_percent)
        facies3_cut = _top_percent_cut(p800_max, facies3_domain, top_percent)
        facies2_hotspot = facies2_domain & np.isfinite(facies2_cut) & (p1000_max >= facies2_cut)
        facies3_hotspot = facies3_domain & np.isfinite(facies3_cut) & (p800_max >= facies3_cut)

        if not np.any(facies2_hotspot):
            raise ValueError("Facies II hotspot mask is empty.")
        if not np.any(facies3_hotspot):
            raise ValueError("Facies III hotspot mask is empty.")

        p1000_f2 = _masked_profiles(ds3[p1000_3d_name], facies2_hotspot)
        p800_f3 = _masked_profiles(ds3[p800_3d_name], facies3_hotspot)
        rey_f2 = _masked_profiles(ds3[rey_3d_name], facies2_hotspot)
        rey_f3 = _masked_profiles(ds3[rey_3d_name], facies3_hotspot)

        return {
            "ages": ages,
            "p1000_facies2": p1000_f2,
            "p800_facies3": p800_f3,
            "rey_facies2": rey_f2,
            "rey_facies3": rey_f3,
            "facies2_peak_age": _peak_age_from_profiles(p1000_f2, ages),
            "facies3_peak_age": _peak_age_from_profiles(p800_f3, ages),
            "facies2_cutoff": facies2_cut,
            "facies3_cutoff": facies3_cut,
            "facies2_domain_cells": int(np.sum(facies2_domain)),
            "facies3_domain_cells": int(np.sum(facies3_domain)),
            "facies2_hotspot_cells": int(np.sum(facies2_hotspot)),
            "facies3_hotspot_cells": int(np.sum(facies3_hotspot)),
            "deep_water_cells": int(np.sum(deep_mask)),
            "min_water_depth_m": float(min_water_depth_m),
            "top_percent": float(top_percent),
            "prediction_3d_nc": str(pred3_path),
            "prediction_2d_nc": str(pred2_path),
            "p800_3d_var": p800_3d_name,
            "p1000_3d_var": p1000_3d_name,
            "rey_3d_var": rey_3d_name,
        }
    finally:
        ds3.close()
        ds2.close()


def _plot_predicted_hotspot_profiles(
    context: dict[str, Any],
    output_dir: Path,
    *,
    max_hotspot_lines: int,
    seed: int,
) -> tuple[dict[str, str], pd.DataFrame, pd.DataFrame]:
    ages = np.asarray(context["ages"], dtype=float)
    p1000_f2 = np.asarray(context["p1000_facies2"], dtype=float)
    p800_f3 = np.asarray(context["p800_facies3"], dtype=float)
    rey_f2 = np.asarray(context["rey_facies2"], dtype=float)
    rey_f3 = np.asarray(context["rey_facies3"], dtype=float)
    f2_peak = np.asarray(context["facies2_peak_age"], dtype=float)
    f3_peak = np.asarray(context["facies3_peak_age"], dtype=float)

    _set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 10.2), dpi=190, constrained_layout=True)
    stats_p1000_f2 = _plot_profile_ribbon(
        axes[0, 0],
        ages,
        p1000_f2,
        color=FACIES_COLORS[1],
        light_color=FACIES_LIGHT[1],
        xlabel="P(REY >= 1000 ppm)",
        title="Facies II hotspot P1000 profiles",
        label="Facies II",
        max_lines=max_hotspot_lines,
        seed=seed,
    )
    axes[0, 0].text(
        0.04,
        0.06,
        f"Top {context['top_percent']:g}% cutoff = {context['facies2_cutoff']:.3f}\n"
        f"n cells = {context['facies2_hotspot_cells']:,}",
        transform=axes[0, 0].transAxes,
        fontsize=9.6,
        bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="#BBBBBB", alpha=0.94),
    )
    _add_panel_label(axes[0, 0], "a")

    stats_p800_f3 = _plot_profile_ribbon(
        axes[0, 1],
        ages,
        p800_f3,
        color=FACIES_COLORS[2],
        light_color=FACIES_LIGHT[2],
        xlabel="P(REY >= 800 ppm)",
        title="Facies III hotspot P800 profiles",
        label="Facies III",
        max_lines=max_hotspot_lines,
        seed=seed + 1,
    )
    axes[0, 1].text(
        0.04,
        0.06,
        f"Top {context['top_percent']:g}% cutoff = {context['facies3_cutoff']:.3f}\n"
        f"n cells = {context['facies3_hotspot_cells']:,}",
        transform=axes[0, 1].transAxes,
        fontsize=9.6,
        bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="#BBBBBB", alpha=0.94),
    )
    _add_panel_label(axes[0, 1], "b")

    ax = axes[1, 0]
    stats_rey_f2 = _profile_quantiles(rey_f2)
    stats_rey_f3 = _profile_quantiles(rey_f3)
    for stats, facies_id, label in [
        (stats_rey_f2, 1, "Facies II hotspot"),
        (stats_rey_f3, 2, "Facies III hotspot"),
    ]:
        ax.fill_betweenx(
            ages,
            stats["q25"],
            stats["q75"],
            color=FACIES_COLORS[facies_id],
            alpha=0.15,
            linewidth=0,
        )
        ax.plot(stats["median"], ages, color=FACIES_COLORS[facies_id], linewidth=2.3, label=f"{label} median")
        ax.plot(stats["mean"], ages, color=FACIES_COLORS[facies_id], linewidth=1.15, linestyle="--", alpha=0.78)
    ax.set_xscale("log")
    ax.set_ylim(float(np.nanmax(ages)), float(np.nanmin(ages)))
    ax.set_xlabel("Predicted REY (ppm, log scale)")
    ax.set_ylabel("Apparent age layer (ka)")
    ax.set_title("Predicted REY profiles in hotspot domains")
    ax.legend(frameon=False, loc="lower right", handlelength=1.7)
    _style_axis(ax)
    _add_panel_label(ax, "c")

    ax = axes[1, 1]
    bins = np.r_[ages - np.nanmedian(np.diff(ages)) / 2.0, ages[-1] + np.nanmedian(np.diff(ages)) / 2.0]
    if len(bins) != len(ages) + 1 or not np.all(np.isfinite(bins)):
        bins = np.linspace(float(np.nanmin(ages)), float(np.nanmax(ages)), 16)
    ax.hist(
        f2_peak,
        bins=bins,
        density=True,
        color=FACIES_COLORS[1],
        alpha=0.50,
        label="Facies II max P1000 age",
    )
    ax.hist(
        f3_peak,
        bins=bins,
        density=True,
        color=FACIES_COLORS[2],
        alpha=0.44,
        label="Facies III max P800 age",
    )
    ax.axvline(float(np.nanmedian(f2_peak)), color=FACIES_COLORS[1], linewidth=2.0)
    ax.axvline(float(np.nanmedian(f3_peak)), color=FACIES_COLORS[2], linewidth=2.0)
    ax.set_xlabel("Peak apparent-age layer (ka)")
    ax.set_ylabel("Density")
    ax.set_title("Hotspot peak-age distribution")
    ax.legend(frameon=False, loc="upper left", handlelength=1.4)
    _style_axis(ax, grid_axis="y")
    _add_panel_label(ax, "d")

    profile_rows: list[pd.DataFrame] = []
    for stats, facies_label, variable in [
        (stats_p1000_f2, "Facies II", "prob_rey_ge_1000ppm"),
        (stats_p800_f3, "Facies III", "prob_rey_ge_800ppm"),
        (stats_rey_f2, "Facies II", "pred_rey_ppm"),
        (stats_rey_f3, "Facies III", "pred_rey_ppm"),
    ]:
        tmp = stats.copy()
        tmp.insert(0, "age_ka", ages)
        tmp.insert(1, "facies_label", facies_label)
        tmp.insert(2, "variable", variable)
        profile_rows.append(tmp)
    profile_summary = pd.concat(profile_rows, ignore_index=True)
    profile_summary.to_csv(output_dir / "predicted_hotspot_vertical_profile_summary.csv", index=False)

    peak_rows = [
        {
            "hotspot_domain": "Facies II top-5 P1000",
            "screening_probability": "prob_rey_ge_1000ppm",
            "n_cells": int(len(f2_peak)),
            "peak_age_median_ka": float(np.nanmedian(f2_peak)),
            "peak_age_q25_ka": float(np.nanpercentile(f2_peak, 25)),
            "peak_age_q75_ka": float(np.nanpercentile(f2_peak, 75)),
            "peak_at_oldest_layer_fraction": float(np.mean(f2_peak == np.nanmax(ages))),
        },
        {
            "hotspot_domain": "Facies III top-5 P800",
            "screening_probability": "prob_rey_ge_800ppm",
            "n_cells": int(len(f3_peak)),
            "peak_age_median_ka": float(np.nanmedian(f3_peak)),
            "peak_age_q25_ka": float(np.nanpercentile(f3_peak, 25)),
            "peak_age_q75_ka": float(np.nanpercentile(f3_peak, 75)),
            "peak_at_oldest_layer_fraction": float(np.mean(f3_peak == np.nanmax(ages))),
        },
    ]
    peak_summary = pd.DataFrame(peak_rows)
    peak_summary.to_csv(output_dir / "predicted_hotspot_peak_age_summary.csv", index=False)

    return (
        _save_figure(fig, output_dir, "predicted_hotspot_vertical_profiles"),
        profile_summary,
        peak_summary,
    )


def export_vertical_profile_figures(
    *,
    class_table: Path = DEFAULT_CLASS_TABLE,
    prediction_dir: Path = DEFAULT_PREDICTION_DIR,
    env_nc: Path = DEFAULT_ENV_NC,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    sed_rate_nc: Path | None = DEFAULT_SED_RATE_NC,
    label_col: str = "mech_class_id",
    sed_rate_var: str = "Average sedimentation rate",
    bathymetry_var: str = "Water depth",
    min_water_depth_m: float = 3500.0,
    top_percent: float = 5.0,
    age_bin_width_ka: float = 200.0,
    max_core_lines: int = 120,
    max_hotspot_lines: int = 420,
    seed: int = 20260616,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    observed = _load_class_table(class_table, label_col)
    observed, age_meta = _compute_age_proxy(
        observed,
        sed_rate_nc=sed_rate_nc,
        sed_rate_var=sed_rate_var,
    )
    observed = observed[
        observed["age_ka_proxy"].notna()
        & (observed["age_ka_proxy"] >= AGE_MIN_KA)
        & (observed["age_ka_proxy"] <= AGE_MAX_KA)
        & (observed["rey"] > 0.0)
    ].copy()
    if observed.empty:
        raise ValueError("No observed samples remain inside the 0-2580 ka apparent-age window.")
    observed.to_csv(output_dir / "observed_vertical_profile_points.csv", index=False, encoding="utf-8-sig")

    binned = _binned_facies_stats(observed, bin_width_ka=float(age_bin_width_ka), age_max_ka=AGE_MAX_KA)
    binned.to_csv(output_dir / "observed_vertical_facies_binned_stats.csv", index=False)
    observed_paths = _plot_observed_vertical_profiles(
        observed,
        binned,
        output_dir,
        max_core_lines=int(max_core_lines),
        seed=int(seed),
    )

    context = _load_prediction_context(
        prediction_dir=prediction_dir,
        env_nc=env_nc,
        bathymetry_var=bathymetry_var,
        min_water_depth_m=float(min_water_depth_m),
        top_percent=float(top_percent),
    )
    predicted_paths, profile_summary, peak_summary = _plot_predicted_hotspot_profiles(
        context,
        output_dir,
        max_hotspot_lines=int(max_hotspot_lines),
        seed=int(seed),
    )

    observed_summary = (
        observed.groupby(["facies_id", "facies_label", "facies_name"], dropna=False)
        .agg(
            n_samples=("rey", "size"),
            n_cores=("core_id", "nunique") if "core_id" in observed.columns else ("rey", "size"),
            rey_median=("rey", "median"),
            rey_p90=("rey", lambda s: float(np.nanpercentile(s, 90))),
            age_median_ka=("age_ka_proxy", "median"),
            ge800=("rey", lambda s: int((s >= 800.0).sum())),
            ge1000=("rey", lambda s: int((s >= 1000.0).sum())),
            ge1500=("rey", lambda s: int((s >= 1500.0).sum())),
        )
        .reset_index()
    )
    observed_summary.to_csv(output_dir / "observed_vertical_facies_summary.csv", index=False)

    summary = {
        "status": "completed",
        "class_table": str(class_table),
        "prediction_dir": str(prediction_dir),
        "env_nc": str(env_nc),
        "output_dir": str(output_dir),
        "label_col": label_col,
        "facies_labels": FACIES_LABELS,
        "facies_names": FACIES_NAMES,
        "observed_age_window_ka": [AGE_MIN_KA, AGE_MAX_KA],
        "observed_age_metadata": age_meta,
        "observed_n_samples": int(len(observed)),
        "observed_n_cores": int(observed["core_id"].nunique()) if "core_id" in observed.columns else None,
        "observed_summary": observed_summary.to_dict(orient="records"),
        "prediction_screening": {
            "min_water_depth_m": float(min_water_depth_m),
            "top_percent": float(top_percent),
            "facies2_probability": "max P(REY >= 1000 ppm)",
            "facies3_probability": "max P(REY >= 800 ppm)",
            "facies2_cutoff": float(context["facies2_cutoff"]),
            "facies3_cutoff": float(context["facies3_cutoff"]),
            "facies2_domain_cells": int(context["facies2_domain_cells"]),
            "facies3_domain_cells": int(context["facies3_domain_cells"]),
            "facies2_hotspot_cells": int(context["facies2_hotspot_cells"]),
            "facies3_hotspot_cells": int(context["facies3_hotspot_cells"]),
            "deep_water_cells": int(context["deep_water_cells"]),
            "peak_age_summary": peak_summary.to_dict(orient="records"),
        },
        "artifacts": {
            "observed_vertical_profiles": observed_paths,
            "predicted_hotspot_vertical_profiles": predicted_paths,
            "observed_points_csv": str(output_dir / "observed_vertical_profile_points.csv"),
            "observed_binned_stats_csv": str(output_dir / "observed_vertical_facies_binned_stats.csv"),
            "observed_summary_csv": str(output_dir / "observed_vertical_facies_summary.csv"),
            "predicted_profile_summary_csv": str(output_dir / "predicted_hotspot_vertical_profile_summary.csv"),
            "predicted_peak_age_summary_csv": str(output_dir / "predicted_hotspot_peak_age_summary.csv"),
        },
    }
    _write_json(output_dir / "vertical_profile_figure_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export observed and predicted vertical REY/facies profile figures.")
    parser.add_argument("--class-table", type=Path, default=DEFAULT_CLASS_TABLE)
    parser.add_argument("--prediction-dir", type=Path, default=DEFAULT_PREDICTION_DIR)
    parser.add_argument("--env-nc", type=Path, default=DEFAULT_ENV_NC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sed-rate-nc", type=Path, default=DEFAULT_SED_RATE_NC)
    parser.add_argument("--label-col", default="mech_class_id")
    parser.add_argument("--sed-rate-var", default="Average sedimentation rate")
    parser.add_argument("--bathymetry-var", default="Water depth")
    parser.add_argument("--min-water-depth-m", type=float, default=3500.0)
    parser.add_argument("--top-percent", type=float, default=5.0)
    parser.add_argument("--age-bin-width-ka", type=float, default=200.0)
    parser.add_argument("--max-core-lines", type=int, default=120)
    parser.add_argument("--max-hotspot-lines", type=int, default=420)
    parser.add_argument("--seed", type=int, default=20260616)
    args = parser.parse_args()

    summary = export_vertical_profile_figures(
        class_table=args.class_table,
        prediction_dir=args.prediction_dir,
        env_nc=args.env_nc,
        output_dir=args.output_dir,
        sed_rate_nc=args.sed_rate_nc,
        label_col=str(args.label_col),
        sed_rate_var=str(args.sed_rate_var),
        bathymetry_var=str(args.bathymetry_var),
        min_water_depth_m=float(args.min_water_depth_m),
        top_percent=float(args.top_percent),
        age_bin_width_ka=float(args.age_bin_width_ka),
        max_core_lines=int(args.max_core_lines),
        max_hotspot_lines=int(args.max_hotspot_lines),
        seed=int(args.seed),
    )
    print(json.dumps(summary, cls=NumpyJSONEncoder, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

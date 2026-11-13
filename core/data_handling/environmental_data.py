
import os
import numpy as np
import xarray as xr
from core.utils.netcdf_io import open_dataset
import pandas as pd
import json
from pathlib import Path
from global_land_mask import globe
from core.utils.logger_config import logger
from core.data.interpolation import interpolate_grid, iterative_imputer_interpolation


def _config_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "y", "t"}:
            return True
        if v in {"false", "0", "no", "n", "f"}:
            return False
    return None


def _detect_categorical_feature(da: xr.DataArray, *, feature_name: str, config: dict | None = None) -> bool:
    """
    Heuristic detection of categorical rasters.

    - Honors explicit config overrides: `categorical`/`is_categorical`/`data_type`.
    - Treats integer/boolean dtypes as categorical.
    - For float rasters, treats as categorical if values are (almost) integers
      and the number of unique values is small.
    """
    cfg = config or {}
    for key in ("categorical", "is_categorical"):
        ov = _config_bool(cfg.get(key))
        if ov is not None:
            return bool(ov)
    dt = cfg.get("data_type")
    if isinstance(dt, str):
        dt_norm = dt.strip().lower()
        if dt_norm in {"categorical", "category", "enum", "class", "label"}:
            return True
        if dt_norm in {"numeric", "continuous", "float"}:
            return False

    try:
        if da.dtype.kind in {"b", "i", "u"}:
            return True
    except Exception:
        pass

    try:
        data = np.asarray(da.data)
        if data.size == 0:
            return False
        flat = data.reshape(-1)
        if flat.size > 200_000:
            rng = np.random.default_rng(0)
            flat = flat[rng.choice(flat.size, size=200_000, replace=False)]
        flat = flat[np.isfinite(flat)]
        if flat.size == 0:
            return False
        frac_int = float(np.mean(np.isclose(flat, np.rint(flat), atol=1e-6)))
        if frac_int < 0.995:
            return False
        uniq = np.unique(np.rint(flat))
        return bool(uniq.size <= 64)
    except Exception as e:
        logger.debug(f"Categorical detection failed for '{feature_name}': {e}")
        return False


def _one_hot_encode_categorical_grid(
    codes_da: xr.DataArray,
    *,
    feature_name: str,
    source_attrs: dict,
    category_labels: dict[int, str] | None = None,
    max_categories: int = 64,
) -> dict[str, xr.DataArray] | None:
    try:
        vals = np.asarray(codes_da.data)
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            return None
        cats = np.unique(np.rint(finite).astype(int))
        if cats.size == 0 or cats.size > max_categories:
            return None

        def _sanitize_var_token(s: str) -> str:
            s2 = "".join((c if (c.isalnum() or c in {"_", "-"}) else "_") for c in str(s))
            s2 = s2.strip("_").replace("-", "_").lower()
            while "__" in s2:
                s2 = s2.replace("__", "_")
            return s2[:80] if len(s2) > 80 else s2

        out: dict[str, xr.DataArray] = {}
        for c in cats.tolist():
            label = None
            if category_labels and int(c) in category_labels:
                label = str(category_labels[int(c)])
            if label:
                token = _sanitize_var_token(label)
                name = f"{feature_name}__cat_{int(c)}__{token}" if token else f"{feature_name}__cat_{int(c)}"
            else:
                name = f"{feature_name}__cat_{int(c)}"
            one = xr.where(
                np.isfinite(codes_da),
                (np.rint(codes_da) == float(c)).astype(np.float32),
                np.nan,
            )
            one.name = name
            one.attrs.update(source_attrs or {})
            one.attrs.update(
                {
                    "is_categorical_onehot": 1,
                    "categorical_source_feature": str(feature_name),
                    "category_code": int(c),
                    "category_count": int(cats.size),
                    "category_label": (str(label) if label else ""),
                }
            )
            out[name] = one
        return out
    except Exception as e:
        logger.warning(f"Failed to one-hot encode categorical grid '{feature_name}': {e}")
        return None


def _parse_category_mapping_string(s: str) -> dict[int, str]:

    import re

    mapping: dict[int, str] = {}
    if not s:
        return mapping
    parts = []
    for chunk in str(s).replace("\n", ";").replace(",", ";").split(";"):
        chunk = chunk.strip()
        if chunk:
            parts.append(chunk)
    for p in parts:
        m = re.match(r"^\s*(-?\d+)\s*:\s*(.*?)\s*$", p)
        if not m:
            continue
        try:
            code = int(m.group(1))
        except Exception:
            continue
        label = m.group(2).strip()
        if label:
            mapping[code] = label
    return mapping


def _get_category_labels(
    *,
    ds_src: xr.Dataset,
    da_src: xr.DataArray,
    feature_name: str,
    config: dict | None = None,
) -> dict[int, str] | None:
    cfg = config or {}

    for key in ("category_labels", "category_map", "category_mapping"):
        v = cfg.get(key)
        if isinstance(v, dict) and v:
            out: dict[int, str] = {}
            for k2, v2 in v.items():
                try:
                    out[int(k2)] = str(v2)
                except Exception:
                    continue
            return out or None
        if isinstance(v, str) and v.strip():
            parsed = _parse_category_mapping_string(v)
            if parsed:
                return parsed

    # 2) CF-style flag_values/flag_meanings (if present)
    try:
        flag_values = da_src.attrs.get("flag_values")
        flag_meanings = da_src.attrs.get("flag_meanings")
        if flag_values is not None and flag_meanings is not None:
            fv = np.asarray(flag_values).reshape(-1).tolist()
            if isinstance(flag_meanings, str):
                fm = [x for x in flag_meanings.split() if x]
            else:
                fm = list(np.asarray(flag_meanings).reshape(-1).tolist())
            if len(fv) == len(fm) and len(fv) > 0:
                out = {}
                for a, b in zip(fv, fm):
                    try:
                        out[int(a)] = str(b)
                    except Exception:
                        continue
                return out or None
    except Exception:
        pass


    try:
        fn = str(feature_name or "").strip().lower()
        candidate_keys = []
        for k in (ds_src.attrs or {}).keys():
            ks = str(k).strip().lower()
            if fn and fn == ks:
                candidate_keys.append(k)
            elif fn and fn in ks:
                candidate_keys.append(k)

        if not candidate_keys:
            for k in (ds_src.attrs or {}).keys():
                ks = str(k).strip().lower()
                if "legend" in ks or "meaning" in ks or "category" in ks:
                    candidate_keys.append(k)
        for k in candidate_keys:
            v = ds_src.attrs.get(k)
            if isinstance(v, str) and ":" in v:
                parsed = _parse_category_mapping_string(v)
                if parsed:
                    return parsed
    except Exception:
        pass


    try:
        fn = str(feature_name or "").strip().lower()
        if fn in {"seafloor lithology", "seabed lithology", "seabed_lithology"}:


            return {
                1: "Gravel and coarser",
                2: "Sand",
                3: "Silt",
                4: "Clay",
                5: "Calcareous ooze",
                6: "Radiolarian ooze",
                7: "Diatom ooze",
                8: "Sponge spicules",
                9: "Mixed calcareous/siliceous ooze",
                10: "Shells and coral fragments",
                11: "Ash and volcanic sand/gravel",
                12: "Siliceous mud",
                13: "Fine-grained calcareous sediment",
            }
    except Exception:
        pass

    return None


def _as_float32_da(da: xr.DataArray, name: str, attrs: dict | None = None) -> xr.DataArray:
    out = da.astype(np.float32)
    out.name = name
    out.attrs.update(attrs or {})
    return out


def _finite_ratio(numerator: xr.DataArray, denominator: xr.DataArray, eps: float = 1e-6) -> xr.DataArray:
    den = denominator.astype(float)
    return xr.where(np.isfinite(den) & (np.abs(den) > eps), numerator.astype(float) / den, np.nan)


def _add_rolling_feature(
    ds: xr.Dataset,
    *,
    source_name: str,
    window: int,
    reducer: str,
    output_name: str,
) -> xr.DataArray | None:
    if source_name not in ds:
        return None
    if "__cat_" in str(source_name) or bool(ds[source_name].attrs.get("is_categorical_onehot", False)):
        return None
    try:
        da = ds[source_name].astype(float)
        min_periods = max(1, int(window * window * 0.35))
        rolled = da.rolling(latitude=int(window), longitude=int(window), center=True, min_periods=min_periods)
        if reducer == "mean":
            out = rolled.mean()
        elif reducer == "std":
            out = rolled.std()
        elif reducer == "range":
            out = rolled.max() - rolled.min()
        else:
            return None
        out = _as_float32_da(
            out,
            output_name,
            {
                "derived_feature": 1,
                "derived_feature_kind": f"rolling_{reducer}",
                "source_feature": str(source_name),
                "window_cells": int(window),
            },
        )
        return out
    except Exception as e:
        logger.warning(f"Failed to derive rolling feature '{output_name}' from '{source_name}': {e}")
        return None


def _distance_to_mask_km(mask_da: xr.DataArray, *, resolution: float, name: str, source_feature: str) -> xr.DataArray | None:
    try:
        from scipy import ndimage
    except Exception as e:
        logger.warning(f"scipy is unavailable; skipping distance feature '{name}': {e}")
        return None

    try:
        mask = np.asarray(mask_da.fillna(0).values > 0.5, dtype=bool)
        if mask.size == 0 or not bool(mask.any()):
            logger.info(f"Skipping distance feature '{name}' because target mask is empty.")
            return None

        lat_vals = np.asarray(mask_da["latitude"].values, dtype=float)
        mean_lat = float(np.nanmean(lat_vals)) if lat_vals.size else 0.0
        lat_km = 111.32 * float(resolution)
        lon_km = 111.32 * max(0.05, float(np.cos(np.deg2rad(mean_lat)))) * float(resolution)
        dist = ndimage.distance_transform_edt(~mask, sampling=(lat_km, lon_km)).astype(np.float32)
        out = xr.DataArray(dist, coords=mask_da.coords, dims=mask_da.dims, name=name)
        out.attrs.update(
            {
                "derived_feature": 1,
                "derived_feature_kind": "distance_to_geomorphic_category_km",
                "source_feature": str(source_feature),
                "units": "km",
                "distance_approximation": "constant-degree cell size at mean latitude",
            }
        )
        return out
    except Exception as e:
        logger.warning(f"Failed to derive distance feature '{name}' from '{source_feature}': {e}")
        return None


def add_derived_environment_features(
    ds: xr.Dataset,
    *,
    resolution: float,
    profile: str = "exploration_v1",
    config: dict | None = None,
) -> tuple[xr.Dataset, dict]:
    """
    Add exploration-oriented derived grids after base variables are aligned.

    These features are deterministic functions of prediction-time environmental
    rasters, so they can be stored in env_gridded.nc and consumed by existing
    dynamic feature discovery without changing training/prediction contracts.
    """
    cfg = dict(config or {})
    profile = str(profile or cfg.get("profile") or "exploration_v1")
    if profile in {"none", "off", "disabled"}:
        return ds, {"enabled": False, "profile": profile, "created_features": []}

    created: list[str] = []
    skipped: list[str] = []

    def add_feature(name: str, da: xr.DataArray | None) -> None:
        if da is None:
            skipped.append(name)
            return
        if name in ds.data_vars:
            skipped.append(name)
            return
        try:
            ds[name] = da
            created.append(name)
        except Exception as e:
            logger.warning(f"Failed to attach derived feature '{name}': {e}")
            skipped.append(name)

    logger.info(f"Adding derived environmental features with profile='{profile}'.")

    rolling_plan = [
        ("Water depth", 9, "mean", "Water depth__mean_w9"),
        ("Water depth", 9, "std", "Water depth__std_w9"),
        ("Water depth", 9, "range", "Water depth__roughness_w9"),
        ("Sediment thickness", 9, "mean", "Sediment thickness__mean_w9"),
        ("Sediment thickness", 9, "std", "Sediment thickness__std_w9"),
        ("Sediment thickness", 9, "range", "Sediment thickness__roughness_w9"),
        ("Gravity anomaly", 9, "std", "Gravity anomaly__std_w9"),
        ("Vertical gravity gradient", 9, "std", "Vertical gravity gradient__std_w9"),
        ("Magnetic anomaly", 9, "std", "Magnetic anomaly__std_w9"),
        ("Total organic carbon", 9, "mean", "Total organic carbon__mean_w9"),
        ("Average sedimentation rate", 9, "mean", "Average sedimentation rate__mean_w9"),
        ("Bottom water dissolved oxygen", 9, "mean", "Bottom water dissolved oxygen__mean_w9"),
    ]
    for source_name, window, reducer, output_name in rolling_plan:
        add_feature(
            output_name,
            _add_rolling_feature(
                ds,
                source_name=source_name,
                window=window,
                reducer=reducer,
                output_name=output_name,
            ),
        )

    composite_plan: list[tuple[str, xr.DataArray | None, dict]] = []
    try:
        if "Bottom phosphate" in ds and "Surface phosphate" in ds:
            composite_plan.append(
                (
                    "Bottom minus surface phosphate",
                    ds["Bottom phosphate"].astype(float) - ds["Surface phosphate"].astype(float),
                    {"derived_feature_kind": "difference", "source_features": "Bottom phosphate|Surface phosphate"},
                )
            )
    except Exception as e:
        logger.warning(f"Failed to prepare phosphate difference feature: {e}")
    try:
        if "Bottom water dissolved oxygen" in ds and "Bottom phosphate" in ds:
            composite_plan.append(
                (
                    "Bottom oxygen to phosphate ratio",
                    _finite_ratio(ds["Bottom water dissolved oxygen"], ds["Bottom phosphate"]),
                    {"derived_feature_kind": "ratio", "source_features": "Bottom water dissolved oxygen|Bottom phosphate"},
                )
            )
    except Exception as e:
        logger.warning(f"Failed to prepare oxygen/phosphate feature: {e}")
    try:
        if "Summer primary productivity" in ds and "Total organic carbon" in ds:
            composite_plan.append(
                (
                    "Productivity TOC index",
                    ds["Summer primary productivity"].astype(float) * ds["Total organic carbon"].astype(float),
                    {"derived_feature_kind": "product", "source_features": "Summer primary productivity|Total organic carbon"},
                )
            )
    except Exception as e:
        logger.warning(f"Failed to prepare productivity*TOC feature: {e}")
    try:
        if "Total organic carbon" in ds and "Average sedimentation rate" in ds:
            ratio = _finite_ratio(ds["Total organic carbon"], xr.where(ds["Average sedimentation rate"] > 0, ds["Average sedimentation rate"], np.nan))
            composite_plan.append(
                (
                    "log1p TOC per sedimentation rate",
                    np.log1p(xr.where(ratio > 0, ratio, np.nan)),
                    {"derived_feature_kind": "log_ratio", "source_features": "Total organic carbon|Average sedimentation rate"},
                )
            )
    except Exception as e:
        logger.warning(f"Failed to prepare TOC/sedimentation feature: {e}")
    try:
        if "Average sedimentation rate" in ds and "Water depth" in ds:
            composite_plan.append(
                (
                    "Sedimentation water depth index",
                    ds["Average sedimentation rate"].astype(float) * np.abs(ds["Water depth"].astype(float)),
                    {"derived_feature_kind": "product", "source_features": "Average sedimentation rate|Water depth_abs"},
                )
            )
    except Exception as e:
        logger.warning(f"Failed to prepare sedimentation*depth feature: {e}")

    for name, da, attrs in composite_plan:
        if da is None:
            add_feature(name, None)
            continue
        base_attrs = {"derived_feature": 1, **attrs}
        add_feature(name, _as_float32_da(da, name, base_attrs))

    distance_sources = {
        "distance_to_spreading_ridge_km": "Geomorphic Features__cat_1__spreading_ridge",
        "distance_to_guyot_km": "Geomorphic Features__cat_12__guyot",
        "distance_to_seamount_km": "Geomorphic Features__cat_14__seamount",
        "distance_to_abyss_km": "Geomorphic Features__cat_15__abyss",
        "distance_to_basin_km": "Geomorphic Features__cat_18__basin",
        "distance_to_trench_km": "Geomorphic Features__cat_21__trench",
    }
    for name, source in distance_sources.items():
        if source not in ds:
            skipped.append(name)
            continue
        add_feature(name, _distance_to_mask_km(ds[source], resolution=resolution, name=name, source_feature=source))

    try:
        if "del3He" in ds and "distance_to_spreading_ridge_km" in ds:
            hydro = _finite_ratio(ds["del3He"], ds["distance_to_spreading_ridge_km"] + 25.0)
            add_feature(
                "del3He ridge proximity index",
                _as_float32_da(
                    hydro,
                    "del3He ridge proximity index",
                    {
                        "derived_feature": 1,
                        "derived_feature_kind": "proximity_weighted_ratio",
                        "source_features": "del3He|distance_to_spreading_ridge_km",
                    },
                ),
            )
    except Exception as e:
        logger.warning(f"Failed to prepare del3He ridge proximity feature: {e}")

    meta = {
        "enabled": True,
        "profile": profile,
        "created_features": created,
        "skipped_features": skipped,
        "n_created": int(len(created)),
    }
    logger.info(f"Derived environmental features added: {len(created)} new variables.")
    return ds, meta


def process_environmental_data(variable_config_list, output_path, resolution,
                               interp_method='nearest',
                               geospatial_bounds=None,
                               use_iterative_imputer=False,
                               enable_derived_features=True,
                               derived_feature_profile="exploration_v1",
                               derived_feature_config=None):
    """
    Loads environmental data based on a configuration list, interpolates each
    variable to a standard grid using xarray.interp, optionally crops to
    geospatial bounds, and saves the result.
    (Optimized Version)

    Args:
        variable_config_list (list): A list of dictionaries, where each dict defines
            a variable to process. Required keys:
            {'file_path': str, 'data_var': str, 'lat_var': str, 'lon_var': str, 'feature_name': str}
        output_path (str): Path to save the processed NetCDF file.
        resolution (float): The desired target grid resolution in degrees.
        interp_method (str, optional): Interpolation method ('nearest' or 'linear') for single variable interpolation. Defaults to 'nearest'.
        geospatial_bounds (dict, optional): Dictionary with 'min_lat', 'max_lat',
            'min_lon', 'max_lon' to crop the data. Defaults to None (no cropping).
        use_iterative_imputer (bool, optional): If True, use multivariate iterative imputer for NaN filling. Defaults to False.

    Returns:
        bool: True if processing was successful, False otherwise.
    """

    logger.debug("Starting environmental data processing...")
    logger.debug(f"Output file path: {output_path}")
    logger.debug(f"Target resolution: {resolution} degrees")
    logger.debug(f"Interpolation method: {interp_method}")
    logger.debug(f"Number of variables to process: {len(variable_config_list)}")
    logger.debug(f"Derived environmental features enabled: {bool(enable_derived_features)}")

    if not variable_config_list:
        logger.error("Error: Variable configuration list is empty.")
        return False


    try:

        epsilon = resolution / 100


        x_target = np.arange(0, 360, resolution)
        y_target = np.arange(-90, 90 + epsilon, resolution)
        logger.debug(f"Target grid defined: Lon points={len(x_target)} (0 to <360), Lat points={len(y_target)} (-90 to 90)")

        target_lat_da = xr.DataArray(y_target, dims='latitude', coords={'latitude': y_target})
        target_lon_da = xr.DataArray(x_target, dims='longitude', coords={'longitude': x_target})
    except Exception as e:
        logger.error(f"Error defining target grid: {e}")
        return False


    processed_arrays = {}
    processed_feature_names = set()

    for config in variable_config_list:
        file_path = config.get('file_path')
        data_var = config.get('data_var')
        lat_var = config.get('lat_var')
        lon_var = config.get('lon_var')
        feature_name = config.get('feature_name')


        if not all([file_path, data_var, lat_var, lon_var, feature_name]):
            logger.warning(f"Warning: Skipping invalid or incomplete config entry: {config}")
            continue
        if feature_name in processed_feature_names:
            logger.warning(f"Warning: Feature name '{feature_name}' is duplicated in config. Skipping subsequent entry.")
            continue
        if not os.path.isfile(file_path):
            logger.error(f"Error: Input file not found for feature '{feature_name}': {file_path}. Skipping.")
            continue

        logger.debug(f"Processing feature: '{feature_name}' from {os.path.basename(file_path)} (Var: {data_var}, Lat: {lat_var}, Lon: {lon_var})")

        try:

            with open_dataset(file_path) as ds:
                logger.debug(f"  Successfully opened {os.path.basename(file_path)}.")


                try:
                    if data_var not in ds.data_vars:
                        raise KeyError(f"Data variable '{data_var}' not found.")

                    if lat_var not in ds.coords and lat_var not in ds.dims and lat_var not in ds.data_vars:
                        raise KeyError(f"Latitude variable '{lat_var}' not found.")
                    if lon_var not in ds.coords and lon_var not in ds.dims and lon_var not in ds.data_vars:
                        raise KeyError(f"Longitude variable '{lon_var}' not found.")
                    logger.debug(f"  All required variables ({data_var}, {lat_var}, {lon_var}) found.")
                except KeyError as ke:
                    logger.error(f"  Error validating variables in {os.path.basename(file_path)}: {ke}. Skipping.")
                    continue


                try:

                    if lat_var in ds.dims and lat_var not in ds.coords:
                        ds = ds.set_coords(lat_var)
                    if lon_var in ds.dims and lon_var not in ds.coords:
                        ds = ds.set_coords(lon_var)
                    da_src = ds[data_var]
                except KeyError:
                    logger.error(f"  Error selecting data variable '{data_var}' after validation. Skipping.")
                    continue
                except Exception as e:
                    logger.error(f"  Error ensuring lat/lon are coordinates for '{data_var}': {e}. Skipping.")
                    continue


                try:
                    if lon_var in ds.coords:
                        lon_coord_da = ds[lon_var]
                        min_lon_val = lon_coord_da.min().item()
                        max_lon_val = lon_coord_da.max().item()
                        needs_conversion = False


                        if pd.notna(min_lon_val) and min_lon_val < 0:
                            logger.debug(f"  Source longitude for '{feature_name}' (min: {min_lon_val:.2f}) seems to be in [-180, 180] range. Converting to [0, 360].")

                            ds = ds.assign_coords(**{lon_var: xr.where(ds[lon_var] < 0, ds[lon_var] + 360, ds[lon_var])})
                            needs_conversion = True
                        # Check if data is 0-360 but might contain values >= 360 (e.g. 360 instead of 0)
                        elif pd.notna(max_lon_val) and max_lon_val >= 360:
                            logger.debug(f"  Source longitude for '{feature_name}' (max: {max_lon_val:.2f}) may exceed or equal 360. Normalizing to [0, 360).")

                            ds = ds.assign_coords(**{lon_var: ds[lon_var] % 360})
                            needs_conversion = True

                        if needs_conversion:
                            ds = ds.sortby(lon_var)
                            da_src = ds[data_var]
                            logger.debug(f"  Longitude for '{feature_name}' processed for [0, 360) range. New range: {ds[lon_var].min().item():.2f} to {ds[lon_var].max().item():.2f}")
                        else:
                            logger.debug(f"  Longitude for '{feature_name}' seems to be already in [0, 360) range or no conversion needed. Range: {min_lon_val:.2f} to {max_lon_val:.2f}")

                            if not ds[lon_var].to_index().is_monotonic_increasing:
                                logger.debug(f"  Sorting longitude for '{feature_name}' as it was not monotonic.")
                                ds = ds.sortby(lon_var)
                                da_src = ds[data_var]


                    else:
                        logger.warning(f"  Warning: Longitude variable '{lon_var}' not found in coordinates for range check/conversion. Assuming it matches target [0, 360) or interpolation will handle.")

                except Exception as lon_fix_e:
                    logger.warning(f"  Warning: Error during longitude processing for '{feature_name}': {lon_fix_e}. Proceeding with original/current longitudes.")


                rename_map = {}
                final_lat_var = lat_var
                final_lon_var = lon_var

                if lat_var != 'latitude' and lat_var in ds.coords:
                    rename_map[lat_var] = 'latitude'
                    final_lat_var = 'latitude'
                if lon_var != 'longitude' and lon_var in ds.coords:
                    rename_map[lon_var] = 'longitude'
                    final_lon_var = 'longitude'

                if rename_map:
                    logger.debug(f"  Renaming source coordinates for interpolation: {rename_map}")
                    try:

                        ds_renamed = ds.rename(rename_map)

                        da_src = ds_renamed[data_var]
                        logger.debug(f"  Successfully renamed coordinates. Using '{final_lat_var}' and '{final_lon_var}'.")
                    except Exception as rename_e:
                        logger.error(f"  Error: Failed to rename coordinates for interpolation: {rename_e}. Skipping '{feature_name}'.")
                        continue


                try:
                    if final_lat_var in da_src.coords and final_lon_var in da_src.coords:
                        lat_res_src = np.abs(np.diff(da_src[final_lat_var])).mean()
                        lon_res_src = np.abs(np.diff(da_src[final_lon_var])).mean()
                        if pd.notna(lat_res_src) and pd.notna(lon_res_src):
                             res_diff_ratio = max(lat_res_src, lon_res_src) / resolution
                             if res_diff_ratio > 5 or res_diff_ratio < 0.2:
                                logger.warning(f"  Warning: Source resolution (approx. {lat_res_src:.3f}x{lon_res_src:.3f} deg) differs significantly from target ({resolution:.3f} deg) for '{feature_name}'. Interpolation quality may vary.")
                        else:
                            logger.debug(f"  Could not determine source resolution for '{feature_name}' (non-uniform grid?).")
                    else:
                        logger.debug(f"  Could not determine source resolution for '{feature_name}' (coords '{final_lat_var}' or '{final_lon_var}' missing).")
                except Exception as res_e:
                    logger.warning(f"  Warning: Could not calculate source resolution for '{feature_name}': {res_e}")


                try:
                    coords_to_check = {}
                    if final_lat_var in da_src.coords:
                        coords_to_check[final_lat_var] = 'latitude'
                    if final_lon_var in da_src.coords:
                        coords_to_check[final_lon_var] = 'longitude'

                    da_src_processed = da_src

                    for coord_name, standard_name_log in coords_to_check.items():

                        if coord_name not in da_src_processed.coords or not hasattr(da_src_processed[coord_name].to_index(), 'has_duplicates'):
                            logger.debug(f"  Info: Coordinate '{standard_name_log}' (original: {coord_name}) not found or not indexable in '{feature_name}'. Skipping duplicate check for this coordinate.")
                            continue

                        if da_src_processed[coord_name].to_index().has_duplicates:
                            logger.warning(f"  Warning: Duplicate values found in '{standard_name_log}' coordinate for '{feature_name}'. Attempting to handle by taking the first occurrence.")
                            original_da_attrs = da_src_processed.attrs.copy()
                            original_da_name = da_src_processed.name

                            original_latitude_coords = None

                            if final_lat_var in da_src_processed.coords:
                                try:

                                    logger.debug(f"    PRE-BACKUP CHECK for '{final_lat_var}' in '{feature_name}':")
                                    logger.debug(f"      Type(da_src_processed[final_lat_var]): {type(da_src_processed[final_lat_var])}")
                                    logger.debug(f"      da_src_processed[final_lat_var].shape: {da_src_processed[final_lat_var].shape}")
                                    logger.debug(f"      da_src_processed[final_lat_var].dims: {da_src_processed[final_lat_var].dims}")
                                    logger.debug(f"      da_src_processed[final_lat_var].coords: {list(da_src_processed[final_lat_var].coords.keys()) if hasattr(da_src_processed[final_lat_var], 'coords') else 'N/A'}")
                                    if hasattr(da_src_processed[final_lat_var], 'values') and da_src_processed[final_lat_var].size > 0 :
                                        logger.debug(f"      First 5 values of da_src_processed['{final_lat_var}']: {da_src_processed[final_lat_var].data[:5]}")
                                    else:
                                        logger.debug(f"      da_src_processed['{final_lat_var}'] has no values or is empty.")

                                    original_latitude_coords = da_src_processed[final_lat_var].copy(deep=True)


                                    logger.debug(f"    POST-BACKUP CHECK for 'original_latitude_coords' from '{feature_name}':")
                                    logger.debug(f"      Type(original_latitude_coords): {type(original_latitude_coords)}")
                                    logger.debug(f"      original_latitude_coords.shape: {original_latitude_coords.shape}")
                                    logger.debug(f"      original_latitude_coords.dims: {original_latitude_coords.dims}")
                                    logger.debug(f"      original_latitude_coords.coords: {list(original_latitude_coords.coords.keys()) if hasattr(original_latitude_coords, 'coords') else 'N/A'}")
                                    if hasattr(original_latitude_coords, 'values') and original_latitude_coords.size > 0:
                                        logger.debug(f"      First 5 values of original_latitude_coords: {original_latitude_coords.data[:5]}")
                                    else:
                                        logger.debug(f"      original_latitude_coords has no values or is empty.")
                                except Exception as e_backup:
                                    logger.error(f"  ERROR during backup of '{final_lat_var}' for '{feature_name}': {type(e_backup).__name__}: {str(e_backup)}. Full repr: {repr(e_backup)}. Latitude might not be restored correctly.")
                                    original_latitude_coords = None


                            logger.debug(f"  Performing groupby on '{coord_name}' for '{feature_name}' using .first(skipna=False).")
                            da_src_processed = da_src_processed.groupby(coord_name).first(skipna=False)


                            if original_da_name and da_src_processed.name != original_da_name:
                                da_src_processed.name = original_da_name
                                logger.debug(f"  Restored original name '{original_da_name}' to DataArray for '{feature_name}' after groupby.")
                            else:
                                logger.debug(f"  DataArray name for '{feature_name}' after groupby: '{da_src_processed.name}' (original: '{original_da_name}').")


                            logger.debug(f"  After groupby on '{coord_name}' for '{feature_name}': "
                                         f"da_src_processed.coords: {list(da_src_processed.coords.keys())}, "
                                         f"da_src_processed.dims: {da_src_processed.dims}")


                            if original_latitude_coords is not None and \
                               final_lat_var not in da_src_processed.coords and \
                               final_lat_var in da_src_processed.dims:
                                logger.debug(f"  Attempting to restore '{final_lat_var}' coordinates for '{feature_name}' using backed up data.")
                                try:


                                    logger.debug(f"    Current da_src_processed dims: {da_src_processed.dims}")
                                    logger.debug(f"    Backed up '{final_lat_var}' dims: {original_latitude_coords.dims}, shape: {original_latitude_coords.shape}")


                                    if final_lat_var in da_src_processed.dims:
                                        if original_latitude_coords.ndim == 1 and da_src_processed.sizes[final_lat_var] == original_latitude_coords.sizes[original_latitude_coords.dims[0]]:


                                            coord_to_assign = original_latitude_coords.rename({original_latitude_coords.dims[0]: final_lat_var})
                                            da_src_processed = da_src_processed.assign_coords({final_lat_var: coord_to_assign})
                                            logger.debug(f"    Successfully restored '{final_lat_var}' as coordinate using 1D backup aligned to '{final_lat_var}' dim.")
                                        elif original_latitude_coords.ndim > 1 and final_lat_var in original_latitude_coords.dims and coord_name in original_latitude_coords.dims:


                                            logger.warning(f"    Attempting to assign multi-dimensional backed-up '{final_lat_var}'. This might not be correctly aligned if its structure was complex. Backup dims: {original_latitude_coords.dims}")


                                            if original_latitude_coords.dims == (final_lat_var,) and original_latitude_coords.sizes[final_lat_var] == da_src_processed.sizes[final_lat_var]:
                                                da_src_processed = da_src_processed.assign_coords({final_lat_var: original_latitude_coords})
                                                logger.debug(f"    Successfully restored '{final_lat_var}' (assumed 1D and aligned).")
                                            else:


                                                aligned_backup_lat = None
                                                if len(original_latitude_coords.dims) == 1 and final_lat_var in da_src_processed.dims:

                                                    dim_name_in_backup = original_latitude_coords.dims[0]
                                                    if original_latitude_coords.sizes[dim_name_in_backup] == da_src_processed.sizes[final_lat_var]:
                                                        aligned_backup_lat = original_latitude_coords.rename({dim_name_in_backup: final_lat_var})
                                                        logger.debug(f"    Aligning 1D backup by renaming its dimension '{dim_name_in_backup}' to '{final_lat_var}'.")
                                                    else:
                                                        logger.warning(f"    Size mismatch for 1D backup of '{final_lat_var}'. Backup size {original_latitude_coords.sizes[dim_name_in_backup]} for dim '{dim_name_in_backup}', target dim '{final_lat_var}' size {da_src_processed.sizes[final_lat_var]}. Cannot directly assign.")

                                                if aligned_backup_lat is not None:
                                                    da_src_processed = da_src_processed.assign_coords({final_lat_var: aligned_backup_lat})
                                                    logger.debug(f"    Successfully restored '{final_lat_var}' as coordinate via aligned 1D backup.")
                                                else:
                                                    logger.warning(f"    Could not directly assign backed-up '{final_lat_var}'. Its dimensions/size might not match '{final_lat_var}' dim in processed data. Backup dims: {original_latitude_coords.dims}, shape: {original_latitude_coords.shape}. Target dim '{final_lat_var}' size: {da_src_processed.sizes.get(final_lat_var)}.")

                                        else:
                                            logger.warning(f"    Cannot restore '{final_lat_var}': it is not a dimension in the processed data, or backup structure mismatch. Processed dims: {da_src_processed.dims}")


                                    if final_lat_var in da_src_processed.coords:
                                        logger.debug(f"    Successfully restored '{final_lat_var}' as coordinate.")
                                        if da_src_processed[final_lat_var].size > 0:
                                             logger.debug(f"      Restored '{final_lat_var}' sample (first 5): {da_src_processed[final_lat_var].data[:5]}")
                                    else:
                                        logger.warning(f"    Failed to restore '{final_lat_var}' as coordinate using assign_coords, though it was attempted.")
                                except Exception as e_assign:
                                    logger.warning(f"    Exception while trying to restore '{final_lat_var}' using assign_coords: {e_assign}")
                            elif original_latitude_coords is None and final_lat_var not in da_src_processed.coords:
                                logger.warning(f"  Cannot restore '{final_lat_var}' coordinate for '{feature_name}' as no backup was made (e.g., it wasn't a coord initially or an error occurred during backup).")
                            elif final_lat_var in da_src_processed.coords:
                                 logger.debug(f"  '{final_lat_var}' coordinate is already present for '{feature_name}' after groupby. No restoration needed.")


                            logger.debug(f"  Final check of coords for '{feature_name}' after all restoration attempts: {list(da_src_processed.coords.keys())}")
                            logger.debug(f"  Successfully handled duplicates in '{standard_name_log}' for '{feature_name}' using groupby().first().")


                            da_src_processed.attrs.update(original_da_attrs)


                            # Verify uniqueness and monotonicity.
                            if not da_src_processed[coord_name].to_index().is_unique:
                                logger.error(f"  Error: Duplicate values persist in '{standard_name_log}' for '{feature_name}' even after groupby().first(). Skipping this feature.")
                                raise ValueError(f"Failed to remove duplicates in {standard_name_log} after groupby().first().")

                            if not da_src_processed[coord_name].to_index().is_monotonic_increasing:
                                logger.warning(f"  Warning: Coordinate '{standard_name_log}' for '{feature_name}' is not monotonic increasing after groupby().first(). Attempting to sort.")
                                da_src_processed = da_src_processed.sortby(coord_name)
                                if not da_src_processed[coord_name].to_index().is_monotonic_increasing:
                                    logger.error(f"  Error: Failed to sort coordinate '{standard_name_log}' for '{feature_name}' after groupby().first(). Skipping.")
                                    raise ValueError(f"Failed to sort {standard_name_log} after groupby().first().")

                            logger.debug(f"  Coordinates for '{standard_name_log}' in '{feature_name}' are now unique and monotonic after groupby().first() and potential sort.")
                        else:
                            logger.debug(f"  Coordinates for '{standard_name_log}' in '{feature_name}' are unique. No processing needed.")

                        da_src = da_src_processed

                except ValueError as ve:
                    logger.error(f"  Skipping '{feature_name}' due to issue with duplicate coordinate handling: {ve}")
                    continue
                except Exception as unique_coord_e:
                    logger.error(f"  Error during unique coordinate check/handling for '{feature_name}': {unique_coord_e}. Skipping feature.")
                    continue


                def _log_coord_diagnostics(data_array, coord_name_to_log, actual_coord_name_in_da, current_file, current_feature):
                    pass


                original_attrs = da_src.attrs.copy()
                values_src_flat = da_src.values.ravel()
                valid_mask = ~np.isnan(values_src_flat)

                if not np.any(valid_mask):
                    logger.warning(f"  Warning: All source data for '{feature_name}' is NaN after potential duplicate removal. Interpolation skipped. Creating NaN array.")

                    interpolated_da = xr.DataArray(
                        np.full((len(y_target), len(x_target)), np.nan),
                        coords=[('latitude', y_target), ('longitude', x_target)],
                        name=feature_name
                    )
                else:
                    logger.debug(f"  Starting interpolation for '{feature_name}' using xarray.interp (method='{interp_method}', fill_value=NaN)...")


                    _log_coord_diagnostics(da_src, 'latitude', final_lat_var, os.path.basename(file_path), feature_name)

                    _log_coord_diagnostics(da_src, 'longitude', final_lon_var, os.path.basename(file_path), feature_name)


                    try:


                        logger.debug(f"  PRE-INTERP CHECK for '{feature_name}': "
                                     f"da_src.coords: {list(da_src.coords.keys())}, "
                                     f"da_src.dims: {da_src.dims}, "
                                     f"final_lat_var='{final_lat_var}', final_lon_var='{final_lon_var}'")
                        if final_lat_var not in da_src.coords or final_lon_var not in da_src.coords:
                            logger.error(f"  Error: Coordinates '{final_lat_var}' or '{final_lon_var}' missing from da_src for '{feature_name}'. Skipping interpolation. (This log confirms the check condition)")
                            continue

                        is_categorical = False
                        try:
                            is_categorical = _detect_categorical_feature(
                                da_src,
                                feature_name=str(feature_name),
                                config=dict(config),
                            )
                        except Exception:
                            is_categorical = False
                        interp_method_this = "nearest" if is_categorical else interp_method
                        if is_categorical and interp_method_this != interp_method:
                            logger.info(
                                f"  Feature '{feature_name}' detected as categorical; "
                                f"overriding interp_method '{interp_method}' -> '{interp_method_this}'."
                            )

                        interpolated_da = da_src.interp(
                            latitude=target_lat_da,
                            longitude=target_lon_da,
                            method=interp_method_this,
                            assume_sorted=False,
                            kwargs={"fill_value": np.nan}
                        )


                        interpolated_da.name = feature_name
                        logger.debug(f"  Interpolation complete for '{feature_name}'. NaNs preserved where interpolation is not possible.")

                        if is_categorical:
                            codes_da = xr.where(np.isfinite(interpolated_da), np.rint(interpolated_da), np.nan)
                            codes_da.name = feature_name

                            category_labels = None
                            try:
                                category_labels = _get_category_labels(
                                    ds_src=ds,
                                    da_src=da_src,
                                    feature_name=str(feature_name),
                                    config=dict(config),
                                )
                            except Exception:
                                category_labels = None

                            onehot = _one_hot_encode_categorical_grid(
                                codes_da,
                                feature_name=str(feature_name),
                                source_attrs=dict(original_attrs),
                                category_labels=category_labels,
                                max_categories=int(config.get("max_categories", 64) or 64),
                            )
                            if onehot:
                                for k, v in onehot.items():
                                    if k in processed_arrays:
                                        logger.warning(f"  Warning: One-hot variable name collision: '{k}'. Skipping.")
                                        continue
                                    processed_arrays[k] = v
                                processed_feature_names.add(feature_name)
                                logger.debug(
                                    f"  Stored categorical feature '{feature_name}' as {len(onehot)} one-hot channels."
                                )
                                continue
                            else:
                                logger.warning(
                                    f"  Categorical feature '{feature_name}' not one-hot encoded "
                                    "(too many categories or empty); keeping as numeric codes."
                                )
                                interpolated_da = codes_da
                                interpolated_da.attrs.update(
                                    {
                                        "is_categorical_codes": 1,
                                        "categorical_source_feature": str(feature_name),
                                    }
                                )

                    except ValueError as interp_ve:
                        logger.error(f"  VALUE ERROR during xarray interpolation for '{feature_name}': {interp_ve}")
                        logger.error(f"  This often indicates issues with coordinate uniqueness or monotonicity even after pre-processing.")
                        logger.error(f"  Re-logging coordinate diagnostics due to interpolation failure for '{feature_name}':")
                        _log_coord_diagnostics(da_src, 'latitude', final_lat_var, os.path.basename(file_path), feature_name)
                        _log_coord_diagnostics(da_src, 'longitude', final_lon_var, os.path.basename(file_path), feature_name)
                        continue
                    except Exception as interp_e:
                        logger.error(f"  Error during xarray interpolation for '{feature_name}': {interp_e}")
                        continue


                interpolated_da.attrs.update(original_attrs)
                logger.debug(f"  Preserved original attributes for '{feature_name}'.")


                processed_arrays[feature_name] = interpolated_da
                processed_feature_names.add(feature_name)
                logger.debug(f"  Stored processed DataArray for '{feature_name}', shape: {interpolated_da.shape}")


        except FileNotFoundError:
            logger.error(f"Error: Input file not found: {file_path}. Skipping.")
            continue
        except Exception as e:
            logger.error(f"  Error processing feature '{feature_name}' from {os.path.basename(file_path)}: {e}")
            continue


    if not processed_arrays:
        logger.error("Error: No environmental features were successfully processed.")
        return False

    logger.debug(f"Finished processing features. Successfully processed: {len(processed_feature_names)}.")
    logger.debug("Merging processed DataArrays into a single Dataset...")

    try:

        ds_out = xr.Dataset(processed_arrays)


        land_mask = None
        try:
            logger.info("Generating land mask using global-land-mask...")

            lats = ds_out.latitude.values
            lons = ds_out.longitude.values


            lons_180 = np.where(lons > 180, lons - 360, lons)


            lon_grid, lat_grid = np.meshgrid(lons_180, lats)


            mask_np = globe.is_land(lat_grid, lon_grid)

            # Create a temporary DataArray with the non-monotonic [-180, 180] longitude
            temp_mask = xr.DataArray(
                mask_np,
                coords=[('latitude', lats), ('lon_180', lons_180)],
                dims=['latitude', 'lon_180']
            )


            # This correctly reorders the mask data to match the monotonic [0, 360) coordinate.
            land_mask = temp_mask.assign_coords(
                longitude=('lon_180', lons)
            ).swap_dims(
                {'lon_180': 'longitude'}
            ).sortby('longitude')

            logger.info("Land mask generated and sorted successfully using global-land-mask.")
        except NameError:
            logger.warning("`global-land-mask` library not found. Please install it (`pip install global-land-mask`). Proceeding without land mask.")
            land_mask = None
        except Exception as e:
            logger.warning(f"Could not generate land mask using global-land-mask: {e}. Proceeding without it.", exc_info=True)
            land_mask = None


        onehot_vars = [
            v
            for v in ds_out.data_vars
            if bool(ds_out[v].attrs.get("is_categorical_onehot", False)) or "__cat_" in str(v)
        ]
        numeric_vars = [v for v in ds_out.data_vars if v not in onehot_vars]

        has_nans_numeric = any(ds_out[var].isnull().any().item() for var in numeric_vars) if numeric_vars else False
        has_nans_onehot = any(ds_out[var].isnull().any().item() for var in onehot_vars) if onehot_vars else False
        interpolation_method_attr = f"xarray.interp(method='{interp_method}')"
        title_attr = "Processed Environmental Data from Configuration (xarray.interp)"

        if has_nans_numeric:
            if use_iterative_imputer:
                logger.info("Using multivariate iterative imputer to fill remaining NaNs...")
                ds_out = iterative_imputer_interpolation(ds_out, variables=list(numeric_vars), land_mask=land_mask)
                interpolation_method_attr = 'iterative_imputer'
                title_attr = "Processed Environmental Data (Iterative Imputer)"
            else:
                logger.info(f"Using single-variable interpolation (method='{interp_method}') to fill remaining NaNs...")
                ds_out = interpolate_grid(ds_out, method=interp_method, variables=list(numeric_vars), land_mask=land_mask)
                interpolation_method_attr = f"interpolate_grid(method='{interp_method}')"
                title_attr = f"Processed Environmental Data (interpolate_grid, method='{interp_method}')"
        elif has_nans_onehot:
            logger.info(
                "Only categorical one-hot variables contain NaNs; skipping final interpolation "
                "to avoid corrupting one-hot channels."
            )
        else:
            logger.info("No NaNs found in the merged dataset. Skipping final interpolation step.")
        logger.info("Land mask was used during interpolation but will not be saved as a data variable in the final output.")


        for feature_name_iter in ds_out.data_vars:
            is_onehot = bool(ds_out[feature_name_iter].attrs.get("is_categorical_onehot", False)) or "__cat_" in str(feature_name_iter)
            ds_out[feature_name_iter].attrs['interpolation_method'] = (
                "categorical_onehot(no_fill)" if is_onehot else interpolation_method_attr
            )
            ds_out[feature_name_iter].attrs['interpolation_fill_value'] = (
                'none' if is_onehot else ('filled' if has_nans_numeric else 'none')
            )
            ds_out[feature_name_iter].attrs['interpolation_resolution_degrees'] = resolution

            original_config = next((c for c in variable_config_list if c.get('feature_name') == feature_name_iter), None)
            if original_config:
                 ds_out[feature_name_iter].attrs['source_file'] = os.path.basename(original_config.get('file_path', 'N/A'))
                 ds_out[feature_name_iter].attrs['source_variable'] = original_config.get('data_var', 'N/A')


        ds_out["latitude"].attrs = {"units": "degrees_north", "standard_name": "latitude", "axis": "Y"}
        ds_out["longitude"].attrs = {"units": "degrees_east", "standard_name": "longitude", "axis": "X"}
        ds_out.attrs = {
            "title": title_attr,
            "history": f"Created on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}.",
            "processed_features": json.dumps(list(processed_feature_names))
        }

        logger.debug("Sanitizing attributes for all data variables and coordinates before saving...")
        for var_name_to_sanitize in list(ds_out.data_vars) + list(ds_out.coords):
            target_obj = ds_out[var_name_to_sanitize]
            original_obj_attrs = target_obj.attrs.copy()
            sanitized_attrs = {}
            attr_type = "Variable" if var_name_to_sanitize in ds_out.data_vars else "Coordinate"

            for k, v in original_obj_attrs.items():
                if isinstance(v, np.ndarray):
                    try:

                        if k == 'actual_range' and v.size >= 2 :

                            sanitized_attrs[k] = f"{v.item(0)}|{v.item(-1)}"
                            logger.debug(f"  Sanitized {attr_type} '{var_name_to_sanitize}' attr '{k}': numpy.ndarray to simple string -> {sanitized_attrs[k]}")
                        else:
                            v_list = v.tolist()
                            sanitized_attrs[k] = json.dumps(v_list)
                            logger.debug(f"  Sanitized {attr_type} '{var_name_to_sanitize}' attr '{k}': numpy.ndarray to JSON string -> {sanitized_attrs[k]}")
                    except Exception as e_conv_array:
                        logger.warning(f"  Warning: Could not convert numpy.ndarray attribute '{k}' for {attr_type} '{var_name_to_sanitize}': {e_conv_array}. Keeping original.")
                        sanitized_attrs[k] = v
                elif isinstance(v, list):
                    try:
                        sanitized_attrs[k] = json.dumps(v)
                        logger.debug(f"  Sanitized {attr_type} '{var_name_to_sanitize}' attr '{k}': list to JSON string -> {sanitized_attrs[k]}")
                    except Exception as e_conv_list:
                        logger.warning(f"  Warning: Could not convert list attribute '{k}' for {attr_type} '{var_name_to_sanitize}' to JSON string: {e_conv_list}. Keeping original.")
                        sanitized_attrs[k] = v
                elif isinstance(v, (np.float16, np.float32, np.float64, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.bool_)):
                    try:
                        sanitized_attrs[k] = v.item()
                        logger.debug(f"  Sanitized {attr_type} '{var_name_to_sanitize}' attr '{k}': numpy scalar to Python type -> {sanitized_attrs[k]} (type: {type(sanitized_attrs[k])})")
                    except Exception as e_conv_scalar:
                        logger.warning(f"  Warning: Could not convert numpy scalar attribute '{k}' for {attr_type} '{var_name_to_sanitize}': {e_conv_scalar}. Keeping original.")
                        sanitized_attrs[k] = v
                elif pd.isna(v):
                    try:
                        sanitized_attrs[k] = None
                        logger.debug(f"  Sanitized {attr_type} '{var_name_to_sanitize}' attr '{k}': pandas NA type to None")
                    except Exception as e_conv_na:
                        logger.warning(f"  Warning: Could not convert pandas NA attribute '{k}' for {attr_type} '{var_name_to_sanitize}': {e_conv_na}. Keeping original.")
                        sanitized_attrs[k] = v
                else:

                    sanitized_attrs[k] = v

            target_obj.attrs = sanitized_attrs
        logger.debug("Finished sanitizing attributes.")

        logger.debug("Saving processed data to NetCDF...")


        if geospatial_bounds and isinstance(geospatial_bounds, dict):
            min_lat_input = geospatial_bounds.get('min_lat')
            max_lat_input = geospatial_bounds.get('max_lat')
            min_lon_input = geospatial_bounds.get('min_lon')
            max_lon_input = geospatial_bounds.get('max_lon')


            internal_lat_coord = 'latitude'
            internal_lon_coord = 'longitude'


            valid_for_cropping = True
            if internal_lat_coord not in ds_out.coords or internal_lon_coord not in ds_out.coords:
                 logger.error(f"Error: Standard coordinates '{internal_lat_coord}' or '{internal_lon_coord}' not found in the processed dataset. Skipping crop.")
                 valid_for_cropping = False

            if valid_for_cropping and not all(isinstance(val, (int, float)) for val in [min_lat_input, max_lat_input, min_lon_input, max_lon_input]):
                logger.warning("Warning: Invalid geospatial_bounds values (not all numeric). Cropping skipped.")
                valid_for_cropping = False

            if valid_for_cropping and min_lat_input > max_lat_input:
                logger.warning(f"Warning: Invalid latitude bounds (min_lat {min_lat_input} > max_lat {max_lat_input}). Cropping skipped.")
                valid_for_cropping = False

            # ds_out is now expected to be in 0-360 longitude range.
            # User-provided bounds min_lon_input, max_lon_input are also expected to be 0-360.


            if valid_for_cropping:
                user_lon_min_orig = geospatial_bounds.get('min_lon')
                user_lon_max_orig = geospatial_bounds.get('max_lon')
                logger.debug(f"Applying geospatial crop: Lat [{min_lat_input}, {max_lat_input}], User Lon Input [{user_lon_min_orig}, {user_lon_max_orig}]")

                try:


                    cropped_ds_lat = ds_out.sel(
                        **{internal_lat_coord: slice(min_lat_input, max_lat_input)}
                    )


                    # Converting -180 -> 180 on a 0-360 grid collapses the interval to a point (180),

                    user_lon_min_orig = float(user_lon_min_orig)
                    user_lon_max_orig = float(user_lon_max_orig)
                    span = (user_lon_max_orig - user_lon_min_orig) if (user_lon_min_orig <= user_lon_max_orig) else ((user_lon_max_orig + 360.0) - user_lon_min_orig)
                    full_lon_range = bool(span >= 359.999)
                    if full_lon_range:
                        logger.debug("  Longitude bounds span ~360°, skipping lon cropping (full globe).")
                        ds_out = cropped_ds_lat
                        if ds_out[internal_lat_coord].size == 0 or ds_out[internal_lon_coord].size == 0:
                            logger.warning(
                                "Warning: Cropping resulted in an empty dataset (either latitude or longitude dimension is zero). "
                                f"Final lat shape: {ds_out[internal_lat_coord].shape}, lon shape: {ds_out[internal_lon_coord].shape}"
                            )
                        else:
                            logger.debug(
                                f"Data cropped (lat only). New shape for a sample variable: "
                                f"{ds_out[list(ds_out.data_vars)[0]].shape if ds_out.data_vars else 'N/A'}"
                            )
                    else:

                        sel_lon_min_0_360 = user_lon_min_orig + 360 if user_lon_min_orig < 0 else user_lon_min_orig
                        sel_lon_max_0_360 = user_lon_max_orig + 360 if user_lon_max_orig < 0 else user_lon_max_orig

                        logger.debug(
                            f"  User lon range [{user_lon_min_orig}, {user_lon_max_orig}] converted for 0-360 grid selection as: "
                            f"[{sel_lon_min_0_360}, {sel_lon_max_0_360}]"
                        )


                        if user_lon_min_orig <= user_lon_max_orig:
                            # Subcase 1.1: Standard segment on 0-360 grid.
                            if sel_lon_min_0_360 <= sel_lon_max_0_360:
                                logger.debug(f"    Selecting single segment on 0-360 grid: slice({sel_lon_min_0_360}, {sel_lon_max_0_360})")
                                cropped_ds_final = cropped_ds_lat.sel(
                                    **{internal_lon_coord: slice(sel_lon_min_0_360, sel_lon_max_0_360)}
                                )
                            # Subcase 1.2: Wraps around 0/360 (e.g. user [-20, 20] -> sel_min=340, sel_max=20).
                            else:
                                logger.debug(
                                    f"    Selecting two segments on 0-360 grid (crossing prime meridian/0deg): "
                                    f"slice({sel_lon_min_0_360}, end) and slice(start, {sel_lon_max_0_360})"
                                )
                                part1_ds = cropped_ds_lat.sel(**{internal_lon_coord: slice(sel_lon_min_0_360, None)})
                                part2_ds = cropped_ds_lat.sel(**{internal_lon_coord: slice(None, sel_lon_max_0_360)})

                                valid_parts_to_concat = []
                                if part1_ds[internal_lon_coord].size > 0:
                                    valid_parts_to_concat.append(part1_ds)
                                    logger.debug(f"      Part 1 ({sel_lon_min_0_360} to end) has data.")
                                else:
                                    logger.debug(f"      Part 1 ({sel_lon_min_0_360} to end) is empty.")
                                if part2_ds[internal_lon_coord].size > 0:
                                    valid_parts_to_concat.append(part2_ds)
                                    logger.debug(f"      Part 2 (start to {sel_lon_max_0_360}) has data.")
                                else:
                                    logger.debug(f"      Part 2 (start to {sel_lon_max_0_360}) is empty.")

                                if not valid_parts_to_concat:
                                    logger.warning(
                                        f"    Warning: Both parts of 0/360 wrap-around slice resulted in empty data for user range "
                                        f"[{user_lon_min_orig}, {user_lon_max_orig}]."
                                    )
                                    cropped_ds_final = cropped_ds_lat.isel(**{internal_lon_coord: slice(0, 0)})
                                elif len(valid_parts_to_concat) == 1:
                                    cropped_ds_final = valid_parts_to_concat[0]
                                else:
                                    cropped_ds_final = xr.concat(valid_parts_to_concat, dim=internal_lon_coord)

                        # Case 2: User selection crosses the 180th meridian (min > max): single segment on 0-360 grid.
                        else:
                            logger.debug(f"    User selection crosses 180th meridian: user range [{user_lon_min_orig}, {user_lon_max_orig}]")
                            logger.debug(f"    This corresponds to a single segment on 0-360 grid: slice({sel_lon_min_0_360}, {sel_lon_max_0_360})")
                            cropped_ds_final = cropped_ds_lat.sel(
                                **{internal_lon_coord: slice(sel_lon_min_0_360, sel_lon_max_0_360)}
                            )

                        ds_out = cropped_ds_final

                        if ds_out[internal_lat_coord].size == 0 or ds_out[internal_lon_coord].size == 0:
                            logger.warning(
                                "Warning: Cropping resulted in an empty dataset (either latitude or longitude dimension is zero). "
                                f"Final lat shape: {ds_out[internal_lat_coord].shape}, lon shape: {ds_out[internal_lon_coord].shape}"
                            )
                        else:
                            logger.debug(
                                f"Data cropped. New shape for a sample variable: "
                                f"{ds_out[list(ds_out.data_vars)[0]].shape if ds_out.data_vars else 'N/A'}"
                            )

                except Exception as crop_e:
                    logger.error(f"Error during geospatial crop application (0-360 grid): {crop_e}. Proceeding with uncropped data.")


        elif geospatial_bounds is not None:


            logger.warning("Warning: geospatial_bounds was provided but not in the expected dictionary format or failed initial validation. Cropping skipped.")


        derived_feature_meta = {
            "enabled": bool(enable_derived_features),
            "profile": str(derived_feature_profile or "exploration_v1"),
            "created_features": [],
            "skipped_features": [],
            "n_created": 0,
        }
        if enable_derived_features:
            try:
                ds_out, derived_feature_meta = add_derived_environment_features(
                    ds_out,
                    resolution=float(resolution),
                    profile=str(derived_feature_profile or "exploration_v1"),
                    config=derived_feature_config,
                )
                processed_feature_names.update(list(derived_feature_meta.get("created_features", []) or []))
                title_attr = f"{title_attr} + derived exploration features"
            except Exception as e_derived:
                logger.warning(f"Failed to add derived environmental features: {e_derived}", exc_info=True)
                derived_feature_meta = {
                    "enabled": bool(enable_derived_features),
                    "profile": str(derived_feature_profile or "exploration_v1"),
                    "error": str(e_derived),
                    "created_features": [],
                    "skipped_features": [],
                    "n_created": 0,
                }


            for feature_name_iter in ds_out.data_vars:
                if not bool(ds_out[feature_name_iter].attrs.get("derived_feature", False)):
                    continue
                ds_out[feature_name_iter].attrs["interpolation_resolution_degrees"] = resolution
                ds_out[feature_name_iter].attrs["interpolation_method"] = "derived_after_base_interpolation"
                ds_out[feature_name_iter].attrs["interpolation_fill_value"] = "derived"

        ds_out.attrs = {
            "title": title_attr,
            "history": f"Created on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}.",
            "processed_features": json.dumps(list(ds_out.data_vars)),
            "derived_feature_meta": json.dumps(derived_feature_meta, ensure_ascii=False),
        }


        output_dir = os.path.dirname(output_path)
        if output_dir: os.makedirs(output_dir, exist_ok=True)

        ds_out.to_netcdf(output_path)
        try:
            processed_dir = Path(output_path).parent
            run_dir = processed_dir.parent
            manifest = {
                "artifact_type": "env_feature_manifest",
                "env_gridded_nc": str(output_path),
                "resolution": float(resolution),
                "interp_method": str(interp_method),
                "use_iterative_imputer": bool(use_iterative_imputer),
                "derived_features": derived_feature_meta,
                "geospatial_bounds": geospatial_bounds,
                "feature_names": list(ds_out.data_vars),
                "n_features": int(len(ds_out.data_vars)),
                "latitude_size": int(ds_out.sizes.get("latitude", 0)),
                "longitude_size": int(ds_out.sizes.get("longitude", 0)),
                "variable_configs": variable_config_list,
            }
            manifest_path = processed_dir / "env_feature_manifest.json"
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)
            report_path = run_dir / "RUN_REPORT.md"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("# Environment Preprocessing Run\n\n")
                f.write(f"- Output: `{output_path}`\n")
                f.write(f"- Feature manifest: `{manifest_path}`\n")
                f.write(f"- Feature count: `{len(ds_out.data_vars)}`\n")
                f.write(f"- Resolution: `{resolution}` degrees\n")
                f.write(f"- Interpolation: `{interp_method}`\n")
                f.write(f"- Derived features: `{derived_feature_meta}`\n")
                if geospatial_bounds:
                    f.write(f"- Bounds: `{geospatial_bounds}`\n")
        except Exception as e_manifest:
            logger.warning(f"Failed to write env feature manifest/RUN_REPORT: {e_manifest}")


        logger.debug(f"Processed environmental data successfully saved to {output_path}")

    except MemoryError as mem_err:
        logger.error(f"Error: Memory error during merging or saving: {mem_err}. Try reducing resolution or processing fewer variables.")
        return False
    except OSError as os_err:
        logger.error(f"Error: OS error during file saving (e.g., disk full, permissions): {os_err}")
        return False
    except Exception as e:
        logger.error(f"Error merging or saving processed data to NetCDF: {e}")
        return False


    logger.debug("Environmental data processing finished successfully.")
    return True

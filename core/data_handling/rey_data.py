
import os
import numpy as np
import xarray as xr
from core.utils.netcdf_io import open_dataset
import pandas as pd


import re
import json
from pathlib import Path

from core.data_handling.mechanism_clustering import (
    EXPLORATION_4CLASS_NAMES,
    GRADE_GEOCHEM_CLASS_NAMES,
    REE_Y_FEATURE_COLS,
    fit_predict_grade_geochem_classification,
    fit_predict_mechanism_kmeans,
    fit_predict_two_stage_mechanism_kmeans,
)
from core.data_handling.mechanism_features import (
    add_mechanism_proxy_features,
    add_two_stage_mechanism_features,
    select_mechanism_feature_cols,
)
from core.utils.logger_config import logger


def _normalize_site_group_id(row: pd.Series) -> str:
    def _clean_text(v):
        if pd.isna(v):
            return ""
        s = str(v).strip().upper()
        s = re.sub(r"\s+", "", s)
        s = re.sub(r"[^A-Z0-9]+", "", s)

        m = re.match(r"^(U?\d{3,5})(?:[A-Z]\d*)?$", s)
        if m:
            return m.group(1)
        return s

    for col in ("site_hole", "core_id"):
        if col in row.index:
            s = _clean_text(row[col])
            if s:
                return s
    try:
        lon = float(row.get("lon"))
        lat = float(row.get("lat"))
        if np.isfinite(lon) and np.isfinite(lat):
            return f"COORD_{round(lat, 3):.3f}_{round(lon, 3):.3f}"
    except Exception:
        pass
    return "UNKNOWN_SITE"

def _dms_to_decimal(dms):
    """Converts DMS format lat/lon to decimal degrees."""
    try:
        if not isinstance(dms, str) or not dms.strip(): return None
        dms = dms.replace("′′", '"').replace("′", "'").strip()
        regex = re.compile(r"(\d+)[°]\s*(\d+)?(?:\.(\d{1,10}))?[']?\s*([NSEW])?")
        match = regex.match(dms)
        if not match:
            try: return float(dms)
            except ValueError: return None
        degrees = float(match.group(1))
        minutes = float(match.group(2)) if match.group(2) else 0
        seconds_str = match.group(3)
        seconds = 0
        if seconds_str:
             try:
                 sec_val = float(f"0.{seconds_str}") * 60
                 if sec_val >= 60: minutes += 1; seconds = 0
                 else: seconds = sec_val
             except ValueError: seconds = 0
        decimal = degrees + minutes / 60 + seconds / 3600
        direction = match.group(4)
        if direction and direction in 'SW': decimal *= -1
        return decimal
    except Exception: return None

def process_rey_data_points(
    input_file_path,
    output_path,
    lon_col,
    lat_col,
    depth_col,
    rey_col_name,
    land_mask_da=None,
    geospatial_bounds=None,
    core_id_col=None,
    site_hole_col=None,
    extra_cols_to_keep=None,
    enable_mechanism_clustering: bool = False,
    mechanism_mode: str = "two_stage",
    mechanism_feature_cols=None,
    mechanism_n_classes: int = 3,
    mechanism_stage2_k: int = 2,
    mechanism_label_col: str = "mech_class_id",
    mechanism_prob_prefix: str = "mech_class_prob_",
    mechanism_order_by_col: str = "rey",
    mechanism_class_names=None,
    mechanism_random_state: int = 42,
    mechanism_kmeans_n_init: int = 20,
    mechanism_kmeans_max_iter: int = 300,
    mechanism_kmeans_init: str = "k-means++",
    mechanism_log1p_transform: bool = True,
    mechanism_scaler_kind: str = "robust",
    mechanism_softmax_temperature: float | None = None,
    mechanism_max_missing_rate: float = 0.30,
    mechanism_max_abs_corr: float = 0.95,
    mechanism_artifacts_path: str | None = None,


    mechanism_fit_time_domain: str = "quaternary",
    mechanism_fit_max_age_ka_proxy: float = 2580.0,
    mechanism_sedimentation_rate_grid_path: str | None = None,
):
    """
    Loads raw REY data from Excel or CSV, processes using specified columns,
    optionally filters by geospatial_bounds and land_mask,
    and saves as a point-based Parquet file. Does NOT perform gridding or binning.

    Args:
        input_file_path (str): Path to the input Excel or CSV file.
        output_path (str): Path to save the output Parquet file.
        lon_col (str): Name of the longitude column in the input file.
        lat_col (str): Name of the latitude column in the input file.
        depth_col (str): Name of the depth column in the input file.
        rey_col_name (str): Name of the REY sum column.
        land_mask_da (xr.DataArray, optional): Land mask to apply. Defaults to None.
        geospatial_bounds (dict, optional): Dictionary with 'min_lat', 'max_lat',
            'min_lon', 'max_lon' to filter the points. Defaults to None.
    """
    logger.debug(f"Starting REY point data processing...")
    logger.debug(f"Input file: {input_file_path}")
    logger.debug(f"Output file path: {output_path}")
    logger.debug(f"Using columns - Lon: '{lon_col}', Lat: '{lat_col}', Depth: '{depth_col}'")
    if core_id_col:
        logger.debug(f"Using core_id column: '{core_id_col}'")
    if site_hole_col:
        logger.debug(f"Using site_hole column: '{site_hole_col}'")

    rare_earth_data = None
    try:

        if not os.path.exists(input_file_path):
            raise FileNotFoundError(f"REY input file not found: {input_file_path}")


        logger.debug(f"Attempting to load file: {input_file_path}")
        file_ext = os.path.splitext(input_file_path)[1].lower()
        if file_ext == '.xlsx':
            try:
                rare_earth_data = pd.read_excel(input_file_path)
                logger.debug(f"Successfully loaded Excel file.")
            except Exception as excel_e:
                raise ValueError(f"Error reading Excel file: {excel_e}")
        elif file_ext == '.csv':
            try:
                rare_earth_data = pd.read_csv(input_file_path)
                logger.debug("Successfully loaded CSV file (UTF-8 attempt).")
            except UnicodeDecodeError:
                logger.warning("Warning: Standard UTF-8 decoding failed for CSV. Trying 'ISO-8859-1'.")
                try:
                    rare_earth_data = pd.read_csv(input_file_path, encoding='ISO-8859-1')
                    logger.debug("Successfully loaded CSV file (ISO-8859-1 attempt).")
                except Exception as csv_e2:
                     raise ValueError(f"Error reading CSV file with fallback encoding: {csv_e2}")
            except Exception as csv_e:
                 raise ValueError(f"Error reading CSV file: {csv_e}")
        else:
            raise ValueError(f"Unsupported file type '{file_ext}'. Please provide .xlsx or .csv.")

    except FileNotFoundError as fnf_err:
        logger.error(f"Error: {fnf_err}")
        return False
    except ValueError as val_err:
        logger.error(f"Error loading or parsing input file: {val_err}")
        return False
    except Exception as load_err:
        logger.error(f"Unexpected error loading input file {input_file_path}: {load_err}")
        return False


    if rare_earth_data is None:
        logger.error("Error: Data loading failed, cannot proceed.")
        return False

    try:

        logger.debug("Validating required columns...")


        required_input_cols = [lon_col, lat_col, depth_col, rey_col_name]
        missing_cols = [col for col in required_input_cols if col not in rare_earth_data.columns]
        if missing_cols:
            raise KeyError(f"Specified columns not found in the input file: {', '.join(missing_cols)}. Available: {list(rare_earth_data.columns)}")
        logger.debug("All required columns found.")


        logger.debug(f"Converting coordinates from columns '{lon_col}' and '{lat_col}'...")

        try:
            rare_earth_data["decimal_lon"] = rare_earth_data[lon_col].apply(_dms_to_decimal)
            rare_earth_data["decimal_lat"] = rare_earth_data[lat_col].apply(_dms_to_decimal)

            if rare_earth_data["decimal_lon"].isnull().sum() > len(rare_earth_data) * 0.5:
                logger.warning("Warning: DMS conversion resulted in many NaNs. Attempting direct numeric conversion for longitude.")
                rare_earth_data["decimal_lon"] = pd.to_numeric(rare_earth_data[lon_col], errors='coerce')
            if rare_earth_data["decimal_lat"].isnull().sum() > len(rare_earth_data) * 0.5:
                logger.warning("Warning: DMS conversion resulted in many NaNs. Attempting direct numeric conversion for latitude.")
                rare_earth_data["decimal_lat"] = pd.to_numeric(rare_earth_data[lat_col], errors='coerce')
        except Exception as coord_conv_e:
            logger.warning(f"Warning: Error during coordinate conversion ({coord_conv_e}). Attempting direct numeric conversion.")
            rare_earth_data["decimal_lon"] = pd.to_numeric(rare_earth_data[lon_col], errors='coerce')
            rare_earth_data["decimal_lat"] = pd.to_numeric(rare_earth_data[lat_col], errors='coerce')

        logger.debug("Adjusting longitude range to [0, 360) for consistency with grid...")
        lon_numeric = pd.to_numeric(rare_earth_data["decimal_lon"], errors='coerce')
        rare_earth_data["decimal_lon"] = np.mod(lon_numeric, 360)


        logger.debug("Cleaning data (converting types and dropping rows with NaN in required columns)...")

        rare_earth_data[depth_col] = pd.to_numeric(rare_earth_data[depth_col], errors='coerce')
        rare_earth_data[rey_col_name] = pd.to_numeric(rare_earth_data[rey_col_name], errors='coerce')


        required_processed_cols = ["decimal_lon", "decimal_lat", depth_col, rey_col_name]
        initial_rows = len(rare_earth_data)

        rare_earth_data_cleaned = rare_earth_data.dropna(subset=required_processed_cols)
        cleaned_rows = len(rare_earth_data_cleaned)
        logger.debug(f"Data cleaning: Initial rows={initial_rows}, Cleaned rows={cleaned_rows}.")
        if cleaned_rows == 0:
            logger.error("Error: No valid data remaining after cleaning (check coordinate conversion and numeric columns).")
            return False


        if geospatial_bounds and isinstance(geospatial_bounds, dict):
            min_lat_bound = geospatial_bounds.get('min_lat')
            max_lat_bound = geospatial_bounds.get('max_lat')
            min_lon_bound = geospatial_bounds.get('min_lon')
            max_lon_bound = geospatial_bounds.get('max_lon')

            if all(isinstance(val, (int, float)) for val in [min_lat_bound, max_lat_bound, min_lon_bound, max_lon_bound]):
                logger.debug(f"Applying geospatial filter: Lat [{min_lat_bound}, {max_lat_bound}], Lon [{min_lon_bound}, {max_lon_bound}]")

                lat_condition = (rare_earth_data_cleaned['decimal_lat'] >= min_lat_bound) & \
                                (rare_earth_data_cleaned['decimal_lat'] <= max_lat_bound)


                min_lon_in = float(min_lon_bound)
                max_lon_in = float(max_lon_bound)

                span = (max_lon_in - min_lon_in) if (min_lon_in <= max_lon_in) else ((max_lon_in + 360.0) - min_lon_in)
                full_lon_range = bool(span >= 359.999)

                if full_lon_range:
                    logger.debug("  Longitude bounds span ~360°, skipping lon filtering (full globe).")
                    lon_condition = True
                else:

                    min_lon_360 = float(np.mod(min_lon_in, 360.0))
                    max_lon_360 = float(np.mod(max_lon_in, 360.0))
                    if min_lon_360 <= max_lon_360:
                        logger.debug(f"  Longitude point filter (0-360 grid): {min_lon_360} to {max_lon_360}")
                        lon_condition = (rare_earth_data_cleaned['decimal_lon'] >= min_lon_360) & \
                                        (rare_earth_data_cleaned['decimal_lon'] <= max_lon_360)
                    else:
                        # Wrap-around on 0-360 grid.
                        logger.debug(f"  Handling longitude wrap-around for point filter (0-360 grid): {min_lon_360} to {max_lon_360}")
                        lon_condition = (rare_earth_data_cleaned['decimal_lon'] >= min_lon_360) | \
                                        (rare_earth_data_cleaned['decimal_lon'] <= max_lon_360)

                initial_points_before_geofilter = len(rare_earth_data_cleaned)
                rare_earth_data_cleaned = rare_earth_data_cleaned[lat_condition & lon_condition]
                points_removed_geofilter = initial_points_before_geofilter - len(rare_earth_data_cleaned)
                logger.debug(f"Geospatial filter applied. Removed {points_removed_geofilter} points. Points remaining: {len(rare_earth_data_cleaned)}")

                if len(rare_earth_data_cleaned) == 0:
                    logger.error("Error: No valid data remaining after applying geospatial filter.")
                    return False
            else:
                logger.warning("Warning: Invalid geospatial_bounds values for point filtering. Filter skipped.")
        elif geospatial_bounds is not None:
            logger.warning("Warning: geospatial_bounds for points provided but not in expected dictionary format. Filter skipped.")


        if land_mask_da is not None:
            logger.debug("Applying land mask...")
            if not isinstance(land_mask_da, xr.DataArray):
                logger.warning("Warning: land_mask_da is not an xarray.DataArray. Skipping masking.")
            elif 'decimal_lon' not in rare_earth_data_cleaned.columns or 'decimal_lat' not in rare_earth_data_cleaned.columns:
                logger.warning("Warning: Cleaned data does not contain 'decimal_lon' or 'decimal_lat'. Skipping masking.")
            else:
                initial_points_before_mask = len(rare_earth_data_cleaned)
                try:

                    lon_coord_name = None
                    lat_coord_name = None
                    possible_lon_names = ['lon', 'longitude', 'x']
                    possible_lat_names = ['lat', 'latitude', 'y']
                    mask_coords_lower = {str(c).lower(): c for c in land_mask_da.coords}

                    for name in possible_lon_names:
                        if name in mask_coords_lower:
                            lon_coord_name = mask_coords_lower[name]
                            break
                    for name in possible_lat_names:
                        if name in mask_coords_lower:
                            lat_coord_name = mask_coords_lower[name]
                            break

                    if not lon_coord_name or not lat_coord_name:
                        logger.error(f"Error: Could not determine longitude/latitude coordinate names from mask coords: {list(land_mask_da.coords)}. Skipping masking.")
                    else:
                        logger.debug(f"Using mask coordinates: Lon='{lon_coord_name}', Lat='{lat_coord_name}'")


                        df_lons = rare_earth_data_cleaned['decimal_lon'].to_xarray()
                        df_lats = rare_earth_data_cleaned['decimal_lat'].to_xarray()


                        mask_values = land_mask_da.sel({lon_coord_name: df_lons, lat_coord_name: df_lats}, method='nearest')


                        is_land = mask_values.fillna(False).astype(bool)


                        rare_earth_data_cleaned = rare_earth_data_cleaned[~is_land.values]

                        final_points_after_mask = len(rare_earth_data_cleaned)
                        points_removed = initial_points_before_mask - final_points_after_mask
                        logger.debug(f"Land mask applied. Removed {points_removed} points located on land (mask value True/1). Points remaining: {final_points_after_mask}")


                        if final_points_after_mask == 0:
                            logger.error("Error: No valid data remaining after applying land mask.")
                            return False

                except Exception as mask_err:
                    logger.error(f"Error applying land mask: {mask_err}. Skipping masking.")


        logger.debug("Selecting required columns and preparing for point-based output...")


        try:
            cols_to_keep = [
                "decimal_lon",
                "decimal_lat",
                depth_col,
                rey_col_name,
            ]
            if core_id_col and core_id_col in rare_earth_data_cleaned.columns:
                cols_to_keep.append(core_id_col)
            if site_hole_col and site_hole_col in rare_earth_data_cleaned.columns and site_hole_col not in cols_to_keep:
                cols_to_keep.append(site_hole_col)
            if extra_cols_to_keep:
                for c in list(extra_cols_to_keep):
                    if c in rare_earth_data_cleaned.columns and c not in cols_to_keep:
                        cols_to_keep.append(c)


            if enable_mechanism_clustering:
                proxy_raw_cols = [

                    "La",
                    "Ce",
                    "Pr",
                    "Nd",
                    "Sm",
                    "Eu",
                    "Gd",
                    "Tb",
                    "Dy",
                    "Er",
                    "Ho",
                    "Tm",
                    "Yb",
                    "Lu",
                    "Y",

                    "U",
                    "Th",
                    "Mo",
                    "V",
                    "Zn",
                    "Ni",
                    "Co",
                    "Cu",

                    "MnO",
                    "Fe2O3",
                    "P2O5",
                    "TiO2",
                    "Al2O3",
                    "CaO",
                    "SiO2",
                    "K2O",
                    "MgO",
                ]
                for c in proxy_raw_cols:
                    if c in rare_earth_data_cleaned.columns and c not in cols_to_keep:
                        cols_to_keep.append(c)

            output_df = rare_earth_data_cleaned[cols_to_keep].rename(columns={
                "decimal_lon": "lon",
                "decimal_lat": "lat",
                depth_col: "depth",
                rey_col_name: "rey"
            })
            if core_id_col and core_id_col in output_df.columns:
                output_df = output_df.rename(columns={core_id_col: "core_id"})
            if site_hole_col:
                if site_hole_col in output_df.columns:
                    output_df = output_df.rename(columns={site_hole_col: "site_hole"})
                elif site_hole_col == core_id_col and "core_id" in output_df.columns:


                    output_df["site_hole"] = output_df["core_id"]


            if "core_id" in output_df.columns:
                output_df["core_id"] = output_df["core_id"].astype(str).str.strip()
                output_df.loc[output_df["core_id"].isin(["", "nan", "None", "<NA>"]), "core_id"] = np.nan
            if "site_hole" in output_df.columns:
                output_df["site_hole"] = output_df["site_hole"].astype(str).str.strip()
                output_df.loc[output_df["site_hole"].isin(["", "nan", "None", "<NA>"]), "site_hole"] = np.nan


            non_numeric_cols = {"core_id", "site_hole", "Lithology"}
            for c in list(output_df.columns):
                if c in non_numeric_cols:
                    continue
                if output_df[c].dtype == object:
                    output_df[c] = pd.to_numeric(output_df[c], errors="coerce")
        except KeyError as ke:
            raise KeyError(f"Error selecting/renaming columns for output: Column '{ke}' not found.")
        except Exception as rename_e:
            raise RuntimeError(f"Unexpected error during final column selection/renaming: {rename_e}")


        if enable_mechanism_clustering:
            try:
                mode = str(mechanism_mode or "two_stage").strip().lower()
                if mode not in ("two_stage", "single_stage", "grade_geochem"):
                    raise ValueError(
                        f"Unknown mechanism_mode='{mechanism_mode}' "
                        "(expected 'two_stage', 'single_stage', or 'grade_geochem')"
                    )


                output_df = add_mechanism_proxy_features(output_df)

                output_df = add_two_stage_mechanism_features(output_df)

                if mode in ("two_stage", "grade_geochem"):

                    stage1_feature_cols = ["log_ratio_cao_al", "log_ratio_tio2_al"]
                    missing_stage1 = [c for c in stage1_feature_cols if c not in output_df.columns]
                    if missing_stage1:
                        raise ValueError(
                            "Two-stage clustering requires CaO/Al2O3 and TiO2/Al2O3 ratio features. "
                            f"Missing: {missing_stage1}. Ensure CaO/Al2O3/TiO2 columns are present in the input."
                        )


                    if mechanism_feature_cols is None:
                        mechanism_feature_cols = [

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

                    candidates = [c for c in list(mechanism_feature_cols) if c in output_df.columns]
                    fs = select_mechanism_feature_cols(
                        output_df,
                        candidates=candidates,
                        max_missing_rate=float(mechanism_max_missing_rate),
                        max_abs_corr=float(mechanism_max_abs_corr),
                    )
                    stage2_feature_cols = fs.selected
                    if len(stage2_feature_cols) < 3:
                        raise ValueError(
                            "Too few Stage-2 mechanism features available after filtering. "
                            f"candidates={candidates} selected={stage2_feature_cols}"
                        )

                    stage2_k = 2 if mode == "grade_geochem" else int(mechanism_stage2_k or 2)
                    n_classes_total = 1 + stage2_k
                    if mode == "grade_geochem":
                        mechanism_class_names = GRADE_GEOCHEM_CLASS_NAMES
                    elif mechanism_class_names is not None and len(list(mechanism_class_names)) != n_classes_total:
                        logger.warning(
                            "Two-stage mechanism clustering: class_names length does not match 1+stage2_k; "
                            "falling back to autogenerated names."
                        )
                        mechanism_class_names = None

                    stage1_fit_mask = None
                    stage2_fit_mask = None
                    fit_domain = str(mechanism_fit_time_domain or "quaternary").strip().lower()
                    if fit_domain not in ("all", "full", "quaternary"):
                        raise ValueError(
                            "mechanism_fit_time_domain must be one of: 'quaternary', 'all'. "
                            f"Got: {mechanism_fit_time_domain!r}"
                        )

                    if fit_domain == "quaternary":


                        max_age = float(mechanism_fit_max_age_ka_proxy or 2580.0)
                        sed_path = mechanism_sedimentation_rate_grid_path
                        default_sed = os.path.join("data", "external", "grids", "Average sedimentation rate.nc")
                        if not sed_path:
                            sed_path = default_sed if os.path.exists(default_sed) else None
                        if not sed_path:
                            raise ValueError(
                                "mechanism_fit_time_domain='quaternary' requires a sedimentation rate grid. "
                                "Provide mechanism_sedimentation_rate_grid_path or place the default grid at "
                                f"{default_sed}."
                            )


                        sed_ds = open_dataset(str(sed_path))
                        try:
                            sed_var = "Average sedimentation rate"
                            if sed_var not in sed_ds.data_vars:
                                sed_var = list(sed_ds.data_vars)[0]
                            sed_field = sed_ds[sed_var]

                            lon = pd.to_numeric(output_df["lon"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
                            lat = pd.to_numeric(output_df["lat"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
                            lon_for_interp = ((lon + 180.0) % 360.0) - 180.0

                            sed = sed_field.interp(
                                lon=("points", lon_for_interp),
                                lat=("points", lat),
                                method="linear",
                            ).values
                            sed = np.asarray(sed, dtype=np.float64)
                            if np.isnan(sed).any():
                                mask_nan = np.isnan(sed)
                                sed_near = sed_field.interp(
                                    lon=("points", lon_for_interp[mask_nan]),
                                    lat=("points", lat[mask_nan]),
                                    method="nearest",
                                ).values
                                sed[mask_nan] = np.asarray(sed_near, dtype=np.float64)
                        finally:
                            try:
                                sed_ds.close()
                            except Exception:
                                pass

                        sed_min = 1e-3
                        sed = np.where(np.isfinite(sed), sed, np.nan)
                        sed = np.where(np.isfinite(sed), sed, sed_min)
                        sed = np.clip(sed, sed_min, None)

                        depth_m = pd.to_numeric(output_df["depth"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
                        age_ka_proxy = (depth_m * 100.0) / sed
                        quat_mask = np.isfinite(age_ka_proxy) & (age_ka_proxy <= max_age)
                        stage1_fit_mask = quat_mask.astype(bool)
                        stage2_fit_mask = quat_mask.astype(bool)

                    if mode == "grade_geochem":
                        out_df, artifacts = fit_predict_grade_geochem_classification(
                            output_df,
                            stage1_feature_cols=stage1_feature_cols,
                            stage2_feature_cols=stage2_feature_cols,
                            grade_feature_cols=REE_Y_FEATURE_COLS,
                            mech_label_col="mech_class_id",
                            mech_prob_prefix="mech_class_prob_",
                            grade_label_col="grade_geochem_class_id",
                            grade_name_col="grade_geochem_class_name",
                            grade_prob_prefix="grade_geochem_class_prob_",
                            endmember_col="mech_endmember_id",
                            subclass_col="mech_subclass_id",
                            rey_col="rey",
                            random_state=int(mechanism_random_state),
                            n_init=int(mechanism_kmeans_n_init),
                            max_iter=int(mechanism_kmeans_max_iter),
                            init=str(mechanism_kmeans_init or "k-means++"),
                            scaler_kind=str(mechanism_scaler_kind or "robust"),
                            softmax_temperature=mechanism_softmax_temperature,
                            stage1_fit_mask=stage1_fit_mask,
                            stage2_fit_mask=stage2_fit_mask,
                        )
                        output_df = out_df
                        logger.info(
                            "Grade geochem classification: assigned "
                            "'grade_geochem_class_id' with n_classes=5."
                        )
                    else:
                        out_df, clustering_meta, artifacts = fit_predict_two_stage_mechanism_kmeans(
                            output_df,
                            stage1_feature_cols=stage1_feature_cols,
                            stage2_feature_cols=stage2_feature_cols,
                            stage2_k=stage2_k,
                            label_col=str(mechanism_label_col),
                            prob_prefix=str(mechanism_prob_prefix),
                            endmember_col="mech_endmember_id",
                            subclass_col="mech_subclass_id",
                            carbonate_order_by_col="ratio_cao_al",
                            stage2_order_by_col=("proxy_metal_index" if "proxy_metal_index" in output_df.columns else stage2_feature_cols[0]),
                            class_names=(list(mechanism_class_names) if mechanism_class_names is not None else None),
                            random_state=int(mechanism_random_state),
                            n_init=int(mechanism_kmeans_n_init),
                            max_iter=int(mechanism_kmeans_max_iter),
                            init=str(mechanism_kmeans_init or "k-means++"),
                            scaler_kind=str(mechanism_scaler_kind or "robust"),
                            softmax_temperature=mechanism_softmax_temperature,
                            stage1_fit_mask=stage1_fit_mask,
                            stage2_fit_mask=stage2_fit_mask,
                        )
                        output_df = out_df
                        logger.info(
                            f"Two-stage mechanism clustering: assigned '{mechanism_label_col}' with n_classes={clustering_meta.n_classes} "
                            f"(stage2_k={stage2_k})."
                        )

                    artifacts.setdefault("feature_selection", {})
                    artifacts["feature_selection"].update(
                        {
                            "mode": mode,
                            "stage1_fixed_features": stage1_feature_cols,
                            "stage2_candidates": candidates,
                            "stage2_selected": stage2_feature_cols,
                            "dropped_missing": fs.dropped_missing,
                            "dropped_correlated": fs.dropped_correlated,
                            "max_missing_rate": float(mechanism_max_missing_rate),
                            "max_abs_corr": float(mechanism_max_abs_corr),
                            "note": (
                                "ΣREY/'rey' is excluded from mechanism clustering features. "
                                "For grade_geochem mode, REE+Y element columns are used only for within-mechanism grade subclassing."
                            ),
                        }
                    )
                else:

                    if mechanism_feature_cols is None:
                        mechanism_feature_cols = [
                            "proxy_ce_anom",
                            "proxy_eu_anom",
                            "proxy_y_ho",
                            "proxy_la_yb",
                            "proxy_dy_yb",
                            "proxy_u_th",
                            "Mo",
                            "V",
                            "Zn",
                            "MnO",
                            "Fe2O3",
                            "P2O5",
                            "TiO2",
                            "Al2O3",
                            "CaO",
                        ]
                    candidates = [c for c in list(mechanism_feature_cols) if c in output_df.columns]
                    fs = select_mechanism_feature_cols(
                        output_df,
                        candidates=candidates,
                        max_missing_rate=float(mechanism_max_missing_rate),
                        max_abs_corr=float(mechanism_max_abs_corr),
                    )
                    feature_cols = fs.selected
                    if len(feature_cols) < 3:
                        raise ValueError(
                            "Too few mechanism_feature_cols available after filtering. "
                            f"candidates={candidates} selected={feature_cols}"
                        )
                    out_df, clustering_meta, artifacts = fit_predict_mechanism_kmeans(
                        output_df,
                        feature_cols=feature_cols,
                        n_classes=int(mechanism_n_classes),
                        label_col=str(mechanism_label_col),
                        prob_prefix=str(mechanism_prob_prefix),
                        order_by_col=str(mechanism_order_by_col),
                        class_names=(list(mechanism_class_names) if mechanism_class_names is not None else None),
                        random_state=int(mechanism_random_state),
                        n_init=int(mechanism_kmeans_n_init),
                        max_iter=int(mechanism_kmeans_max_iter),
                        init=str(mechanism_kmeans_init or "k-means++"),
                        log1p_transform=bool(mechanism_log1p_transform),
                        scaler_kind=str(mechanism_scaler_kind or "robust"),
                        softmax_temperature=mechanism_softmax_temperature,
                    )
                    output_df = out_df
                    logger.info(
                        f"Mechanism clustering (single-stage): assigned '{mechanism_label_col}' with n_classes={clustering_meta.n_classes} "
                        f"using {len(clustering_meta.feature_cols)} features."
                    )
                    artifacts.setdefault("feature_selection", {})
                    artifacts["feature_selection"].update(
                        {
                            "mode": "single_stage",
                            "candidates": candidates,
                            "selected": feature_cols,
                            "dropped_missing": fs.dropped_missing,
                            "dropped_correlated": fs.dropped_correlated,
                            "max_missing_rate": float(mechanism_max_missing_rate),
                            "max_abs_corr": float(mechanism_max_abs_corr),
                            "note": "ΣREY/'rey' is excluded from clustering features; it may still be used downstream for evaluation/post-hoc interpretation.",
                        }
                    )

                artifacts_path = mechanism_artifacts_path
                if not artifacts_path:
                    if mode == "grade_geochem":
                        artifacts_path = str(Path(output_path).with_name("geochem_classification.json"))
                    else:
                        artifacts_path = str(Path(output_path).with_suffix("").as_posix()) + "_mechanism_clustering.json"
                Path(artifacts_path).parent.mkdir(parents=True, exist_ok=True)
                with open(artifacts_path, "w", encoding="utf-8") as f:
                    json.dump(artifacts, f, ensure_ascii=False, indent=2)
                logger.info(f"Mechanism/geochem classification: saved artifacts to {artifacts_path}")
            except Exception as mech_e:
                logger.error(f"Mechanism clustering failed: {mech_e}")
                return False

        if "site_group" not in output_df.columns:
            output_df["site_group"] = output_df.apply(_normalize_site_group_id, axis=1)

        if enable_mechanism_clustering and "grade_geochem_class_id" in output_df.columns:
            processed_dir = Path(output_path).parent
            try:
                class_table_path = processed_dir / "geochem_class_table.parquet"
                output_df.to_parquet(class_table_path, index=False)

                label = pd.to_numeric(output_df["grade_geochem_class_id"], errors="coerce")
                summary_rows = []
                for class_id, class_name in enumerate(GRADE_GEOCHEM_CLASS_NAMES):
                    m = label == int(class_id)
                    vals = pd.to_numeric(output_df.loc[m, "rey"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
                    vals = vals[np.isfinite(vals)]
                    summary_rows.append(
                        {
                            "class_id": int(class_id),
                            "class_name": str(class_name),
                            "n": int(np.sum(m)),
                            "rey_median": float(np.nanmedian(vals)) if vals.size else np.nan,
                            "rey_p90": float(np.nanpercentile(vals, 90)) if vals.size else np.nan,
                            "p_ge_1000": float(np.mean(vals >= 1000.0)) if vals.size else np.nan,
                            "p_ge_1500": float(np.mean(vals >= 1500.0)) if vals.size else np.nan,
                        }
                    )
                summary_df = pd.DataFrame(summary_rows)
                summary_df.to_csv(processed_dir / "geochem_class_summary.csv", index=False)
                if "exploration_class_id" in output_df.columns:
                    exploration_label = pd.to_numeric(output_df["exploration_class_id"], errors="coerce")
                    exploration_rows = []
                    for class_id, class_name in enumerate(EXPLORATION_4CLASS_NAMES):
                        m = exploration_label == int(class_id)
                        vals = pd.to_numeric(output_df.loc[m, "rey"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
                        vals = vals[np.isfinite(vals)]
                        exploration_rows.append(
                            {
                                "class_id": int(class_id),
                                "class_name": str(class_name),
                                "n": int(np.sum(m)),
                                "rey_median": float(np.nanmedian(vals)) if vals.size else np.nan,
                                "rey_p90": float(np.nanpercentile(vals, 90)) if vals.size else np.nan,
                                "p_ge_1000": float(np.mean(vals >= 1000.0)) if vals.size else np.nan,
                                "p_ge_1500": float(np.mean(vals >= 1500.0)) if vals.size else np.nan,
                            }
                        )
                    pd.DataFrame(exploration_rows).to_csv(processed_dir / "exploration_class_summary.csv", index=False)
                logger.info(
                    "Grade geochem classification: saved class table and summary under "
                    f"{processed_dir}"
                )
            except Exception as e_cls_out:
                logger.warning(f"Failed to write geochem class table/summary: {e_cls_out}")


        logger.debug(f"Saving processed point data to Parquet: {output_path}")
        output_dir = os.path.dirname(output_path)
        if output_dir: os.makedirs(output_dir, exist_ok=True)

        try:
            output_df.to_parquet(output_path, index=False)
            logger.debug(f"Point data successfully saved to {output_path}")
            return True
        except OSError as os_err:

            logger.error(f"Error saving Parquet file (OS Error): {os_err}")
            return False
        except Exception as save_e:

            logger.error(f"Error saving point data to Parquet: {save_e}")
            return False


    except (KeyError, ValueError, TypeError, RuntimeError) as process_err:

        logger.error(f"Error during data processing: {process_err}")
        return False
    except Exception as e:

        logger.error(f"Unexpected error during data processing/cleaning: {e}")
        return False


    return True

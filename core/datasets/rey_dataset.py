
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import xarray as xr
from core.utils.netcdf_io import open_dataset
from core.utils.logger_config import logger
from core.data_handling.patch_utils import numpy_ffill_bfill_channelwise
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from torchvision import transforms

class REYPointDataset(Dataset):
    """
    PyTorch Dataset for loading REY point data and corresponding gridded environmental patches.
    """
    def __init__(
        self,
        rey_points_path,
        env_grid_path,
        feature_names,
        patch_size=(3, 3),
        transform=None,
        target_transform=None,
        rey_norm=None,
        rey_target_transform_name: str = "log1p",
        rey_target_transform_scale: float = 1000.0,
        rey_target_transform_hinge_beta: float = 0.25,
        depth_transform_method="log1p",
        env_normalization_method="zscore",
        augment_data=False,
        augmentation_degrees=15.0,
        shift_augmentation_pixels=0,
        use_weighted_loss=False,
        weighting_strategy='linear',
        max_weight=5.0,
        use_iterative_imputer: bool = False,
        max_depth_mbsf: float = None,
        max_age_ka_proxy: float = None,
        sed_rate_nc_path: str = None,
        sed_rate_var_name: str = "Average sedimentation rate",
        # The final channel uses age proxy by default and can be overridden with physical depth or zero.
        depth_feature_type_override: str | None = None,
        depth_weighting_scheme: str = None,
        norm_params_override: dict = None,
        # Optional coordinate channels (sin/cos lat/lon) appended after depth channel.
        add_coord_channels: bool = False,


        add_env_missingness_channels: bool = False,


        task_type: str = "regression",
        classification_threshold_ppm: float = 1000.0,
        classification_label_col: str | None = None,
        classification_class_names: list[str] | None = None,
        soft_label_cols: list[str] | None = None,

        use_sample_weights: bool = False,
        sample_weight_col: str | None = None,

        domain_priority_enabled: bool = False,
        domain_priority_mode: str = "loss_weighting",
        training_age_domain_ka: float = 2580.0,
        domain_priority_out_weight: float = 0.2,
    ):
        """
        Args:
            rey_points_path (str): Path to the Parquet file containing point data (lon, lat, depth, rey).
            env_grid_path (str): Path to the NetCDF file containing gridded environmental data.
            feature_names (list): List of environmental variable names to extract from env_grid_path.
            patch_size (tuple): Size of the patch (height, width) to extract around each point. Should be odd numbers.
            transform (callable, optional): Optional transform to be applied on the env_patch tensor.
            target_transform (callable, optional): Optional transform to be applied on the rey target tensor.
            rey_norm (tuple, optional): Tuple of (mean, std) for rey normalization. If None, calculated from data.
            depth_transform_method (str): Method for depth transformation ('log1p', 'sqrt', 'none'). Defaults to 'log1p'.
            env_normalization_method (str): Method for environmental data normalization ('zscore', 'minmax'). Defaults to 'zscore'.
            augment_data (bool): Whether to apply rotation augmentation. Defaults to False.
            augmentation_degrees (float): Maximum degrees for random rotation. Defaults to 15.0.
            shift_augmentation_pixels (int): Maximum number of pixels for random shifting/cropping. Defaults to 0 (disabled).
            use_weighted_loss (bool): If True, calculate and use sample weights for the loss function.
            weighting_strategy (str): Strategy to calculate weights ('linear', 'log').
            max_weight (float): The maximum allowed value for a sample weight.
            use_iterative_imputer (bool): If True, use IterativeImputer for NaN filling.
        """
        logger.info("Dataset Init: Initializing REYPointDataset...")

        self.task_type = str(task_type or "regression").lower()
        if self.task_type not in ("regression", "binary_classification", "multiclass_classification"):
            raise ValueError(f"Dataset Init: Unsupported task_type='{self.task_type}'.")
        self.classification_threshold_ppm = float(classification_threshold_ppm or 1000.0)
        self.classification_label_col = str(classification_label_col).strip() if classification_label_col else None
        self.classification_class_names = list(classification_class_names) if classification_class_names else None
        self.soft_label_cols = [str(c) for c in soft_label_cols] if soft_label_cols else None
        self.num_classes: int | None = None


        self.use_weighted_loss = bool(use_weighted_loss)

        self.use_sample_weights = bool(use_sample_weights)
        self.sample_weight_col = str(sample_weight_col).strip() if sample_weight_col else None
        self.domain_priority_enabled = bool(domain_priority_enabled)
        self.domain_priority_mode = str(domain_priority_mode or "loss_weighting").lower()
        self.training_age_domain_ka = float(training_age_domain_ka or 2580.0)   
        self.domain_priority_out_weight = float(domain_priority_out_weight if domain_priority_out_weight is not None else 0.2)

        if feature_names and isinstance(feature_names, list) and len(feature_names) > 0 and isinstance(feature_names[0], dict):
            if 'name' in feature_names[0]:
                actual_feature_names = [f['name'] for f in feature_names]
                logger.info(f"Dataset Init: Extracted actual feature names (using 'name' key) from list of dicts: {actual_feature_names}")
            elif 'feature_name' in feature_names[0]:
                actual_feature_names = [f['feature_name'] for f in feature_names]
                logger.info(f"Dataset Init: Extracted actual feature names (using 'feature_name' key) from list of dicts: {actual_feature_names}")
            else:
                logger.warning(f"Dataset Init: Warning: feature_names dictionaries lack 'name' or 'feature_name' key. Using as is.")
                actual_feature_names = feature_names
        else:
            actual_feature_names = feature_names
            if isinstance(actual_feature_names, list):
                 logger.info(f"Dataset Init: Using provided feature names (assumed to be list of strings or list of dicts without 'name'/'feature_name'): {actual_feature_names}")
            elif actual_feature_names is None:
                 logger.info(f"Dataset Init: Feature names provided is None. Using as is.")
            else:
                 logger.info(f"Dataset Init: Feature names provided is not a list: {actual_feature_names}. Using as is.")

        logger.info(f"Dataset Init: Loading REY points from: {rey_points_path}")
        try:
            self.rey_data = pd.read_parquet(rey_points_path)
            logger.info(f"Dataset Init: Loaded {len(self.rey_data)} REY data points.")
            self.num_points_loaded = int(len(self.rey_data))
        except Exception as e:
            logger.error(f"Dataset Init: Failed to load REY points Parquet file: {rey_points_path}. Error: {e}")
            raise


        if "depth" not in self.rey_data.columns:
            logger.error("Dataset Init: 'depth' column is missing in REY points data. This is required.")
            raise KeyError("'depth' column is required in REY points Parquet data.")


        self.rey_data = self.rey_data.rename(columns={"depth": "depth_physical"})

        if max_depth_mbsf is not None:
            before_count = len(self.rey_data)
            self.rey_data = self.rey_data[self.rey_data["depth_physical"] <= float(max_depth_mbsf)].copy()
            after_count = len(self.rey_data)
            logger.info(
                f"Dataset Init: Applied max_depth_mbsf={max_depth_mbsf}. "
                f"Filtered REY points from {before_count} to {after_count}."
            )
        else:
            logger.info("Dataset Init: max_depth_mbsf not specified. No physical-depth upper bound will be applied.")


        depth_nonneg = np.clip(self.rey_data["depth_physical"].astype(float), 0.0, None)
        if max_depth_mbsf is not None and float(max_depth_mbsf) > 0:
            self.rey_data["depth_clipped"] = np.clip(depth_nonneg, 0.0, float(max_depth_mbsf))
        else:
            self.rey_data["depth_clipped"] = depth_nonneg


        depth_max_obs = float(np.nanmax(self.rey_data["depth_clipped"].to_numpy(dtype=float))) if len(self.rey_data) else 0.0
        depth_max_obs = depth_max_obs if depth_max_obs > 0 else 1.0
        self.max_depth_mbsf = float(max_depth_mbsf) if (max_depth_mbsf is not None and float(max_depth_mbsf) > 0) else depth_max_obs
        self.rey_data["z_geom_norm"] = self.rey_data["depth_clipped"] / self.max_depth_mbsf
        logger.info(f"Dataset Init: Physical-depth stats after filtering: max_depth_mbsf_used={self.max_depth_mbsf:.3f} (for z_geom_norm only).")


        self.sed_rate_nc_path = sed_rate_nc_path
        self.sed_rate_var_name = sed_rate_var_name
        self.has_time_depth = False  # Back-compat flag name (represents age/time proxy availability)
        self.has_age_proxy = False
        self.depth_feature_type = "depth_mbsf"
        self.max_age_ka_proxy = float(max_age_ka_proxy) if max_age_ka_proxy is not None else None

        if self.sed_rate_nc_path is not None and os.path.exists(self.sed_rate_nc_path):
            try:
                logger.info(f"Dataset Init: Loading sedimentation rate grid from: {self.sed_rate_nc_path}")
                sed_ds = open_dataset(self.sed_rate_nc_path)
                if self.sed_rate_var_name not in sed_ds.data_vars:
                    logger.warning(
                        f"Dataset Init: Sedimentation rate variable '{self.sed_rate_var_name}' not found in "
                        f"{self.sed_rate_nc_path}. Available variables: {list(sed_ds.data_vars)}. "
                        "Proceeding without age-proxy feature."
                    )
                else:
                    sed_field = sed_ds[self.sed_rate_var_name]
                    # NOTE: The sedimentation-rate grid is a regular lat/lon raster.


                    rey_lon = self.rey_data["lon"].astype(float).values
                    rey_lat = self.rey_data["lat"].astype(float).values

                    rey_lon_for_interp = ((rey_lon + 180.0) % 360.0) - 180.0

                    logger.info("Dataset Init: Interpolating sedimentation rate at REY point locations...")
                    sed_values = sed_field.interp(
                        lon=("points", rey_lon_for_interp),
                        lat=("points", rey_lat),
                        method="linear",
                    ).values
                    if np.isnan(sed_values).any():
                        logger.warning(
                            "Dataset Init: Linear interpolation for sedimentation rate produced NaNs; "
                            "falling back to nearest-neighbor for those points."
                        )
                        mask_nan = np.isnan(sed_values)
                        sed_values_nearest = sed_field.interp(
                            lon=("points", rey_lon_for_interp[mask_nan]),
                            lat=("points", rey_lat[mask_nan]),
                            method="nearest",
                        ).values
                        sed_values[mask_nan] = sed_values_nearest

                    sed_min = 1e-3
                    sed_values = np.asarray(sed_values, dtype=np.float64)
                    mask_bad = ~np.isfinite(sed_values)
                    if mask_bad.any():
                        logger.warning(
                            "Dataset Init: Sedimentation rate interpolation still has non-finite values "
                            f"after nearest fallback (n={int(mask_bad.sum())}); filling them with sed_min={sed_min}."
                        )
                        sed_values[mask_bad] = sed_min
                    sed_values = np.clip(sed_values, sed_min, None)
                    self.rey_data["sed_rate_raw"] = sed_values
                    # Age proxy in ka: sed_rate is assumed cm/ka, depth is meters.

                    self.rey_data["age_ka_proxy"] = (self.rey_data["depth_clipped"].astype(float) * 100.0) / self.rey_data["sed_rate_raw"]
                    self.has_time_depth = True
                    self.has_age_proxy = True
                    self.depth_feature_type = "age_ka_proxy"
                    logger.info(
                        "Dataset Init: Sedimentation rate interpolation complete (age_ka_proxy enabled). "
                        f"Min={float(np.nanmin(sed_values)):.4e}, "
                        f"Max={float(np.nanmax(sed_values)):.4e}, "
                        f"Median={float(np.nanmedian(sed_values)):.4e}."
                    )

                try:
                    sed_ds.close()
                except Exception:
                    pass
            except Exception as e_sed:
                try:
                    sed_ds.close()
                except Exception:
                    pass
                logger.warning(
                    f"Dataset Init: Failed to load or interpolate sedimentation rate from "
                    f"{self.sed_rate_nc_path}: {e_sed}. Proceeding with geometric depth only."
                )
        else:
            if self.sed_rate_nc_path is None:
                logger.info("Dataset Init: sed_rate_nc_path not provided. Using physical depth only (no age proxy).")
            else:
                logger.warning(
                    f"Dataset Init: sed_rate_nc_path='{self.sed_rate_nc_path}' does not exist. "
                    "Using physical depth only (no age proxy)."
                )


        if self.max_age_ka_proxy is not None:
            if not self.has_age_proxy or "age_ka_proxy" not in self.rey_data.columns:
                raise ValueError(
                    "Dataset Init: max_age_ka_proxy was provided, but age proxy could not be computed. "
                    "Ensure sedimentation_rate_grid_path points to a valid sedimentation-rate NetCDF."
                )
            before_count = len(self.rey_data)
            self.rey_data = self.rey_data[self.rey_data["age_ka_proxy"] <= float(self.max_age_ka_proxy)].copy()
            after_count = len(self.rey_data)
            logger.info(
                f"Dataset Init: Applied max_age_ka_proxy={self.max_age_ka_proxy} ka. "
                f"Filtered REY points from {before_count} to {after_count}."
            )
            if after_count == 0:
                raise ValueError("Dataset Init: No samples remain after max_age_ka_proxy filtering.")

        self.num_points_after_filters = int(len(self.rey_data))


        override = str(depth_feature_type_override).strip().lower() if depth_feature_type_override else None
        if override in ("age_ka_proxy", "age", "time"):
            if self.has_age_proxy and ("age_ka_proxy" in self.rey_data.columns):
                self.rey_data["depth"] = self.rey_data["age_ka_proxy"]
                self.depth_feature_type = "age_ka_proxy"
                logger.info("Dataset Init: depth_feature_type_override=age_ka_proxy (using age proxy as depth feature).")
            else:
                logger.warning(
                    "Dataset Init: depth_feature_type_override requested age_ka_proxy but it is unavailable; "
                    "falling back to physical depth."
                )
                self.rey_data["depth"] = self.rey_data["depth_clipped"]
                self.depth_feature_type = "depth_mbsf"
        elif override in ("depth_mbsf", "depth", "physical", "mbsf"):
            self.rey_data["depth"] = self.rey_data["depth_clipped"]
            self.depth_feature_type = "depth_mbsf"
            logger.info("Dataset Init: depth_feature_type_override=depth_mbsf (using physical depth as depth feature).")
        elif override in ("zero", "none", "constant0"):
            self.rey_data["depth"] = 0.0
            self.depth_feature_type = "zero"
            logger.info("Dataset Init: depth_feature_type_override=zero (depth feature set to constant 0).")
        else:
            if self.has_age_proxy and "age_ka_proxy" in self.rey_data.columns:
                self.rey_data["depth"] = self.rey_data["age_ka_proxy"]
                self.depth_feature_type = "age_ka_proxy"
                logger.info("Dataset Init: Using age proxy (age_ka_proxy, ka) as modeling depth feature.")
            else:
                self.rey_data["depth"] = self.rey_data["depth_clipped"]
                self.depth_feature_type = "depth_mbsf"
                logger.info("Dataset Init: Using physical depth (mbsf) as modeling depth feature.")


        self.rey_data["domain_weight"] = 1.0
        if self.domain_priority_enabled:
            if (not self.has_age_proxy) or ("age_ka_proxy" not in self.rey_data.columns):
                logger.warning(
                    "Dataset Init: domain_priority_enabled=True but age_ka_proxy is unavailable; "
                    "falling back to uniform domain weights."
                )
            else:
                in_domain = (self.rey_data["age_ka_proxy"].astype(float) <= float(self.training_age_domain_ka)).astype(float)
                if self.domain_priority_mode == "two_stage_finetune":
                    domain_w = in_domain
                else:
                    out_w = float(self.domain_priority_out_weight)
                    if (not np.isfinite(out_w)) or (out_w < 0.0):
                        out_w = 0.2
                    domain_w = np.where(in_domain > 0.5, 1.0, out_w).astype(float)
                self.rey_data["domain_weight"] = domain_w
                logger.info(
                    f"Dataset Init: Domain-priority weights enabled (mode={self.domain_priority_mode}, "
                    f"training_age_domain_ka={self.training_age_domain_ka:g}, out_weight={self.domain_priority_out_weight:g}). "
                    f"In-domain fraction={float(np.mean(in_domain)):.3f}"
                )


            if self.use_sample_weights and (not self.use_weighted_loss):
                self.rey_data["sample_weight"] = self.rey_data["domain_weight"].astype(float)


        depth_array = self.rey_data["depth"].astype(float).values
        finite_mask = np.isfinite(depth_array)
        if not np.all(finite_mask):
            num_bad = np.size(depth_array) - np.count_nonzero(finite_mask)
            logger.warning(
                f"Dataset Init: Detected {num_bad} non-finite depth-feature values after proxy computation. "
                "Replacing them with the median of finite depths."
            )
            if np.any(finite_mask):
                median_depth = float(np.nanmedian(depth_array[finite_mask]))
            else:
                median_depth = 0.0
            depth_array[~finite_mask] = median_depth
            self.rey_data["depth"] = depth_array

        logger.info(f"Dataset Init: Loading environmental grid from: {env_grid_path}")
        try:

            # This avoids accidentally promoting to float64 and holding a duplicate xarray copy in memory

            with open_dataset(env_grid_path) as _env_ds:
                temp_env_ds = _env_ds[actual_feature_names]
                logger.info(f"Dataset Init: Loaded environmental grid with variables: {list(temp_env_ds.data_vars)}")

                if 'longitude' in temp_env_ds.coords:
                    lon_coord_check = temp_env_ds['longitude']
                    min_lon_val_check = lon_coord_check.min().item()
                    if pd.notna(min_lon_val_check) and min_lon_val_check < 0:
                        logger.info(
                            f"Dataset Init:   Environmental grid longitude (min: {min_lon_val_check:.2f}) seems to be [-180, 180]. Converting to [0, 360]."
                        )
                        temp_env_ds = temp_env_ds.assign_coords(
                            longitude=xr.where(temp_env_ds['longitude'] < 0, temp_env_ds['longitude'] + 360, temp_env_ds['longitude'])
                        )
                        temp_env_ds = temp_env_ds.sortby('longitude')
                        logger.info(
                            f"Dataset Init:   Environmental grid longitude converted. New range: {temp_env_ds['longitude'].min().item():.2f} to {temp_env_ds['longitude'].max().item():.2f}"
                        )
                    elif pd.notna(lon_coord_check.max().item()) and lon_coord_check.max().item() >= 360:
                        logger.info(
                            f"Dataset Init:   Environmental grid longitude (max: {lon_coord_check.max().item():.2f}) may exceed or equal 360. Normalizing to [0, 360)."
                        )
                        temp_env_ds = temp_env_ds.assign_coords(longitude=temp_env_ds['longitude'] % 360)
                        temp_env_ds = temp_env_ds.sortby('longitude')
                        logger.info(
                            f"Dataset Init:   Environmental grid longitude normalized. New range: {temp_env_ds['longitude'].min().item():.2f} to {temp_env_ds['longitude'].max().item():.2f}"
                        )

                temp_env_ds = temp_env_ds.load()

                self.env_lat = temp_env_ds['latitude'].values
                self.env_lon = temp_env_ds['longitude'].values

                env_arrays = []
                for var_name in actual_feature_names:
                    try:
                        a = temp_env_ds[var_name].to_numpy()
                    except Exception:
                        a = temp_env_ds[var_name].values
                    env_arrays.append(np.asarray(a, dtype=np.float32))
                self.env_data_np = np.ascontiguousarray(np.stack(env_arrays, axis=0), dtype=np.float32)
                logger.info(
                    f"Dataset Init: Converted env_data to NumPy array with shape: {self.env_data_np.shape}, dtype={self.env_data_np.dtype}"
                )

            # Back-compat attribute: older code referenced the xarray Dataset.
            self.env_data_full_grid_xr = None
            self.lat_res = abs(self.env_lat[1] - self.env_lat[0]) if len(self.env_lat) > 1 else 0
            self.lon_res = abs(self.env_lon[1] - self.env_lon[0]) if len(self.env_lon) > 1 else 0
            logger.info(f"Dataset Init: Grid resolution (lat, lon): ({self.lat_res}, {self.lon_res})")

        except Exception as e:
            logger.error(f"Dataset Init: Failed to load environmental grid NetCDF file: {env_grid_path}. Error: {e}")
            raise

        self.feature_names = actual_feature_names
        self.patch_size = patch_size
        self.pad_rows = patch_size[0] // 2
        self.pad_cols = patch_size[1] // 2
        self.transform = transform
        self.target_transform = target_transform
        self.rey_target_transform_name = str(rey_target_transform_name or "log1p")
        self.rey_target_transform_scale = float(rey_target_transform_scale or 1000.0)
        self.rey_target_transform_hinge_beta = float(rey_target_transform_hinge_beta or 0.0)
        if self.rey_target_transform_name not in ("log1p", "asinh", "hinged_log"):
            raise ValueError(f"Unsupported REY target transform: {self.rey_target_transform_name}")
        self.depth_transform_method = depth_transform_method
        self.env_normalization_method = env_normalization_method
        self.augment_data = augment_data
        self.augmentation_degrees = augmentation_degrees
        self.augmentation_transform = None
        self.shift_augmentation_pixels = shift_augmentation_pixels
        self.use_weighted_loss = bool(use_weighted_loss)
        self.weighting_strategy = weighting_strategy
        self.max_weight = max_weight

        self.depth_weighting_scheme = depth_weighting_scheme
        self.use_iterative_imputer = use_iterative_imputer
        self.imputer = None
        self.add_coord_channels = bool(add_coord_channels)
        self.coord_feature_names = ["sin_lat", "cos_lat", "sin_lon", "cos_lon"] if self.add_coord_channels else []
        self.add_env_missingness_channels = bool(add_env_missingness_channels)
        self.env_missingness_feature_names = (
            [f"{name}__missing" for name in self.feature_names]
            if self.add_env_missingness_channels
            else []
        )
        if self.add_env_missingness_channels:
            logger.info(
                "Dataset Init: Environmental missingness mask channels enabled "
                f"({len(self.env_missingness_feature_names)} channels)."
            )

        if self.use_iterative_imputer:
            logger.info("Dataset Init: Initializing and fitting IterativeImputer...")


            num_channels, height, width = self.env_data_np.shape
            imputer_data = self.env_data_np.reshape(num_channels, -1).T


            valid_rows_mask = ~np.all(np.isnan(imputer_data), axis=1)
            self.imputer = IterativeImputer(max_iter=10, random_state=0)
            self.imputer.fit(imputer_data[valid_rows_mask])
            logger.info("Dataset Init: IterativeImputer fitted successfully.")

        if self.augment_data:
            self.augmentation_transform = transforms.RandomRotation(
                degrees=self.augmentation_degrees,
                interpolation=transforms.InterpolationMode.BILINEAR
            )
            logger.info(f"Dataset Init: Rotation augmentation enabled with random rotations up to {self.augmentation_degrees} degrees.")
        else:
            logger.info("Dataset Init: Rotation augmentation is disabled.")

        if self.shift_augmentation_pixels > 0:
            logger.info(f"Dataset Init: Shift/crop augmentation enabled with max {self.shift_augmentation_pixels} pixels.")
        else:
            logger.info("Dataset Init: Shift/crop augmentation is disabled.")

        logger.info(f"Dataset Init: Depth transform method set to: {self.depth_transform_method}")
        logger.info(f"Dataset Init: Environment normalization method set to: {self.env_normalization_method}")

        self.channel_means = []
        self.channel_stds = []
        self.channel_mins = []
        self.channel_maxs = []


        use_norm_override = False
        if norm_params_override is not None and isinstance(norm_params_override, dict):
            try:
                expected_len = len(self.feature_names) + 1

                rey_cfg = norm_params_override.get("rey", {}) if isinstance(norm_params_override.get("rey", {}), dict) else {}
                rey_mean = float(rey_cfg.get("mean"))
                rey_std = float(rey_cfg.get("std"))
                if rey_std < 1e-6:
                    rey_std = 1.0
                self.rey_target_transform_name = str(rey_cfg.get("transform", "log1p"))
                self.rey_target_transform_scale = float(rey_cfg.get("scale", 1000.0) or 1000.0)
                self.rey_target_transform_hinge_beta = float(rey_cfg.get("hinge_beta", 0.25) or 0.0)

                if self.env_normalization_method == "zscore":
                    means = list(norm_params_override.get("channel_means", []))
                    stds = list(norm_params_override.get("channel_stds", []))
                    if len(means) != expected_len or len(stds) != expected_len:
                        raise ValueError(f"zscore stats length mismatch (expected {expected_len}).")
                    self.channel_means = [float(x) for x in means]
                    self.channel_stds = [float(x) for x in stds]
                elif self.env_normalization_method == "minmax":
                    mins = list(norm_params_override.get("channel_mins", []))
                    maxs = list(norm_params_override.get("channel_maxs", []))
                    if len(mins) != expected_len or len(maxs) != expected_len:
                        raise ValueError(f"minmax stats length mismatch (expected {expected_len}).")
                    self.channel_mins = [float(x) for x in mins]
                    self.channel_maxs = [float(x) for x in maxs]
                else:
                    raise ValueError(f"Unknown env_normalization_method '{self.env_normalization_method}' for override.")

                self.rey_mean = rey_mean
                self.rey_std = rey_std

                self.norm_params = dict(norm_params_override)
                use_norm_override = True
                logger.info("Dataset Init: Using provided norm_params_override (no stats recomputation).")
            except Exception as e_override:
                logger.warning(f"Dataset Init: Invalid norm_params_override; falling back to recompute stats. Error: {e_override}")

        if use_norm_override:
            if self.use_weighted_loss:
                self._calculate_weights()
            logger.info(f"Dataset initialized successfully (norm override). Patch size: {self.patch_size}, Padding: ({self.pad_rows}, {self.pad_cols})")
            return


        all_feature_names_for_norm = self.feature_names + ['depth']

        feature_to_channel_idx = {str(n): int(i) for i, n in enumerate(self.feature_names)}


        original_depth_data = self.rey_data['depth']
        if self.depth_transform_method == "log1p":
            self.rey_data['depth'] = np.log1p(original_depth_data)
        elif self.depth_transform_method == "sqrt":
            self.rey_data['depth'] = np.sqrt(original_depth_data)

        if self.env_normalization_method == "zscore":
            logger.info("Dataset Init: Calculating per-channel statistics (Z-Score) for all features (env + depth)...")
            for var_name in all_feature_names_for_norm:
                if var_name == 'depth':
                    depth_vals = self.rey_data['depth'].astype(float).values    
                    channel_data_flat_valid = depth_vals[np.isfinite(depth_vals)]
                else:
                    is_onehot = "__cat_" in str(var_name)
                    if is_onehot:
                        ch_mean = 0.0
                        ch_std = 1.0
                        self.channel_means.append(ch_mean)
                        self.channel_stds.append(ch_std)
                        logger.info(f"Dataset Init:   Feature '{var_name}' (One-Hot): Mean={ch_mean:.4f}, Std={ch_std:.4f}")
                        continue
                    ch_idx = feature_to_channel_idx.get(str(var_name), None)
                    if ch_idx is None:
                        logger.warning(f"Dataset Init: Feature '{var_name}' missing channel index; defaulting stats mean=0,std=1.")
                        ch_mean = 0.0
                        ch_std = 1.0
                        self.channel_means.append(ch_mean)
                        self.channel_stds.append(ch_std)
                        continue
                    channel = self.env_data_np[int(ch_idx)]
                    ch_mean = float(np.nanmean(channel))
                    ch_std = float(np.nanstd(channel))
                    if not np.isfinite(ch_mean):
                        ch_mean = 0.0
                    if (not np.isfinite(ch_std)) or ch_std < 1e-6:
                        ch_std = 1.0
                    self.channel_means.append(ch_mean)
                    self.channel_stds.append(ch_std)
                    logger.info(f"Dataset Init:   Feature '{var_name}' (Z-Score): Mean={ch_mean:.4f}, Std={ch_std:.4f}")
                    continue

                if channel_data_flat_valid.size == 0:
                    ch_mean = 0.0
                    ch_std = 1.0
                    logger.warning(f"Dataset Init: Feature '{var_name}' has no valid values for Z-Score; defaulting mean=0,std=1.")
                else:
                    ch_mean = float(np.mean(channel_data_flat_valid))
                    ch_std = float(np.std(channel_data_flat_valid))
                if ch_std < 1e-6: ch_std = 1.0

                self.channel_means.append(ch_mean)
                self.channel_stds.append(ch_std)
                logger.info(f"Dataset Init:   Feature '{var_name}' (Z-Score): Mean={ch_mean:.4f}, Std={ch_std:.4f}")

        elif self.env_normalization_method == "minmax":
            logger.info("Dataset Init: Calculating per-channel statistics (Min-Max) for all features (env + depth)...")
            for var_name in all_feature_names_for_norm:
                if var_name == 'depth':
                    depth_vals = self.rey_data['depth'].astype(float).values    
                    channel_data_flat_valid = depth_vals[np.isfinite(depth_vals)]
                else:
                    is_onehot = "__cat_" in str(var_name)
                    if is_onehot:
                        ch_min, ch_max = 0.0, 1.0
                        self.channel_mins.append(ch_min)
                        self.channel_maxs.append(ch_max)
                        logger.info(f"Dataset Init:   Feature '{var_name}' (One-Hot): Min={ch_min:.4f}, Max={ch_max:.4f}")
                        continue
                    ch_idx = feature_to_channel_idx.get(str(var_name), None)
                    if ch_idx is None:
                        logger.warning(f"Dataset Init: Feature '{var_name}' missing channel index; defaulting stats min=0,max=1.")
                        ch_min, ch_max = 0.0, 1.0
                        self.channel_mins.append(ch_min)
                        self.channel_maxs.append(ch_max)
                        continue
                    channel = self.env_data_np[int(ch_idx)]
                    try:
                        ch_min = float(np.nanmin(channel))
                        ch_max = float(np.nanmax(channel))
                    except Exception:
                        ch_min, ch_max = 0.0, 1.0
                    if (not np.isfinite(ch_min)) or (not np.isfinite(ch_max)) or (ch_max - ch_min) < 1e-6:
                        ch_min, ch_max = 0.0, 1.0
                    self.channel_mins.append(ch_min)
                    self.channel_maxs.append(ch_max)
                    logger.info(f"Dataset Init:   Feature '{var_name}' (Min-Max): Min={ch_min:.4f}, Max={ch_max:.4f}")
                    continue

                if channel_data_flat_valid.size == 0:
                    ch_min, ch_max = 0.0, 1.0
                    logger.warning(f"Dataset Init: Feature '{var_name}' has no valid values for Min-Max; defaulting min=0,max=1.")
                else:
                    ch_min = float(np.min(channel_data_flat_valid))
                    ch_max = float(np.max(channel_data_flat_valid))
                if ch_max - ch_min < 1e-6: ch_min, ch_max = 0.0, 1.0

                self.channel_mins.append(ch_min)
                self.channel_maxs.append(ch_max)
                logger.info(f"Dataset Init:   Feature '{var_name}' (Min-Max): Min={ch_min:.4f}, Max={ch_max:.4f}")


        self.rey_data['depth'] = original_depth_data

        if self.task_type == "binary_classification":

            self.rey_mean, self.rey_std = 0.0, 1.0
            self.norm_params = {
                "rey": {
                    "mean": self.rey_mean,
                    "std": self.rey_std,
                    "transform": None,
                    "task_type": "binary_classification",
                    "threshold_ppm": float(self.classification_threshold_ppm),
                }
            }
            logger.info(
                "Dataset Init: Using binary classification target "
                f"(ΣREY >= {self.classification_threshold_ppm:g} ppm)."
            )
        elif self.task_type == "multiclass_classification":
            if not self.classification_label_col:
                raise ValueError("Dataset Init: multiclass_classification requires classification_label_col.")
            if self.classification_label_col not in self.rey_data.columns:
                raise ValueError(
                    "Dataset Init: multiclass_classification requires label column "
                    f"'{self.classification_label_col}' in rey_points.parquet."
                )

            labels = pd.to_numeric(self.rey_data[self.classification_label_col], errors="coerce")
            if labels.isna().any():
                n_bad = int(labels.isna().sum())
                raise ValueError(
                    f"Dataset Init: label column '{self.classification_label_col}' has {n_bad} missing/non-numeric values."
                )
            labels_int = labels.astype(int)
            if (labels_int < 0).any():
                raise ValueError(f"Dataset Init: label column '{self.classification_label_col}' contains negative labels.")
            n_classes = int(labels_int.max()) + 1
            if n_classes < 2:
                raise ValueError(f"Dataset Init: multiclass_classification requires >=2 classes, got {n_classes}.")
            self.rey_data[self.classification_label_col] = labels_int
            self.num_classes = int(n_classes)

            self.rey_mean, self.rey_std = 0.0, 1.0
            self.norm_params = {
                "rey": {
                    "mean": self.rey_mean,
                    "std": self.rey_std,
                    "transform": None,
                    "task_type": "multiclass_classification",
                    "num_classes": int(self.num_classes),
                    "class_names": self.classification_class_names,
                }
            }
            logger.info(
                "Dataset Init: Using multiclass classification target "
                f"(label_col='{self.classification_label_col}', num_classes={self.num_classes})."
            )
        else:
            if rey_norm:
                self.rey_mean, self.rey_std = rey_norm
                logger.info(f"Dataset Init: Using provided REY normalization: mean={self.rey_mean}, std={self.rey_std}")
            else:

                original_rey_data = self.rey_data['rey']
                transformed_rey_for_norm = self._transform_rey_target(original_rey_data)
                logger.info(
                    f"Dataset Init: Applying {self.rey_target_transform_name} "
                    "transformation for REY normalization."
                )

                self.rey_mean = transformed_rey_for_norm.mean()
                self.rey_std = transformed_rey_for_norm.std()
                if self.rey_std < 1e-6: self.rey_std = 1.0
                logger.info(
                    "Dataset Init: Calculated REY normalization "
                    f"(on {self.rey_target_transform_name} transformed data): "
                    f"mean={self.rey_mean:.4f}, std={self.rey_std:.4f}"
                )

            self.norm_params = {
                "rey": {
                    "mean": self.rey_mean,
                    "std": self.rey_std,
                    "transform": self.rey_target_transform_name,
                    "scale": self.rey_target_transform_scale,
                    "hinge_beta": self.rey_target_transform_hinge_beta,
                    "task_type": "regression",
                }
            }
        if self.env_normalization_method == "zscore":
            self.norm_params['channel_means'] = self.channel_means
            self.norm_params['channel_stds'] = self.channel_stds
        elif self.env_normalization_method == "minmax":
            self.norm_params['channel_mins'] = self.channel_mins
            self.norm_params['channel_maxs'] = self.channel_maxs
        logger.info(f"Dataset initialized successfully. Patch size: {self.patch_size}, Padding: ({self.pad_rows}, {self.pad_cols})")

        if self.sample_weight_col and self.sample_weight_col in self.rey_data.columns:
            weights = pd.to_numeric(self.rey_data[self.sample_weight_col], errors="coerce").astype(float).to_numpy()
            weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 1.0)
            weights = weights / max(float(np.mean(weights)), 1e-12)
            self.rey_data["sample_weight"] = np.clip(weights, 1e-6, float(self.max_weight))
            logger.info(
                f"Dataset Init: Using sample weights from column '{self.sample_weight_col}'. "
                f"Min={self.rey_data['sample_weight'].min():.4f}, "
                f"Max={self.rey_data['sample_weight'].max():.4f}, "
                f"Mean={self.rey_data['sample_weight'].mean():.4f}"
            )
        elif self.sample_weight_col:
            logger.warning(
                f"Dataset Init: sample_weight_col='{self.sample_weight_col}' was requested but is not present; "
                "falling back to default weighting behavior."
            )

        if self.use_weighted_loss and "sample_weight" not in self.rey_data.columns:
            self._calculate_weights()

    def _calculate_weights(self):
        """Calculates sample weights based on the REY target values."""
        logger.info(f"Calculating sample weights using '{self.weighting_strategy}' strategy...")
        rey_values = self.rey_data['rey'].values

        if self.weighting_strategy == 'linear':
            weights = rey_values
        elif self.weighting_strategy == 'log':
            weights = np.log1p(rey_values)
        else:
            logger.warning(f"Unknown weighting strategy: {self.weighting_strategy}. Defaulting to uniform weights.")
            weights = np.ones_like(rey_values)


        if self.depth_weighting_scheme:
            try:
                depth_phys = self.rey_data.get("depth_physical", self.rey_data.get("depth", None))
                if depth_phys is not None:
                    depth_vals = depth_phys.astype(float).values
                    depth_weights = np.ones_like(depth_vals, dtype=float)

                    if self.depth_weighting_scheme == "shallow_priority":

                        depth_weights[(depth_vals >= 0.0) & (depth_vals < 3.0)] = 1.4
                        depth_weights[(depth_vals >= 3.0) & (depth_vals < 6.0)] = 1.3
                        depth_weights[(depth_vals >= 6.0) & (depth_vals < 9.0)] = 1.15
                        depth_weights[(depth_vals >= 9.0) & (depth_vals < 12.0)] = 1.0
                        depth_weights[(depth_vals >= 12.0) & (depth_vals <= 15.0)] = 0.8
                    else:
                        logger.warning(f"Unknown depth_weighting_scheme '{self.depth_weighting_scheme}'. Ignoring depth weights.")
                        depth_weights = np.ones_like(depth_vals, dtype=float)

                    weights = weights * depth_weights
                    logger.info(
                        f"Depth-based weighting '{self.depth_weighting_scheme}' applied. "
                        f"Depth weight range: min={depth_weights.min():.3f}, max={depth_weights.max():.3f}"
                    )
            except Exception as e_dw:
                logger.warning(f"Failed to apply depth-based weighting scheme '{self.depth_weighting_scheme}': {e_dw}")


        if self.domain_priority_enabled and "domain_weight" in self.rey_data.columns:
            try:
                domain_w = self.rey_data["domain_weight"].astype(float).values
                if len(domain_w) == len(weights):
                    weights = weights * domain_w
                else:
                    logger.warning(
                        "Domain-priority weights are present but length mismatch was detected; ignoring domain weights."
                    )
            except Exception as e_dom:
                logger.warning(f"Failed to apply domain-priority weights: {e_dom}")


        weights = weights / np.mean(weights)


        weights = np.maximum(weights, 1e-6)


        weights = np.clip(weights, a_min=None, a_max=self.max_weight)

        self.rey_data['sample_weight'] = weights
        logger.info(f"Sample weights calculated and clipped at {self.max_weight}. Min: {np.min(weights):.4f}, Max: {np.max(weights):.4f}, Mean: {np.mean(weights):.4f}")

    def __len__(self):
        return len(self.rey_data)

    def _normalize_depth(self, depth):

        # Kept for potential backward compatibility checks but should not be used in the main flow.
        logger.warning("_normalize_depth is called but is obsolete in the new architecture.")
        return 0.0

    def _normalize_rey(self, rey):
        if self.task_type == "binary_classification":

            return 1.0 if float(rey) >= float(self.classification_threshold_ppm) else 0.0
        if self.task_type == "multiclass_classification":

            return float(rey)

        transformed_rey = self._transform_rey_target(rey)
        return (transformed_rey - self.rey_mean) / self.rey_std

    def _denormalize_rey(self, norm_rey):
        if self.task_type == "binary_classification":

            return np.asarray(norm_rey, dtype=float)
        if self.task_type == "multiclass_classification":

            return np.asarray(norm_rey, dtype=float)
        # Inverse of normalization, then inverse of transform.
        linear_denorm = (norm_rey * self.rey_std) + self.rey_mean
        return self._inverse_transform_rey_target(linear_denorm)

    def _transform_rey_target(self, rey):
        rey_arr = np.asarray(rey, dtype=float)
        rey_arr = np.clip(rey_arr, 0.0, None)
        transform = str(getattr(self, "rey_target_transform_name", "log1p"))
        if transform == "log1p":
            return np.log1p(rey_arr)
        if transform == "asinh":
            scale = max(float(getattr(self, "rey_target_transform_scale", 1000.0)), 1e-6)
            return np.arcsinh(rey_arr / scale)
        if transform == "hinged_log":
            threshold = max(float(getattr(self, "rey_target_transform_scale", 1000.0)), 1e-6)
            beta = max(float(getattr(self, "rey_target_transform_hinge_beta", 0.25)), 0.0)
            base = np.log1p(rey_arr)
            hinge_start = np.log1p(threshold)
            return base + beta * np.maximum(0.0, base - hinge_start)
        raise ValueError(f"Unsupported REY target transform: {transform}")

    def _inverse_transform_rey_target(self, transformed):
        arr = np.asarray(transformed, dtype=float)
        transform = str(getattr(self, "rey_target_transform_name", "log1p"))
        if transform == "log1p":
            return np.expm1(arr)
        if transform == "asinh":
            scale = max(float(getattr(self, "rey_target_transform_scale", 1000.0)), 1e-6)
            return np.sinh(arr) * scale
        if transform == "hinged_log":
            threshold = max(float(getattr(self, "rey_target_transform_scale", 1000.0)), 1e-6)
            beta = max(float(getattr(self, "rey_target_transform_hinge_beta", 0.25)), 0.0)
            hinge_start = np.log1p(threshold)
            base = np.where(
                arr <= hinge_start,
                arr,
                (arr + beta * hinge_start) / max(1.0 + beta, 1e-6),
            )
            return np.expm1(base)
        raise ValueError(f"Unsupported REY target transform: {transform}")

    def _denormalize_depth(self, norm_depth):
        """Denormalizes depth values back to their original scale."""
        transformed_depth = (norm_depth * self.depth_std) + self.depth_mean

        if self.depth_transform_method == "log1p":
            return np.expm1(transformed_depth)
        elif self.depth_transform_method == "sqrt":
            return np.square(transformed_depth)
        elif self.depth_transform_method == "none":
            return transformed_depth
        else:
            return np.expm1(transformed_depth)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        point_data = self.rey_data.iloc[idx]
        lon, lat = point_data['lon'], point_data['lat']
        depth = point_data['depth']
        if self.task_type == "multiclass_classification":
            rey = point_data[self.classification_label_col]
        else:
            rey = point_data['rey']

        if (self.use_weighted_loss or self.use_sample_weights) and 'sample_weight' in point_data:
            sample_weight = point_data['sample_weight']
        else:
            sample_weight = 1.0

        lat_idx = np.searchsorted(self.env_lat, lat - self.lat_res / 2)
        lon_idx = np.searchsorted(self.env_lon, lon - self.lon_res / 2)

        lat_idx = np.clip(lat_idx, 0, len(self.env_lat) - 1)
        lon_idx = np.clip(lon_idx, 0, len(self.env_lon) - 1)


        if self.shift_augmentation_pixels > 0:
            row_offset = np.random.randint(-self.shift_augmentation_pixels, self.shift_augmentation_pixels + 1)
            col_offset = np.random.randint(-self.shift_augmentation_pixels, self.shift_augmentation_pixels + 1)
            lat_idx += row_offset
            lon_idx += col_offset

        row_start = lat_idx - self.pad_rows
        row_end = lat_idx + self.pad_rows + 1
        col_start = lon_idx - self.pad_cols
        col_end = lon_idx + self.pad_cols + 1

        try:
            actual_row_start = max(0, row_start)
            actual_row_end = min(len(self.env_lat), row_end)
            actual_col_start = max(0, col_start)
            actual_col_end = min(len(self.env_lon), col_end)

            env_patch_extracted_np = self.env_data_np[:, actual_row_start:actual_row_end, actual_col_start:actual_col_end]

            pad_top = max(0, -row_start)
            pad_bottom = max(0, row_end - len(self.env_lat))
            pad_left = max(0, -col_start)
            pad_right = max(0, col_end - len(self.env_lon))

            env_patch_np = env_patch_extracted_np

            if any([pad_top > 0, pad_bottom > 0, pad_left > 0, pad_right > 0]):
                env_patch_np = np.pad(
                    env_patch_extracted_np,
                    ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
                    mode='edge'
                )

            if env_patch_np.shape[1:] != self.patch_size:
                 logger.warning(f"Initial patch extraction and padding resulted in shape {env_patch_np.shape[1:]} (target {self.patch_size}) for point idx {idx}. Applying fallback padding to fix shape.")
                 target_shape_patch = np.full((len(self.feature_names), self.patch_size[0], self.patch_size[1]), np.nan)
                 src_h, src_w = env_patch_np.shape[1], env_patch_np.shape[2]
                 tgt_h, tgt_w = self.patch_size
                 copy_h = min(src_h, tgt_h)
                 copy_w = min(src_w, tgt_w)
                 target_shape_patch[:, :copy_h, :copy_w] = env_patch_np[:, :copy_h, :copy_w]
                 env_patch_np = target_shape_patch
                 if np.isnan(env_patch_np).any():
                      logger.debug(f"NaNs present after shape correction for idx {idx}. Proceeding to NaN fill.")

            env_missingness_np = None
            if self.add_env_missingness_channels:
                env_missingness_np = np.isnan(env_patch_np).astype(np.float32)


            if np.isnan(env_patch_np).any():
                if self.use_iterative_imputer and self.imputer:
                    logger.debug(f"Dataset GetItem: NaNs detected in patch for idx {idx}. Using IterativeImputer.")
                    original_shape = env_patch_np.shape
                    patch_reshaped = env_patch_np.reshape(original_shape[0], -1).T


                    imputed_patch = self.imputer.transform(patch_reshaped)


                    env_patch_np = imputed_patch.T.reshape(original_shape)
                else:

                    logger.debug(f"Dataset GetItem: NaNs detected in patch for idx {idx}. Using fallback NaN fill.")
                    env_patch_np = numpy_ffill_bfill_channelwise(env_patch_np)
                    if np.isnan(env_patch_np).any():
                        logger.debug(f"Dataset GetItem: NaNs still present after spatial fill for idx {idx}. Filling with global channel means.")
                        for c in range(env_patch_np.shape[0]):
                            if np.isnan(env_patch_np[c, :, :]).any():
                                fill_value = self.channel_means[c] if self.env_normalization_method == "zscore" else self.channel_mins[c]
                                np.nan_to_num(env_patch_np[c, :, :], copy=False, nan=fill_value)

            env_patch_np_standardized = np.zeros_like(env_patch_np, dtype=np.float32)
            if self.env_normalization_method == "zscore":
                for c in range(env_patch_np.shape[0]):
                    channel_mean = self.channel_means[c]
                    channel_std = self.channel_stds[c]

                    env_patch_np_standardized[c, :, :] = (env_patch_np[c, :, :] - channel_mean) / channel_std
            elif self.env_normalization_method == "minmax":
                for c in range(env_patch_np.shape[0]):
                    channel_min = self.channel_mins[c]
                    channel_max = self.channel_maxs[c]
                    channel_range = channel_max - channel_min
                    if channel_range < 1e-6:
                        env_patch_np_standardized[c, :, :] = 0.0
                        logger.debug(f"Dataset GetItem: Channel {c} range is zero for Min-Max. Setting standardized patch to 0.0.")
                    else:
                        env_patch_np_standardized[c, :, :] = (env_patch_np[c, :, :] - channel_min) / channel_range
            else:
                logger.error(f"Dataset GetItem: Unknown env_normalization_method: {self.env_normalization_method}. Returning non-standardized patch for env data.")
                env_patch_np_standardized = env_patch_np.astype(np.float32)

            env_patch = torch.tensor(env_patch_np_standardized, dtype=torch.float32)

            if env_missingness_np is not None:
                env_missingness = torch.tensor(env_missingness_np, dtype=torch.float32)
                env_patch = torch.cat((env_patch, env_missingness), dim=0)

            if self.augmentation_transform:
                env_patch = self.augmentation_transform(env_patch)

        except Exception as e:
            logger.error(f"Error extracting or standardizing patch for point idx {idx} (Lon={lon}, Lat={lat}): {e}")
            env_patch_np_zeros = np.zeros((len(self.feature_names), self.patch_size[0], self.patch_size[1]), dtype=np.float32)
            env_patch_np_standardized_error = np.zeros_like(env_patch_np_zeros, dtype=np.float32)

            if self.env_normalization_method == "zscore" and self.channel_means and self.channel_stds and len(self.channel_means) == env_patch_np_zeros.shape[0] and len(self.channel_stds) == env_patch_np_zeros.shape[0]:
                for c in range(env_patch_np_zeros.shape[0]):
                    channel_mean = self.channel_means[c]
                    channel_std = self.channel_stds[c]
                    env_patch_np_standardized_error[c, :, :] = (env_patch_np_zeros[c, :, :] - channel_mean) / channel_std
            elif self.env_normalization_method == "minmax" and self.channel_mins and self.channel_maxs and len(self.channel_mins) == env_patch_np_zeros.shape[0] and len(self.channel_maxs) == env_patch_np_zeros.shape[0]:
                 for c in range(env_patch_np_zeros.shape[0]):
                    channel_min = self.channel_mins[c]
                    channel_max = self.channel_maxs[c]
                    channel_range = channel_max - channel_min
                    if channel_range < 1e-6:
                        env_patch_np_standardized_error[c, :, :] = 0.0
                    else:
                        env_patch_np_standardized_error[c, :, :] = (env_patch_np_zeros[c, :, :] - channel_min) / channel_range
            else:
                 logger.warning(f"Dataset GetItem (Error Fallback): Could not apply preferred normalization '{self.env_normalization_method}' due to missing or mismatched params. Defaulting to zeros.")
                 env_patch_np_standardized_error = env_patch_np_zeros
            env_patch = torch.tensor(env_patch_np_standardized_error, dtype=torch.float32)
            if self.add_env_missingness_channels:
                env_missingness = torch.ones(
                    (len(self.feature_names), self.patch_size[0], self.patch_size[1]),
                    dtype=torch.float32,
                )
                env_patch = torch.cat((env_patch, env_missingness), dim=0)


        transformed_depth_val = depth
        if self.depth_transform_method == "log1p":
            transformed_depth_val = np.log1p(depth)
        elif self.depth_transform_method == "sqrt":
            transformed_depth_val = np.sqrt(depth)

        norm_depth_val = 0.0
        depth_channel_idx = len(self.feature_names)
        if self.env_normalization_method == "zscore":
            norm_depth_val = (transformed_depth_val - self.channel_means[depth_channel_idx]) / self.channel_stds[depth_channel_idx]
        elif self.env_normalization_method == "minmax":
            depth_range = self.channel_maxs[depth_channel_idx] - self.channel_mins[depth_channel_idx]
            if depth_range > 1e-6:
                norm_depth_val = (transformed_depth_val - self.channel_mins[depth_channel_idx]) / depth_range


        depth_channel = torch.full((1, self.patch_size[0], self.patch_size[1]), float(norm_depth_val), dtype=torch.float32)


        combined_patch = torch.cat((env_patch, depth_channel), dim=0)


        if self.add_coord_channels:
            lon_val = float(lon)
            lat_val = float(lat)
            lon_norm = ((lon_val + 180.0) % 360.0) - 180.0
            lon_rad = np.deg2rad(lon_norm)
            lat_rad = np.deg2rad(lat_val)
            coord_vals = np.array(
                [np.sin(lat_rad), np.cos(lat_rad), np.sin(lon_rad), np.cos(lon_rad)],
                dtype=np.float32,
            )
            coord_channels = torch.from_numpy(coord_vals).view(4, 1, 1).expand(4, self.patch_size[0], self.patch_size[1])
            combined_patch = torch.cat((combined_patch, coord_channels), dim=0)

        if self.task_type == "multiclass_classification":
            target_rey = torch.tensor(int(self._normalize_rey(rey)), dtype=torch.int64)
        else:
            target_rey = torch.tensor(self._normalize_rey(rey), dtype=torch.float32)

        if self.transform:
            combined_patch = self.transform(combined_patch)
        if self.target_transform and self.task_type != "multiclass_classification":
            target_rey = self.target_transform(target_rey)

        sample_weight = torch.tensor(sample_weight, dtype=torch.float32)

        if self.task_type == "multiclass_classification" and self.soft_label_cols:
            soft_vals = pd.to_numeric(point_data[self.soft_label_cols], errors="coerce").to_numpy(dtype=np.float32)
            soft_vals = np.where(np.isfinite(soft_vals) & (soft_vals >= 0.0), soft_vals, 0.0)
            soft_sum = float(np.sum(soft_vals))
            if soft_sum <= 1e-12:
                soft_vals = np.zeros(len(self.soft_label_cols), dtype=np.float32)
                cls_idx = int(target_rey.item())
                if 0 <= cls_idx < len(soft_vals):
                    soft_vals[cls_idx] = 1.0
            else:
                soft_vals = soft_vals / soft_sum
            soft_target = torch.tensor(soft_vals, dtype=torch.float32)
            return combined_patch, target_rey.unsqueeze(0), sample_weight.unsqueeze(0), soft_target

        return combined_patch, target_rey.unsqueeze(0), sample_weight.unsqueeze(0)

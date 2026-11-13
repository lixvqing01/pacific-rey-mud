
import os
import time
import numpy as np
import torch
import xarray as xr
from core.utils.netcdf_io import open_dataset
from torch.amp import autocast
from global_land_mask import globe

from core.utils.logger_config import logger
from core.models.cnn_mlp import MixtureOfProfilesModel, HierarchicalMixtureOfProfilesModel
from core.data_handling.patch_utils import numpy_ffill_bfill_channelwise
from core.utils.config_models import TrainingArtifacts, PredictionGridSettings, NormalizationParams
from core.data.interpolation import interpolate_grid, iterative_imputer_interpolation


class DebugHook:
    """Detect anomalous tensors during a model forward pass."""
    def __init__(self, logger, layer_name):
        self.logger = logger
        self.layer_name = layer_name
        self.call_count = 0

    def check_tensor_anomalies(self, tensor, context=""):
        """Check a tensor for missing and infinite values."""
        if tensor is None:
            return


        if torch.isnan(tensor).any():
            self.logger.error(f"[{self.layer_name}] {context} contains NaN values! "
                            f"Tensor shape: {tensor.shape}, "
                            f"NaN count: {torch.isnan(tensor).sum()}")

        if torch.isinf(tensor).any():
            self.logger.error(f"[{self.layer_name}] {context} contains infinite values! "
                            f"Tensor shape: {tensor.shape}, "
                            f"Infinity count: {torch.isinf(tensor).sum()}")

    def check_batchnorm_input_variance(self, tensor):
        """Inspect the variance of BatchNorm input channels."""
        if tensor is None or tensor.dim() < 4:
            return


        batch_size, channels, height, width = tensor.shape


        tensor_reshaped = tensor.permute(1, 0, 2, 3).reshape(channels, -1)

        for ch in range(channels):
            channel_data = tensor_reshaped[ch]
            if len(channel_data) > 1:
                variance = torch.var(channel_data)
                if variance < 1e-8:
                    self.logger.warning(f"[{self.layer_name}] BatchNorm input channel {ch} has near-zero variance: {variance:.10f}")

    def __call__(self, module, input, output):
        """Apply the forward hook."""
        self.call_count += 1


        if isinstance(input, tuple) and len(input) > 0:
            input_tensor = input[0]
        else:
            input_tensor = input


        self.check_tensor_anomalies(input_tensor, f"input (call {self.call_count})")


        if isinstance(module, torch.nn.BatchNorm2d):
            self.check_batchnorm_input_variance(input_tensor)


        self.check_tensor_anomalies(output, f"output (call {self.call_count})")


def setup_model_debug_hooks(model, logger):
    """Register model debugging hooks."""
    hooks = []


    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            hook = module.register_forward_hook(DebugHook(logger, f"BatchNorm2d({name})"))
            hooks.append(hook)
            logger.info(f"Registered debug hook for BatchNorm2d layer {name} ")

    return hooks

class GridPredictor:
    def __init__(self, training_artifacts: TrainingArtifacts, env_gridded_nc_path: str, device, batch_size=64, progress_callback=None, pred_config=None):
        self.logger = logger
        self.device = device
        self.batch_size = batch_size
        self.progress_callback = progress_callback
        self.pred_config = pred_config
        self.enable_mc_dropout = bool(getattr(pred_config, "enable_mc_dropout", False))
        self.mc_samples = int(getattr(pred_config, "mc_dropout_samples", 0) or 0)
        self.depth_bins_m = getattr(pred_config, "depth_bins_m", None) or [0.0, 3.0, 6.0, 15.0]
        self.depth_transform = None

        self.logger.info(f"GridPredictor: Initializing... Device: {self.device}")


        self.logger.debug("GridPredictor: Unpacking training artifacts...")
        model_init_config = training_artifacts.model_init_config
        norm_params = training_artifacts.norm_params
        model_path = training_artifacts.model_path


        self.target_task_type = str(getattr(norm_params.rey, "task_type", None) or "regression").lower()
        if self.target_task_type not in ("regression", "binary_classification", "multiclass_classification"):
            self.logger.warning(
                f"GridPredictor: Unknown target task_type='{self.target_task_type}', falling back to 'regression'."
            )
            self.target_task_type = "regression"

        self.patch_size = tuple(model_init_config.patch_size)
        self.pad_rows = self.patch_size[0] // 2
        self.pad_cols = self.patch_size[1] // 2

        self.feature_names = model_init_config.feature_names
        self.num_features = len(self.feature_names)
        self.coord_feature_names = getattr(model_init_config, "coord_feature_names", None) or []
        self.coord_feature_dim = int(getattr(model_init_config, "coord_feature_dim", 0) or 0)
        self.add_coord_channels = bool(self.coord_feature_dim > 0)
        self.add_env_missingness_channels = bool(getattr(model_init_config, "add_env_missingness_channels", False))
        self.env_missingness_feature_names = getattr(model_init_config, "env_missingness_feature_names", None) or []

        self.sed_rate_index = None
        try:
            self.sed_rate_index = self.feature_names.index("Average sedimentation rate")
        except ValueError:
            self.sed_rate_index = None
        self.logger.debug(f"GridPredictor: Using {self.num_features} features: {self.feature_names}")


        self.depth_feature_type = getattr(norm_params, "depth_feature_type", None)
        self.depth_min = getattr(norm_params, "depth_feature_min", None)
        self.depth_max = getattr(norm_params, "depth_feature_max", None)
        if self.depth_min is None and self.depth_max is None:
            # Backward-compatible aliases for older artifacts.
            self.depth_min = getattr(norm_params, "depth_min", None)
            self.depth_max = getattr(norm_params, "depth_max", None)

        if self.depth_feature_type is None:


            self.depth_feature_type = "legacy_time_depth" if self.sed_rate_index is not None else "depth_mbsf"

        if self.depth_min is not None and self.depth_max is not None:
            self.logger.info(
                "GridPredictor: Loaded depth-feature range from training artifacts: "
                f"type={self.depth_feature_type}, min={self.depth_min}, max={self.depth_max}"
            )


        self.logger.debug("GridPredictor: Loading model...")
        model_type = getattr(model_init_config, "model_type", "mixture_profiles").lower()
        hierarchical_multiclass = bool(getattr(model_init_config, "hierarchical_multiclass", False))
        self.hierarchical_multiclass = bool(hierarchical_multiclass) and (self.target_task_type == "multiclass_classification")
        self.hierarchical_grade_multitask = bool(getattr(model_init_config, "hierarchical_grade_multitask", False)) and (
            self.hierarchical_multiclass
        )

        # Classification head config (kept backward-compatible with older regression-only artifacts)
        num_classes = int(getattr(model_init_config, "num_classes", 1) or 1)
        class_head_hidden_dim = int(getattr(model_init_config, "class_head_hidden_dim", 64) or 64)
        use_high_grade_ranking_head = bool(getattr(model_init_config, "use_high_grade_ranking_head", False))
        high_grade_ranking_head_hidden_dim = int(
            getattr(model_init_config, "high_grade_ranking_head_hidden_dim", 64) or 64
        )
        if self.target_task_type == "multiclass_classification":
            if num_classes < 2:
                raise ValueError(
                    "GridPredictor: multiclass_classification requires num_classes>=2 in training artifacts."
                )
        else:

            num_classes = 1

        common_kwargs = {
            "input_channels": model_init_config.input_channels,
            "device": self.device,
            "cnn_filters": model_init_config.cnn_filters,
            "kernel_size": model_init_config.kernel_size,
            "mlp_hidden_layers": model_init_config.mlp_hidden_layers,
            "dropout_prob": model_init_config.dropout_prob,
            "linear_attention_heads": getattr(model_init_config, 'linear_attention_heads', 4),
            "use_linear_attention": bool(getattr(model_init_config, "use_linear_attention", True)),

            "categorical_channel_indices": getattr(model_init_config, "categorical_channel_indices", None),
            "categorical_compress_dim": getattr(model_init_config, "categorical_compress_dim", None),
            "categorical_compress_dropout": getattr(model_init_config, "categorical_compress_dropout", None),

            "use_multiscale_env": getattr(model_init_config, "use_multiscale_env", False),
            "multiscale_kernel_size": getattr(model_init_config, "multiscale_kernel_size", 5),
            "use_film": getattr(model_init_config, "use_film", False),
            "film_embedding_dim": getattr(model_init_config, "film_embedding_dim", 32),
            "depth_feature_order": getattr(model_init_config, "depth_feature_order", 1),

            "coord_feature_dim": self.coord_feature_dim,
            "use_coord_branch": bool(getattr(model_init_config, "use_coord_branch", False)),
            "coord_branch_hidden_dim": int(getattr(model_init_config, "coord_branch_hidden_dim", 8) or 8),
            "coord_branch_dropout": float(getattr(model_init_config, "coord_branch_dropout", 0.5) or 0.5),
            "coord_branch_weight": float(getattr(model_init_config, "coord_branch_weight", 1.0) or 1.0),

            "num_classes": num_classes,
            "class_head_hidden_dim": class_head_hidden_dim,
            "use_high_grade_ranking_head": use_high_grade_ranking_head,
            "high_grade_ranking_head_hidden_dim": high_grade_ranking_head_hidden_dim,
        }
        if model_type == "mixture_profiles":
            num_profiles = getattr(model_init_config, "num_profiles", 3) or 3
            env_embedding_dim = getattr(model_init_config, "env_embedding_dim", 64) or 64
            profile_hidden_layers = getattr(model_init_config, "profile_hidden_layers", None) or [32, 16]
            if self.target_task_type == "multiclass_classification" and hierarchical_multiclass:
                kwargs_h = dict(common_kwargs)
                kwargs_h.pop("num_classes", None)
                kwargs_h.pop("class_head_hidden_dim", None)
                num_subclasses = 2 if self.hierarchical_grade_multitask else max(2, int(num_classes) - 1)
                self.logger.info(
                    "GridPredictor: Using HierarchicalMixtureOfProfilesModel "
                    f"(mode={'grade_multitask_3head' if self.hierarchical_grade_multitask else 'legacy_2head'}, "
                    f"num_classes={int(num_classes)}, num_subclasses={int(num_subclasses)})."
                )
                self.model = HierarchicalMixtureOfProfilesModel(
                    num_profiles=num_profiles,
                    env_embedding_dim=env_embedding_dim,
                    profile_hidden_layers=profile_hidden_layers,
                    num_subclasses=int(num_subclasses),
                    endmember_head_hidden_dim=int(getattr(model_init_config, "endmember_head_hidden_dim", 64) or 64),
                    subclass_head_hidden_dim=int(getattr(model_init_config, "subclass_head_hidden_dim", 64) or 64),
                    hierarchical_grade_multitask=bool(self.hierarchical_grade_multitask),
                    grade_head_hidden_dim=int(getattr(model_init_config, "grade_head_hidden_dim", 64) or 64),
                    **kwargs_h,
                )
            else:
                self.logger.info("GridPredictor: Using MixtureOfProfilesModel (Mixture-of-Profiles Env×Depth).")
                self.model = MixtureOfProfilesModel(
                    num_profiles=num_profiles,
                    env_embedding_dim=env_embedding_dim,
                    profile_hidden_layers=profile_hidden_layers,
                    **common_kwargs,
                )
        else:
            raise ValueError(
                f"GridPredictor: Unsupported model_type='{model_type}'. "
                "Only 'mixture_profiles' is supported for prediction."
            )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()
        self.logger.debug(f"GridPredictor: Model loaded from {model_path} and set to eval mode.")

        self.debug_hooks = []
        if bool(getattr(self.pred_config, "enable_debug_hooks", False)):
            self.logger.info("GridPredictor: Setting up debug hooks for model...")
            self.debug_hooks = setup_model_debug_hooks(self.model, self.logger)
            self.logger.info(f"GridPredictor: Registered {len(self.debug_hooks)} debug hooks.")
        else:
            self.logger.debug("GridPredictor: Debug hooks disabled.")


        self.logger.debug(f"GridPredictor: Loading environmental data from {env_gridded_nc_path}...")
        env_ds_full = open_dataset(env_gridded_nc_path)[self.feature_names].load()

        if pred_config and pred_config.use_iterative_imputer:
            self.logger.info("GridPredictor: Performing iterative imputer interpolation...")
            env_ds_full = iterative_imputer_interpolation(
                dataset=env_ds_full,
                variables=self.feature_names
            )
            self.logger.info("GridPredictor: Iterative imputer interpolation complete.")
        elif pred_config and pred_config.perform_interpolation:
            self.logger.info(f"GridPredictor: Performing interpolation with method '{pred_config.interpolation_method}'...")
            env_ds_full = interpolate_grid(
                dataset=env_ds_full,
                method=pred_config.interpolation_method,
                variables=self.feature_names
            )
            self.logger.info("GridPredictor: Interpolation complete.")
        else:
            self.logger.info("GridPredictor: Interpolation is disabled.")

        if 'longitude' in env_ds_full.coords:
            lon_coord = env_ds_full['longitude']
            if lon_coord.min().item() < 0:
                self.logger.debug("GridPredictor: Converting environmental grid longitude from -180/180 to 0/360 range.")
                env_ds_full = env_ds_full.assign_coords(longitude=xr.where(lon_coord < 0, lon_coord + 360, lon_coord))
                env_ds_full = env_ds_full.sortby('longitude')

        self.env_data_np = env_ds_full.to_array(dim="variable").data
        self.env_lat = env_ds_full['latitude'].values
        self.env_lon = env_ds_full['longitude'].values
        self.lat_res = abs(self.env_lat[1] - self.env_lat[0]) if len(self.env_lat) > 1 else 0
        self.lon_res = abs(self.env_lon[1] - self.env_lon[0]) if len(self.env_lon) > 1 else 0
        env_ds_full.close()
        self.logger.debug(f"GridPredictor: Environmental data loaded. Shape: {self.env_data_np.shape}, Lon range: [{self.env_lon.min() if len(self.env_lon)>0 else 'N/A'}, {self.env_lon.max() if len(self.env_lon)>0 else 'N/A'}]")


        if hasattr(norm_params, 'channel_means') and hasattr(norm_params, 'channel_stds'):
            self.logger.debug("GridPredictor: Loading unified normalization parameters (new format).")
            self.channel_means = np.array(norm_params.channel_means, dtype=float)
            self.channel_stds = np.array(norm_params.channel_stds, dtype=float)
            self.depth_transform = getattr(norm_params, "depth_transform_method", None)
        elif hasattr(norm_params, 'env_channel_means') and hasattr(norm_params, 'depth'):
            self.logger.debug("GridPredictor: Loading separated normalization parameters (old format) and unifying them.")
            self.channel_means = np.array(list(norm_params.env_channel_means) + [norm_params.depth['mean']], dtype=float)
            self.channel_stds = np.array(list(norm_params.env_channel_stds) + [norm_params.depth['std']], dtype=float)
            self.depth_transform = getattr(norm_params.depth, "transform_method", None)
        else:
            raise ValueError("NormalizationParams structure in artifacts is unrecognized.")


        if not self.depth_transform:
            self.depth_transform = "log1p"
            self.logger.warning("GridPredictor: depth_transform_method missing in artifacts; defaulting to 'log1p' (training default).")

        self.rey_mean = norm_params.rey.mean
        self.rey_std = norm_params.rey.std
        self.rey_transform = norm_params.rey.transform
        self.rey_transform_scale = float(getattr(norm_params.rey, "scale", 1000.0) or 1000.0)
        self.rey_transform_hinge_beta = float(getattr(norm_params.rey, "hinge_beta", 0.25) or 0.0)
        self.target_task_type = str(getattr(norm_params.rey, "task_type", None) or "regression").lower()
        if self.target_task_type not in ("regression", "binary_classification", "multiclass_classification"):
            self.logger.warning(
                f"GridPredictor: Unknown target task_type='{self.target_task_type}', "
                "falling back to 'regression'."
            )
            self.target_task_type = "regression"
        self.target_threshold_ppm = getattr(norm_params.rey, "threshold_ppm", None)
        self.target_num_classes = int(getattr(norm_params.rey, "num_classes", 0) or 0)
        self.target_class_names = getattr(norm_params.rey, "class_names", None)
        self.logger.debug(f"GridPredictor: REY denormalization configured with transform: '{self.rey_transform}'")
        self.logger.debug("GridPredictor: Normalization parameters stored.")

    @staticmethod
    def convert_lon_range(lon_array, target_range='0_360'):
        if target_range == '0_360':
            return np.where(lon_array < 0, lon_array + 360, lon_array)
        return lon_array

    def _denormalize_rey(self, norm_rey_values):
        norm_rey_np = np.asarray(norm_rey_values).astype(np.float64)
        linear_denorm = (norm_rey_np * self.rey_std) + self.rey_mean
        self.logger.debug(f"_denormalize_rey: linear_denorm stats: min={np.min(linear_denorm)}, max={np.max(linear_denorm)}, mean={np.mean(linear_denorm)}")
        if self.rey_transform == 'log1p':
            return np.expm1(linear_denorm)
        if self.rey_transform == 'asinh':
            return np.sinh(linear_denorm) * max(float(self.rey_transform_scale), 1e-6)
        if self.rey_transform == 'hinged_log':
            threshold = max(float(self.rey_transform_scale), 1e-6)
            beta = max(float(self.rey_transform_hinge_beta), 0.0)
            hinge_start = np.log1p(threshold)
            base = np.where(
                linear_denorm <= hinge_start,
                linear_denorm,
                (linear_denorm + beta * hinge_start) / max(1.0 + beta, 1e-6),
            )
            return np.expm1(base)
        else:
            return linear_denorm

    def _decode_model_output(self, preds_norm: np.ndarray, depth_factors: list[float] | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Decode model outputs into physical space:
        - regression: denormalize to ppm and apply multiplicative depth attenuation.
        - binary_classification: apply depth attenuation on logits (toward 0), then sigmoid -> probability.

        Returns (decoded_mean, decoded_std_or_None).
        """
        if self.target_task_type == "binary_classification":
            logits = np.asarray(preds_norm, dtype=np.float64)
            if depth_factors:
                logits = logits * np.asarray(depth_factors, dtype=np.float64)
            logits = np.clip(logits, -60.0, 60.0)
            prob = 1.0 / (1.0 + np.exp(-logits))
            return prob.astype(np.float64), None


        preds_denorm = self._denormalize_rey(preds_norm)
        if depth_factors:
            preds_denorm = preds_denorm * np.asarray(depth_factors, dtype=np.float64)
        return preds_denorm.astype(np.float64), None

    def _enable_dropout(self):
        """
        Enable dropout layers while keeping BatchNorm layers in eval mode.
        Used for MC Dropout inference.
        """
        self.model.eval()
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()
            elif isinstance(module, torch.nn.BatchNorm2d):
                module.eval()

    def _depth_extrapolation_factor(self, depth_feature_val: float) -> float:
        """
        Compute an attenuation factor for predictions at a given depth
        based on the depth-feature range observed during training.

        - If depth_min/max are not available or the value lies within
          [depth_min, depth_max], returns 1.0.
        - Outside this range, applies a smooth exponential decay based
          on relative distance to the nearest bound.
        """
        if self.depth_min is None or self.depth_max is None:
            return 1.0
        try:
            if not np.isfinite(depth_feature_val):
                return 1.0
        except Exception:
            return 1.0

        try:
            d_min = float(self.depth_min)
            d_max = float(self.depth_max)
            if d_max <= d_min:
                return 1.0
            if d_min <= depth_feature_val <= d_max:
                return 1.0

            if depth_feature_val < d_min:
                delta = d_min - depth_feature_val
            else:
                delta = depth_feature_val - d_max

            depth_span = d_max - d_min

            factor = float(np.exp(-delta / depth_span))
            return max(0.0, min(1.0, factor))
        except Exception as e:
            self.logger.warning(
                "GridPredictor: Failed to compute depth-feature extrapolation factor "
                f"for value={depth_feature_val} (type={self.depth_feature_type}): {e}"
            )
            return 1.0

    def predict_3d_grid(self, pred_grid_settings: PredictionGridSettings):

        return self._predict_3d_grid_v2(pred_grid_settings)

        bounds = pred_grid_settings.geospatial_bounds
        if bounds and all(v is not None for v in [bounds.min_lon, bounds.max_lon, bounds.min_lat, bounds.max_lat]):
            self.logger.debug(f"Using user-defined geospatial bounds: {bounds}")
            min_lon, max_lon, min_lat, max_lat = bounds.min_lon, bounds.max_lon, bounds.min_lat, bounds.max_lat
        else:
            self.logger.debug("Geospatial bounds not provided or incomplete. Using full extent of environmental data.")
            min_lon, max_lon = self.env_lon.min(), self.env_lon.max()
            min_lat, max_lat = self.env_lat.min(), self.env_lat.max()

        lon_res = pred_grid_settings.lon_res
        lat_res = pred_grid_settings.lat_res
        depth_levels = pred_grid_settings.depth_levels

        max_lon_adjusted = max_lon
        if max_lon < min_lon:
            self.logger.debug(f"Cross-meridian longitude detected. Adjusting max_lon from {max_lon} to {max_lon + 360}.")
            max_lon_adjusted = max_lon + 360

        num_lon_points = int(np.floor((max_lon_adjusted - min_lon) / lon_res)) + 1
        num_lat_points = int(np.floor((max_lat - min_lat) / lat_res)) + 1

        pred_lons_raw = np.linspace(min_lon, min_lon + (num_lon_points - 1) * lon_res, num_lon_points)
        pred_lons = (pred_lons_raw + 180) % 360 - 180
        pred_lats = np.linspace(min_lat, min_lat + (num_lat_points - 1) * lat_res, num_lat_points)

        land_mask = None
        if self.pred_config and self.pred_config.use_land_mask:
            self.logger.info("Generating land mask for the prediction grid...")
            lon_grid, lat_grid = np.meshgrid(pred_lons, pred_lats)
            land_mask = globe.is_land(lat_grid, lon_grid)
            self.logger.info(f"Land mask generated. Land points: {np.sum(land_mask)}/{land_mask.size}")

        pred_lons_0_360 = self.convert_lon_range(pred_lons, target_range='0_360')

        prediction_points_coords = [
            (lon_v, lat_v, depth_v, (lat_idx, lon_idx))
            for lat_idx, lat_v in enumerate(pred_lats)
            for lon_idx, lon_v in enumerate(pred_lons_0_360)
            for depth_v in depth_levels
        ]

        if not prediction_points_coords:
            self.logger.warning("GridPredictor: No 3D prediction points generated based on grid definition.")
            return None

        num_total_points = len(prediction_points_coords)
        self.logger.debug(f"GridPredictor: Total {num_total_points} 3D points for prediction.")
        all_grid_predictions_denorm_array = np.full(num_total_points, np.nan)
        all_grid_uncertainty_array = np.full(num_total_points, np.nan) if (self.enable_mc_dropout and self.mc_samples > 1) else None

        for i in range(0, num_total_points, self.batch_size):
            batch_coords = prediction_points_coords[i:i + self.batch_size]

            batch_combined_patches_list = []
            batch_depth_factors = []
            valid_point_indices_in_batch = []

            for idx_in_batch, (lon_val, lat_val, depth_val, (lat_idx_grid, lon_idx_grid)) in enumerate(batch_coords):
                if land_mask is not None and land_mask[lat_idx_grid, lon_idx_grid]:
                    continue

                valid_point_indices_in_batch.append(idx_in_batch)

                lat_idx = np.searchsorted(self.env_lat, lat_val - self.lat_res / 2)
                lon_idx = np.searchsorted(self.env_lon, lon_val - self.lon_res / 2)
                lat_idx = np.clip(lat_idx, 0, len(self.env_lat) - 1)
                lon_idx = np.clip(lon_idx, 0, len(self.env_lon) - 1)

                row_start = lat_idx - self.pad_rows
                row_end = lat_idx + self.pad_rows + 1
                col_start = lon_idx - self.pad_cols
                col_end = lon_idx + self.pad_cols + 1

                actual_row_start = max(0, row_start)
                actual_row_end = min(len(self.env_lat), row_end)
                actual_col_start = max(0, col_start)
                actual_col_end = min(len(self.env_lon), col_end)

                env_patch_extracted_np = self.env_data_np[:, actual_row_start:actual_row_end, actual_col_start:actual_col_end]


                nan_count_in_patch = np.isnan(env_patch_extracted_np).sum()
                if nan_count_in_patch > 0:
                    self.logger.debug(f"Depth: {depth_val}m, Lat: {lat_val:.2f}, Lon: {lon_val:.2f} - Initial patch has {nan_count_in_patch} NaNs.")

                pad_top = max(0, -row_start)
                pad_bottom = max(0, row_end - len(self.env_lat))
                pad_left = max(0, -col_start)
                pad_right = max(0, col_end - len(self.env_lon))

                current_env_patch_np = env_patch_extracted_np
                if any([pad_top > 0, pad_bottom > 0, pad_left > 0, pad_right > 0]):
                    current_env_patch_np = np.pad(
                        env_patch_extracted_np,
                        ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
                        mode='edge'
                    )

                if current_env_patch_np.shape[1:] != self.patch_size:
                    target_shape_patch = np.full((self.num_features, self.patch_size[0], self.patch_size[1]), np.nan)
                    copy_h = min(current_env_patch_np.shape[1], self.patch_size[0])
                    copy_w = min(current_env_patch_np.shape[2], self.patch_size[1])
                    target_shape_patch[:, :copy_h, :copy_w] = current_env_patch_np[:, :copy_h, :copy_w]
                    current_env_patch_np = target_shape_patch

                env_missingness_np = None
                if self.add_env_missingness_channels:
                    env_missingness_np = np.isnan(current_env_patch_np).astype(np.float32)

                if np.isnan(current_env_patch_np).any():
                    current_env_patch_np = numpy_ffill_bfill_channelwise(current_env_patch_np)

                if np.isnan(current_env_patch_np).any():
                    for c_idx in range(self.num_features):
                        if np.isnan(current_env_patch_np[c_idx, :, :]).any():
                            current_env_patch_np[c_idx, :, :] = np.nan_to_num(
                                current_env_patch_np[c_idx, :, :],
                                copy=False,
                                nan=self.channel_means[c_idx]
                            )

                env_patch_stdized = np.zeros_like(current_env_patch_np, dtype=np.float32)
                for c_idx in range(self.num_features):
                    ch_m, ch_s = self.channel_means[c_idx], self.channel_stds[c_idx]
                    env_patch_stdized[c_idx, :, :] = (current_env_patch_np[c_idx, :, :] - ch_m) / (ch_s if ch_s > 1e-6 else 1.0)

                depth_channel_idx = self.num_features
                depth_mean = self.channel_means[depth_channel_idx]
                depth_std = self.channel_stds[depth_channel_idx]

                # --- Build the model's depth feature (physical depth or age proxy) ---
                depth_feature_raw = float(depth_val)
                if self.depth_feature_type in ("age_ka_proxy", "legacy_time_depth"):
                    if self.sed_rate_index is None:
                        self.logger.warning(
                            "GridPredictor: depth_feature_type requires sedimentation rate, "
                            "but 'Average sedimentation rate' is not present in env features; "
                            "falling back to physical depth."
                        )
                    else:
                        sed_rate_val = self.env_data_np[self.sed_rate_index, lat_idx, lon_idx]
                        if not np.isfinite(sed_rate_val):
                            sed_rate_val = (
                                self.channel_means[self.sed_rate_index]
                                if self.sed_rate_index < len(self.channel_means)
                                else 1.0
                            )
                        sed_rate_val = max(float(sed_rate_val), 1e-3)
                        if self.depth_feature_type == "age_ka_proxy":
                            # sed_rate is cm/ka, depth is meters: age_ka = depth(m)*100 / sed(cm/ka)
                            depth_feature_raw = float(depth_val) * 100.0 / sed_rate_val
                        else:

                            depth_feature_raw = float(depth_val) / sed_rate_val


                if self.depth_transform == "log1p":
                    depth_val_transformed = np.log1p(depth_feature_raw)
                elif self.depth_transform == "sqrt":
                    depth_val_transformed = np.sqrt(max(depth_feature_raw, 0.0))
                else:
                    depth_val_transformed = depth_feature_raw

                norm_d_val = (depth_val_transformed - depth_mean) / (depth_std if depth_std > 1e-6 else 1.0)

                depth_channel = np.full((1, self.patch_size[0], self.patch_size[1]), norm_d_val, dtype=np.float32)
                env_model_channels = (
                    np.concatenate((env_patch_stdized, env_missingness_np), axis=0)
                    if env_missingness_np is not None
                    else env_patch_stdized
                )
                combined_patch = np.concatenate((env_model_channels, depth_channel), axis=0)


                if self.add_coord_channels and self.coord_feature_dim == 4:
                    lon_norm = ((float(lon_val) + 180.0) % 360.0) - 180.0
                    lon_rad = np.deg2rad(lon_norm)
                    lat_rad = np.deg2rad(float(lat_val))
                    coord_vals = np.array(
                        [np.sin(lat_rad), np.cos(lat_rad), np.sin(lon_rad), np.cos(lon_rad)],
                        dtype=np.float32,
                    )
                    coord_channels = np.tile(coord_vals.reshape(4, 1, 1), (1, self.patch_size[0], self.patch_size[1]))
                    combined_patch = np.concatenate((combined_patch, coord_channels), axis=0)


                if np.isnan(combined_patch).any():
                    self.logger.warning(f"Depth: {depth_val}m - Patch still contains NaNs after all processing steps. Applying final nan_to_num.")

                    final_patch = np.copy(combined_patch)

                    num_total_channels = final_patch.shape[0]
                    for c_idx in range(num_total_channels):
                        if np.isnan(final_patch[c_idx, :, :]).any():
                            fill_value = self.channel_means[c_idx] if c_idx < len(self.channel_means) else 0
                            final_patch[c_idx, :, :] = np.nan_to_num(
                                final_patch[c_idx, :, :],
                                copy=False,
                                nan=fill_value
                            )
                    batch_combined_patches_list.append(final_patch)
                else:
                    batch_combined_patches_list.append(combined_patch)


                depth_factor = self._depth_extrapolation_factor(depth_feature_raw)
                batch_depth_factors.append(depth_factor)

            if batch_combined_patches_list:
                batch_combined_patches_tensor = torch.tensor(np.array(batch_combined_patches_list), dtype=torch.float32).to(self.device)

                if self.enable_mc_dropout and self.mc_samples > 1:
                    mc_preds_denorm = []
                    for _ in range(self.mc_samples):
                        self._enable_dropout()
                        with torch.no_grad(), autocast(device_type=self.device.type, enabled=torch.cuda.is_available()):
                            preds_norm_mc = self.model(batch_combined_patches_tensor)
                        preds_denorm_mc = self._denormalize_rey(preds_norm_mc.squeeze(-1).cpu().numpy())
                        if batch_depth_factors:
                            depth_factors_np = np.array(batch_depth_factors, dtype=np.float32)
                            preds_denorm_mc = preds_denorm_mc * depth_factors_np
                        mc_preds_denorm.append(preds_denorm_mc)
                    mc_preds_denorm = np.stack(mc_preds_denorm, axis=0)
                    preds_denorm = mc_preds_denorm.mean(axis=0)
                    preds_std = mc_preds_denorm.std(axis=0)
                else:
                    self.model.eval()
                    with torch.no_grad(), autocast(device_type=self.device.type, enabled=torch.cuda.is_available()):
                        self.logger.debug(f"GridPredictor: Predicting batch {i//self.batch_size + 1}, Combined patch shape: {batch_combined_patches_tensor.shape}")
                        if torch.isnan(batch_combined_patches_tensor).any():
                            self.logger.error(f"GridPredictor: Combined patch input contains NaN values! NaN count: {torch.isnan(batch_combined_patches_tensor).sum()}")
                        preds_norm = self.model(batch_combined_patches_tensor)
                        if torch.isnan(preds_norm).any():
                            self.logger.error(f"GridPredictor: Model output contains NaNs! Shape: {preds_norm.shape}, NaN count: {torch.isnan(preds_norm).sum()}")
                        if torch.isinf(preds_norm).any():
                            self.logger.error(f"GridPredictor: Model output contains infinities! Shape: {preds_norm.shape}, Infinity count: {torch.isinf(preds_norm).sum()}")

                    preds_denorm = self._denormalize_rey(preds_norm.squeeze(-1).cpu().numpy())
                    if batch_depth_factors:
                        depth_factors_np = np.array(batch_depth_factors, dtype=np.float32)
                        preds_denorm = preds_denorm * depth_factors_np
                    preds_std = None

                global_indices_to_update = [i + j for j in valid_point_indices_in_batch]
                all_grid_predictions_denorm_array[global_indices_to_update] = preds_denorm
                if all_grid_uncertainty_array is not None and preds_std is not None:
                    all_grid_uncertainty_array[global_indices_to_update] = preds_std

            if self.progress_callback:
                current_progress_value = (i + len(batch_coords)) / num_total_points * 0.7 + 0.1
                self.progress_callback(current_progress_value, f"Predicting 3D grid: {i + len(batch_coords)}/{num_total_points} points processed.")

        self.logger.debug(f"GridPredictor: Finished predicting {num_total_points} points.")

        all_grid_predictions_denorm_list = all_grid_predictions_denorm_array.tolist()

        if not all_grid_predictions_denorm_list or np.all(np.isnan(all_grid_predictions_denorm_array)):
            self.logger.warning("GridPredictor: No valid predictions were generated.")
            return None

        predicted_rey_grid_np = np.array(all_grid_predictions_denorm_list).reshape(num_lat_points, num_lon_points, len(depth_levels))

        predicted_rey_grid_np_transposed = predicted_rey_grid_np.transpose(2, 0, 1)
        data_vars = {
            "predicted_rey": (("depth", "latitude", "longitude"), predicted_rey_grid_np_transposed)
        }
        if all_grid_uncertainty_array is not None:
            pred_std_grid_np = np.array(all_grid_uncertainty_array.tolist()).reshape(num_lat_points, num_lon_points, len(depth_levels))
            pred_std_grid_np_transposed = pred_std_grid_np.transpose(2, 0, 1)
            data_vars["predicted_rey_std"] = (("depth", "latitude", "longitude"), pred_std_grid_np_transposed)
        pred_grid_ds = xr.Dataset(
            data_vars,
            coords={"depth": depth_levels, "latitude": pred_lats, "longitude": pred_lons}
        )

        pred_data = pred_grid_ds["predicted_rey"].data
        pred_min, pred_max, pred_mean = np.nanmin(pred_data), np.nanmax(pred_data), np.nanmean(pred_data)
        pred_nan_count, pred_total_count = np.isnan(pred_data).sum(), pred_data.size
        self.logger.debug(f"GridPredictor: Predicted REY data stats: min={pred_min}, max={pred_max}, mean={pred_mean}, nan_count={pred_nan_count}/{pred_total_count}")

        pred_grid_ds["longitude"].attrs = {'units': 'degrees_east', 'long_name': 'Longitude'}
        pred_grid_ds["latitude"].attrs = {'units': 'degrees_north', 'long_name': 'Latitude'}
        pred_grid_ds["depth"].attrs = {'units': 'm', 'long_name': 'Depth'}
        pred_grid_ds["predicted_rey"].attrs = {'units': 'ppm', 'long_name': 'Predicted REY Concentration (mean)'}
        if "predicted_rey_std" in pred_grid_ds:
            pred_grid_ds["predicted_rey_std"].attrs = {'units': 'ppm', 'long_name': 'Prediction uncertainty (std, MC Dropout)'}
        pred_grid_ds.attrs['description'] = '3D gridded REY predictions from CNN-MLP model.'
        pred_grid_ds.attrs['creation_date'] = time.strftime('%Y-%m-%d %H:%M:%S')


        if "predicted_rey_std" in pred_grid_ds and self.depth_bins_m:
            try:
                depth_values = np.asarray(depth_levels, dtype=float)
                for i in range(len(self.depth_bins_m) - 1):
                    d_min = self.depth_bins_m[i]
                    d_max = self.depth_bins_m[i + 1]
                    mask = (depth_values >= d_min) & (depth_values < d_max)
                    if not np.any(mask):
                        continue
                    std_layer = pred_grid_ds["predicted_rey_std"].isel(depth=mask).mean(dim="depth", skipna=True)
                    mean_layer = pred_grid_ds["predicted_rey"].isel(depth=mask).mean(dim="depth", skipna=True)
                    layer_name_std = f"uncertainty_std_{int(d_min)}_{int(d_max)}m"
                    layer_name_mean = f"pred_mean_{int(d_min)}_{int(d_max)}m"
                    pred_grid_ds[layer_name_std] = std_layer
                    pred_grid_ds[layer_name_mean] = mean_layer
                    pred_grid_ds[layer_name_std].attrs = {"long_name": f"MC std {d_min}-{d_max} m", "units": "ppm"}
                    pred_grid_ds[layer_name_mean].attrs = {"long_name": f"Mean REY {d_min}-{d_max} m", "units": "ppm"}
            except Exception as e_bin:
                self.logger.warning(f"GridPredictor: Failed to aggregate uncertainty by depth bins: {e_bin}")

        return pred_grid_ds

    def _predict_3d_grid_v2(self, pred_grid_settings: PredictionGridSettings):
        """
        v2 prediction implementation.

        Key differences vs legacy:
        - Supports `vertical_axis_mode` in PredictionGridSettings: 'depth_mbsf' or 'age_ka'
        - Writes auxiliary variables for interpretability:
          - `age_ka_proxy` (when predicting by depth)
          - `depth_mbsf` (when predicting by age)
          - `is_in_training_age_domain`, `depth_extrapolation_factor`, `sedimentation_rate`
        - Optionally masks out-of-domain points based on `training_age_domain_ka`
        """
        self.logger.info("GridPredictor(v2): Starting 3D grid prediction...")

        def _vertical_mode_and_levels():
            mode = getattr(pred_grid_settings, "vertical_axis_mode", None) or "depth_mbsf"
            mode = str(mode).lower()
            if mode not in ("depth_mbsf", "age_ka"):
                raise ValueError(f"Unsupported vertical_axis_mode='{mode}'.")

            if mode == "age_ka":
                levels = list(getattr(pred_grid_settings, "age_levels_ka", None) or [])
                if not levels:
                    raise ValueError("vertical_axis_mode='age_ka' requires non-empty age_levels_ka.")
            else:
                levels = list(getattr(pred_grid_settings, "depth_levels", None) or [])
                if not levels:
                    raise ValueError("vertical_axis_mode='depth_mbsf' requires non-empty depth_levels.")

            return mode, np.asarray(levels, dtype=float)

        bounds = pred_grid_settings.geospatial_bounds
        if bounds and all(v is not None for v in [bounds.min_lon, bounds.max_lon, bounds.min_lat, bounds.max_lat]):
            self.logger.debug(f"Using user-defined geospatial bounds: {bounds}")
            min_lon, max_lon, min_lat, max_lat = bounds.min_lon, bounds.max_lon, bounds.min_lat, bounds.max_lat
        else:
            self.logger.debug("Geospatial bounds not provided or incomplete. Using full extent of environmental data.")
            min_lon, max_lon = float(self.env_lon.min()), float(self.env_lon.max())
            min_lat, max_lat = float(self.env_lat.min()), float(self.env_lat.max())

        lon_res = float(pred_grid_settings.lon_res)
        lat_res = float(pred_grid_settings.lat_res)
        vertical_mode, levels = _vertical_mode_and_levels()

        max_lon_adjusted = max_lon
        if max_lon < min_lon:
            self.logger.debug(f"Cross-meridian longitude detected. Adjusting max_lon from {max_lon} to {max_lon + 360}.")
            max_lon_adjusted = max_lon + 360

        num_lon_points = int(np.floor((max_lon_adjusted - min_lon) / lon_res)) + 1
        num_lat_points = int(np.floor((max_lat - min_lat) / lat_res)) + 1

        pred_lons_raw = np.linspace(min_lon, min_lon + (num_lon_points - 1) * lon_res, num_lon_points)
        pred_lons = (pred_lons_raw + 180) % 360 - 180
        pred_lats = np.linspace(min_lat, min_lat + (num_lat_points - 1) * lat_res, num_lat_points)
        pred_lons_0_360 = self.convert_lon_range(pred_lons, target_range="0_360")

        land_mask = None
        if self.pred_config and self.pred_config.use_land_mask:
            self.logger.info("Generating land mask for the prediction grid...")
            lon_grid, lat_grid = np.meshgrid(pred_lons, pred_lats)
            land_mask = globe.is_land(lat_grid, lon_grid)
            self.logger.info(f"Land mask generated. Land points: {np.sum(land_mask)}/{land_mask.size}")


        lat_idx_arr = np.searchsorted(self.env_lat, pred_lats - self.lat_res / 2)
        lon_idx_arr = np.searchsorted(self.env_lon, pred_lons_0_360 - self.lon_res / 2)
        lat_idx_arr = np.clip(lat_idx_arr, 0, len(self.env_lat) - 1)
        lon_idx_arr = np.clip(lon_idx_arr, 0, len(self.env_lon) - 1)

        depth_channel_idx = self.num_features
        depth_mean = float(self.channel_means[depth_channel_idx])
        depth_std = float(self.channel_stds[depth_channel_idx])

        max_training_age_ka = float(getattr(self.pred_config, "training_age_domain_ka", 2580.0) or 2580.0)
        restrict_to_domain = bool(getattr(self.pred_config, "restrict_to_training_age_domain", False))
        apply_attenuation = bool(getattr(self.pred_config, "apply_depth_extrapolation_attenuation", True))

        num_levels = int(levels.size)
        is_multiclass = self.target_task_type == "multiclass_classification"
        if is_multiclass:
            n_cls = int(self.target_num_classes or 0)
            if n_cls < 2:
                raise ValueError("GridPredictor(v2): multiclass_classification requires num_classes>=2 in artifacts.")
            pred_prob_grid = np.full((n_cls, num_levels, num_lat_points, num_lon_points), np.nan, dtype=np.float32)
            pred_prob_std_grid = (
                np.full((n_cls, num_levels, num_lat_points, num_lon_points), np.nan, dtype=np.float32)
                if (self.enable_mc_dropout and self.mc_samples > 1)
                else None
            )
            pred_class_grid = np.full((num_levels, num_lat_points, num_lon_points), -1, dtype=np.int16)
            pred_entropy_grid = np.full((num_levels, num_lat_points, num_lon_points), np.nan, dtype=np.float32)
            pred_ranking_score_grid = (
                np.full((num_levels, num_lat_points, num_lon_points), np.nan, dtype=np.float32)
                if bool(getattr(self.model, "use_high_grade_ranking_head", False))
                else None
            )
            pred_mean_grid = None
            pred_std_grid = None
        else:
            pred_mean_grid = np.full((num_levels, num_lat_points, num_lon_points), np.nan, dtype=np.float32)
            pred_std_grid = (
                np.full((num_levels, num_lat_points, num_lon_points), np.nan, dtype=np.float32)
                if (self.enable_mc_dropout and self.mc_samples > 1)
                else None
            )
            pred_prob_grid = None
            pred_prob_std_grid = None
            pred_class_grid = None
            pred_entropy_grid = None
            pred_ranking_score_grid = None
        depth_extrap_factor = np.full((num_levels, num_lat_points, num_lon_points), np.nan, dtype=np.float32)
        in_domain_mask = np.zeros((num_levels, num_lat_points, num_lon_points), dtype=np.uint8)

        age_ka_proxy_field = (
            np.full((num_levels, num_lat_points, num_lon_points), np.nan, dtype=np.float32)
            if vertical_mode == "depth_mbsf"
            else None
        )
        depth_mbsf_field = (
            np.full((num_levels, num_lat_points, num_lon_points), np.nan, dtype=np.float32)
            if vertical_mode == "age_ka"
            else None
        )
        sed_rate_grid = (
            np.full((num_lat_points, num_lon_points), np.nan, dtype=np.float32)
            if self.sed_rate_index is not None
            else None
        )
        env_center_valid_fraction = np.full((num_lat_points, num_lon_points), np.nan, dtype=np.float32)
        env_center_any_missing = np.zeros((num_lat_points, num_lon_points), dtype=np.uint8)
        sed_rate_was_filled = (
            np.zeros((num_lat_points, num_lon_points), dtype=np.uint8)
            if self.sed_rate_index is not None
            else None
        )

        def _extract_env_patch_stdized(lat_idx: int, lon_idx: int) -> np.ndarray:
            row_start = lat_idx - self.pad_rows
            row_end = lat_idx + self.pad_rows + 1
            col_start = lon_idx - self.pad_cols
            col_end = lon_idx + self.pad_cols + 1

            actual_row_start = max(0, row_start)
            actual_row_end = min(len(self.env_lat), row_end)
            actual_col_start = max(0, col_start)
            actual_col_end = min(len(self.env_lon), col_end)

            env_patch_extracted_np = self.env_data_np[:, actual_row_start:actual_row_end, actual_col_start:actual_col_end]

            pad_top = max(0, -row_start)
            pad_bottom = max(0, row_end - len(self.env_lat))
            pad_left = max(0, -col_start)
            pad_right = max(0, col_end - len(self.env_lon))

            current_env_patch_np = env_patch_extracted_np
            if any([pad_top > 0, pad_bottom > 0, pad_left > 0, pad_right > 0]):
                current_env_patch_np = np.pad(
                    env_patch_extracted_np,
                    ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
                    mode="edge",
                )

            if current_env_patch_np.shape[1:] != self.patch_size:
                target_shape_patch = np.full((self.num_features, self.patch_size[0], self.patch_size[1]), np.nan, dtype=np.float32)
                copy_h = min(current_env_patch_np.shape[1], self.patch_size[0])
                copy_w = min(current_env_patch_np.shape[2], self.patch_size[1])
                target_shape_patch[:, :copy_h, :copy_w] = current_env_patch_np[:, :copy_h, :copy_w]
                current_env_patch_np = target_shape_patch

            env_missingness_np = None
            if self.add_env_missingness_channels:
                env_missingness_np = np.isnan(current_env_patch_np).astype(np.float32)

            if np.isnan(current_env_patch_np).any():
                current_env_patch_np = numpy_ffill_bfill_channelwise(current_env_patch_np)

            if np.isnan(current_env_patch_np).any():
                for c_idx in range(self.num_features):
                    if np.isnan(current_env_patch_np[c_idx, :, :]).any():
                        current_env_patch_np[c_idx, :, :] = np.nan_to_num(
                            current_env_patch_np[c_idx, :, :],
                            copy=False,
                            nan=float(self.channel_means[c_idx]),
                        )

            env_patch_stdized = np.zeros_like(current_env_patch_np, dtype=np.float32)
            for c_idx in range(self.num_features):
                ch_m, ch_s = float(self.channel_means[c_idx]), float(self.channel_stds[c_idx])
                env_patch_stdized[c_idx, :, :] = (current_env_patch_np[c_idx, :, :] - ch_m) / (ch_s if ch_s > 1e-6 else 1.0)

            if env_missingness_np is not None:
                return np.concatenate((env_patch_stdized, env_missingness_np), axis=0)
            return env_patch_stdized

        batch_patches = []
        batch_targets = []
        batch_depth_factors = []

        warned_missing_sed = False

        def _flush_batch():
            nonlocal batch_patches, batch_targets, batch_depth_factors
            if not batch_patches:
                return

            batch_tensor = torch.tensor(np.array(batch_patches), dtype=torch.float32).to(self.device)
            depth_factors_to_apply = batch_depth_factors if apply_attenuation else None

            def _softmax(z: np.ndarray) -> np.ndarray:
                z = np.asarray(z, dtype=np.float64)
                z = z - np.max(z, axis=1, keepdims=True)
                ez = np.exp(np.clip(z, -60.0, 60.0))
                return ez / np.sum(ez, axis=1, keepdims=True)

            def _hierarchical_logits(preds_norm):
                if self.hierarchical_grade_multitask:
                    end_logits, sub_logits, grade_logits = preds_norm
                    pe = torch.softmax(end_logits, dim=1)
                    ps = torch.softmax(sub_logits, dim=1)
                    pg = torch.softmax(grade_logits, dim=1)
                    p_non = pe[:, 1:2]
                    p_full = torch.cat(
                        [
                            pe[:, 0:1],
                            p_non * ps[:, 0:1] * pg[:, 0:1],
                            p_non * ps[:, 0:1] * pg[:, 1:2],
                            p_non * ps[:, 1:2] * pg[:, 0:1],
                            p_non * ps[:, 1:2] * pg[:, 1:2],
                        ],
                        dim=1,
                    )
                else:
                    end_logits, sub_logits = preds_norm
                    pe = torch.softmax(end_logits, dim=1)
                    ps = torch.softmax(sub_logits, dim=1)
                    p_full = torch.cat([pe[:, 0:1], pe[:, 1:2] * ps], dim=1)
                return torch.log(torch.clamp(p_full, 1e-12, 1.0)).detach().cpu().numpy()

            if self.enable_mc_dropout and self.mc_samples > 1:
                mc_preds_decoded = []
                mc_ranking_scores = []
                for _ in range(self.mc_samples):
                    self._enable_dropout()
                    with torch.no_grad(), autocast(device_type=self.device.type, enabled=torch.cuda.is_available()):
                        if is_multiclass and bool(getattr(self.model, "use_high_grade_ranking_head", False)):
                            preds_norm_mc, comps_mc = self.model(batch_tensor, return_components=True)
                        else:
                            preds_norm_mc = self.model(batch_tensor)
                            comps_mc = None
                    if is_multiclass:
                        if self.hierarchical_multiclass:
                            logits = _hierarchical_logits(preds_norm_mc)
                        else:
                            logits = preds_norm_mc.detach().cpu().numpy()
                        if depth_factors_to_apply:
                            df = np.asarray(depth_factors_to_apply, dtype=np.float64).reshape(-1, 1)
                            logits = logits * df
                        mc_preds_decoded.append(_softmax(logits).astype(np.float64))
                        rank_logit = comps_mc.get("high_grade_ranking_logit", None) if isinstance(comps_mc, dict) else None
                        if rank_logit is not None:
                            rlog = rank_logit.view(-1).detach().cpu().numpy().astype(np.float64)
                            if depth_factors_to_apply:
                                rlog = rlog * np.asarray(depth_factors_to_apply, dtype=np.float64)
                            mc_ranking_scores.append(1.0 / (1.0 + np.exp(-np.clip(rlog, -60.0, 60.0))))
                    else:
                        preds_np = preds_norm_mc.squeeze(-1).cpu().numpy()
                        if self.target_task_type == "binary_classification":
                            logits = np.asarray(preds_np, dtype=np.float64)
                            if depth_factors_to_apply:
                                logits = logits * np.asarray(depth_factors_to_apply, dtype=np.float64)
                            logits = np.clip(logits, -60.0, 60.0)
                            decoded = 1.0 / (1.0 + np.exp(-logits))
                        else:
                            decoded = self._denormalize_rey(preds_np)
                            if depth_factors_to_apply:
                                decoded = decoded * np.asarray(depth_factors_to_apply, dtype=np.float64)
                        mc_preds_decoded.append(np.asarray(decoded, dtype=np.float64))
                mc_preds_decoded = np.stack(mc_preds_decoded, axis=0)
                preds_denorm = mc_preds_decoded.mean(axis=0)
                preds_std = mc_preds_decoded.std(axis=0)
                ranking_scores = (
                    np.stack(mc_ranking_scores, axis=0).mean(axis=0)
                    if mc_ranking_scores
                    else None
                )
            else:
                self.model.eval()
                with torch.no_grad(), autocast(device_type=self.device.type, enabled=torch.cuda.is_available()):
                    if is_multiclass and bool(getattr(self.model, "use_high_grade_ranking_head", False)):
                        preds_norm, comps_pred = self.model(batch_tensor, return_components=True)
                    else:
                        preds_norm = self.model(batch_tensor)
                        comps_pred = None
                if is_multiclass:
                    if self.hierarchical_multiclass:
                        logits = _hierarchical_logits(preds_norm)
                    else:
                        logits = preds_norm.detach().cpu().numpy()
                    if depth_factors_to_apply:
                        df = np.asarray(depth_factors_to_apply, dtype=np.float64).reshape(-1, 1)
                        logits = logits * df
                    preds_denorm = _softmax(logits).astype(np.float64)
                    rank_logit = comps_pred.get("high_grade_ranking_logit", None) if isinstance(comps_pred, dict) else None
                    if rank_logit is not None:
                        rlog = rank_logit.view(-1).detach().cpu().numpy().astype(np.float64)
                        if depth_factors_to_apply:
                            rlog = rlog * np.asarray(depth_factors_to_apply, dtype=np.float64)
                        ranking_scores = 1.0 / (1.0 + np.exp(-np.clip(rlog, -60.0, 60.0)))
                    else:
                        ranking_scores = None
                    preds_std = None
                else:
                    ranking_scores = None
                    preds_np = preds_norm.squeeze(-1).cpu().numpy()
                    if self.target_task_type == "binary_classification":
                        logits = np.asarray(preds_np, dtype=np.float64)
                        if depth_factors_to_apply:
                            logits = logits * np.asarray(depth_factors_to_apply, dtype=np.float64)
                        logits = np.clip(logits, -60.0, 60.0)
                        preds_denorm = 1.0 / (1.0 + np.exp(-logits))
                    else:
                        preds_denorm = self._denormalize_rey(preds_np)
                        if depth_factors_to_apply:
                            preds_denorm = preds_denorm * np.asarray(depth_factors_to_apply, dtype=np.float64)
                    preds_std = None

            for j, (lvl_i, lat_i, lon_i) in enumerate(batch_targets):
                if is_multiclass:
                    probs_j = np.asarray(preds_denorm[j], dtype=np.float64).reshape(-1)
                    for c in range(pred_prob_grid.shape[0]):
                        pred_prob_grid[c, lvl_i, lat_i, lon_i] = float(probs_j[c]) if c < probs_j.size else np.nan
                        if pred_prob_std_grid is not None and preds_std is not None:
                            pred_prob_std_grid[c, lvl_i, lat_i, lon_i] = float(preds_std[j, c]) if (preds_std.ndim == 2 and c < preds_std.shape[1]) else np.nan
                    pred_class_grid[lvl_i, lat_i, lon_i] = int(np.argmax(probs_j))
                    p = np.clip(probs_j, 1e-12, 1.0)
                    pred_entropy_grid[lvl_i, lat_i, lon_i] = float(-np.sum(p * np.log(p)))
                    if pred_ranking_score_grid is not None and ranking_scores is not None:
                        pred_ranking_score_grid[lvl_i, lat_i, lon_i] = float(ranking_scores[j])
                else:
                    pred_mean_grid[lvl_i, lat_i, lon_i] = float(preds_denorm[j])
                    if pred_std_grid is not None and preds_std is not None:
                        pred_std_grid[lvl_i, lat_i, lon_i] = float(preds_std[j])

            batch_patches = []
            batch_targets = []
            batch_depth_factors = []

        total_cells = num_lat_points * num_lon_points
        processed_cells = 0

        for lat_i, _ in enumerate(pred_lats):
            env_lat_idx = int(lat_idx_arr[lat_i])
            for lon_i, _ in enumerate(pred_lons_0_360):
                if land_mask is not None and land_mask[lat_i, lon_i]:
                    processed_cells += 1
                    continue

                env_lon_idx = int(lon_idx_arr[lon_i])

                try:
                    center_vec = self.env_data_np[:, env_lat_idx, env_lon_idx]
                    center_valid = np.isfinite(center_vec)
                    env_center_valid_fraction[lat_i, lon_i] = float(np.mean(center_valid))
                    env_center_any_missing[lat_i, lon_i] = 0 if bool(np.all(center_valid)) else 1
                except Exception:
                    pass

                env_patch_stdized = _extract_env_patch_stdized(env_lat_idx, env_lon_idx)

                sed_rate_val = np.nan
                if self.sed_rate_index is not None:
                    sed_rate_raw = float(self.env_data_np[self.sed_rate_index, env_lat_idx, env_lon_idx])
                    was_missing = not np.isfinite(sed_rate_raw)
                    if sed_rate_was_filled is not None:
                        sed_rate_was_filled[lat_i, lon_i] = 1 if was_missing else 0

                    sed_rate_val = sed_rate_raw
                    if was_missing:
                        sed_rate_val = float(self.channel_means[self.sed_rate_index]) if self.sed_rate_index < len(self.channel_means) else np.nan
                    if np.isfinite(sed_rate_val):
                        sed_rate_val = max(sed_rate_val, 1e-3)
                    if sed_rate_grid is not None:
                        sed_rate_grid[lat_i, lon_i] = sed_rate_val

                for lvl_i, lvl_v in enumerate(levels):
                    depth_mbsf = np.nan
                    age_ka = np.nan

                    if vertical_mode == "depth_mbsf":
                        depth_mbsf = float(lvl_v)
                        if np.isfinite(sed_rate_val):
                            age_ka = depth_mbsf * 100.0 / sed_rate_val
                        if age_ka_proxy_field is not None:
                            age_ka_proxy_field[lvl_i, lat_i, lon_i] = age_ka
                    else:
                        age_ka = float(lvl_v)
                        if np.isfinite(sed_rate_val):
                            depth_mbsf = age_ka * sed_rate_val / 100.0
                        if depth_mbsf_field is not None:
                            depth_mbsf_field[lvl_i, lat_i, lon_i] = depth_mbsf


                    is_in_domain = bool(np.isfinite(age_ka) and age_ka <= max_training_age_ka)
                    if vertical_mode == "age_ka" and not np.isfinite(depth_mbsf):
                        is_in_domain = False
                    in_domain_mask[lvl_i, lat_i, lon_i] = 1 if is_in_domain else 0
                    if restrict_to_domain and not is_in_domain:
                        continue
                    if vertical_mode == "age_ka" and not np.isfinite(depth_mbsf):
                        continue

                    depth_feature_raw = depth_mbsf
                    if self.depth_feature_type == "age_ka_proxy":
                        if np.isfinite(age_ka):
                            depth_feature_raw = age_ka
                        else:
                            if not warned_missing_sed:
                                self.logger.warning(
                                    "GridPredictor(v2): depth_feature_type='age_ka_proxy' but sedimentation rate is missing/non-finite; "
                                    "falling back to physical depth for some points."
                                )
                                warned_missing_sed = True
                            depth_feature_raw = depth_mbsf
                    elif self.depth_feature_type == "legacy_time_depth":
                        if self.sed_rate_index is None or not np.isfinite(sed_rate_val):
                            if not warned_missing_sed:
                                self.logger.warning(
                                    "GridPredictor(v2): depth_feature_type requires sedimentation rate but it is unavailable; "
                                    "falling back to physical depth for some points."
                                )
                                warned_missing_sed = True
                            depth_feature_raw = depth_mbsf
                        else:
                            depth_feature_raw = float(depth_mbsf) / float(sed_rate_val)

                    if self.depth_transform == "log1p":
                        depth_val_transformed = np.log1p(max(float(depth_feature_raw), 0.0))
                    elif self.depth_transform == "sqrt":
                        depth_val_transformed = np.sqrt(max(float(depth_feature_raw), 0.0))
                    else:
                        depth_val_transformed = float(depth_feature_raw)

                    norm_d_val = (depth_val_transformed - depth_mean) / (depth_std if depth_std > 1e-6 else 1.0)
                    depth_channel = np.full((1, self.patch_size[0], self.patch_size[1]), norm_d_val, dtype=np.float32)
                    combined_patch = np.concatenate((env_patch_stdized, depth_channel), axis=0)


                    if self.add_coord_channels and self.coord_feature_dim == 4:
                        lon_cell = float(self.env_lon[lon_i])
                        lat_cell = float(self.env_lat[lat_i])
                        lon_norm = ((lon_cell + 180.0) % 360.0) - 180.0
                        lon_rad = np.deg2rad(lon_norm)
                        lat_rad = np.deg2rad(lat_cell)
                        coord_vals = np.array(
                            [np.sin(lat_rad), np.cos(lat_rad), np.sin(lon_rad), np.cos(lon_rad)],
                            dtype=np.float32,
                        )
                        coord_channels = np.tile(coord_vals.reshape(4, 1, 1), (1, self.patch_size[0], self.patch_size[1]))
                        combined_patch = np.concatenate((combined_patch, coord_channels), axis=0)

                    if np.isnan(combined_patch).any():
                        final_patch = np.copy(combined_patch)
                        num_total_channels = final_patch.shape[0]
                        for c_idx in range(num_total_channels):
                            if np.isnan(final_patch[c_idx, :, :]).any():
                                fill_value = float(self.channel_means[c_idx]) if c_idx < len(self.channel_means) else 0.0
                                final_patch[c_idx, :, :] = np.nan_to_num(final_patch[c_idx, :, :], copy=False, nan=fill_value)
                        combined_patch = final_patch

                    batch_patches.append(combined_patch)
                    batch_targets.append((lvl_i, lat_i, lon_i))

                    depth_factor = float(self._depth_extrapolation_factor(float(depth_feature_raw)))
                    batch_depth_factors.append(depth_factor)
                    depth_extrap_factor[lvl_i, lat_i, lon_i] = depth_factor

                    if len(batch_patches) >= self.batch_size:
                        _flush_batch()

                processed_cells += 1
                if self.progress_callback and processed_cells % max(1, int(total_cells / 50)) == 0:
                    pct = int(20 + 60 * (processed_cells / total_cells))
                    self.progress_callback(pct, f"Predicting grid... ({processed_cells}/{total_cells} cells)")

        _flush_batch()

        if (not is_multiclass and np.all(np.isnan(pred_mean_grid))) or (is_multiclass and np.all(np.isnan(pred_prob_grid))):
            self.logger.warning("GridPredictor(v2): No valid predictions were generated.")
            return None

        vertical_dim = "depth_mbsf" if vertical_mode == "depth_mbsf" else "age_ka"
        coords = {vertical_dim: levels.tolist(), "latitude": pred_lats, "longitude": pred_lons}

        data_vars = {
            "depth_extrapolation_factor": ((vertical_dim, "latitude", "longitude"), depth_extrap_factor),
            "is_in_training_age_domain": ((vertical_dim, "latitude", "longitude"), in_domain_mask),
        }
        if is_multiclass:
            class_names_norm = [str(x) for x in (self.target_class_names or [])]
            is_grade_geochem = class_names_norm[:5] == [
                "carbonate_background",
                "class1_low_grade",
                "class1_high_grade",
                "class2_low_grade",
                "class2_high_grade",
            ]
            is_exploration_4class = class_names_norm[:4] == [
                "carbonate_background",
                "class1_enriched",
                "class2_low_grade",
                "class2_high_grade",
            ]
            grade_aliases = (
                {
                    0: "prob_carbonate_background",
                    1: "prob_class1_enriched",
                    2: "prob_class2_low_grade",
                    3: "prob_class2_high_grade",
                }
                if is_exploration_4class
                else {
                    0: "prob_carbonate_background",
                    1: "prob_class1_low_grade",
                    2: "prob_class1_high_grade",
                    3: "prob_class2_low_grade",
                    4: "prob_class2_high_grade",
                }
            )
            for c in range(pred_prob_grid.shape[0]):
                data_vars[f"predicted_prob_class{c}"] = ((vertical_dim, "latitude", "longitude"), pred_prob_grid[c])
                if (is_grade_geochem or is_exploration_4class) and c in grade_aliases:
                    data_vars[grade_aliases[c]] = ((vertical_dim, "latitude", "longitude"), pred_prob_grid[c])
                if pred_prob_std_grid is not None:
                    data_vars[f"predicted_prob_class{c}_std"] = ((vertical_dim, "latitude", "longitude"), pred_prob_std_grid[c])
            data_vars["predicted_class_argmax"] = ((vertical_dim, "latitude", "longitude"), pred_class_grid)
            if is_grade_geochem or is_exploration_4class:
                data_vars["pred_grade_geochem_class_id"] = ((vertical_dim, "latitude", "longitude"), pred_class_grid)
            if is_exploration_4class:
                data_vars["pred_exploration_class_id"] = ((vertical_dim, "latitude", "longitude"), pred_class_grid)
                high_grade_score_grid = (
                    np.clip(pred_ranking_score_grid, 0.0, 1.0).astype(np.float32)
                    if pred_ranking_score_grid is not None
                    else np.clip(pred_prob_grid[3], 0.0, 1.0).astype(np.float32)
                )
                data_vars["prob_class2_high_grade"] = (
                    (vertical_dim, "latitude", "longitude"),
                    np.clip(pred_prob_grid[3], 0.0, 1.0).astype(np.float32),
                )
                if pred_ranking_score_grid is not None:
                    data_vars["high_grade_ranking_score"] = (
                        (vertical_dim, "latitude", "longitude"),
                        high_grade_score_grid,
                    )
                data_vars["high_grade_prospectivity_score"] = (
                    (vertical_dim, "latitude", "longitude"),
                    high_grade_score_grid,
                )
            data_vars["predicted_class_entropy"] = ((vertical_dim, "latitude", "longitude"), pred_entropy_grid)
            data_vars["prediction_entropy"] = ((vertical_dim, "latitude", "longitude"), pred_entropy_grid)
            data_vars["prediction_confidence"] = (
                (vertical_dim, "latitude", "longitude"),
                np.nanmax(pred_prob_grid, axis=0).astype(np.float32),
            )
        else:
            pred_var_name = "predicted_rey"
            pred_std_name = "predicted_rey_std"
            if self.target_task_type == "binary_classification":
                thr = float(self.target_threshold_ppm or 1000.0)
                thr_tag = str(int(round(thr))) if abs(thr - round(thr)) < 1e-6 else str(thr).replace(".", "p")
                pred_var_name = f"predicted_prob_ge_{thr_tag}ppm"
                pred_std_name = f"{pred_var_name}_std"
            data_vars[pred_var_name] = ((vertical_dim, "latitude", "longitude"), pred_mean_grid)
            if pred_std_grid is not None:
                data_vars[pred_std_name] = ((vertical_dim, "latitude", "longitude"), pred_std_grid)
        if sed_rate_grid is not None:
            data_vars["sedimentation_rate"] = (("latitude", "longitude"), sed_rate_grid)
        if age_ka_proxy_field is not None:
            data_vars["age_ka_proxy"] = ((vertical_dim, "latitude", "longitude"), age_ka_proxy_field)
        if depth_mbsf_field is not None:
            data_vars["depth_mbsf"] = ((vertical_dim, "latitude", "longitude"), depth_mbsf_field)
        data_vars["env_center_valid_fraction"] = (("latitude", "longitude"), env_center_valid_fraction)
        data_vars["env_center_any_missing"] = (("latitude", "longitude"), env_center_any_missing)
        if sed_rate_was_filled is not None:
            data_vars["sedimentation_rate_was_filled"] = (("latitude", "longitude"), sed_rate_was_filled)

        pred_grid_ds = xr.Dataset(data_vars, coords=coords)

        pred_grid_ds["longitude"].attrs = {"units": "degrees_east", "long_name": "Longitude"}
        pred_grid_ds["latitude"].attrs = {"units": "degrees_north", "long_name": "Latitude"}
        if vertical_dim == "depth_mbsf":
            pred_grid_ds[vertical_dim].attrs = {"units": "m", "long_name": "Sub-bottom depth (mbsf)"}
        else:
            pred_grid_ds[vertical_dim].attrs = {"units": "ka", "long_name": "Age (kilo-annum, proxy)"}

        if is_multiclass:
            if self.target_class_names:
                pred_grid_ds.attrs["class_names"] = ",".join(self.target_class_names)
            for c in range(int(self.target_num_classes or 0)):
                class_label = (
                    self.target_class_names[c]
                    if self.target_class_names and c < len(self.target_class_names)
                    else f"class{c}"
                )
                prob_name = f"predicted_prob_class{c}"
                if prob_name in pred_grid_ds:
                    pred_grid_ds[prob_name].attrs = {"units": "1", "long_name": f"Predicted probability of {class_label}"}
                alias_name = {
                    "carbonate_background": "prob_carbonate_background",
                    "class1_enriched": "prob_class1_enriched",
                    "class1_low_grade": "prob_class1_low_grade",
                    "class1_high_grade": "prob_class1_high_grade",
                    "class2_low_grade": "prob_class2_low_grade",
                    "class2_high_grade": "prob_class2_high_grade",
                }.get(str(class_label))
                if alias_name and alias_name in pred_grid_ds:
                    pred_grid_ds[alias_name].attrs = {
                        "units": "1",
                        "long_name": f"Predicted probability of {class_label}",
                    }
                prob_std_name = f"{prob_name}_std"
                if prob_std_name in pred_grid_ds:
                    pred_grid_ds[prob_std_name].attrs = {
                        "units": "1",
                        "long_name": f"Prediction uncertainty (std, MC Dropout) for {class_label}",
                    }
            if "predicted_class_argmax" in pred_grid_ds:
                pred_grid_ds["predicted_class_argmax"].attrs = {"units": "1", "long_name": "Predicted class index (argmax)"}
            if "predicted_class_entropy" in pred_grid_ds:
                pred_grid_ds["predicted_class_entropy"].attrs = {"units": "nats", "long_name": "Predictive entropy over classes"}
            if "prediction_entropy" in pred_grid_ds:
                pred_grid_ds["prediction_entropy"].attrs = {"units": "nats", "long_name": "Predictive entropy over classes"}
            if "prediction_confidence" in pred_grid_ds:
                pred_grid_ds["prediction_confidence"].attrs = {"units": "1", "long_name": "Maximum predicted class probability"}
            if "pred_grade_geochem_class_id" in pred_grid_ds:
                pred_grid_ds["pred_grade_geochem_class_id"].attrs = {
                    "units": "class_id",
                    "long_name": "Predicted grade geochemical class id",
                }
            if "pred_exploration_class_id" in pred_grid_ds:
                pred_grid_ds["pred_exploration_class_id"].attrs = {
                    "units": "class_id",
                    "long_name": "Predicted exploration class id",
                }
            if "high_grade_prospectivity_score" in pred_grid_ds:
                formula = (
                    "learned_high_grade_ranking_head"
                    if "high_grade_ranking_score" in pred_grid_ds
                    else "prob_class2_high"
                )
                pred_grid_ds["high_grade_prospectivity_score"].attrs.update(
                    {
                        "units": "1",
                        "long_name": "High-grade prospectivity score",
                        "formula": formula,
                        "source": "single_model",
                    }
                )
                pred_grid_ds.attrs["high_grade_score_formula"] = formula
                pred_grid_ds.attrs["high_grade_score_source"] = "single_model"
            if "high_grade_ranking_score" in pred_grid_ds:
                pred_grid_ds["high_grade_ranking_score"].attrs.update(
                    {
                        "units": "1",
                        "long_name": "Learned high-grade ranking-head score",
                        "formula": "sigmoid(high_grade_ranking_logit)",
                    }
                )
        elif self.target_task_type == "binary_classification":
            pred_grid_ds[pred_var_name].attrs = {
                "units": "1",
                "long_name": f"Predicted probability: ΣREY ≥ {float(self.target_threshold_ppm or 1000.0):g} ppm",
            }
            if pred_std_name in pred_grid_ds:
                pred_grid_ds[pred_std_name].attrs = {"units": "1", "long_name": "Prediction uncertainty (std, MC Dropout)"}
        else:
            pred_grid_ds[pred_var_name].attrs = {"units": "ppm", "long_name": "Predicted REY Concentration (mean)"}
            if pred_std_name in pred_grid_ds:
                pred_grid_ds[pred_std_name].attrs = {"units": "ppm", "long_name": "Prediction uncertainty (std, MC Dropout)"}
        pred_grid_ds["depth_extrapolation_factor"].attrs = {
            "long_name": "Depth-feature extrapolation attenuation factor",
            "description": "Multiplier applied to predictions when depth-feature is outside the training range.",
        }
        pred_grid_ds["is_in_training_age_domain"].attrs = {
            "long_name": "In-domain flag based on age proxy",
            "training_age_domain_ka": max_training_age_ka,
        }
        if "sedimentation_rate" in pred_grid_ds:
            pred_grid_ds["sedimentation_rate"].attrs = {"units": "cm/ka", "long_name": "Average sedimentation rate"}
        if "env_center_valid_fraction" in pred_grid_ds:
            pred_grid_ds["env_center_valid_fraction"].attrs = {
                "units": "1",
                "long_name": "Fraction of environmental features that are finite at the grid cell center",
                "description": "Computed before any patch-level NaN filling; values < 1 indicate missing environmental fields at that location.",
            }
        if "env_center_any_missing" in pred_grid_ds:
            pred_grid_ds["env_center_any_missing"].attrs = {
                "units": "1",
                "long_name": "Flag: any missing environmental feature at cell center (0/1)",
                "description": "Computed before any patch-level NaN filling.",
            }
        if "sedimentation_rate_was_filled" in pred_grid_ds:
            pred_grid_ds["sedimentation_rate_was_filled"].attrs = {
                "units": "1",
                "long_name": "Flag: sedimentation rate missing and replaced by mean (0/1)",
                "description": "Indicates where age↔depth conversion used a filled value due to missing sedimentation rate.",
            }
        if "age_ka_proxy" in pred_grid_ds:
            pred_grid_ds["age_ka_proxy"].attrs = {"units": "ka", "long_name": "Age proxy derived from depth and sedimentation rate"}
        if "depth_mbsf" in pred_grid_ds and vertical_dim != "depth_mbsf":
            pred_grid_ds["depth_mbsf"].attrs = {"units": "m", "long_name": "Depth below seafloor derived from age and sedimentation rate"}

        pred_grid_ds.attrs["description"] = "3D gridded REY predictions from CNN-MLP model."
        pred_grid_ds.attrs["creation_date"] = time.strftime("%Y-%m-%d %H:%M:%S")
        pred_grid_ds.attrs["vertical_axis_mode"] = vertical_mode
        pred_grid_ds.attrs["depth_feature_type_used"] = self.depth_feature_type
        pred_grid_ds.attrs["training_age_domain_ka"] = max_training_age_ka
        pred_grid_ds.attrs["age_proxy_formula"] = "age_ka = depth_mbsf(m)*100 / sed_rate(cm/ka)"
        pred_grid_ds.attrs["depth_formula_from_age"] = "depth_mbsf(m) = age_ka*sed_rate(cm/ka)/100"

        pred_grid_ds.attrs["apply_depth_extrapolation_attenuation"] = int(bool(apply_attenuation))
        pred_grid_ds.attrs["restrict_to_training_age_domain"] = int(bool(restrict_to_domain))


        mean_var = "predicted_rey"
        std_var = "predicted_rey_std"
        units = "ppm"
        if self.target_task_type == "binary_classification":
            thr = float(self.target_threshold_ppm or 1000.0)
            thr_tag = str(int(round(thr))) if abs(thr - round(thr)) < 1e-6 else str(thr).replace(".", "p")
            mean_var = f"predicted_prob_ge_{thr_tag}ppm"
            std_var = f"{mean_var}_std"
            units = "1"

        if std_var in pred_grid_ds:
            try:
                if vertical_dim == "depth_mbsf":
                    bins = list(getattr(self.pred_config, "depth_bins_m", None) or self.depth_bins_m or [])
                    unit_tag = "m"
                    dim_name = vertical_dim
                    layer_unit = "m"
                else:
                    bins = list(getattr(self.pred_config, "age_bins_ka", None) or [])
                    unit_tag = "ka"
                    dim_name = vertical_dim
                    layer_unit = "ka"

                if bins and len(bins) >= 2:
                    level_values = np.asarray(levels, dtype=float)
                    for b_i in range(len(bins) - 1):
                        b_min = float(bins[b_i])
                        b_max = float(bins[b_i + 1])
                        mask = (level_values >= b_min) & (level_values < b_max)
                        if not np.any(mask):
                            continue
                        std_layer = pred_grid_ds[std_var].isel({dim_name: mask}).mean(dim=dim_name, skipna=True)
                        mean_layer = pred_grid_ds[mean_var].isel({dim_name: mask}).mean(dim=dim_name, skipna=True)
                        layer_name_std = f"uncertainty_std_{int(b_min)}_{int(b_max)}{unit_tag}"
                        layer_name_mean = f"pred_mean_{int(b_min)}_{int(b_max)}{unit_tag}"
                        pred_grid_ds[layer_name_std] = std_layer
                        pred_grid_ds[layer_name_mean] = mean_layer
                        pred_grid_ds[layer_name_std].attrs = {"long_name": f"MC std {b_min}-{b_max} {layer_unit}", "units": units}
                        pred_grid_ds[layer_name_mean].attrs = {"long_name": f"Mean {mean_var} {b_min}-{b_max} {layer_unit}", "units": units}
            except Exception as e_bin:
                self.logger.warning(f"GridPredictor(v2): Failed to aggregate by vertical bins: {e_bin}")

        return pred_grid_ds

    def cleanup_debug_hooks(self):
        """Remove registered debugging hooks."""
        if hasattr(self, 'debug_hooks'):
            for hook in self.debug_hooks:
                hook.remove()
            self.logger.info(f"GridPredictor: Removed {len(self.debug_hooks)} debug hooks")
            self.debug_hooks = []

    def __del__(self):
        """Release debugging hooks when the predictor is destroyed."""
        try:
            self.cleanup_debug_hooks()
        except:
            pass

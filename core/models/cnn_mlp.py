import torch
import torch.nn as nn
import numpy as np
from core.utils.logger_config import logger
from core.datasets.rey_dataset import REYPointDataset

class LinearSelfAttention(nn.Module):
    """
    A computationally efficient Self-Attention mechanism with linear complexity.
    This avoids forming the large N x N attention matrix, making it suitable for large feature maps.
    The implementation is based on the idea of changing the order of matrix multiplications.
    """
    def __init__(self, in_channels, heads=4):
        super(LinearSelfAttention, self).__init__()
        self.heads = int(heads)
        if self.heads <= 0:
            raise ValueError(f"LinearSelfAttention heads must be > 0, got {heads}")
        if int(in_channels) % self.heads != 0:
            raise ValueError(
                f"LinearSelfAttention requires in_channels divisible by heads; got in_channels={in_channels}, heads={heads}"
            )
        self.scale = (in_channels // self.heads) ** -0.5


        self.to_qkv = nn.Conv2d(in_channels, in_channels * 3, 1, bias=False)
        self.to_out = nn.Conv2d(in_channels, in_channels, 1)

        logger.debug(f"LinearSelfAttention initialized with in_channels={in_channels}, heads={heads}")

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.reshape(b, self.heads, c // self.heads, h * w), qkv)


        k = k.softmax(dim=-1)


        context = torch.einsum('bhdn,bhen->bhde', k, v)


        out = torch.einsum('bhde,bhdn->bhen', context, q)

        out = out.reshape(b, c, h, w)
        return self.to_out(out)

class FactorizedEnvDepthModel(nn.Module):
    """
    A factorized model that separates environmental (spatial) effects
    from depth (vertical/stratigraphic) effects:

        REY(x, y, z) ≈ EnvScore(x, y) * DepthProfile(z)

    The input format remains identical to CNNModel: a combined patch
    where the last channel is the (normalized) depth channel broadcast
    over the spatial dimensions, so existing pipelines stay compatible.
    """

    def __init__(
        self,
        input_channels: int,
        device,
        cnn_filters=None,
        kernel_size: int = 3,
        mlp_hidden_layers=None,
        dropout_prob: float = 0.3,
        linear_attention_heads: int = 4,
        use_linear_attention: bool = True,
        depth_mlp_hidden_layers=None,
        use_multiscale_env: bool = False,
        multiscale_kernel_size: int = 5,
        use_film: bool = False,
        film_embedding_dim: int = 32,
        depth_feature_order: int = 1,
        combination_type: str = "product",
        **kwargs,
    ):
        super(FactorizedEnvDepthModel, self).__init__()
        self.device = device
        self.use_multiscale_env = bool(use_multiscale_env)
        self.use_film = bool(use_film)

        self.depth_feature_order = max(1, int(depth_feature_order))


        self.combination_type = (combination_type or "product").lower()

        if cnn_filters is None:
            cnn_filters = [64, 128, 128]
        if mlp_hidden_layers is None:
            mlp_hidden_layers = [128, 64]
        if depth_mlp_hidden_layers is None:
            depth_mlp_hidden_layers = [32, 16]

        if input_channels < 2:
            raise ValueError(
                f"FactorizedEnvDepthModel expects at least 2 channels "
                f"(env + depth), got input_channels={input_channels}."
            )

        logger.debug(
            "Initializing FactorizedEnvDepthModel with "
            f"input_channels={input_channels}, cnn_filters={cnn_filters}, "
            f"kernel_size={kernel_size}, mlp_hidden_layers={mlp_hidden_layers}, "
            f"depth_mlp_hidden_layers={depth_mlp_hidden_layers}, "
            f"linear_attention_heads={linear_attention_heads}, "
            f"combination_type={self.combination_type}"
        )


        env_input_channels = input_channels - 1


        env_cnn_layers = []
        in_channels_env = env_input_channels
        for idx_layer, out_channels in enumerate(cnn_filters):
            env_cnn_layers.extend(
                [
                    nn.Conv2d(in_channels_env, out_channels, kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2) if idx_layer < len(cnn_filters) - 1 else nn.Identity(),
                ]
            )
            in_channels_env = out_channels
        self.env_cnn_backbone = nn.Sequential(*env_cnn_layers)


        if self.use_multiscale_env:
            ms_cnn_layers = []
            in_channels_ms = env_input_channels
            k_ms = multiscale_kernel_size
            for idx_layer, out_channels in enumerate(cnn_filters):
                ms_cnn_layers.extend(
                    [
                        nn.Conv2d(in_channels_ms, out_channels, k_ms, padding=k_ms // 2),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(2, 2) if idx_layer < len(cnn_filters) - 1 else nn.Identity(),
                    ]
                )
                in_channels_ms = out_channels
            self.env_cnn_backbone_ms = nn.Sequential(*ms_cnn_layers)
        else:
            self.env_cnn_backbone_ms = None


        cnn_output_channels = cnn_filters[-1]
        self.use_linear_attention = bool(use_linear_attention)
        if self.use_linear_attention:
            self.env_global_attention = LinearSelfAttention(cnn_output_channels, heads=linear_attention_heads)
            if self.use_multiscale_env:
                self.env_global_attention_ms = LinearSelfAttention(cnn_output_channels, heads=linear_attention_heads)
        else:
            self.env_global_attention = nn.Identity()
            self.env_global_attention_ms = nn.Identity() if self.use_multiscale_env else None
        self.env_adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))


        env_feature_dim = cnn_output_channels * (2 if self.use_multiscale_env else 1)
        env_mlp_layers = []
        current_size_env = env_feature_dim
        for hidden_size in mlp_hidden_layers:
            env_mlp_layers.extend(
                [
                    nn.Linear(current_size_env, hidden_size),
                    nn.LeakyReLU(inplace=True),
                    nn.Dropout(dropout_prob),
                ]
            )
            current_size_env = hidden_size

        env_mlp_layers.append(nn.Linear(current_size_env, 1))
        self.env_head = nn.Sequential(*env_mlp_layers)


        self.depth_hidden_dims = list(depth_mlp_hidden_layers)
        self.depth_linears = nn.ModuleList()
        in_depth = self.depth_feature_order
        for hidden_size in self.depth_hidden_dims:
            self.depth_linears.append(nn.Linear(in_depth, hidden_size))
            in_depth = hidden_size
        self.depth_out = nn.Linear(in_depth, 1)
        self.depth_activation = nn.LeakyReLU(inplace=True)
        self.depth_dropout = nn.Dropout(dropout_prob)


        if self.use_film and self.depth_hidden_dims:
            self.film_embedding = nn.Sequential(
                nn.Linear(env_feature_dim, film_embedding_dim),
                nn.LeakyReLU(inplace=True),
            )
            self.film_gamma = nn.Linear(film_embedding_dim, self.depth_hidden_dims[0])
            self.film_beta = nn.Linear(film_embedding_dim, self.depth_hidden_dims[0])
        else:
            self.film_embedding = None
            self.film_gamma = None
            self.film_beta = None


        if getattr(self, "use_interaction_head", False):
            if "interaction_hidden_layers" in kwargs and kwargs["interaction_hidden_layers"] is not None:
                interaction_hidden_layers = list(kwargs["interaction_hidden_layers"])
            else:
                interaction_hidden_layers = [64, 32]
            interaction_embedding_dim = kwargs.get("interaction_embedding_dim", 32)

            self.interaction_env_proj = nn.Linear(env_feature_dim, interaction_embedding_dim)
            inter_layers = []
            in_inter = interaction_embedding_dim + self.depth_feature_order
            for hidden_size in interaction_hidden_layers:
                inter_layers.extend(
                    [
                        nn.Linear(in_inter, hidden_size),
                        nn.LeakyReLU(inplace=True),
                        nn.Dropout(dropout_prob),
                    ]
                )
                in_inter = hidden_size
            inter_layers.append(nn.Linear(in_inter, 1))
            self.interaction_head = nn.Sequential(*inter_layers)
        else:
            self.interaction_env_proj = None
            self.interaction_head = None

        logger.debug("FactorizedEnvDepthModel initialization complete.")

    @staticmethod
    def _split_env_and_depth(combined_patch: torch.Tensor):
        """
        Split a combined patch tensor into env_patch and depth_scalar.

        combined_patch: [batch, channels_total, H, W]
        env_patch:      [batch, channels_total-1, H, W]
        depth_scalar:   [batch, 1]  (spatial mean of last channel)
        """
        if combined_patch.dim() != 4:
            raise ValueError(
                f"Expected combined_patch with 4 dimensions "
                f"[batch, channels, height, width], got shape={combined_patch.shape}."
            )

        batch_size, channels_total, height, width = combined_patch.shape
        if channels_total < 2:
            raise ValueError(
                f"FactorizedEnvDepthModel requires at least 2 channels "
                f"(env + depth), but got channels_total={channels_total}."
            )

        env_channels = channels_total - 1
        env_patch = combined_patch[:, :env_channels, :, :]
        depth_channel = combined_patch[:, env_channels:, :, :]

        # Depth channel is expected to be spatially constant; we take the mean
        depth_scalar = depth_channel.mean(dim=(2, 3))
        return env_patch, depth_scalar

    def forward(self, combined_patch: torch.Tensor, return_components: bool = False):
        """
        Forward pass with factorized Env and Depth contributions.

        Returns:
            prediction: [batch, 1]
            optionally a dict with 'env_score' and 'depth_factor' for diagnostics.
        """
        env_patch, depth_scalar = self._split_env_and_depth(combined_patch)


        env_features = self.env_cnn_backbone(env_patch)
        env_attended = self.env_global_attention(env_features)
        env_vec_main = torch.flatten(self.env_adaptive_pool(env_attended), 1)

        if self.use_multiscale_env and self.env_cnn_backbone_ms is not None:
            env_features_ms = self.env_cnn_backbone_ms(env_patch)
            env_attended_ms = self.env_global_attention_ms(env_features_ms)
            env_vec_ms = torch.flatten(self.env_adaptive_pool(env_attended_ms), 1)
            env_vec = torch.cat([env_vec_main, env_vec_ms], dim=1)
        else:
            env_vec = env_vec_main

        env_score = self.env_head(env_vec)


        d = depth_scalar
        depth_features = [d]
        for power in range(2, self.depth_feature_order + 1):
            depth_features.append(d ** power)
        x_depth = torch.cat(depth_features, dim=1)
        gamma = beta = None
        if self.use_film and self.film_embedding is not None:
            emb = self.film_embedding(env_vec)
            gamma = self.film_gamma(emb)
            beta = self.film_beta(emb)

        for idx, linear in enumerate(self.depth_linears):
            x_depth = linear(x_depth)
            if idx == 0 and gamma is not None and beta is not None:

                x_depth = x_depth * (1.0 + gamma) + beta
            x_depth = self.depth_activation(x_depth)
            x_depth = self.depth_dropout(x_depth)


        depth_factor = self.depth_out(x_depth)


        if self.combination_type == "additive":
            prediction = env_score + depth_factor
        elif self.combination_type == "product":
            prediction = env_score * depth_factor
        else:
            raise ValueError(
                f"Unsupported combination_type='{self.combination_type}'. "
                "Use 'product' or 'additive'."
            )

        if return_components:
            components = {"env_score": env_score, "depth_factor": depth_factor}
            return prediction, components

        return prediction

    @staticmethod
    def get_spatial_info_for_sample(dataset: REYPointDataset, sample_idx: int, patch_size_model: tuple):
        try:
            point_data = dataset.rey_data.iloc[sample_idx]
            center_lon, center_lat = point_data['lon'], point_data['lat']

            env_lon_res = dataset.lon_res
            env_lat_res = dataset.lat_res

            half_width_lon = (patch_size_model[1] / 2.0) * env_lon_res
            half_height_lat = (patch_size_model[0] / 2.0) * env_lat_res

            min_lon = center_lon - half_width_lon + (env_lon_res / 2.0)
            max_lon = center_lon + half_width_lon - (env_lon_res / 2.0)
            min_lat = center_lat - half_height_lat + (env_lat_res / 2.0)
            max_lat = center_lat + half_height_lat - (env_lat_res / 2.0)

            patch_lon_coords = np.linspace(min_lon, max_lon, patch_size_model[1])
            patch_lat_coords = np.linspace(max_lat, min_lat, patch_size_model[0])

            return {
                "center_lon": center_lon,
                "center_lat": center_lat,
                "patch_lon_coords": patch_lon_coords,
                "patch_lat_coords": patch_lat_coords,
                "lon_resolution_of_input_patch_pixel": env_lon_res,
                "lat_resolution_of_input_patch_pixel": env_lat_res,
                "original_rey_point_data": point_data.to_dict(),
            }
        except Exception as e:
            logger.error(f"Error getting spatial info for sample {sample_idx}: {e}")
            return None


class MixtureOfProfilesModel(nn.Module):
    """
    Mixture-of-Profiles (MoP) model.

    y_norm(x, y, z) ≈ Σ_k π_k(x, y) * g_k(z)

    - Env branch: CNN + linear self-attention + pooling -> env embedding ->
      mixture weights π_k(x, y) (softmax over K profiles).
    - Depth branch: polynomial features of normalized depth -> MLP -> g_k(z)
      (one scalar per profile).
    """

    def __init__(
        self,
        input_channels,
        device,
        cnn_filters=None,
        kernel_size=3,
        mlp_hidden_layers=None,
        dropout_prob=0.3,
        linear_attention_heads=4,
        use_linear_attention: bool = True,
        use_multiscale_env=False,
        multiscale_kernel_size=5,
        depth_feature_order=2,
        num_profiles=3,
        env_embedding_dim=64,
        profile_hidden_layers=None,

        num_classes: int = 1,
        class_head_hidden_dim: int = 64,
        use_high_grade_ranking_head: bool = False,
        high_grade_ranking_head_hidden_dim: int = 64,

        coord_feature_dim: int = 0,
        use_coord_branch: bool = False,
        coord_branch_hidden_dim: int = 8,
        coord_branch_dropout: float = 0.5,
        coord_branch_weight: float = 1.0,
        **kwargs,
    ):
        super(MixtureOfProfilesModel, self).__init__()
        self.device = device
        self.use_multiscale_env = use_multiscale_env
        self.use_linear_attention = bool(use_linear_attention)
        self.num_profiles = int(num_profiles)
        if self.num_profiles < 2:
            raise ValueError(
                f"MixtureOfProfilesModel requires num_profiles >= 2, got {self.num_profiles}."
            )

        if cnn_filters is None:
            cnn_filters = [64, 128, 128]
        if mlp_hidden_layers is None:
            mlp_hidden_layers = [128, 64]

        self.depth_feature_order = max(1, int(depth_feature_order))
        self.env_embedding_dim = int(env_embedding_dim)
        self.num_classes = int(num_classes or 1)
        if self.num_classes < 1:
            raise ValueError(f"MixtureOfProfilesModel requires num_classes >= 1, got {self.num_classes}.")
        self.class_head_hidden_dim = int(class_head_hidden_dim or 64)
        self.use_high_grade_ranking_head = bool(use_high_grade_ranking_head) and (self.num_classes > 1)
        self.high_grade_ranking_head_hidden_dim = int(high_grade_ranking_head_hidden_dim or 64)

        self.coord_feature_dim = int(coord_feature_dim or 0)
        self.use_coord_branch = bool(use_coord_branch) and (self.coord_feature_dim > 0)
        self.coord_branch_hidden_dim = int(coord_branch_hidden_dim or 8)
        self.coord_branch_dropout = float(coord_branch_dropout) if coord_branch_dropout is not None else 0.5
        self.coord_branch_weight = float(coord_branch_weight) if coord_branch_weight is not None else 1.0

        if input_channels < 2:
            raise ValueError(
                "MixtureOfProfilesModel expects at least 2 channels "
                f"(env + depth), got input_channels={input_channels}."
            )

        logger.debug(
            "Initializing MixtureOfProfilesModel with "
            f"input_channels={input_channels}, cnn_filters={cnn_filters}, "
            f"kernel_size={kernel_size}, depth_feature_order={self.depth_feature_order}, "
            f"num_profiles={self.num_profiles}, env_embedding_dim={self.env_embedding_dim}"
        )

        if input_channels < (1 + 1 + self.coord_feature_dim):
            raise ValueError(
                "MixtureOfProfilesModel input_channels is inconsistent with coord_feature_dim. "
                f"Need >= env+depth+coord = (1+1+{self.coord_feature_dim}), got input_channels={input_channels}."
            )

        env_input_channels = input_channels - (1 + self.coord_feature_dim)
        self.env_input_channels = int(env_input_channels)


        raw_cat_idx = kwargs.get("categorical_channel_indices", None)
        try:
            cat_idx = [int(i) for i in (raw_cat_idx or [])]
        except Exception:
            cat_idx = []
        cat_idx = sorted({i for i in cat_idx if 0 <= i < int(env_input_channels)})
        self.categorical_channel_indices = cat_idx
        self.categorical_compress_dim = int(kwargs.get("categorical_compress_dim") or 0)
        _raw_cat_drop = kwargs.get("categorical_compress_dropout", None)
        if _raw_cat_drop is None:
            _raw_cat_drop = dropout_prob if dropout_prob is not None else 0.25
        self.categorical_compress_dropout = float(_raw_cat_drop)
        self._numeric_channel_indices = [i for i in range(int(env_input_channels)) if i not in set(cat_idx)]

        if self.categorical_channel_indices and self.categorical_compress_dim > 0:
            in_cat = len(self.categorical_channel_indices)
            self.categorical_compress = nn.Sequential(
                nn.Conv2d(in_cat, self.categorical_compress_dim, kernel_size=1),
                nn.BatchNorm2d(self.categorical_compress_dim),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=self.categorical_compress_dropout),
            )
            env_cnn_in_channels = len(self._numeric_channel_indices) + self.categorical_compress_dim
            logger.info(
                "MixtureOfProfilesModel: Enabled categorical compression "
                f"(cat_in={in_cat} -> cat_dim={self.categorical_compress_dim}, "
                f"env_in={int(env_input_channels)} -> env_cnn_in={env_cnn_in_channels})."
            )
        else:
            self.categorical_compress = None
            env_cnn_in_channels = int(env_input_channels)


        env_cnn_layers = []
        in_channels_env = env_cnn_in_channels
        for idx_layer, out_channels in enumerate(cnn_filters):
            env_cnn_layers.extend(
                [
                    nn.Conv2d(in_channels_env, out_channels, kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2) if idx_layer < len(cnn_filters) - 1 else nn.Identity(),
                ]
            )
            in_channels_env = out_channels
        self.env_cnn_backbone = nn.Sequential(*env_cnn_layers)

        if self.use_multiscale_env:
            ms_cnn_layers = []
            in_channels_ms = env_cnn_in_channels
            k_ms = multiscale_kernel_size
            for idx_layer, out_channels in enumerate(cnn_filters):
                ms_cnn_layers.extend(
                    [
                        nn.Conv2d(in_channels_ms, out_channels, k_ms, padding=k_ms // 2),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(2, 2) if idx_layer < len(cnn_filters) - 1 else nn.Identity(),
                    ]
                )
                in_channels_ms = out_channels
            self.env_cnn_backbone_ms = nn.Sequential(*ms_cnn_layers)
        else:
            self.env_cnn_backbone_ms = None


        cnn_output_channels = cnn_filters[-1]
        if self.use_linear_attention:
            self.env_global_attention = LinearSelfAttention(cnn_output_channels, heads=linear_attention_heads)
            if self.use_multiscale_env:
                self.env_global_attention_ms = LinearSelfAttention(cnn_output_channels, heads=linear_attention_heads)
            else:
                self.env_global_attention_ms = None
        else:
            self.env_global_attention = nn.Identity()
            self.env_global_attention_ms = nn.Identity() if self.use_multiscale_env else None
        self.env_adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))


        env_feature_dim = cnn_output_channels * (2 if self.use_multiscale_env else 1)
        env_mlp_layers = []
        current_size_env = env_feature_dim
        if mlp_hidden_layers is None:
            mlp_hidden_layers = [128, 64]
        for hidden_size in mlp_hidden_layers:
            env_mlp_layers.extend(
                [
                    nn.Linear(current_size_env, hidden_size),
                    nn.LeakyReLU(inplace=True),
                    nn.Dropout(dropout_prob),
                ]
            )
            current_size_env = hidden_size
        env_mlp_layers.append(nn.Linear(current_size_env, self.env_embedding_dim))
        env_mlp_layers.append(nn.LeakyReLU(inplace=True))
        self.env_embedding_mlp = nn.Sequential(*env_mlp_layers)


        self.profile_classifier = nn.Linear(self.env_embedding_dim, self.num_profiles)


        self.use_env_scale = True
        self.env_scale_head = nn.Linear(self.env_embedding_dim, 1)


        if profile_hidden_layers is None:
            profile_hidden_layers = [32, 16]
        self.profile_hidden_layers = list(profile_hidden_layers)
        depth_layers = []
        in_depth = self.depth_feature_order
        for hidden_size in self.profile_hidden_layers:
            depth_layers.extend(
                [
                    nn.Linear(in_depth, hidden_size),
                    nn.LeakyReLU(inplace=True),
                    nn.Dropout(dropout_prob),
                ]
            )
            in_depth = hidden_size
        depth_layers.append(nn.Linear(in_depth, self.num_profiles))
        self.depth_mlp = nn.Sequential(*depth_layers)


        if self.num_classes > 1:

            class_in_dim = self.env_embedding_dim + self.num_profiles + self.num_profiles + 1 + self.depth_feature_order
            self.class_head = nn.Sequential(
                nn.Linear(class_in_dim, self.class_head_hidden_dim),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(dropout_prob),
                nn.Linear(self.class_head_hidden_dim, self.num_classes),
            )
            if self.use_high_grade_ranking_head:
                self.high_grade_ranking_head = nn.Sequential(
                    nn.Linear(class_in_dim, self.high_grade_ranking_head_hidden_dim),
                    nn.LeakyReLU(inplace=True),
                    nn.Dropout(dropout_prob),
                    nn.Linear(self.high_grade_ranking_head_hidden_dim, 1),
                )
            else:
                self.high_grade_ranking_head = None
        else:
            self.class_head = None
            self.high_grade_ranking_head = None


        if self.use_coord_branch:
            self.coord_mlp = nn.Sequential(
                nn.Linear(self.coord_feature_dim, self.coord_branch_hidden_dim),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(self.coord_branch_dropout),
                nn.Linear(self.coord_branch_hidden_dim, 1),
            )

            try:
                last = self.coord_mlp[-1]
                if isinstance(last, nn.Linear):
                    nn.init.zeros_(last.weight)
                    nn.init.zeros_(last.bias)
            except Exception:
                pass
        else:
            self.coord_mlp = None

        logger.debug("MixtureOfProfilesModel initialization complete.")

    def _split_env_depth_coord(self, combined_patch: torch.Tensor):
        if combined_patch.ndim != 4:
            raise ValueError(f"Expected combined_patch [B,C,H,W], got shape {tuple(combined_patch.shape)}")

        b, c, h, w = combined_patch.shape
        c_env = int(self.env_input_channels)
        c_coord = int(self.coord_feature_dim)
        if c < (c_env + 1 + c_coord):
            raise ValueError(
                f"combined_patch has too few channels (c={c}) for env={c_env}, depth=1, coord={c_coord}."
            )

        env_patch = combined_patch[:, :c_env, :, :]
        depth_channel = combined_patch[:, c_env : c_env + 1, :, :]
        depth_scalar = depth_channel.mean(dim=(2, 3))

        coord_features = None
        if c_coord > 0:
            coord_channels = combined_patch[:, c_env + 1 : c_env + 1 + c_coord, :, :]
            coord_features = coord_channels.mean(dim=(2, 3))

        return env_patch, depth_scalar, coord_features

    def forward(self, combined_patch: torch.Tensor, return_components: bool = False):
        """
        Forward pass.

        combined_patch: [batch, channels_total, H, W]
        Returns:
          - prediction: [batch, 1] in normalized REY space.
          - optionally a dict with 'pi' and 'profiles' for diagnostics.
        """
        env_patch, depth_scalar, coord_features = self._split_env_depth_coord(combined_patch)

        if self.categorical_compress is not None and self.categorical_channel_indices:
            cat_patch = env_patch[:, self.categorical_channel_indices, :, :]
            num_patch = (
                env_patch[:, self._numeric_channel_indices, :, :]
                if self._numeric_channel_indices
                else None
            )
            cat_emb = self.categorical_compress(cat_patch)
            env_patch = torch.cat([num_patch, cat_emb], dim=1) if num_patch is not None else cat_emb


        env_features = self.env_cnn_backbone(env_patch)
        env_attended = self.env_global_attention(env_features)
        env_vec_main = torch.flatten(self.env_adaptive_pool(env_attended), 1)

        if self.use_multiscale_env and self.env_cnn_backbone_ms is not None:
            env_features_ms = self.env_cnn_backbone_ms(env_patch)
            env_attended_ms = self.env_global_attention_ms(env_features_ms)
            env_vec_ms = torch.flatten(self.env_adaptive_pool(env_attended_ms), 1)
            env_vec = torch.cat([env_vec_main, env_vec_ms], dim=1)
        else:
            env_vec = env_vec_main

        env_embedding = self.env_embedding_mlp(env_vec)
        pi_logits = self.profile_classifier(env_embedding)
        pi = torch.softmax(pi_logits, dim=1)


        d = depth_scalar
        depth_features = [d]
        for power in range(2, self.depth_feature_order + 1):
            depth_features.append(d ** power)
        x_depth = torch.cat(depth_features, dim=1)

        profiles = self.depth_mlp(x_depth)


        y_norm_base = torch.sum(pi * profiles, dim=1, keepdim=True)


        if self.use_env_scale and self.env_scale_head is not None:
            scale_logits = self.env_scale_head(env_embedding)
            scale = 1.0 + torch.nn.functional.softplus(scale_logits)
        else:
            scale = 1.0

        prediction = y_norm_base * scale

        coord_bias = None
        if self.use_coord_branch and self.coord_mlp is not None and coord_features is not None:
            coord_bias = self.coord_mlp(coord_features)
            prediction = prediction + (self.coord_branch_weight * coord_bias)


        class_logits = None
        high_grade_ranking_logit = None
        if self.class_head is not None:
            class_features = torch.cat([env_embedding, pi, profiles, scale, x_depth], dim=1)
            class_logits = self.class_head(class_features)
            if self.high_grade_ranking_head is not None:
                high_grade_ranking_logit = self.high_grade_ranking_head(class_features)

        if return_components:
            components = {
                "pi": pi,
                "profiles": profiles,
                "env_embedding": env_embedding,
                "y_env_depth": y_norm_base,
                "env_scale": scale,
                "coord_features": coord_features,
                "coord_bias": coord_bias,
                "class_logits": class_logits,
                "high_grade_ranking_logit": high_grade_ranking_logit,
                "x_depth": x_depth,
            }
            return (class_logits if class_logits is not None else prediction), components

        return class_logits if class_logits is not None else prediction


class HierarchicalMixtureOfProfilesModel(nn.Module):
    """
    Hierarchical multiclass model for two-stage and grade-aware labels:

    - Head A (endmember): carbonate vs non-carbonate (2 logits)
    - Head B (subclass): non-carbonate mechanism subclasses (K logits), or class1/class2 in grade multitask mode
    - Head C (grade, optional): low vs high grade inside class1/class2 (2 logits)

    Legacy two-head probabilities are composed as:
      p(class0=carbonate) = p_end[0]
      p(class(1+i))       = p_end[1] * p_sub[i]

    Grade multitask probabilities are composed as:
      p(class0=carbonate) = p_end[0]
      p(class1_low)       = p_end[1] * p_sub[class1] * p_grade[low]
      p(class1_high)      = p_end[1] * p_sub[class1] * p_grade[high]
      p(class2_low)       = p_end[1] * p_sub[class2] * p_grade[low]
      p(class2_high)      = p_end[1] * p_sub[class2] * p_grade[high]

    Notes
    - This is designed for the repo's two-stage mechanism clustering labels where class0 is carbonate.
    - Training should apply endmember loss on all samples and subclass loss only on non-carbonate samples.
    """

    def __init__(
        self,
        *,
        input_channels: int,
        device,
        num_subclasses: int,
        cnn_filters=None,
        kernel_size=3,
        mlp_hidden_layers=None,
        dropout_prob=0.3,
        linear_attention_heads=4,
        use_multiscale_env=False,
        multiscale_kernel_size=5,
        depth_feature_order=2,
        num_profiles=3,
        env_embedding_dim=64,
        profile_hidden_layers=None,
        endmember_head_hidden_dim: int = 64,
        subclass_head_hidden_dim: int = 64,
        hierarchical_grade_multitask: bool = False,
        grade_head_hidden_dim: int = 64,
        coord_feature_dim: int = 0,
        use_coord_branch: bool = False,
        coord_branch_hidden_dim: int = 8,
        coord_branch_dropout: float = 0.5,
        coord_branch_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.device = device
        self.hierarchical_grade_multitask = bool(hierarchical_grade_multitask)
        self.num_subclasses = int(num_subclasses or 0)
        if self.num_subclasses < 2:
            raise ValueError(f"HierarchicalMixtureOfProfilesModel requires num_subclasses>=2, got {self.num_subclasses}.")
        if self.hierarchical_grade_multitask and self.num_subclasses != 2:
            raise ValueError(
                "HierarchicalMixtureOfProfilesModel grade multitask mode requires "
                f"num_subclasses=2 for class1/class2, got {self.num_subclasses}."
            )


        self.backbone = MixtureOfProfilesModel(
            input_channels=input_channels,
            device=device,
            cnn_filters=cnn_filters,
            kernel_size=kernel_size,
            mlp_hidden_layers=mlp_hidden_layers,
            dropout_prob=dropout_prob,
            linear_attention_heads=linear_attention_heads,
            use_multiscale_env=use_multiscale_env,
            multiscale_kernel_size=multiscale_kernel_size,
            depth_feature_order=depth_feature_order,
            num_profiles=num_profiles,
            env_embedding_dim=env_embedding_dim,
            profile_hidden_layers=profile_hidden_layers,
            num_classes=1,
            class_head_hidden_dim=1,
            coord_feature_dim=coord_feature_dim,
            use_coord_branch=use_coord_branch,
            coord_branch_hidden_dim=coord_branch_hidden_dim,
            coord_branch_dropout=coord_branch_dropout,
            coord_branch_weight=coord_branch_weight,
            **kwargs,
        )

        k = int(self.backbone.num_profiles)
        depth_order = int(self.backbone.depth_feature_order)
        env_dim = int(self.backbone.env_embedding_dim)


        feature_dim = env_dim + k + k + 1 + depth_order

        def _make_head(out_dim: int, hidden_dim: int) -> nn.Module:
            h = int(hidden_dim or 64)
            return nn.Sequential(
                nn.Linear(feature_dim, h),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(float(dropout_prob) if dropout_prob is not None else 0.25),
                nn.Linear(h, int(out_dim)),
            )

        self.endmember_head = _make_head(2, int(endmember_head_hidden_dim or 64))
        self.subclass_head = _make_head(self.num_subclasses, int(subclass_head_hidden_dim or 64))
        self.grade_head = (
            _make_head(2, int(grade_head_hidden_dim or subclass_head_hidden_dim or 64))
            if self.hierarchical_grade_multitask
            else None
        )

    def forward(self, combined_patch: torch.Tensor):
        _, comp = self.backbone(combined_patch, return_components=True)
        env_embedding = comp["env_embedding"]
        pi = comp["pi"]
        profiles = comp["profiles"]
        scale = comp["env_scale"]
        x_depth = comp["x_depth"]
        features = torch.cat([env_embedding, pi, profiles, scale, x_depth], dim=1)
        end_logits = self.endmember_head(features)
        sub_logits = self.subclass_head(features)
        if self.grade_head is not None:
            grade_logits = self.grade_head(features)
            return end_logits, sub_logits, grade_logits
        return end_logits, sub_logits

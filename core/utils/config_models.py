from pydantic import BaseModel, Field
from typing import List, Dict, Tuple, Optional, Literal

class TargetNormalizationParams(BaseModel):
    mean: float
    std: float
    transform: Optional[str] = None
    scale: Optional[float] = None
    hinge_beta: Optional[float] = None


    task_type: Optional[str] = None
    threshold_ppm: Optional[float] = None
    num_classes: Optional[int] = None
    class_names: Optional[List[str]] = None

class NormalizationParams(BaseModel):
    channel_means: List[float]
    channel_stds: List[float]
    rey: TargetNormalizationParams


    depth_feature_type: Optional[str] = None
    depth_feature_min: Optional[float] = None
    depth_feature_max: Optional[float] = None

    # Backward-compatible aliases (older artifacts stored these as depth_min/depth_max).
    depth_min: Optional[float] = None
    depth_max: Optional[float] = None

    depth_transform_method: Optional[str] = None

class ModelInitConfig(BaseModel):


    model_type: str = "mixture_profiles"
    input_channels: int
    cnn_filters: List[int]
    kernel_size: int
    mlp_hidden_layers: List[int]
    dropout_prob: float
    attention_bottleneck_factor: Optional[int] = None
    depth_mlp_output_dim: Optional[int] = None
    patch_size: List[int]
    feature_names: List[str]
    linear_attention_heads: Optional[int] = 4

    use_linear_attention: Optional[bool] = True

    use_multiscale_env: Optional[bool] = False
    multiscale_kernel_size: Optional[int] = 5
    use_film: Optional[bool] = False
    film_embedding_dim: Optional[int] = 32
    depth_feature_order: Optional[int] = 1

    num_profiles: Optional[int] = None
    env_embedding_dim: Optional[int] = None
    profile_hidden_layers: Optional[List[int]] = None

    coord_feature_dim: Optional[int] = None
    coord_feature_names: Optional[List[str]] = None
    add_env_missingness_channels: Optional[bool] = None
    env_missingness_feature_names: Optional[List[str]] = None
    use_coord_branch: Optional[bool] = None
    coord_branch_hidden_dim: Optional[int] = None
    coord_branch_dropout: Optional[float] = None
    coord_branch_weight: Optional[float] = None

    num_classes: Optional[int] = None
    class_head_hidden_dim: Optional[int] = None
    use_high_grade_ranking_head: Optional[bool] = None
    high_grade_ranking_head_hidden_dim: Optional[int] = None
    # Hierarchical multiclass configuration. Legacy artifacts use two heads

    hierarchical_multiclass: Optional[bool] = None
    hierarchical_grade_multitask: Optional[bool] = None
    endmember_head_hidden_dim: Optional[int] = None
    subclass_head_hidden_dim: Optional[int] = None
    grade_head_hidden_dim: Optional[int] = None


    categorical_channel_indices: Optional[List[int]] = None


    categorical_compress_dim: Optional[int] = None
    categorical_compress_dropout: Optional[float] = None

class TrainingArtifacts(BaseModel):
    model_path: str
    model_init_config: ModelInitConfig
    norm_params: NormalizationParams


class CascadeStageArtifacts(BaseModel):
    stage_name: str
    training_artifacts_path: str
    model_path: str
    classification_label_col: str
    class_names: List[str]
    num_classes: int


class CascadeTrainingArtifacts(BaseModel):
    artifact_type: Literal["cascade_grade_geochem_classification", "cascade_exploration_4class"] = "cascade_grade_geochem_classification"
    final_label_col: str
    final_class_names: List[str]
    final_num_classes: int
    combination_rule: str
    stage_main: CascadeStageArtifacts
    stage_class1_grade: Optional[CascadeStageArtifacts] = None
    stage_class2_grade: CascadeStageArtifacts
    high_grade_score_formula: Optional[str] = None
    high_grade_score_source: Optional[str] = None


class CascadeEnsembleMember(BaseModel):
    seed: int
    split_random_state: Optional[int] = None
    run_dir: str
    cascade_artifacts_path: str
    weight: float = 1.0


class CascadeEnsembleArtifacts(BaseModel):
    artifact_type: Literal["cascade_grade_geochem_ensemble", "cascade_exploration_4class_ensemble"] = "cascade_grade_geochem_ensemble"
    final_label_col: str
    final_class_names: List[str]
    final_num_classes: int
    combination_rule: str
    ensemble_rule: str = "arithmetic_mean_final_probabilities"
    members: List[CascadeEnsembleMember]
    recommended_high_grade_threshold: Optional[float] = None
    recommended_threshold_metric: Optional[str] = None
    recommended_threshold_source: Optional[str] = None
    high_grade_score_formula: Optional[str] = None
    high_grade_score_source: Optional[str] = None

class GeospatialBounds(BaseModel):
    min_lon: Optional[float] = None
    max_lon: Optional[float] = None
    min_lat: Optional[float] = None
    max_lat: Optional[float] = None

class PredictionGridSettings(BaseModel):
    geospatial_bounds: Optional[GeospatialBounds] = None
    lon_res: float
    lat_res: float
    vertical_axis_mode: Literal["depth_mbsf", "age_ka"] = Field(
        default="depth_mbsf",
        description="Vertical axis for prediction outputs: 'depth_mbsf' (meters below seafloor) or 'age_ka' (time slices).",
    )
    depth_levels: List[float] = Field(
        default_factory=list,
        description="Physical depth levels (mbsf, meters). Used when vertical_axis_mode='depth_mbsf'.",
    )
    age_levels_ka: List[float] = Field(
        default_factory=list,
        description="Age levels (ka). Used when vertical_axis_mode='age_ka'.",
    )

class PlottingConfig(BaseModel):
    figsize_pred_map: Tuple[int, int] = (12, 8)
    pred_map_qmin: int = 2
    pred_map_qmax: int = 98
    pred_map_cmap: str = "viridis"
    pred_map_coastlines: bool = True
    pred_map_borders: bool = True
    pred_map_land: bool = True
    pred_map_ocean: bool = True
    pred_map_gridlines: bool = True
    dpi_pred_map: int = 150

class PredictionConfig(BaseModel):
    training_artifacts_path: str
    env_gridded_nc_path: str
    batch_size: int
    prediction_grid_settings: PredictionGridSettings
    plotting: PlottingConfig

    enable_mc_dropout: bool = Field(default=False, description="Enable MC Dropout at prediction time to estimate uncertainty.")
    mc_dropout_samples: int = Field(default=30, description="Number of MC Dropout forward passes for uncertainty.")
    depth_bins_m: Optional[List[float]] = Field(default_factory=lambda: [0.0, 3.0, 6.0, 15.0], description="Depth bin edges for per-layer aggregation.")
    age_bins_ka: Optional[List[float]] = Field(
        default_factory=lambda: [0.0, 1000.0, 2580.0, 10000.0],
        description="Age bin edges (ka) for per-layer aggregation when vertical_axis_mode='age_ka'.",
    )

    perform_interpolation: bool = Field(default=False, description="Interpolate missing values in the input environmental grid")
    interpolation_method: Literal["linear", "nearest", "cubic"] = Field(default="linear", description="Interpolation method for missing values")
    use_land_mask: bool = Field(default=True, description="Apply a land mask to exclude land cells")
    use_iterative_imputer: bool = Field(default=False, description="Use multivariate iterative imputation for missing values")


    output_mode: Literal["slices_3d", "summaries_2d", "both"] = Field(
        default="summaries_2d",
        description="Prediction outputs: full 3D slices, 2D age-window summaries, or both.",
    )


    enable_2d_age_summaries: bool = Field(
        default=True,
        description="Generate 2D summary maps aggregated over age windows (requires vertical age proxy).",
    )

    holocene_end_ka: float = Field(default=11.7, description="Holocene window end (ka) for near-modern proxy.")
    lgc_end_ka: float = Field(default=130.0, description="Last glacial cycle window end (ka) for sensitivity reporting.")

    enable_holocene_summary: bool = Field(default=True, description="Generate Holocene (0–11.7 ka) summary maps.")
    enable_lgc_summary: bool = Field(default=True, description="Generate Last glacial cycle (0–130 ka) summary maps.")
    enable_quaternary_summary: bool = Field(default=True, description="Generate Quaternary-domain (0–training_age_domain_ka) summary maps.")

    summary_quantile: float = Field(default=0.95, description="Quantile for 'high-value' summary map (e.g., p95).")
    summary_sampling_scheme: Literal["linear", "quadratic"] = Field(
        default="linear",
        description="How to sample ages inside each window for summaries. 'quadratic' concentrates samples near 0 ka.",
    )

    summary_n_samples_holocene: int = Field(default=5, description="Number of age samples within Holocene window.")
    summary_n_samples_lgc: int = Field(default=7, description="Number of age samples within 0–130 ka window.")
    summary_n_samples_quaternary: int = Field(default=9, description="Number of age samples within 0–training_age_domain_ka window.")


    summary_plot_shared_scale: bool = Field(
        default=True,
        description="If True, use shared color scales across windows for mean/pXX maps so they are visually comparable.",
    )
    summary_generate_delta_maps: bool = Field(
        default=True,
        description="If True, generate window-to-window difference maps (e.g., Holocene minus Quaternary).",
    )
    summary_generate_age_sensitivity_maps: bool = Field(
        default=True,
        description="If True, generate age-sensitivity maps within each window (std and quantile spread across sampled ages).",
    )
    summary_include_env_diagnostics_maps: bool = Field(
        default=True,
        description="If True, include environmental coverage/quality diagnostics (e.g., missing flags, sedimentation-rate filling).",
    )


    enable_threshold_exceedance_maps: bool = Field(
        default=False,
        description="If True, generate threshold exceedance frequency/any-exceed maps within each age window based on regression predictions.",
    )
    threshold_exceedance_ppm: float = Field(
        default=1000.0,
        description="Threshold (ppm) used for exceedance summaries (e.g., >=1000 ppm).",
    )
    threshold_exceedance_include_any: bool = Field(
        default=True,
        description="If True, generate a binary 'any exceedance within window' map (0/1).",
    )
    threshold_exceedance_include_freq: bool = Field(
        default=True,
        description="If True, generate an exceedance frequency map (fraction of sampled ages with pred>=threshold).",
    )
    threshold_exceedance_include_p95: bool = Field(
        default=False,
        description="If True, generate a high-quantile map of the exceedance indicator within window (useful when n_samples is large).",
    )


    enable_multiclass_target_threshold_maps: bool = Field(
        default=True,
        description="If True (and task is multiclass), generate threshold-based one-vs-rest hotspot maps for the target class within each age window.",
    )
    multiclass_target_class_index: int = Field(
        default=1,
        description="Target class index for one-vs-rest hotspot maps (e.g., class1 = enriched endmember).",
    )
    multiclass_target_threshold_source: Literal[
        "manual",
        "from_val_recommendation_f0_5",
        "from_val_recommendation_f1",
        "from_val_recommendation_f2",
    ] = Field(
        default="from_val_recommendation_f1",
        description="How to choose the probability threshold for the target class hotspot maps.",
    )
    multiclass_target_threshold: float = Field(
        default=0.5,
        description="Manual probability threshold for target class hotspot maps when multiclass_target_threshold_source='manual'.",
    )
    multiclass_target_include_any: bool = Field(
        default=True,
        description="If True, output a binary any-hotspot map (any sampled age in window exceeds threshold).",
    )
    multiclass_target_include_freq: bool = Field(
        default=True,
        description="If True, output a hotspot frequency map (fraction of sampled ages exceeding threshold).",
    )

    training_age_domain_ka: float = Field(
        default=2580.0,
        description="Training-domain age threshold (ka) used for in-domain vs out-of-domain diagnostics in prediction.",
    )
    restrict_to_training_age_domain: bool = Field(
        default=True,
        description="If True, set predictions to NaN when age_ka_proxy exceeds training_age_domain_ka.",
    )
    mask_out_of_domain_in_maps: bool = Field(
        default=True,
        description="If True, apply the in-domain mask when generating maps (NetCDF retains full fields unless restrict_to_training_age_domain=True).",
    )
    apply_depth_extrapolation_attenuation: bool = Field(
        default=True,
        description="Apply a smooth attenuation to predictions when the depth-feature is outside the training range.",
    )


    dominant_confidence_threshold: float = Field(
        default=0.75,
        description="For paper-ready mechanism partition maps, mask dominant-class pixels where pmax < threshold.",
    )
    enable_debug_hooks: bool = Field(
        default=False,
        description="Enable heavy debug hooks during prediction (BatchNorm anomaly hooks).",
    )

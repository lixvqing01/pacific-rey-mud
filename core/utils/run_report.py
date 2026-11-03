from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional


def _fmt(x: Any) -> str:
    if x is None:
        return "N/A"
    if isinstance(x, float):
        return f"{x:.6g}"
    return str(x)


def _extract_test_metrics(run_summary: Dict[str, Any]) -> Dict[str, Any]:
    return (run_summary.get("results", {}) or {}).get("test_metrics", {}) or {}


def _extract_test_metrics_in_domain(run_summary: Dict[str, Any]) -> Dict[str, Any]:
    return (run_summary.get("results", {}) or {}).get("test_metrics_in_domain", {}) or {}


def _extract_test_in_domain_n(run_summary: Dict[str, Any]) -> Optional[int]:
    n = (run_summary.get("results", {}) or {}).get("test_in_domain_n", None)
    try:
        return int(n) if n is not None else None
    except Exception:
        return None


def _extract_training_hyperparams(run_summary: Dict[str, Any]) -> Dict[str, Any]:
    cfg = run_summary.get("config", {}) or {}
    training = cfg.get("training", {}) or {}
    return training.get("hyperparams", {}) or {}

def _extract_training_split_cfg(run_summary: Dict[str, Any]) -> Dict[str, Any]:
    cfg = run_summary.get("config", {}) or {}
    training = cfg.get("training", {}) or {}
    split_keys = [
        "split_strategy",
        "spatial_block_deg",
        "run_spatial_block_cv",
        "spatial_block_cv_folds",
        "spatial_block_cv_max_folds",
        "spatial_block_cv_only",
    ]
    out: Dict[str, Any] = {}
    for k in split_keys:
        if k in training:
            out[k] = training.get(k)
    return out


def _extract_data_processing(run_summary: Dict[str, Any]) -> Dict[str, Any]:
    cfg = run_summary.get("config", {}) or {}
    return cfg.get("data_processing", {}) or {}


def make_training_run_report_md(
    run_summary: Dict[str, Any],
    *,
    dataset_summary: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Create a paper-oriented, self-contained Markdown report for a training run.
    Keep it compact and explicit about scientific assumptions.
    """
    run_id = run_summary.get("run_id", "N/A")
    ts_start = run_summary.get("timestamp_start")
    ts_end = run_summary.get("timestamp_end")
    duration = run_summary.get("duration_seconds")

    test_metrics = _extract_test_metrics(run_summary)
    test_metrics_in_domain = _extract_test_metrics_in_domain(run_summary)
    test_in_domain_n = _extract_test_in_domain_n(run_summary)
    hp = _extract_training_hyperparams(run_summary)
    split_cfg = _extract_training_split_cfg(run_summary)
    dp = _extract_data_processing(run_summary)

    feature_names: List[str] = []
    try:
        model_cfg = run_summary.get("MODEL_CONFIG", {}) or run_summary.get("model_config", {}) or {}
        feature_names = list(model_cfg.get("feature_names") or [])
    except Exception:
        feature_names = []

    depth_feature_type = None
    try:
        norm = run_summary.get("NORMALIZATION_PARAMS", {}) or {}
        depth_feature_type = norm.get("depth_feature_type") or norm.get("depth_feature", {}).get("type")
    except Exception:
        depth_feature_type = None

    lines: List[str] = []
    lines.append(f"# Training Run Report — `{run_id}`")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Status: `{run_summary.get('status')}`")
    lines.append(f"- Start: `{_fmt(ts_start)}`")
    lines.append(f"- End: `{_fmt(ts_end)}`")
    lines.append(f"- Duration (s): `{_fmt(duration)}`")
    lines.append("")
    lines.append("## Key Results (Test Set)")
    task_type = str(hp.get("task_type", "regression") or "regression").lower()
    if task_type == "multiclass_classification":
        lines.append(f"- Accuracy: `{_fmt(test_metrics.get('accuracy'))}`")
        lines.append(f"- Balanced accuracy: `{_fmt(test_metrics.get('balanced_accuracy'))}`")
        lines.append(f"- Macro-F1: `{_fmt(test_metrics.get('macro_f1'))}`")
        lines.append(f"- Log-loss: `{_fmt(test_metrics.get('log_loss'))}`")
        lines.append(f"- Num classes: `{_fmt(test_metrics.get('num_classes'))}`")
        if any(k in test_metrics for k in ["high_grade_recall", "high_grade_precision", "high_grade_f1", "high_grade_ap"]):
            lines.append(f"- High-grade recall: `{_fmt(test_metrics.get('high_grade_recall'))}`")
            lines.append(f"- High-grade precision: `{_fmt(test_metrics.get('high_grade_precision'))}`")
            lines.append(f"- High-grade F1: `{_fmt(test_metrics.get('high_grade_f1'))}`")
            lines.append(f"- High-grade AP: `{_fmt(test_metrics.get('high_grade_ap'))}`")
    elif task_type == "binary_classification":
        lines.append(f"- ROC-AUC: `{_fmt(test_metrics.get('roc_auc'))}`")
        lines.append(f"- PR-AUC: `{_fmt(test_metrics.get('pr_auc'))}`")
        lines.append(f"- Brier: `{_fmt(test_metrics.get('brier'))}`")
        lines.append(f"- Accuracy@0.5: `{_fmt(test_metrics.get('accuracy'))}`")
        lines.append(f"- F1@0.5: `{_fmt(test_metrics.get('f1'))}`")
    else:
        lines.append(f"- R2: `{_fmt(test_metrics.get('r2'))}`")
        lines.append(f"- MAE (ppm): `{_fmt(test_metrics.get('mae'))}`")   
        lines.append(f"- MSE (ppm2): `{_fmt(test_metrics.get('mse'))}`")  
    lines.append("")

    if isinstance(test_metrics_in_domain, dict) and test_metrics_in_domain and (test_in_domain_n is None or test_in_domain_n > 0):
        domain_ka = None
        try:
            domain_ka = float(hp.get("training_age_domain_ka")) if hp.get("training_age_domain_ka") is not None else None
        except Exception:
            domain_ka = None
        domain_ka = 2580.0 if domain_ka is None else domain_ka

        lines.append(f"## Key Results (In-Domain Test Set: age_ka_proxy ≤ {domain_ka:g} ka)")
        if test_in_domain_n is not None:
            lines.append(f"- In-domain N: `{_fmt(test_in_domain_n)}`")
        if task_type == "multiclass_classification":
            lines.append(f"- Accuracy: `{_fmt(test_metrics_in_domain.get('accuracy'))}`")
            lines.append(f"- Balanced accuracy: `{_fmt(test_metrics_in_domain.get('balanced_accuracy'))}`")
            lines.append(f"- Macro-F1: `{_fmt(test_metrics_in_domain.get('macro_f1'))}`")
            lines.append(f"- Log-loss: `{_fmt(test_metrics_in_domain.get('log_loss'))}`")
            lines.append(f"- Num classes: `{_fmt(test_metrics_in_domain.get('num_classes'))}`")
        elif task_type == "binary_classification":
            lines.append(f"- ROC-AUC: `{_fmt(test_metrics_in_domain.get('roc_auc'))}`")
            lines.append(f"- PR-AUC: `{_fmt(test_metrics_in_domain.get('pr_auc'))}`")
            lines.append(f"- Brier: `{_fmt(test_metrics_in_domain.get('brier'))}`")
            lines.append(f"- Accuracy@0.5: `{_fmt(test_metrics_in_domain.get('accuracy'))}`")
            lines.append(f"- F1@0.5: `{_fmt(test_metrics_in_domain.get('f1'))}`")
        else:
            lines.append(f"- R2: `{_fmt(test_metrics_in_domain.get('r2'))}`")
            lines.append(f"- MAE (ppm): `{_fmt(test_metrics_in_domain.get('mae'))}`")
            lines.append(f"- MSE (ppm2): `{_fmt(test_metrics_in_domain.get('mse'))}`")
        lines.append("")
    lines.append("## Data & Scientific Definitions")
    lines.append("- `depth` refers to **sub-bottom depth** (mbsf) in the original REY table; it is **not** water depth.")
    lines.append("- If a sedimentation-rate raster is available, the model uses an **age proxy** as the depth feature:")
    lines.append("  - `age_ka_proxy = depth_mbsf(m) * 100 / sed_rate(cm/ka)`")
    lines.append("- Interpretability note: modern seafloor environmental rasters are assumed to correlate with controls on REY enrichment; this assumption weakens for older sediments.")
    lines.append("")
    lines.append("## Filtering / Time Window")
    lines.append(f"- `max_training_age_ka_proxy`: `{_fmt(dp.get('max_training_age_ka_proxy'))}` (Quaternary boundary default = 2580 ka)")
    lines.append(f"- `max_training_depth_mbsf`: `{_fmt(dp.get('max_training_depth_mbsf'))}` (optional physical filter)")
    if task_type == "multiclass_classification" and hp.get("training_age_domain_ka") is not None:
        lines.append(f"- `training_age_domain_ka` (evaluation boundary): `{_fmt(hp.get('training_age_domain_ka'))}`")
    lines.append("")
    lines.append("## Model & Training")
    lines.append(f"- Model type: `{_fmt(hp.get('model_type'))}`")
    lines.append(f"- Patch size: `{_fmt(hp.get('patch_size'))}`")
    lines.append(f"- Batch size: `{_fmt(hp.get('batch_size'))}`; epochs: `{_fmt(hp.get('epochs'))}`; lr: `{_fmt(hp.get('learning_rate'))}`")
    lines.append(f"- Depth feature type (artifacts): `{_fmt(depth_feature_type)}`")
    lines.append("")
    if split_cfg:
        lines.append("## Split Strategy (Leakage Control)")
        lines.append(f"- `split_strategy`: `{_fmt(split_cfg.get('split_strategy'))}`")
        if split_cfg.get("split_strategy") == "spatial_block":
            lines.append(f"- `spatial_block_deg`: `{_fmt(split_cfg.get('spatial_block_deg'))}` (lat/lon degree grid for group blocking)")
            if split_cfg.get("train_sampling_strategy"):
                lines.append(f"- `train_sampling_strategy`: `{_fmt(split_cfg.get('train_sampling_strategy'))}`")
                if split_cfg.get("train_sampling_strategy") == "block_balanced":
                    lines.append(f"- `block_balance_alpha`: `{_fmt(split_cfg.get('block_balance_alpha'))}`")
                    lines.append(f"- `block_balance_min_count`: `{_fmt(split_cfg.get('block_balance_min_count'))}`")
            lines.append(f"- `run_spatial_block_cv`: `{_fmt(split_cfg.get('run_spatial_block_cv'))}`")
            if split_cfg.get("run_spatial_block_cv"):
                lines.append(f"- `spatial_block_cv_folds`: `{_fmt(split_cfg.get('spatial_block_cv_folds'))}`")
                lines.append(f"- `spatial_block_cv_max_folds`: `{_fmt(split_cfg.get('spatial_block_cv_max_folds'))}`")
                lines.append(f"- `spatial_block_cv_only`: `{_fmt(split_cfg.get('spatial_block_cv_only'))}`")
        split_diag = (run_summary.get("results", {}) or {}).get("split_group_diagnostics")
        if isinstance(split_diag, dict):
            lines.append(f"- Group column: `{_fmt(split_diag.get('group_col'))}`")
            lines.append(f"- Split leakage detected: `{_fmt(split_diag.get('leakage_detected'))}`")
            lines.append(f"- Groups crossing splits: `{_fmt(split_diag.get('n_leaking_groups'))}`")
        lines.append("")
    if feature_names:
        lines.append("## Environmental Features (from processed `env_gridded.nc`)")
        lines.append(f"- Count: `{len(feature_names)}`")
        lines.append("- Names: " + ", ".join([f"`{n}`" for n in feature_names]))
        lines.append("")

    if dataset_summary:
        lines.append("## Dataset Summary")
        for k in [
            "num_points_total",
            "num_points_after_filters",
            "depth_mbsf_min",
            "depth_mbsf_max",
            "age_ka_proxy_min",
            "age_ka_proxy_p50",
            "age_ka_proxy_max",
            "split_sizes",
        ]:
            if k in dataset_summary:
                lines.append(f"- `{k}`: `{_fmt(dataset_summary.get(k))}`")
        lines.append("")

    lines.append("## Outputs")
    lines.append("- `run_summary.json`: machine-readable summary + artifact paths")
    lines.append("- `training_artifacts.json`: model init config + normalization params + model path")
    lines.append("- `plots/diagnostics/classification/*`: confusion matrices, calibration, confidence/entropy histograms (multiclass runs)")
    lines.append("- `tables/test/*`: classification report + confusion matrix (multiclass runs)")
    lines.append("- `tables/test/grade_class_rey_summary.csv`: REY median and P>=1000/P>=1500 by true/predicted class")
    lines.append("- `tables/test/high_grade_metrics.json`: any-high-grade one-vs-rest metrics for 5-class runs")
    lines.append("- `splits/split_group_*`: site/core group split diagnostics and leakage checks")
    lines.append("- `plots/diagnostics/age_binned_*.png`: time-scale diagnostics; `*_extended` includes >2580 ka samples (evaluation only)")
    lines.append("- `splits/spatial_blocks.csv` (if spatial_block): per-sample block assignment for reproducible splitting")
    lines.append("- `cv/cv_summary.json` (if spatial block CV enabled): fold-wise spatial generalization diagnostic")
    lines.append("")
    lines.append("## Notes for Publication")
    lines.append("- Report time-binned performance (Quaternary vs older) and disclose the age-proxy definition and its dependence on the sedimentation-rate raster.")
    lines.append("- When presenting extended (>2580 ka) evaluation, describe it as **out-of-domain** diagnostics (training remains Quaternary-filtered).")


    try:
        boot = (run_summary.get("results", {}) or {}).get("target_class_bootstrap")
        if not isinstance(boot, dict):
            boot = (run_summary.get("results", {}) or {}).get("class1_bootstrap")
        if isinstance(boot, dict) and boot.get("method") == "cluster_bootstrap":
            thr = boot.get("threshold")
            npos = boot.get("n_pos")
            ap = boot.get("ap") if isinstance(boot.get("ap"), dict) else None
            f1 = boot.get("f1") if isinstance(boot.get("f1"), dict) else None
            if ap and f1:
                lines.append("- Target-class stability: cluster bootstrap by site/core group is available for rare-class uncertainty.")
                lines.append(
                    "  - "
                    f"target_class={_fmt(boot.get('target_class'))}, threshold={_fmt(thr)}, n_pos={_fmt(npos)}, "
                    f"AP median={_fmt(ap.get('p50'))} (95% CI {_fmt(ap.get('p2_5'))}–{_fmt(ap.get('p97_5'))}), "
                    f"F1 median={_fmt(f1.get('p50'))} (95% CI {_fmt(f1.get('p2_5'))}–{_fmt(f1.get('p97_5'))})"
                )
    except Exception:
        pass

    lines.append("")

    return "\n".join(lines)


def make_prediction_run_report_md(run_summary: Dict[str, Any]) -> str:
    run_id = run_summary.get("run_id", "N/A")
    ts_start = run_summary.get("timestamp_start")
    ts_end = run_summary.get("timestamp_end")
    duration = run_summary.get("duration_seconds")

    inputs = run_summary.get("inputs", {}) or {}
    pred_cfg = inputs.get("prediction_config", {}) or {}
    results = run_summary.get("results", {}) or {}
    target_task_type = str(results.get("target_task_type") or "").lower()

    lines: List[str] = []
    lines.append(f"# Prediction Run Report — `{run_id}`")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Status: `{run_summary.get('status')}`")
    lines.append(f"- Start: `{_fmt(ts_start)}`")
    lines.append(f"- End: `{_fmt(ts_end)}`")
    lines.append(f"- Duration (s): `{_fmt(duration)}`")
    lines.append("")
    lines.append("## Inputs")
    lines.append(f"- Training artifacts: `{_fmt(inputs.get('source_training_artifacts_path'))}`")
    lines.append(f"- Env grid: `{_fmt(pred_cfg.get('env_gridded_nc_path'))}`")
    lines.append("")
    lines.append("## Vertical Axis & Scientific Interpretation")
    try:
        grid_cfg = (pred_cfg.get("prediction_grid_settings") or {}) if isinstance(pred_cfg, dict) else {}
    except Exception:
        grid_cfg = {}
    lines.append(f"- `vertical_axis_mode`: `{_fmt((grid_cfg or {}).get('vertical_axis_mode'))}`")
    lines.append("- Depth definition: **sub-bottom depth** (mbsf), not water depth.")
    lines.append("- If `vertical_axis_mode='age_ka'`, maps are time slices (ka) and depth is derived via sedimentation rate.")
    lines.append("- If training used `age_ka_proxy` as the depth feature, the proxy is: `age_ka = depth_mbsf(m)*100 / sed_rate(cm/ka)`.")
    lines.append("")
    lines.append("## Domain / Extrapolation Controls")
    lines.append(f"- `training_age_domain_ka`: `{_fmt(pred_cfg.get('training_age_domain_ka'))}`")
    lines.append(f"- `restrict_to_training_age_domain`: `{_fmt(pred_cfg.get('restrict_to_training_age_domain'))}`")
    lines.append(f"- `mask_out_of_domain_in_maps`: `{_fmt(pred_cfg.get('mask_out_of_domain_in_maps'))}`")
    lines.append(f"- `apply_depth_extrapolation_attenuation`: `{_fmt(pred_cfg.get('apply_depth_extrapolation_attenuation'))}`")
    if target_task_type == "multiclass_classification":
        thr_info = results.get("multiclass_target_threshold") or {}
        lines.append(f"- `multiclass_target_class_index`: `{_fmt(thr_info.get('multiclass_target_class_index'))}`")
        lines.append(f"- `multiclass_target_threshold`: `{_fmt(thr_info.get('multiclass_target_threshold'))}`")
        lines.append(f"- `multiclass_target_threshold_source`: `{_fmt(thr_info.get('multiclass_target_threshold_source'))}`")
    lines.append("")
    lines.append("## Outputs")
    lines.append("- Prediction outputs are **classification-only 2D age-window summaries** (no 3D slices).")
    lines.append(f"- 2D summaries: `{_fmt(results.get('predicted_summary_2d_nc_path'))}`")
    lines.append(f"- 2D summary maps: `{_fmt(results.get('summary_2d_maps_dir_path'))}`")
    lines.append("")
    lines.append("## NetCDF Variables for Publication")
    if target_task_type == "multiclass_classification":
        lines.append("- `predicted_prob_class{i}_mean_0_{window}`: window-mean P(class i)")
        lines.append("- `predicted_prob_class{i}_p{XX}_0_{window}`: window-high quantile P(class i) (e.g. p95)")
        lines.append("- `predicted_prob_class{i}_std_0_{window}` (optional): mean MC Dropout std for P(class i)")
        lines.append("- `predicted_target_class{k}_hotspot_freq_thr{t}_0_{window}`: hotspot frequency for target class (one-vs-rest)")
        lines.append("- `predicted_target_class{k}_hotspot_any_thr{t}_0_{window}`: any-hotspot (0/1) for target class")
        lines.append("- `predicted_class_entropy_mean_0_{window}` / `..._p{XX}_0_{window}`")
        lines.append("- `predicted_class_argmax_freq_class{i}_0_{window}`: fraction of ages where argmax==i")
        lines.append("- `in_domain_fraction_0_{window}`: fraction of sampled ages within training domain")
    else:
        lines.append("- (Non-multiclass task: variable naming may differ.)")
    lines.append("- `is_in_training_age_domain`: in-domain mask based on age proxy")
    lines.append("- `depth_extrapolation_factor`: attenuation factor when outside training depth-feature range")
    lines.append("- `sedimentation_rate` (if available): used for depth→age conversion")
    lines.append("- `age_ka_proxy` or `depth_mbsf` (depending on vertical axis): conversion reference fields")
    lines.append("")
    lines.append("## Notes")
    lines.append("- For journal figures, report whether maps are depth slices or age slices, and disclose the sedimentation-rate raster used for the proxy.")
    if bool(pred_cfg.get("restrict_to_training_age_domain")):
        lines.append(
            "- Domain policy: **Quaternary-only** outputs when `restrict_to_training_age_domain=true` "
            "(age > `training_age_domain_ka` is set to NaN; maps only show in-domain values)."
        )
    elif bool(pred_cfg.get("mask_out_of_domain_in_maps")):
        lines.append(
            "- Domain policy: maps are masked using `is_in_training_age_domain`, but NetCDF may retain out-of-domain fields "
            "unless `restrict_to_training_age_domain=true`."
        )
    else:
        lines.append(
            "- Domain policy: out-of-domain ages are not masked; interpret results cautiously and use `is_in_training_age_domain`."
        )
    lines.append("")
    return "\n".join(lines)

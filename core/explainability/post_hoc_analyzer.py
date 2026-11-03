import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.inspection import permutation_importance

from core.utils.logger_config import logger
from core.visualization.publication_style import publication_style, save_figure
from core.visualization.shap_plots import plot_shap_summary_optimized


def _safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(name))


def _split_categorical_feature_name(name: str) -> tuple[str | None, str | None, str | None]:
    s = str(name)
    if "__cat_" not in s:
        return None, None, None

    base, rest = s.split("__cat_", 1)
    cat_id = None
    label = None
    if "__" in rest:
        cat_id, label = rest.split("__", 1)
    else:
        cat_id = rest
    return base.strip() or None, (cat_id.strip() if cat_id is not None else None), (label.strip() if label is not None else None)


def _group_feature_names(feature_names: list[str]) -> dict[str, dict]:
    groups: dict[str, dict] = {}
    for i, fn in enumerate(list(feature_names)):
        base, cat_id, label = _split_categorical_feature_name(fn)
        if base is None:
            groups.setdefault(fn, {"name": fn, "indices": [], "is_categorical": False, "categories": []})
            groups[fn]["indices"].append(i)
            continue
        g = groups.setdefault(base, {"name": base, "indices": [], "is_categorical": True, "categories": []})
        g["indices"].append(i)
        g["categories"].append({"feature": fn, "idx": i, "cat_id": cat_id, "label": label})

    for g in groups.values():
        if g.get("is_categorical"):
            g["categories"] = sorted(
                g.get("categories") or [],
                key=lambda d: (
                    int(d["cat_id"]) if str(d.get("cat_id") or "").isdigit() else 10**9,
                    str(d.get("label") or ""),
                ),
            )
    return groups


def _is_one_hot_categorical_feature(name: str) -> bool:
    return "__cat_" in str(name)


DEFAULT_OPTIMIZED_SHAP_SCHEME = "marine_geology_process_chain"


def _default_optimized_shap_colors_map() -> dict[str, str]:
    return {
        "Tectono-Geomorphic Setting": "#D6B85A",
        "Sedimentary Burial & Stratigraphic Context": "#7CBF9E",
        "Ocean Circulation & Transport Dynamics": "#7FA6D6",
        "Biogeochemical Environment": "#E38A8A",
        "Benthic Ecological Response": "#9C8DC2",
        "Other": "#C7C7C7",
    }


def _default_optimized_shap_category(feature_name: str) -> str:
    base, _, _ = _split_categorical_feature_name(str(feature_name))
    key = (base or str(feature_name)).strip().lower()

    if "geomorphic features" in key or "seafloor lithology" in key:
        return "Tectono-Geomorphic Setting"

    if (
        "water depth" in key
        or "gravity anomaly" in key
        or "magnetic anomaly" in key
        or "vertical gravity gradient" in key
    ):
        return "Tectono-Geomorphic Setting"

    if (
        "age_ka_proxy" in key
        or "average sedimentation rate" in key
        or "sediment thickness" in key
    ):
        return "Sedimentary Burial & Stratigraphic Context"

    if "bottom current speed sd" in key or "del3he" in key:
        return "Ocean Circulation & Transport Dynamics"

    if (
        "bottom phosphate" in key
        or "surface phosphate" in key
        or "total organic carbon" in key
        or "summer primary productivity" in key
        or "bottom water dissolved oxygen" in key
    ):
        return "Biogeochemical Environment"

    if "macrobenthic biomass" in key:
        return "Benthic Ecological Response"

    return "Other"


def _get_norm_params_from_plot_config(plot_config: dict | None) -> dict | None:
    """
    Retrieve normalization parameters (as a plain dict) from plot_config.

    Expected keys (any one):
      - plot_config["norm_params"] (dict-like)
      - plot_config["training_artifacts"]["norm_params"] (dict-like)
      - plot_config["training_artifacts_obj"].norm_params (pydantic model)
    """
    if not plot_config:
        return None

    norm_params = plot_config.get("norm_params")
    if isinstance(norm_params, dict):
        return norm_params

    ta = plot_config.get("training_artifacts")
    if isinstance(ta, dict) and isinstance(ta.get("norm_params"), dict):
        return ta.get("norm_params")

    ta_obj = plot_config.get("training_artifacts_obj")
    if ta_obj is not None and hasattr(ta_obj, "norm_params"):
        try:
            norm_obj = getattr(ta_obj, "norm_params")
            if hasattr(norm_obj, "model_dump"):
                return norm_obj.model_dump()
        except Exception:
            return None

    return None


def _denormalize_feature_values(
    values_norm: np.ndarray,
    *,
    feature_idx: int,
    feature_names: list[str],
    norm_params: dict | None,
) -> np.ndarray:
    if norm_params is None:
        return values_norm

    try:
        if "channel_means" in norm_params and "channel_stds" in norm_params:
            means = np.asarray(norm_params["channel_means"], dtype=float)
            stds = np.asarray(norm_params["channel_stds"], dtype=float)
            if feature_idx >= means.size or feature_idx >= stds.size:
                return values_norm
            values_trans = values_norm * float(stds[feature_idx]) + float(means[feature_idx])
        elif "channel_mins" in norm_params and "channel_maxs" in norm_params:
            mins = np.asarray(norm_params["channel_mins"], dtype=float)
            maxs = np.asarray(norm_params["channel_maxs"], dtype=float)
            if feature_idx >= mins.size or feature_idx >= maxs.size:
                return values_norm
            values_trans = values_norm * (float(maxs[feature_idx]) - float(mins[feature_idx])) + float(
                mins[feature_idx]
            )
        else:
            return values_norm


        depth_feature_type = str(norm_params.get("depth_feature_type") or "")
        depth_transform_method = str(norm_params.get("depth_transform_method") or "")
        feature_name = feature_names[feature_idx] if feature_idx < len(feature_names) else ""
        if feature_name in {"age_ka_proxy", "depth"} or feature_idx == len(feature_names) - 1:
            if depth_feature_type == "age_ka_proxy" or feature_name == "age_ka_proxy":
                if depth_transform_method == "log1p":
                    return np.expm1(values_trans)
                if depth_transform_method == "sqrt":
                    return np.square(values_trans)
                return values_trans

            if depth_transform_method == "log1p":
                return np.expm1(values_trans)
            if depth_transform_method == "sqrt":
                return np.square(values_trans)
            return values_trans

        return values_trans
    except Exception:
        return values_norm


def _denormalize_target_values(y_norm: np.ndarray, norm_params: dict | None) -> np.ndarray:
    """
    Convert model output from normalized space back to physical ΣREY (ppm).

    Training target is typically z-score of log1p(ΣREY).
    """
    if norm_params is None:
        return y_norm

    rey_cfg = norm_params.get("rey") if isinstance(norm_params.get("rey"), dict) else {}
    try:
        mean = float(rey_cfg.get("mean"))
        std = float(rey_cfg.get("std")) if float(rey_cfg.get("std")) != 0 else 1.0
        transform = str(rey_cfg.get("transform") or "")
        scale = float(rey_cfg.get("scale", 1000.0) or 1000.0)
        hinge_beta = float(rey_cfg.get("hinge_beta", 0.25) or 0.0)
    except Exception:
        return y_norm

    y_trans = y_norm * std + mean
    if transform == "log1p":
        return np.expm1(y_trans)
    if transform == "asinh":
        return np.sinh(y_trans) * max(scale, 1e-6)
    if transform == "hinged_log":
        hinge_start = np.log1p(max(scale, 1e-6))
        beta = max(hinge_beta, 0.0)
        base = np.where(
            y_trans <= hinge_start,
            y_trans,
            (y_trans + beta * hinge_start) / max(1.0 + beta, 1e-6),
        )
        return np.expm1(base)
    return y_trans


def _feature_axis_label(feature_name: str, norm_params: dict | None) -> str:
    if feature_name == "age_ka_proxy":
        return "Age proxy (ka)"
    if feature_name == "depth":
        return "Depth (m bsf)"
    if feature_name == "Water depth":
        return "Water depth (m)"
    if feature_name == "Average sedimentation rate":
        return "Sedimentation rate (cm/ka)"


    return str(feature_name)


def _plot_categorical_pdp_grouped(
    model,
    X_sample: np.ndarray,
    group_info: dict,
    feature_names: list[str],
    pdp_dir: Path,
    plot_config: dict,
    norm_params: dict | None,
    task_type: str,
    target_label: str,
) -> dict:
    """
    Compute and plot grouped categorical PDP showing all categories in one bar chart.

    Args:
        model: Wrapped model with predict() method
        X_sample: Sample data for PDP computation
        group_info: Dict with keys 'name', 'categories', 'indices'
        feature_names: List of all feature names
        pdp_dir: Output directory for plots and CSVs
        plot_config: Plot configuration dict
        norm_params: Normalization parameters (optional)
        task_type: 'regression' or 'classification'
        target_label: Y-axis label for plots

    Returns:
        Dict with 'plot_paths', 'csv_path', and 'data' keys
    """
    logger.info(f"Computing grouped categorical PDP for '{group_info['name']}' with {len(group_info['categories'])} categories...")


    X_base = np.asarray(X_sample, dtype=float)
    group_indices = [int(c["idx"]) for c in (group_info.get("categories") or []) if "idx" in c]

    category_data = []
    for cat in group_info["categories"]:
        idx = int(cat["idx"])
        cat_id = cat.get("cat_id", "")
        label = cat.get("label") or f"cat_{cat_id}"

        try:
            X_forced = np.array(X_base, copy=True)
            if group_indices:
                X_forced[:, group_indices] = 0.0
            X_forced[:, idx] = 1.0

            y_forced = np.asarray(model.predict(X_forced), dtype=float).ravel()
            pdp_val = float(np.nanmean(y_forced))

            support_frac = None
            try:
                support_frac = float(np.nanmean(np.asarray(X_base[:, idx], dtype=float)))
            except Exception:
                support_frac = None

            row = {
                "category_id": str(cat_id),
                "label": label.replace("_", " "),
                "pdp_value": pdp_val,
            }
            if support_frac is not None and np.isfinite(support_frac):
                row["support_frac"] = support_frac
            category_data.append(row)
        except Exception as e:
            logger.warning(f"Failed to compute forced-category PDP for '{label}': {e}")
            continue

    if not category_data:
        logger.warning(f"No valid PDP data for group '{group_info['name']}'")
        return {"plot_paths": [], "csv_path": None, "data": None}


    df = pd.DataFrame(category_data)
    df = df.sort_values("pdp_value", ascending=False)


    baseline = None
    try:
        baseline = float(np.nanmean(np.asarray(model.predict(X_sample), dtype=float).ravel()))
    except Exception:
        baseline = None
    if baseline is not None and np.isfinite(baseline):
        df["baseline"] = baseline
        df["delta_from_baseline"] = df["pdp_value"].astype(float) - float(baseline)


    with publication_style(plot_config):
        fig, ax = plt.subplots(figsize=(9.5, max(5.5, len(df) * 0.35)))

        use_delta = bool(plot_config.get("pdp_categorical_use_delta", True))
        if use_delta and ("delta_from_baseline" in df.columns) and baseline is not None and np.isfinite(baseline):
            x = pd.to_numeric(df["delta_from_baseline"], errors="coerce").astype(float).to_numpy()
            vmax = float(np.nanmax(np.abs(x))) if np.isfinite(x).any() else 1.0
            vmax = vmax if (np.isfinite(vmax) and vmax > 0) else 1.0
            norm = matplotlib.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
            cmap = plt.cm.RdBu_r
            colors = cmap(norm(x))
            ax.barh(df["label"], x, color=colors, alpha=0.88, edgecolor="black", linewidth=0.5)
            ax.axvline(0.0, color="#111111", lw=1.0, alpha=0.65)
            ax.set_xlabel(f"Δ {target_label} (vs baseline {baseline:.3f})", fontsize=12)
        else:
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df)))
            ax.barh(df["label"], df["pdp_value"], color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
            ax.set_xlabel(target_label, fontsize=12)

        ax.set_ylabel("")
        ax.set_title(f"Partial dependence: {group_info['name']} (categorical)", fontsize=14)
        ax.grid(True, axis="x", alpha=0.3, linestyle="--")
        ax.tick_params(axis="both", labelsize=11)
        fig.tight_layout()


        safe_name = _safe_filename(group_info['name'])
        base_path = pdp_dir / f"pdp_categorical_{safe_name}"
        save_paths = save_figure(fig, path_no_ext=base_path, plot_config=plot_config)
        logger.info(f"Saved grouped categorical PDP to {save_paths}")
        plt.close(fig)


    csv_path = pdp_dir / f"pdp_values_{safe_name}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved PDP values to {csv_path}")

    return {
        "plot_paths": save_paths,
        "csv_path": str(csv_path),
        "data": df,
    }


class ModelWrapperForSklearn:
    """A wrapper to make the PyTorch model compatible with scikit-learn's inspection tools."""
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()

    def predict(self, X):


        import torch
        X_tensor = torch.from_numpy(X).float().to(self.device)
        with torch.no_grad():
            out = self.model(X_tensor)
            if isinstance(out, dict):
                for k0 in ("logits", "preds", "pred", "y_pred", "output"):
                    v0 = out.get(k0)
                    if torch.is_tensor(v0):
                        out = v0
                        break
            elif isinstance(out, (tuple, list)):
                for v0 in out:
                    if torch.is_tensor(v0):
                        out = v0
                        break
            if not torch.is_tensor(out):
                raise TypeError(f"ModelWrapperForSklearn.predict: output is not a torch.Tensor: {type(out)}")
            predictions = out.cpu().numpy().ravel()
        return predictions

def analyze_partial_dependence(model, X_train, feature_names, output_dir, plot_config):
    """
    Manually calculates and plots geology-focused, individual 1D and 2D PDPs
    to bypass scikit-learn's estimator checks and gain full plotting control.
    """
    logger.info("Post-hoc Analyzer: Starting Manual & Geology-Focused Partial Dependence Analysis...")
    results = {"pdp_1d_plot_paths": [], "pdp_2d_plot_paths": []}
    pdp_dir = Path(output_dir) / "partial_dependence"
    pdp_dir.mkdir(parents=True, exist_ok=True)

    X_sample = X_train[:plot_config.get("pdp_sample_size", 200)]

    from sklearn.inspection import partial_dependence


    norm_params = _get_norm_params_from_plot_config(plot_config)
    task_type = str((plot_config or {}).get("task_type") or "regression").lower()
    target_label = str((plot_config or {}).get("target_label") or ("Predicted ΣREY (ppm)" if task_type == "regression" else "Predicted P(class)"))


    logger.info("Grouping features for systematic PDP generation...")
    feature_groups = _group_feature_names(list(feature_names))
    pdp_csv_paths = []

    for group_name, group_info in feature_groups.items():
        if group_info.get("is_categorical"):

            try:
                result = _plot_categorical_pdp_grouped(
                    model,
                    X_sample,
                    group_info,
                    list(feature_names),
                    pdp_dir,
                    plot_config,
                    norm_params,
                    task_type,
                    target_label,
                )
                if result["plot_paths"]:
                    results["pdp_1d_plot_paths"].extend(result["plot_paths"])
                if result["csv_path"]:
                    pdp_csv_paths.append(result["csv_path"])
            except Exception as e:
                logger.error(f"Failed to generate grouped categorical PDP for '{group_name}': {e}", exc_info=True)

        else:

            feature_idx = group_info["indices"][0]
            feature_name = group_info["name"]

            try:

                pdp_results_raw = partial_dependence(
                    model,
                    X_sample,
                    features=[feature_idx],
                    kind='average',
                    grid_resolution=plot_config.get("pdp_grid_resolution", 50)
                )

                pdp_values = pdp_results_raw['average'][0]
                feature_grid = pdp_results_raw['grid_values'][0]


                feature_grid_plot = _denormalize_feature_values(
                    feature_grid,
                    feature_idx=feature_idx,
                    feature_names=list(feature_names),
                    norm_params=norm_params,
                )
                pdp_values_plot = (
                    _denormalize_target_values(pdp_values, norm_params)
                    if task_type == "regression"
                    else pdp_values
                )


                pdp_df = pd.DataFrame({
                    "feature_value": feature_grid_plot,
                    "pdp_value": pdp_values_plot,
                })
                safe_feature_name = _safe_filename(feature_name)
                csv_path = pdp_dir / f"pdp_values_{safe_feature_name}.csv"
                pdp_df.to_csv(csv_path, index=False)
                pdp_csv_paths.append(str(csv_path))
                logger.info(f"Saved PDP values to {csv_path}")


                with publication_style(plot_config):
                    fig, ax = plt.subplots(figsize=(8.5, 5.5))
                    ax.plot(feature_grid_plot, pdp_values_plot, color="royalblue", lw=2.2, label="Partial dependence")


                    if bool(plot_config.get("pdp_enable_knee_detection", False)):
                        try:
                            from kneed import KneeLocator

                            knee_locator = KneeLocator(
                                feature_grid_plot,
                                pdp_values_plot,
                                curve="concave",
                                direction="increasing",
                            )
                            if knee_locator.knee:
                                threshold_x = float(knee_locator.knee)
                                ax.axvline(
                                    x=threshold_x,
                                    color="crimson",
                                    linestyle="--",
                                    lw=1.6,
                                    label=f"Threshold ≈ {threshold_x:.2f}",
                                )
                        except Exception:
                            pass

                    ax.set_title(f"Partial dependence: {_feature_axis_label(feature_name, norm_params)}")
                    ax.set_xlabel(_feature_axis_label(feature_name, norm_params))
                    ax.set_ylabel(target_label)
                    ax.grid(True, which="both", alpha=0.3, linestyle="--")
                    ax.legend(frameon=False, loc="best")
                    fig.tight_layout()

                    save_paths = save_figure(
                        fig,
                        path_no_ext=pdp_dir / f"pdp_numeric_{safe_feature_name}",
                        plot_config=plot_config,
                    )
                    results["pdp_1d_plot_paths"].append(
                        next((p for p in save_paths if p.lower().endswith(".png")), save_paths[0])
                    )
                    logger.info(f"Saved numeric PDP for '{feature_name}' to {save_paths}")

                    plt.close(fig)

            except Exception as e:
                logger.error(f"Failed to generate numeric PDP for feature '{feature_name}': {e}", exc_info=True)


    results["pdp_csv_paths"] = pdp_csv_paths


    from itertools import combinations


    n_top_features_for_pdp_2d = min(plot_config.get("pdp_2d_top_n_features", 5), len(feature_names))
    top_feature_indices = list(range(n_top_features_for_pdp_2d))

    feature_pairs_indices = list(combinations(top_feature_indices, 2))

    logger.info(f"Systematically generating 2D PDPs for all {len(feature_pairs_indices)} pairs from the top {n_top_features_for_pdp_2d} features...")

    for pair_indices in feature_pairs_indices:
        pair_names = (feature_names[pair_indices[0]], feature_names[pair_indices[1]])
        try:
            pdp_results_raw_2d = partial_dependence(
                model,
                X_sample,
                features=pair_indices,
                kind='average',
                grid_resolution=plot_config.get("pdp_grid_resolution_2d", 25)
            )

            pdp_values_2d = pdp_results_raw_2d['average'][0]
            grid_x, grid_y = np.meshgrid(pdp_results_raw_2d['grid_values'][0], pdp_results_raw_2d['grid_values'][1])


            grid_x_plot = _denormalize_feature_values(
                grid_x,
                feature_idx=int(pair_indices[0]),
                feature_names=list(feature_names),
                norm_params=norm_params,
            )
            grid_y_plot = _denormalize_feature_values(
                grid_y,
                feature_idx=int(pair_indices[1]),
                feature_names=list(feature_names),
                norm_params=norm_params,
            )
            pdp_values_2d_plot = (
                _denormalize_target_values(pdp_values_2d, norm_params)
                if task_type == "regression"
                else pdp_values_2d
            )


            with publication_style(plot_config):
                fig, ax = plt.subplots(figsize=(7.2, 5.6))
                contour = ax.contourf(
                    grid_x_plot,
                    grid_y_plot,
                    pdp_values_2d_plot.T,
                    levels=15,
                    cmap="viridis",
                )
                cbar = fig.colorbar(contour, ax=ax)
                cbar.set_label(target_label)

                ax.set_title(f"2D partial dependence: {pair_names[0]} vs {pair_names[1]}")
                ax.set_xlabel(_feature_axis_label(pair_names[0], norm_params))
                ax.set_ylabel(_feature_axis_label(pair_names[1], norm_params))
                fig.tight_layout()

                safe_pair_name = f'{"_and_".join(_safe_filename(p) for p in pair_names)}'
                save_paths = save_figure(
                    fig,
                    path_no_ext=pdp_dir / f"pdp_2d_{safe_pair_name}",
                    plot_config=plot_config,
                )
                results["pdp_2d_plot_paths"].append(
                    next((p for p in save_paths if p.lower().endswith(".png")), save_paths[0])
                )
                logger.info(f"Saved 2D PDP for pair {pair_names} to {save_paths}")

                plt.close(fig)

        except Exception as e:
            logger.error(f"Failed to generate manual 2D PDP for pair {pair_names}: {e}", exc_info=True)

    return results


def analyze_permutation_importance(model, X_test, y_test, feature_names, output_dir, plot_config):
    """
    Calculates and plots permutation feature importance.
    """
    logger.info("Post-hoc Analyzer: Starting Permutation Importance Analysis...")
    results = {}
    perm_dir = Path(output_dir) / "permutation_importance"
    perm_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info("Calculating permutation importance...")

        n_samples = min(plot_config.get("permutation_importance_samples", 500), len(y_test))
        X_test_sample, y_test_sample = X_test[:n_samples], y_test[:n_samples]

        scoring = None
        try:
            scoring = (plot_config or {}).get("permutation_scoring", None)
        except Exception:
            scoring = None

        n_jobs = 1
        try:
            n_jobs = int((plot_config or {}).get("permutation_n_jobs", 1) or 1)
        except Exception:
            n_jobs = 1

        r = permutation_importance(
            model,
            X_test_sample,
            y_test_sample,
            n_repeats=plot_config.get("permutation_n_repeats", 10),
            random_state=42,
            n_jobs=n_jobs,
            scoring=scoring,
        )


        sorted_indices = r.importances_mean.argsort()[::-1]


        importance_df = pd.DataFrame({
            'feature': [feature_names[i] for i in sorted_indices],
            'importance_mean': r.importances_mean[sorted_indices],
            'importance_std': r.importances_std[sorted_indices]
        })

        csv_path = perm_dir / "permutation_importance.csv"
        importance_df.to_csv(csv_path, index=False)
        results["permutation_importance_csv_path"] = str(csv_path)
        logger.info(f"Saved permutation importance data to {csv_path}")


        with publication_style(plot_config):
            fig, ax = plt.subplots(figsize=(8.0, max(5.0, len(feature_names) * 0.33)))

            importance_df.sort_values(by="importance_mean", ascending=True, inplace=True)

            ax.barh(
                importance_df["feature"],
                importance_df["importance_mean"],
                xerr=importance_df["importance_std"],
                align="center",
                color="#b2df8a",
                edgecolor="black",
                linewidth=0.6,
            )
            ax.set_title("Permutation importance")
            xlab = "Decrease in R² (higher = more important)"
            try:
                tt = str((plot_config or {}).get("task_type") or "regression").lower()
                if tt != "regression":
                    xlab = "Decrease in AP (higher = more important)"
            except Exception:
                pass
            ax.set_xlabel(xlab)
            ax.grid(True, axis="x")
            fig.tight_layout()

            save_paths = save_figure(
                fig,
                path_no_ext=perm_dir / "permutation_importance_plot",
                plot_config=plot_config,
            )
            results["permutation_importance_plot_path"] = next(
                (p for p in save_paths if p.lower().endswith(".png")), save_paths[0]
            )
            logger.info(f"Saved permutation importance plot to {save_paths}")
            plt.close(fig)

    except Exception as e:
        logger.error(f"Failed to calculate permutation importance: {e}", exc_info=True)
        results["permutation_importance_csv_path"] = None
        results["permutation_importance_plot_path"] = None

    return results

def analyze_shap_dependence(model_wrapper, X_data, feature_names, output_dir, plot_config):
   """
   Generates geology-focused, individual 2D SHAP dependence plots with enhanced aesthetics.
   """
   logger.info("Post-hoc Analyzer: Starting Geology-Focused SHAP Dependence Analysis...")
   results = {"shap_dependence_plot_paths": []}
   shap_dep_dir = Path(output_dir) / "shap_dependence"
   shap_dep_dir.mkdir(parents=True, exist_ok=True)
   norm_params = _get_norm_params_from_plot_config(plot_config)

   try:
       import shap
       sample_size = plot_config.get("shap_dependence_sample_size", 200)
       dot_size = plot_config.get("shap_dependence_dot_size", 30)
       X_sample = X_data[:sample_size]

       X_sample_plot = np.asarray(X_sample, dtype=float)
       if norm_params is not None:
           try:
               X_phys = np.empty_like(X_sample_plot, dtype=float)
               for j in range(X_phys.shape[1]):
                   X_phys[:, j] = _denormalize_feature_values(
                       X_sample_plot[:, j],
                       feature_idx=j,
                       feature_names=list(feature_names),
                       norm_params=norm_params,
                   )
               X_sample_plot = X_phys
           except Exception:
               X_sample_plot = np.asarray(X_sample, dtype=float)

       explainer = shap.KernelExplainer(model_wrapper.predict, shap.sample(X_sample, 50))
       shap_values = explainer.shap_values(X_sample, nsamples='auto')
       if isinstance(shap_values, list) and shap_values:
           shap_values = shap_values[0]
       task_type = str((plot_config or {}).get("task_type") or "regression").lower()


       shap_values_plot = shap_values
       if task_type == "regression" and norm_params is not None and isinstance(norm_params.get("rey"), dict):
           try:
               target_std = float(norm_params["rey"].get("std")) if float(norm_params["rey"].get("std")) != 0 else 1.0
               shap_values_plot = np.asarray(shap_values, dtype=float) * target_std
           except Exception:
               shap_values_plot = shap_values


       logger.info("Systematically generating SHAP dependence plots for all features...")

       for feature_idx, feature_name in enumerate(feature_names):
           try:
               with publication_style(plot_config):
                   fig, ax = plt.subplots(figsize=(7.8, 5.6))


                   shap.dependence_plot(
                       feature_name,
                       shap_values_plot,
                       X_sample_plot,
                       feature_names=feature_names,
                       interaction_index="auto",
                       ax=ax,
                       show=False,
                       alpha=0.8,
                       dot_size=dot_size,
                   )


                   try:
                       interaction_feature_name = ax.get_figure().axes[1].get_ylabel()
                       if "SHAP" in interaction_feature_name:
                           interaction_feature_name = "itself"
                   except Exception:
                       interaction_feature_name = "itself"


                   try:
                       fig_axes = ax.get_figure().axes
                       if len(fig_axes) > 1:
                           fig_axes[1].set_ylabel(_feature_axis_label(interaction_feature_name, norm_params))
                   except Exception:
                       pass

                   ax.set_title(
                       f"SHAP dependence: {_feature_axis_label(feature_name, norm_params)} "
                       f"(color: {_feature_axis_label(interaction_feature_name, norm_params)})"
                   )
                   ax.set_xlabel(_feature_axis_label(feature_name, norm_params))
                   ylab = "SHAP value (Δ log1p ΣREY)" if task_type == "regression" else "SHAP value (Δ P(class))"
                   ax.set_ylabel(ylab)
                   fig.tight_layout()

                   safe_feature_name = _safe_filename(feature_name)
                   safe_interaction_name = _safe_filename(interaction_feature_name)
                   save_paths = save_figure(
                       fig,
                       path_no_ext=shap_dep_dir / f"shap_dependence_{safe_feature_name}_vs_{safe_interaction_name}",
                       plot_config=plot_config,
                   )
                   results["shap_dependence_plot_paths"].append(
                       next((p for p in save_paths if p.lower().endswith(".png")), save_paths[0])
                   )
                   logger.info(
                       f"Saved SHAP dependence plot for '{feature_name}' vs '{interaction_feature_name}' to {save_paths}"
                   )
                   plt.close(fig)

           except Exception as e_inner:
                logger.error(
                    f"Failed to generate SHAP dependence plot for '{feature_name}': {e_inner}",
                    exc_info=True,
                )


       try:
           scheme = str((plot_config or {}).get("optimized_shap_scheme") or DEFAULT_OPTIMIZED_SHAP_SCHEME).strip().lower()
           if scheme != DEFAULT_OPTIMIZED_SHAP_SCHEME:
               logger.warning(
                   f"Unknown optimized_shap_scheme={scheme!r}; falling back to {DEFAULT_OPTIMIZED_SHAP_SCHEME!r}."
               )

           user_category_map = (plot_config or {}).get("optimized_shap_category_map") or {}
           category_map: dict[str, str] = {}
           other_defaults: set[str] = set()
           for fn in feature_names:
               base, _, _ = _split_categorical_feature_name(str(fn))
               mapped = user_category_map.get(str(fn))
               if mapped is None and base is not None:
                   mapped = user_category_map.get(str(base))
               if mapped is None:
                   mapped = _default_optimized_shap_category(str(base or fn))
                   if mapped == "Other":
                       other_defaults.add(str(base or fn))
               mapped = str(mapped)
               category_map[str(fn)] = mapped
               if base is not None:
                   category_map.setdefault(str(base), mapped)

           if other_defaults:
               unknown_preview = ", ".join(sorted(other_defaults)[:8])
               logger.warning(
                   "Optimized SHAP category mapping used 'Other' for "
                   f"{len(other_defaults)} feature(s): {unknown_preview}"
               )

           colors_map = _default_optimized_shap_colors_map()
           colors_map.update((plot_config or {}).get("optimized_shap_colors_map") or {})

           task_type = str((plot_config or {}).get("task_type") or "regression").lower()
           target_label = str((plot_config or {}).get("target_label") or "").strip()
           if task_type == "multiclass_classification" and target_label:
               title = f"SHAP + Marginal Trend Summary ({target_label})"
           elif task_type == "multiclass_classification":
               title = "SHAP + Marginal Trend Summary (target probability)"
           else:
               title = None

           optimized_out = plot_shap_summary_optimized(
               X_df=pd.DataFrame(X_sample_plot, columns=list(feature_names)),
               shap_values_matrix=np.asarray(shap_values_plot, dtype=float),
               category_map=category_map,
               colors_map=colors_map,
               r2=(plot_config or {}).get("optimized_shap_r2"),
               figsize=tuple((plot_config or {}).get("optimized_shap_figsize", (12, 9))),
               save_path=Path(shap_dep_dir) / "shap_summary_optimized",
               dpi=int((plot_config or {}).get("optimized_shap_dpi", 300)),
               lowess_frac=float((plot_config or {}).get("optimized_shap_lowess_frac", 0.3)),
               top_n=int((plot_config or {}).get("optimized_shap_top_n", 14)),
               plot_config=plot_config,
               title=title,
           )
           results["shap_summary_optimized_plot_paths"] = list(optimized_out.get("plot_paths") or [])
           results["shap_summary_optimized_plot_path"] = optimized_out.get("primary_path")
       except Exception as e_opt:
           logger.error(f"Failed to generate optimized SHAP summary: {e_opt}", exc_info=True)
           results["shap_summary_optimized_plot_paths"] = []
           results["shap_summary_optimized_plot_path"] = None

   except Exception as e:
       logger.error(f"Failed to run SHAP dependence analysis: {e}", exc_info=True)

   return results

def run_post_hoc_analysis(
    model,
    test_loader,
    feature_names,
    output_dir,
    plot_config,
    task_type: str = "regression",
    target_class_index: int | None = None,
    class_names: list[str] | None = None,
    *,
    group_categorical_features: bool = True,
):
    """
    Main entry point for running all post-hoc analyses (except SHAP).
    """
    logger.info("Starting post-hoc explainability analysis...")
    all_results = {}


    try:
        logger.info("Preparing data for post-hoc analysis by spatial averaging...")
        X_test_list = []
        y_test_list = []
        for patches, targets, _ in test_loader:

            X_test_list.append(patches.mean(axis=(2, 3)).cpu().numpy())
            y_test_list.append(targets.cpu().numpy())

        X_test_agg = np.concatenate(X_test_list, axis=0)
        y_test_agg_raw = np.concatenate(y_test_list, axis=0).ravel()

        task_type_l = str(task_type or (plot_config or {}).get("task_type") or "regression").lower()
        if task_type_l == "multiclass_classification":
            k = 1 if target_class_index is None else int(target_class_index)
            y_test_agg = (np.asarray(y_test_agg_raw, dtype=int) == k).astype(int)
            plot_config = dict(plot_config or {})
            plot_config["task_type"] = task_type_l
            plot_config["permutation_scoring"] = "average_precision"
            if class_names and 0 <= k < len(class_names):
                plot_config["target_label"] = f"Predicted P({class_names[k]})"
            else:
                plot_config["target_label"] = f"Predicted P(class_{k})"
        else:
            y_test_agg = np.asarray(y_test_agg_raw, dtype=float).ravel()
            plot_config = dict(plot_config or {})
            plot_config["task_type"] = "regression"

        logger.info(f"Aggregated test data shape: X={X_test_agg.shape}, y={y_test_agg.shape}")


        class CnnModelWrapperForSklearn:
            def __init__(self, model, device, patch_size):
                self.model = model
                self.device = device
                self.patch_size = patch_size
                self.model.eval()
                if str(task_type_l) == "multiclass_classification":
                    self._estimator_type = "classifier"
                    self.classes_ = np.asarray([0, 1], dtype=int)
                else:
                    self._estimator_type = "regressor"

            def fit(self, X, y):


                self.is_fitted_ = True
                return self

            def score(self, X, y):
                y_pred = self.predict(X)
                if str(task_type_l) == "multiclass_classification":
                    from sklearn.metrics import average_precision_score

                    return float(average_precision_score(y, y_pred))
                from sklearn.metrics import r2_score

                return float(r2_score(y, y_pred))

            def decision_function(self, X):

                return self.predict(X)

            def predict_proba(self, X):

                p = np.asarray(self.predict(X), dtype=float).ravel()
                p = np.clip(p, 0.0, 1.0)
                return np.stack([1.0 - p, p], axis=1)

            def predict(self, X):
                import torch

                X_expanded = np.expand_dims(np.expand_dims(X, axis=2), axis=3)
                X_tiled = np.tile(X_expanded, (1, 1, self.patch_size[0], self.patch_size[1]))
                X_tensor = torch.from_numpy(X_tiled).float().to(self.device)

                with torch.no_grad():
                    out = self.model(X_tensor)
                    if isinstance(out, dict):
                        for k0 in ("logits", "preds", "pred", "y_pred", "output"):
                            v0 = out.get(k0)
                            if torch.is_tensor(v0):
                                out = v0
                                break
                    elif isinstance(out, (tuple, list)):
                        for v0 in out:
                            if torch.is_tensor(v0):
                                out = v0
                                break
                    if str(task_type_l) == "multiclass_classification":
                        k = 1 if target_class_index is None else int(target_class_index)


                        if isinstance(out, (tuple, list)) and len(out) == 2 and torch.is_tensor(out[0]) and torch.is_tensor(out[1]):
                            end_logits, sub_logits = out[0], out[1]
                            if end_logits.ndim != 2 or end_logits.shape[1] != 2:
                                raise ValueError(f"Expected hierarchical end_logits (N,2), got {tuple(end_logits.shape)}")
                            if sub_logits.ndim != 2:
                                raise ValueError(f"Expected hierarchical sub_logits (N,S), got {tuple(sub_logits.shape)}")
                            end_p = torch.softmax(end_logits, dim=1)
                            sub_p = torch.softmax(sub_logits, dim=1)
                            if k == 0:
                                predictions = end_p[:, 0].cpu().numpy().ravel()
                            else:
                                j = k - 1
                                if not (0 <= j < int(sub_p.shape[1])):
                                    raise ValueError(f"target_class_index out of range for subclasses: class{k} (sub={j})")
                                predictions = (end_p[:, 1] * sub_p[:, j]).cpu().numpy().ravel()
                        else:
                            logits = out
                            if isinstance(logits, dict):
                                logits = logits.get("logits", logits)
                            if isinstance(logits, (tuple, list)):
                                logits = logits[0]
                            if not torch.is_tensor(logits) or logits.ndim != 2:
                                raise ValueError(f"Expected logits (N,K), got {type(logits)} {getattr(logits,'shape',None)}")
                            if not (0 <= k < int(logits.shape[1])):
                                raise ValueError(f"target_class_index out of range: {k} for K={int(logits.shape[1])}")
                            predictions = torch.softmax(logits, dim=1)[:, k].cpu().numpy().ravel()
                    else:
                        if not torch.is_tensor(out):
                            raise TypeError(f"Model output is not a torch.Tensor for regression: {type(out)}")
                        predictions = out.cpu().numpy().ravel()
                return predictions


        patch_size = plot_config.get("patch_size", (5, 5))
        device = next(model.parameters()).device
        wrapped_model = CnnModelWrapperForSklearn(model, device, patch_size)


        wrapped_model.fit(X_test_agg, y_test_agg)


        feature_names_eff = list(feature_names)
        X_test_eff = np.asarray(X_test_agg, dtype=float)

        if group_categorical_features:
            try:


                groups = _group_feature_names(feature_names_eff)
                all_results["post_hoc_categorical_groups"] = sorted(
                    [k for k, g in groups.items() if bool(g.get("is_categorical"))]
                )
            except Exception as e_group:
                all_results["post_hoc_categorical_groups"] = []
                all_results["post_hoc_categorical_groups_error"] = str(e_group)

        pdp_results = analyze_partial_dependence(
            wrapped_model,
            X_test_eff,
            feature_names_eff,
            output_dir,
            plot_config,
        )
        all_results.update(pdp_results)

        perm_imp_results = analyze_permutation_importance(
            wrapped_model,
            X_test_eff,
            y_test_agg,
            feature_names_eff,
            output_dir,
            plot_config,
        )
        all_results.update(perm_imp_results)

        shap_dep_results = analyze_shap_dependence(
            wrapped_model,
            X_test_eff,
            feature_names_eff,
            output_dir,
            plot_config,
        )
        all_results.update(shap_dep_results)

    except Exception as e:
        logger.error(f"A critical error occurred in the post-hoc analysis pipeline: {e}", exc_info=True)
        all_results["error_message_post_hoc"] = str(e)

    logger.info("Post-hoc explainability analysis finished.")
    return all_results

import logging
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import shap
import torch

from core.utils.logger_config import logger
from core.models.cnn_mlp import FactorizedEnvDepthModel
from core.visualization.shap_plots import (
    plot_shap_dependence, plot_shap_force, plot_shap_waterfall,
    plot_shap_decision, plot_shap_heatmap, plot_feature_clustering,
    plot_multi_sample_shap_decision, plot_multiple_shap_force_plots
)
from core.visualization.advanced_explainer_plots import (
    plot_shap_interactions,
    plot_partial_dependence_like,
    plot_feature_importance_ranking,
)


def _safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(name))


def _split_categorical_feature_name(name: str) -> tuple[str | None, str | None, str | None]:
    """Parse one-hot categorical feature names.

    Expected pattern:
      "<base>__cat_<id>__<label>"

    Returns: (base, cat_id, label) or (None, None, None).
    """
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


def _build_feature_groups(feature_names: list[str]) -> dict[str, dict]:
    """Group one-hot categorical features back into a single conceptual variable."""
    groups: dict[str, dict] = {}

    for i, fn in enumerate(list(feature_names)):
        base, cat_id, label = _split_categorical_feature_name(fn)
        if base is None:

            groups.setdefault(fn, {"name": fn, "indices": [], "is_categorical": False, "categories": []})
            groups[fn]["indices"].append(i)
            continue

        g = groups.setdefault(
            base,
            {"name": base, "indices": [], "is_categorical": True, "categories": []},
        )
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


def _save_group_importance(
    *,
    shap_values_agg: np.ndarray,
    feature_values_agg: np.ndarray,
    feature_names: list[str],
    output_dir: Path,
    plot_config: dict,
) -> dict:
    """Write grouped importance + per-category SHAP summaries."""
    results: dict = {}

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        groups = _build_feature_groups(feature_names)

        global_importance = np.abs(shap_values_agg).mean(axis=0)

        group_rows = []
        for g in groups.values():
            idxs = list(g.get("indices") or [])
            if not idxs:
                continue
            imp = float(np.sum(global_importance[idxs]))
            group_rows.append(
                {
                    "group": str(g.get("name")),
                    "importance_sum": imp,
                    "n_channels": int(len(idxs)),
                    "is_categorical": bool(g.get("is_categorical")),
                }
            )

        group_df = pd.DataFrame(group_rows)
        group_df.sort_values(by="importance_sum", ascending=False, inplace=True)

        csv_path = output_dir / "shap_feature_group_importance.csv"
        group_df.to_csv(csv_path, index=False)
        results["feature_group_importance_csv_path"] = str(csv_path)


        try:
            import matplotlib.pyplot as _plt

            n_top = int((plot_config or {}).get("group_importance_top_n", 20))
            top_df = group_df.head(max(1, n_top)).copy()
            top_df.sort_values(by="importance_sum", ascending=True, inplace=True)

            fig_h = max(5.0, 0.35 * len(top_df) + 1.0)
            _plt.figure(figsize=(9.0, fig_h))
            colors = ["#1f78b4" if not bool(v) else "#33a02c" for v in top_df["is_categorical"].tolist()]
            _plt.barh(top_df["group"], top_df["importance_sum"], color=colors, edgecolor="black", linewidth=0.5)
            _plt.xlabel("Sum of mean(|SHAP|) across channels")
            _plt.title("Grouped SHAP importance (categoricals merged)")
            _plt.tight_layout()

            fig_path = output_dir / "shap_group_importance.png"
            _plt.savefig(fig_path, dpi=int((plot_config or {}).get("dpi", 300)))
            _plt.close()
            results["feature_group_importance_plot_path"] = str(fig_path)
        except Exception:
            results["feature_group_importance_plot_path"] = None


        cat_dir = output_dir / "categorical_groups"
        cat_dir.mkdir(parents=True, exist_ok=True)
        per_group_csvs = []
        per_group_plots = []
        for g in groups.values():
            if not bool(g.get("is_categorical")):
                continue
            cats = list(g.get("categories") or [])
            if not cats:
                continue

            rows = []
            for c in cats:
                j = int(c["idx"])

                xj = np.asarray(feature_values_agg[:, j], dtype=float)
                active = np.isfinite(xj) & (xj > 0.5)
                n_active = int(active.sum())
                if n_active <= 0:
                    mean_shap = float("nan")
                    mean_abs_shap = float("nan")
                else:
                    sv = np.asarray(shap_values_agg[active, j], dtype=float)
                    mean_shap = float(np.nanmean(sv))
                    mean_abs_shap = float(np.nanmean(np.abs(sv)))
                rows.append(
                    {
                        "group": str(g.get("name")),
                        "category_feature": str(c.get("feature")),
                        "cat_id": c.get("cat_id"),
                        "label": c.get("label"),
                        "n_active": n_active,
                        "mean_shap": mean_shap,
                        "mean_abs_shap": mean_abs_shap,
                    }
                )

            df_cat = pd.DataFrame(rows)
            df_cat.sort_values(by="mean_abs_shap", ascending=False, inplace=True, na_position="last")
            base_safe = _safe_filename(str(g.get("name")))
            out_csv = cat_dir / f"categorical_shap_{base_safe}.csv"
            df_cat.to_csv(out_csv, index=False)
            per_group_csvs.append(str(out_csv))


            try:
                import matplotlib.pyplot as _plt

                df_plot = df_cat.copy()

                df_plot["cat_label"] = df_plot["label"].fillna(df_plot["cat_id"].astype(str))
                df_plot.sort_values(by="mean_shap", ascending=True, inplace=True)

                fig_h = max(3.5, 0.35 * len(df_plot) + 0.8)
                _plt.figure(figsize=(9.0, fig_h))
                vals = df_plot["mean_shap"].to_numpy(dtype=float)
                cols = ["#e31a1c" if v > 0 else "#1f78b4" for v in vals]
                _plt.barh(df_plot["cat_label"], vals, color=cols, edgecolor="black", linewidth=0.5)
                _plt.axvline(0.0, color="black", lw=1.0)
                _plt.xlabel("Mean SHAP (conditional on category present)")
                _plt.title(f"Categorical effect: {g.get('name')}")
                _plt.tight_layout()

                out_png = cat_dir / f"categorical_shap_{base_safe}.png"
                _plt.savefig(out_png, dpi=int((plot_config or {}).get("dpi", 300)))
                _plt.close()
                per_group_plots.append(str(out_png))
            except Exception:
                pass

        results["categorical_group_csv_paths"] = per_group_csvs
        results["categorical_group_plot_paths"] = per_group_plots

    except Exception as e:
        results["feature_group_importance_csv_path"] = None
        results["feature_group_importance_plot_path"] = None
        results["categorical_group_csv_paths"] = []
        results["categorical_group_plot_paths"] = []
        results["feature_grouping_error"] = str(e)

    return results


def _aggregate_by_feature_groups(
    *,
    shap_values_agg: np.ndarray,
    feature_values_agg: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """Collapse one-hot categoricals into group-level features.

    - SHAP aggregation: sum within group (preserves additivity).
    - Feature value aggregation: best-effort (categorical groups use max across one-hot channels).
      This is mainly for compatibility; grouped decision plots can omit feature values.
    """

    groups = _build_feature_groups(feature_names)
    group_names: list[str] = []
    group_member_indices: list[list[int]] = []
    group_meta: dict[str, dict] = {}

    for g in groups.values():
        idxs = [int(i) for i in (g.get("indices") or [])]
        if not idxs:
            continue
        name = str(g.get("name") or "")
        if not name:
            name = f"group_{len(group_names)}"
        group_names.append(name)
        group_member_indices.append(idxs)
        group_meta[name] = {
            "is_categorical": bool(g.get("is_categorical")),
            "indices": idxs,
            "categories": g.get("categories") or [],
        }

    if not group_names:
        raise ValueError("No feature groups could be built (empty feature_names?)")

    shap_values_agg = np.asarray(shap_values_agg, dtype=float)
    feature_values_agg = np.asarray(feature_values_agg, dtype=float)
    if shap_values_agg.ndim != 2:
        raise ValueError(f"Expected shap_values_agg (N,F), got shape={shap_values_agg.shape}")
    if feature_values_agg.ndim != 2:
        raise ValueError(f"Expected feature_values_agg (N,F), got shape={feature_values_agg.shape}")
    if shap_values_agg.shape != feature_values_agg.shape:
        raise ValueError(
            "shap_values_agg and feature_values_agg must match shape; "
            f"got {shap_values_agg.shape} vs {feature_values_agg.shape}"
        )


    shap_grouped_cols = []
    x_grouped_cols = []
    for idxs in group_member_indices:
        sv = np.sum(shap_values_agg[:, idxs], axis=1)
        shap_grouped_cols.append(sv)

        if len(idxs) == 1:
            x_grouped_cols.append(feature_values_agg[:, idxs[0]])
        else:

            x_grouped_cols.append(np.nanmax(feature_values_agg[:, idxs], axis=1))

    shap_grouped = np.stack(shap_grouped_cols, axis=1)
    x_grouped = np.stack(x_grouped_cols, axis=1)
    return shap_grouped, x_grouped, group_names, group_meta

class ShapModelWrapper(torch.nn.Module):
    """
    A wrapper to make our unified model compatible with the SHAP library.
    It takes a single combined patch tensor as input.
    """
    def __init__(self, model, device):
        super(ShapModelWrapper, self).__init__()
        self.model = model
        self.device = device

    def forward(self, combined_patch):
        """
        A simplified forward method that accepts a single tensor.
        """
        try:
            combined_patch = combined_patch.to(self.device, dtype=torch.float32, non_blocking=True).contiguous()
            logger.debug(f"ShapModelWrapper processed combined_patch: shape={combined_patch.shape}, dtype={combined_patch.dtype}, device={combined_patch.device}")
            out = self.model(combined_patch)
            return _unwrap_torch_output(out, context="ShapModelWrapper.forward")
        except Exception as e:
            logger.error("!!! Exception in ShapModelWrapper.forward, re-raising exception to halt execution !!!", exc_info=True)
            raise e


def _unwrap_torch_output(out, *, context: str) -> torch.Tensor:
    """
    Normalize model outputs to a single torch.Tensor.

    Some models return `(preds, aux)` tuples or dict-like outputs; SHAP/sklearn
    wrappers expect a tensor-like output with `.ndim`.
    """
    if torch.is_tensor(out):
        return out

    if isinstance(out, dict):
        for k in ("logits", "preds", "pred", "y_pred", "output"):
            v = out.get(k)
            if torch.is_tensor(v):
                return v

    if isinstance(out, (tuple, list)):
        for v in out:
            if torch.is_tensor(v):
                return v
            if isinstance(v, dict):
                for k in ("logits", "preds", "pred", "y_pred", "output"):
                    vv = v.get(k)
                    if torch.is_tensor(vv):
                        return vv

    raise TypeError(f"{context}: unsupported model output type: {type(out)}")


class MulticlassTargetProbWrapper(torch.nn.Module):
    """
    Wrap a multiclass model to expose a single target class probability as a
    scalar output (N, 1) for SHAP.

    For multiclass mechanism labels, we often care about the rare class (e.g.,
    class1) PR/AP. Explaining P(class_k) matches that objective and avoids
    multi-output SHAP complexity.
    """

    def __init__(self, model, device, target_class_index: int):
        super().__init__()
        self.model = model
        self.device = device
        self.target_class_index = int(target_class_index)

    def forward(self, combined_patch):
        try:
            combined_patch = combined_patch.to(self.device, dtype=torch.float32, non_blocking=True).contiguous()
            out = self.model(combined_patch)


            if isinstance(out, (tuple, list)) and len(out) == 2 and torch.is_tensor(out[0]) and torch.is_tensor(out[1]):
                end_logits, sub_logits = out[0], out[1]
                if end_logits.ndim != 2 or end_logits.shape[1] != 2:
                    raise ValueError(
                        f"Expected hierarchical end_logits shape (N,2), got {tuple(end_logits.shape)}"
                    )
                if sub_logits.ndim != 2:
                    raise ValueError(
                        f"Expected hierarchical sub_logits shape (N,S), got {tuple(sub_logits.shape)}"
                    )
                end_p = torch.softmax(end_logits, dim=1)
                sub_p = torch.softmax(sub_logits, dim=1)
                k = int(self.target_class_index)
                if k == 0:
                    prob = end_p[:, 0:1]
                else:
                    j = k - 1
                    if not (0 <= j < int(sub_p.shape[1])):
                        raise ValueError(f"target_class_index out of range for subclasses: class{k} (sub={j})")
                    prob = end_p[:, 1:2] * sub_p[:, j : j + 1]
                return prob

            logits = _unwrap_torch_output(out, context="MulticlassTargetProbWrapper.forward")
            if logits.ndim != 2:
                raise ValueError(f"Expected model output logits with shape (N, K), got {tuple(logits.shape)}")
            k = int(self.target_class_index)
            if not (0 <= k < int(logits.shape[1])):
                raise ValueError(f"target_class_index out of range: {k} for K={int(logits.shape[1])}")
            prob = torch.softmax(logits, dim=1)[:, k : k + 1]
            return prob
        except Exception as e:
            logger.error("Exception in MulticlassTargetProbWrapper.forward.", exc_info=True)
            raise e


class EnvBranchShapWrapper(torch.nn.Module):
    """
    Wrapper for FactorizedEnvDepthModel to expose only the EnvScore branch
    to SHAP, while keeping the input interface identical (combined_patch).
    """
    def __init__(self, model, device):
        super(EnvBranchShapWrapper, self).__init__()
        self.model = model
        self.device = device

    def forward(self, combined_patch):
        try:
            combined_patch = combined_patch.to(self.device, dtype=torch.float32, non_blocking=True).contiguous()
            preds, components = self.model(combined_patch, return_components=True)
            env_score = components.get("env_score", preds)
            logger.debug(
                f"EnvBranchShapWrapper: combined_patch shape={combined_patch.shape}, "
                f"env_score shape={env_score.shape}"
            )
            return env_score
        except Exception as e:
            logger.error("Exception in EnvBranchShapWrapper.forward.", exc_info=True)
            raise e


class DepthBranchShapWrapper(torch.nn.Module):
    """
    Wrapper for FactorizedEnvDepthModel to expose only the DepthFactor branch
    to SHAP, while keeping the input interface identical (combined_patch).
    """
    def __init__(self, model, device):
        super(DepthBranchShapWrapper, self).__init__()
        self.model = model
        self.device = device

    def forward(self, combined_patch):
        try:
            combined_patch = combined_patch.to(self.device, dtype=torch.float32, non_blocking=True).contiguous()
            preds, components = self.model(combined_patch, return_components=True)
            depth_factor = components.get("depth_factor", preds)
            logger.debug(
                f"DepthBranchShapWrapper: combined_patch shape={combined_patch.shape}, "
                f"depth_factor shape={depth_factor.shape}"
            )
            return depth_factor
        except Exception as e:
            logger.error("Exception in DepthBranchShapWrapper.forward.", exc_info=True)
            raise e

def analyze_shap_values(model, train_loader, test_loader, feature_names, output_dir, plot_config, **kwargs):
    """
    Performs SHAP analysis on the model.

    Args:
        model (torch.nn.Module): The trained model.
        train_loader (DataLoader): DataLoader for background data.
        test_loader (DataLoader): DataLoader for data to explain.
        feature_names (list): List of feature names (for environment patches).
        output_dir (str): Directory to save SHAP plots.
        plot_config (dict): Plotting configuration.
        selected_env_variables (list, optional): List of selected environmental variable names.
                                                Used for patch feature names if provided.
                                                Defaults to feature_names if None.
    """
    logger.info("SHAP Analyzer: Starting SHAP analysis...")

    logger.debug(f"SHAP Analyzer: Received model of type: {type(model)}")
    logger.debug(f"SHAP Analyzer: Received train_loader of type: {type(train_loader)}")
    logger.debug(f"SHAP Analyzer: Received test_loader of type: {type(test_loader)}")

    logger.info(f"SHAP Analyzer: Received feature_names: {feature_names}")
    logger.info(f"SHAP Analyzer: Number of features received: {len(feature_names)}")

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    try:

        logger.debug("SHAP Analyzer: Validating data loaders.")
        if not train_loader or not test_loader:
            logger.warning("SHAP Analyzer: Train or Test loader is empty. Skipping SHAP analysis.")
            return results

        try:

            background_data, _, _ = next(iter(train_loader))
            logger.debug(f"SHAP background_data shape: {getattr(background_data, 'shape', None)}")
        except StopIteration:
            logger.warning("SHAP Analyzer: Train loader is empty, cannot get background data. Skipping SHAP.")
            return results

        try:
            explanation_data, _, _ = next(iter(test_loader))
            logger.debug(f"SHAP explanation_data shape: {getattr(explanation_data, 'shape', None)}")
        except StopIteration:
            logger.warning("SHAP Analyzer: Test loader is empty, cannot get data to explain. Skipping SHAP.")
            return results

        try:
            n_bg = int(getattr(background_data, "shape", [0])[0])
        except Exception:
            n_bg = 0
        try:
            n_exp = int(getattr(explanation_data, "shape", [0])[0])
        except Exception:
            n_exp = 0
        logger.debug(f"SHAP Analyzer: Using {n_bg} samples for background, {n_exp} for SHAP values.")


        device = next(model.parameters()).device
        model.eval()


        task_type = str(kwargs.get("task_type") or "").lower()
        target_class_index = kwargs.get("target_class_index", None)
        if task_type == "multiclass_classification":
            if target_class_index is None:
                target_class_index = 1
            wrapped_model_for_shap = MulticlassTargetProbWrapper(model, device, int(target_class_index)).to(device)
        else:
            wrapped_model_for_shap = ShapModelWrapper(model, device).to(device)
        wrapped_model_for_shap.eval()

        background_data = background_data.to(device)
        explanation_data = explanation_data.to(device)

        logger.debug(f"Background Data Shape: {background_data.shape}, Device: {background_data.device}")
        logger.debug(f"Explanation Data Shape: {explanation_data.shape}, Device: {explanation_data.device}")

        try:
            logger.debug("SHAP Analyzer: Creating SHAP GradientExplainer (unified)...")
            explainer = shap.GradientExplainer(wrapped_model_for_shap, background_data)
            logger.info("SHAP Analyzer: Unified SHAP GradientExplainer created successfully.")
        except Exception as e:
            logger.error("SHAP Analyzer: Failed to create unified SHAP Explainer.", exc_info=True)
            return {"error_message_shap": f"Failed to create explainer: {e}"}


        shap_values_result = None
        logger.debug(f"Final Check - Model device: {device}, Explanation Data device: {explanation_data.device}")

        try:
            logger.debug("SHAP Analyzer: Calculating unified SHAP values...")
            logger.debug(f"Pre-SHAP Calculation - Explanation Data Shape: {explanation_data.shape}, dtype: {explanation_data.dtype}, device: {explanation_data.device}")

            explanation_data.requires_grad_(True)
            shap_values_result = explainer.shap_values(explanation_data)


            if isinstance(shap_values_result, np.ndarray) and shap_values_result.ndim == 5 and shap_values_result.shape[-1] == 1:
                shap_values_result = np.squeeze(shap_values_result, axis=-1)
                logger.debug(f"Squeezed shap_values_result to shape: {getattr(shap_values_result, 'shape', None)}")

            logger.info("SHAP Analyzer: Unified SHAP values calculated successfully.")
            if shap_values_result is None:
                raise ValueError("explainer.shap_values() returned None.")

            if isinstance(shap_values_result, list):
                logger.debug(f"Unified SHAP values result is a list of length {len(shap_values_result)}.")
                for i, item in enumerate(shap_values_result):
                    if hasattr(item, 'shape'):
                        logger.debug(f"  Item {i} type: {type(item)}, shape: {getattr(item, 'shape', None)}")
                    else:
                        logger.debug(f"  Item {i} type: {type(item)}, no shape attribute.")
            elif hasattr(shap_values_result, 'shape'):
                logger.debug(f"Unified SHAP values result type: {type(shap_values_result)}, shape: {getattr(shap_values_result, 'shape', None)}")
            else:
                logger.debug(f"Unified SHAP values result type: {type(shap_values_result)}, no shape attribute.")

        except Exception as e:
            logger.error(f"SHAP unified explainer.shap_values() failed with exception: {e}", exc_info=True)
            error_context = f"Failed on explanation_data with shape: {explanation_data.shape}, dtype: {explanation_data.dtype}."
            full_error_message = f"Unified SHAP value calculation failed: {e}. {error_context}"
            return {"error_message_shap": full_error_message}

        if shap_values_result is None:
            logger.error("SHAP Analyzer: unified shap_values returned None.")
            return {"error_message_shap": "Unified SHAP value calculation returned None."}


        if isinstance(shap_values_result, (list, tuple)):
            if not shap_values_result:
                raise ValueError("Unified SHAP value calculation returned an empty list/tuple.")
            shap_values_result = shap_values_result[0]

        shap_values_arr = np.asarray(shap_values_result)
        if shap_values_arr.ndim >= 4:
            shap_values_agg = np.mean(shap_values_arr, axis=(2, 3))
        elif shap_values_arr.ndim == 2:
            shap_values_agg = shap_values_arr
        else:
            raise ValueError(f"Unexpected SHAP values shape: {shap_values_arr.shape}")
        explanation_data_agg = explanation_data.detach().cpu().numpy().mean(axis=(2, 3))


        depth_feature_type = str(kwargs.get("depth_feature_type") or "")
        depth_feature_name = "age_ka_proxy" if depth_feature_type == "age_ka_proxy" else "depth"
        all_feature_names = list(feature_names) + [depth_feature_name]

        # Optional coordinate channels appended after depth: sin/cos(lat/lon)
        add_coord_channels = bool(kwargs.get("add_coord_channels", False))
        if add_coord_channels:
            all_feature_names += ["sin_lat", "cos_lat", "sin_lon", "cos_lon"]

        # If channel count mismatches (e.g., legacy artifacts), fall back to generic names.
        if shap_values_agg.shape[1] != len(all_feature_names):
            all_feature_names = [f"ch_{i}" for i in range(int(shap_values_agg.shape[1]))]


        plots_shap_dir = Path(output_dir)
        plots_shap_dir.mkdir(parents=True, exist_ok=True)


        try:
            from core.visualization.publication_style import publication_style, save_figure

            with publication_style(plot_config):
                fig = plt.figure(figsize=tuple(plot_config.get("shap_summary_plot_size", (8.3, 6.0))))
                shap.summary_plot(
                    shap_values_agg,
                    features=explanation_data_agg,
                    feature_names=all_feature_names,
                    plot_type=str(plot_config.get("shap_summary_plot_type", "bar")),
                    max_display=int(plot_config.get("shap_max_display", 20)),
                    show=False,
                )
                plt.title("SHAP Summary (mean |SHAP|)")
                plt.tight_layout()

                output_path_base = plots_shap_dir / "shap_summary_unified"
                save_paths = save_figure(fig, path_no_ext=output_path_base, plot_config=plot_config)

            results["shap_summary_plot_path"] = next(
                (p for p in save_paths if p.lower().endswith(".png")),
                save_paths[0] if save_paths else None,
            )
            logger.info(f"SHAP Analyzer: Saved unified SHAP summary plot to {save_paths}")
            plt.close(fig)
        except Exception:
            logger.error("SHAP Analyzer: Failed to generate unified SHAP summary plot.", exc_info=True)
            results["shap_summary_plot_path"] = None


        is_factorized = isinstance(model, FactorizedEnvDepthModel)


        if is_factorized and plot_config.get("run_env_branch_shap", True):
            try:
                logger.info("SHAP Analyzer: Running Env branch SHAP (EnvScore) for FactorizedEnvDepthModel...")
                env_wrapper = EnvBranchShapWrapper(model, device).to(device)
                env_wrapper.eval()

                env_explainer = shap.GradientExplainer(env_wrapper, background_data)
                shap_values_env = env_explainer.shap_values(explanation_data)

                if isinstance(shap_values_env, np.ndarray) and shap_values_env.ndim == 5 and shap_values_env.shape[-1] == 1:
                    shap_values_env = np.squeeze(shap_values_env, axis=-1)
                    logger.debug(f"Env branch: squeezed shap_values_env to shape: {shap_values_env.shape}")

                if isinstance(shap_values_env, (list, tuple)):
                    shap_values_env = shap_values_env[0] if shap_values_env else None
                shap_env_arr = np.asarray(shap_values_env)
                if shap_env_arr.ndim >= 4:
                    shap_env_agg = np.mean(shap_env_arr, axis=(2, 3))
                elif shap_env_arr.ndim == 2:
                    shap_env_agg = shap_env_arr
                else:
                    raise ValueError(f"Unexpected env-branch SHAP shape: {shap_env_arr.shape}")
                explanation_env_agg = explanation_data.detach().cpu().numpy().mean(axis=(2, 3))

                num_env_features = len(feature_names)
                shap_env_agg = shap_env_agg[:, :num_env_features]
                explanation_env_agg = explanation_env_agg[:, :num_env_features]

                from core.visualization.publication_style import publication_style, save_figure

                with publication_style(plot_config):
                    fig = plt.figure(figsize=tuple(plot_config.get("shap_summary_plot_size", (8.3, 6.0))))
                    shap.summary_plot(
                        shap_env_agg,
                        features=explanation_env_agg,
                        feature_names=feature_names,
                        plot_type=str(plot_config.get("shap_summary_plot_type", "bar")),
                        max_display=int(plot_config.get("shap_max_display", 20)),
                        show=False,
                    )
                    plt.title("SHAP Summary (Env Branch)")
                    plt.tight_layout()

                    env_summary_path = plots_shap_dir / "shap_summary_env_branch"
                    save_paths = save_figure(fig, path_no_ext=env_summary_path, plot_config=plot_config)

                results["shap_summary_env_branch_path"] = next(
                    (p for p in save_paths if p.lower().endswith(".png")),
                    save_paths[0] if save_paths else None,
                )
                logger.info(f"SHAP Analyzer: Saved Env-branch SHAP summary plot to {save_paths}")
                plt.close(fig)
            except Exception as e:
                logger.error("SHAP Analyzer: Env branch SHAP analysis failed.", exc_info=True)
                results["shap_summary_env_branch_path"] = None


        if is_factorized and plot_config.get("run_depth_branch_shap", True):
            try:
                logger.info("SHAP Analyzer: Running Depth branch SHAP (DepthFactor) for FactorizedEnvDepthModel...")
                depth_wrapper = DepthBranchShapWrapper(model, device).to(device)
                depth_wrapper.eval()

                depth_explainer = shap.GradientExplainer(depth_wrapper, background_data)
                shap_values_depth = depth_explainer.shap_values(explanation_data)

                if isinstance(shap_values_depth, np.ndarray) and shap_values_depth.ndim == 5 and shap_values_depth.shape[-1] == 1:
                    shap_values_depth = np.squeeze(shap_values_depth, axis=-1)
                    logger.debug(f"Depth branch: squeezed shap_values_depth to shape: {shap_values_depth.shape}")

                if isinstance(shap_values_depth, (list, tuple)):
                    shap_values_depth = shap_values_depth[0] if shap_values_depth else None
                shap_depth_arr = np.asarray(shap_values_depth)
                if shap_depth_arr.ndim >= 4:
                    shap_depth_agg = np.mean(shap_depth_arr, axis=(2, 3))
                elif shap_depth_arr.ndim == 2:
                    shap_depth_agg = shap_depth_arr
                else:
                    raise ValueError(f"Unexpected depth-branch SHAP shape: {shap_depth_arr.shape}")
                explanation_depth_agg = explanation_data.detach().cpu().numpy().mean(axis=(2, 3))

                from core.visualization.publication_style import publication_style, save_figure

                with publication_style(plot_config):
                    fig = plt.figure(figsize=tuple(plot_config.get("shap_summary_plot_size", (8.3, 6.0))))
                    shap.summary_plot(
                        shap_depth_agg,
                        features=explanation_depth_agg,
                        feature_names=all_feature_names,
                        plot_type=str(plot_config.get("shap_summary_plot_type", "bar")),
                        max_display=int(plot_config.get("shap_max_display", 20)),
                        show=False,
                    )
                    plt.title("SHAP Summary (Depth Branch)")
                    plt.tight_layout()

                    depth_summary_path = plots_shap_dir / "shap_summary_depth_branch"
                    save_paths = save_figure(fig, path_no_ext=depth_summary_path, plot_config=plot_config)

                results["shap_summary_depth_branch_path"] = next(
                    (p for p in save_paths if p.lower().endswith(".png")),
                    save_paths[0] if save_paths else None,
                )
                logger.info(f"SHAP Analyzer: Saved Depth-branch SHAP summary plot to {save_paths}")
                plt.close(fig)
            except Exception as e:
                logger.error("SHAP Analyzer: Depth branch SHAP analysis failed.", exc_info=True)
                results["shap_summary_depth_branch_path"] = None


        global_importance = np.abs(shap_values_agg).mean(axis=0)
        sorted_indices = np.argsort(global_importance)[::-1]
        sorted_features: list[str] = [str(all_feature_names[i]) for i in sorted_indices]


        shap_values_sorted = shap_values_agg[:, sorted_indices]
        X_test_sorted_np = explanation_data_agg[:, sorted_indices]
        X_test_sorted = pd.DataFrame(X_test_sorted_np, columns=sorted_features)


        try:
            expected_value_raw = getattr(explainer, "expected_value", 0.0)
            expected_value_scalar = float(np.asarray(expected_value_raw).reshape(-1)[0])
        except Exception:
            expected_value_scalar = 0.0


        try:
            logger.info("SHAP Analyzer: Generating SHAP decision plot...")


            n_samples_decision = min(plot_config.get("decision_plot_samples", 50), shap_values_sorted.shape[0])
            n_features_decision = min(plot_config.get("decision_plot_features", 10), len(sorted_features))


            decision_plot_save_path = plots_shap_dir / "shap_decision_plot"

            plot_shap_decision(
                base_value=expected_value_scalar,
                shap_values=shap_values_sorted[:n_samples_decision, :n_features_decision],
                features=X_test_sorted.iloc[:n_samples_decision, :n_features_decision],
                feature_names=sorted_features[:n_features_decision],
                save_path=decision_plot_save_path,
                show_plot=False,
                title=f"SHAP decision plot - top {n_features_decision}",
                figsize=(8.3, 6.0),
                plot_config=plot_config,
            )

            results["shap_decision_plot_path"] = str(decision_plot_save_path.with_suffix('.png'))
            logger.info(f"SHAP Analyzer: Successfully generated SHAP decision plot.")

        except Exception as e:
            logger.error("SHAP Analyzer: Failed to generate SHAP decision plot.", exc_info=True)
            results["shap_decision_plot_path"] = None


        try:
            logger.info("SHAP Analyzer: Saving feature importance to CSV...")
            importance_df = pd.DataFrame(
                {
                    "feature": sorted_features,
                    "importance": global_importance[sorted_indices],
                }
            )
            csv_save_path = plots_shap_dir / "shap_feature_importance.csv"
            importance_df.to_csv(csv_save_path, index=False)
            results["feature_importance_csv_path"] = str(csv_save_path)
            logger.info(f"SHAP Analyzer: Feature importance saved to {csv_save_path}")
        except Exception:
            logger.error("SHAP Analyzer: Failed to save feature importance CSV.", exc_info=True)
            results["feature_importance_csv_path"] = None


        try:
            group_out_dir = plots_shap_dir / "grouped"
            group_results = _save_group_importance(
                shap_values_agg=shap_values_agg,
                feature_values_agg=explanation_data_agg,
                feature_names=all_feature_names,
                output_dir=group_out_dir,
                plot_config=plot_config,
            )
            results.update(group_results)
        except Exception as e:
            results.setdefault("feature_group_importance_csv_path", None)
            results.setdefault("feature_group_importance_plot_path", None)
            results.setdefault("categorical_group_csv_paths", [])
            results.setdefault("categorical_group_plot_paths", [])
            results["feature_grouping_error"] = str(e)


        try:
            enable_grouped = bool((plot_config or {}).get("group_categorical_shap_plots", True))
            if enable_grouped:
                group_out_dir = plots_shap_dir / "grouped"
                group_out_dir.mkdir(parents=True, exist_ok=True)

                shap_grouped, x_grouped, group_names, group_meta = _aggregate_by_feature_groups(
                    shap_values_agg=shap_values_agg,
                    feature_values_agg=explanation_data_agg,
                    feature_names=all_feature_names,
                )


                did_merge = any(
                    bool(m.get("is_categorical")) and len(m.get("indices") or []) > 1
                    for m in (group_meta or {}).values()
                )

                meta_path = group_out_dir / "feature_groups.json"
                try:
                    with open(meta_path, "w", encoding="utf-8") as f:
                        json.dump(group_meta, f, indent=2, ensure_ascii=True)
                    results["feature_groups_json_path"] = str(meta_path)
                except Exception:
                    results["feature_groups_json_path"] = None

                if did_merge:

                    try:
                        from core.visualization.publication_style import publication_style, save_figure

                        with publication_style(plot_config):
                            fig = plt.figure(figsize=tuple(plot_config.get("shap_summary_plot_size", (8.3, 6.0))))
                            shap.summary_plot(
                                shap_grouped,
                                features=None,
                                feature_names=group_names,
                                plot_type="bar",
                                max_display=int(plot_config.get("shap_max_display", 20)),
                                show=False,
                            )
                            plt.title("SHAP Summary (grouped categoricals)")
                            plt.tight_layout()

                            out_base = group_out_dir / "shap_summary_grouped_bar"
                            save_paths = save_figure(fig, path_no_ext=out_base, plot_config=plot_config)

                        results["shap_summary_grouped_bar_path"] = next(
                            (p for p in save_paths if str(p).lower().endswith(".png")),
                            save_paths[0] if save_paths else None,
                        )
                        plt.close(fig)
                    except Exception:
                        logger.error("SHAP Analyzer: Failed to generate grouped SHAP summary plot.", exc_info=True)
                        results["shap_summary_grouped_bar_path"] = None


                    group_importance = np.abs(shap_grouped).mean(axis=0)
                    group_sorted_idx = np.argsort(group_importance)[::-1]
                    group_sorted_names = [group_names[i] for i in group_sorted_idx]
                    shap_grouped_sorted = shap_grouped[:, group_sorted_idx]


                    try:
                        n_samples_decision_g = min(
                            int(plot_config.get("decision_plot_samples", 50)),
                            int(shap_grouped_sorted.shape[0]),
                        )
                        n_features_decision_g = min(
                            int(plot_config.get("decision_plot_features", 10)),
                            int(len(group_sorted_names)),
                        )

                        decision_plot_save_path_g = group_out_dir / "shap_decision_plot_grouped"
                        plot_shap_decision(
                            base_value=expected_value_scalar,
                            shap_values=shap_grouped_sorted[:n_samples_decision_g, :n_features_decision_g],
                            features=None,
                            feature_names=group_sorted_names[:n_features_decision_g],
                            save_path=decision_plot_save_path_g,
                            show_plot=False,
                            title=f"SHAP decision plot (grouped) - top {n_features_decision_g}",
                            figsize=(8.3, 6.0),
                            plot_config=plot_config,
                        )
                        results["shap_decision_plot_grouped_path"] = str(decision_plot_save_path_g.with_suffix(".png"))
                    except Exception:
                        logger.error("SHAP Analyzer: Failed to generate grouped SHAP decision plot.", exc_info=True)
                        results["shap_decision_plot_grouped_path"] = None


                    try:
                        n_samples_multi_g = min(
                            int(plot_config.get("multi_decision_plot_samples", 50)),
                            int(shap_grouped_sorted.shape[0]),
                        )
                        n_features_multi_g = min(
                            int(plot_config.get("multi_decision_plot_features", 10)),
                            int(len(group_sorted_names)),
                        )

                        multi_save_path_g = group_out_dir / "shap_multi_sample_decision_plot_grouped"
                        plot_multi_sample_shap_decision(
                            base_value=expected_value_scalar,
                            shap_values=shap_grouped_sorted[:n_samples_multi_g, :n_features_multi_g],
                            features=None,
                            feature_names=group_sorted_names[:n_features_multi_g],
                            save_path=multi_save_path_g,
                            show_plot=False,
                            title=f"SHAP decision plot (grouped, multi-sample) - top {n_features_multi_g}",
                            figsize=(8.3, 6.0),
                            plot_config=plot_config,
                        )
                        results["shap_multi_decision_plot_grouped_path"] = str(multi_save_path_g.with_suffix(".png"))
                    except Exception:
                        logger.error("SHAP Analyzer: Failed to generate grouped multi-sample decision plot.", exc_info=True)
                        results["shap_multi_decision_plot_grouped_path"] = None


                    try:
                        max_cluster = int(plot_config.get("grouped_clustering_features", 50))
                        n_cluster = min(max_cluster, int(len(group_sorted_names)))
                        cluster_idx = group_sorted_idx[:n_cluster]
                        cluster_names = [group_names[i] for i in cluster_idx]
                        shap_cluster = shap_grouped[:, cluster_idx]

                        clustering_save_path_g = group_out_dir / "shap_feature_clustering_grouped"
                        plot_feature_clustering(
                            shap_importance=shap_cluster,
                            feature_names=cluster_names,
                            save_path=clustering_save_path_g,
                            show_plot=False,
                            title="Feature clustering (grouped categoricals)",
                            figsize=(8.3, 6.0),
                            metric="correlation",
                            method="average",
                            plot_config=plot_config,
                        )
                        results["feature_clustering_grouped_plot_path"] = str(clustering_save_path_g.with_suffix(".png"))
                    except Exception:
                        logger.error("SHAP Analyzer: Failed to generate grouped feature clustering plot.", exc_info=True)
                        results["feature_clustering_grouped_plot_path"] = None
                else:
                    results.setdefault("shap_summary_grouped_bar_path", None)
                    results.setdefault("shap_decision_plot_grouped_path", None)
                    results.setdefault("shap_multi_decision_plot_grouped_path", None)
                    results.setdefault("feature_clustering_grouped_plot_path", None)
            else:
                results.setdefault("shap_summary_grouped_bar_path", None)
                results.setdefault("shap_decision_plot_grouped_path", None)
                results.setdefault("shap_multi_decision_plot_grouped_path", None)
                results.setdefault("feature_clustering_grouped_plot_path", None)
                results.setdefault("feature_groups_json_path", None)
        except Exception as e:
            logger.error(f"SHAP Analyzer: Grouped SHAP plot generation failed: {e}", exc_info=True)
            results.setdefault("shap_summary_grouped_bar_path", None)
            results.setdefault("shap_decision_plot_grouped_path", None)
            results.setdefault("shap_multi_decision_plot_grouped_path", None)
            results.setdefault("feature_clustering_grouped_plot_path", None)
            results.setdefault("feature_groups_json_path", None)


        try:
            logger.info("SHAP Analyzer: Generating SHAP heatmap...")
            n_samples_heatmap = min(plot_config.get("heatmap_samples", 20), shap_values_sorted.shape[0])
            n_features_heatmap = min(plot_config.get("heatmap_features", 15), len(sorted_features))

            heatmap_explanation = shap.Explanation(
                values=shap_values_sorted[:n_samples_heatmap, :n_features_heatmap],
                data=X_test_sorted.iloc[:n_samples_heatmap, :n_features_heatmap].values,
                feature_names=sorted_features[:n_features_heatmap]
            )

            heatmap_save_path = plots_shap_dir / "shap_heatmap"
            plot_shap_heatmap(
                heatmap_explanation,
                save_path=heatmap_save_path,
                show_plot=False,
                title=f"SHAP heatmap - top {n_features_heatmap}",
                figsize=tuple(plot_config.get("shap_heatmap_figsize", (8.3, 6.0))),
                plot_config=plot_config,
            )
            results["shap_heatmap_plot_path"] = str(heatmap_save_path.with_suffix('.png'))
            logger.info("SHAP Analyzer: Successfully generated SHAP heatmap.")

        except Exception as e:
            logger.error("SHAP Analyzer: Failed to generate SHAP heatmap.", exc_info=True)
            results["shap_heatmap_plot_path"] = None


        generate_no_depth_heatmap = plot_config.get("generate_no_depth_plots", True)

        if generate_no_depth_heatmap:
            try:
                logger.info("SHAP Analyzer: Generating SHAP heatmap without Normalized Depth...")


                if depth_feature_name in sorted_features:
                    depth_idx = sorted_features.index(depth_feature_name)


                    features_no_depth = sorted_features[:depth_idx] + sorted_features[depth_idx+1:]
                    shap_values_no_depth = np.delete(shap_values_sorted, depth_idx, axis=1)
                    X_test_no_depth_df = X_test_sorted.drop(columns=[depth_feature_name])


                    n_samples_heatmap_nd = min(plot_config.get("heatmap_samples", 20), shap_values_no_depth.shape[0])
                    n_features_heatmap_nd = min(plot_config.get("heatmap_features", 15), len(features_no_depth))


                    heatmap_explanation_no_depth = shap.Explanation(
                        values=shap_values_no_depth[:n_samples_heatmap_nd, :n_features_heatmap_nd],
                        data=X_test_no_depth_df.iloc[:n_samples_heatmap_nd, :n_features_heatmap_nd].values,
                        feature_names=features_no_depth[:n_features_heatmap_nd]
                    )


                    heatmap_save_path_no_depth = plots_shap_dir / "shap_heatmap_no_depth"
                    plot_shap_heatmap(
                        heatmap_explanation_no_depth,
                        save_path=heatmap_save_path_no_depth,
                        show_plot=False,
                        title=f"SHAP heatmap (no depth) - top {n_features_heatmap_nd}",
                        figsize=tuple(plot_config.get("shap_heatmap_figsize", (8.3, 6.0))),
                        plot_config=plot_config,
                    )
                    results["shap_heatmap_no_depth_plot_path"] = str(heatmap_save_path_no_depth.with_suffix('.png'))
                    logger.info("SHAP Analyzer: Successfully generated SHAP heatmap without depth.")
                else:
                    logger.warning(f"SHAP Analyzer: '{depth_feature_name}' not found in features, skipping no-depth heatmap.")
                    results["shap_heatmap_no_depth_plot_path"] = None

            except Exception as e:
                logger.error("SHAP Analyzer: Failed to generate SHAP heatmap without depth.", exc_info=True)
                results["shap_heatmap_no_depth_plot_path"] = None


        try:
            logger.info("SHAP Analyzer: Generating feature clustering dendrogram...")
            clustering_save_path = plots_shap_dir / "shap_feature_clustering"

            plot_feature_clustering(
                shap_importance=global_importance[sorted_indices],
                feature_names=sorted_features,
                save_path=clustering_save_path,
                show_plot=False,
                title="Feature clustering (SHAP)",
                figsize=(8.3, 6.0),
                plot_config=plot_config,
            )
            results["feature_clustering_plot_path"] = str(clustering_save_path.with_suffix('.png'))
            logger.info("SHAP Analyzer: Successfully generated feature clustering dendrogram.")

        except Exception as e:
            logger.error("SHAP Analyzer: Failed to generate feature clustering dendrogram.", exc_info=True)
            results["feature_clustering_plot_path"] = None


        try:
            logger.info("SHAP Analyzer: Generating multi-sample SHAP decision plot...")


            n_samples_multi_decision = min(plot_config.get("multi_decision_plot_samples", 50), shap_values_sorted.shape[0])
            n_features_multi_decision = min(plot_config.get("multi_decision_plot_features", 10), len(sorted_features))

            multi_decision_plot_save_path = plots_shap_dir / "shap_multi_sample_decision_plot"

            plot_multi_sample_shap_decision(
                base_value=expected_value_scalar,
                shap_values=shap_values_sorted[:n_samples_multi_decision, :n_features_multi_decision],
                features=X_test_sorted.iloc[:n_samples_multi_decision, :n_features_multi_decision],
                feature_names=sorted_features[:n_features_multi_decision],
                save_path=multi_decision_plot_save_path,
                show_plot=False,
                title=f"SHAP decision plot (multi-sample) - top {n_features_multi_decision}",
                figsize=(8.3, 6.0),
                plot_config=plot_config,
            )

            results["shap_multi_decision_plot_path"] = str(multi_decision_plot_save_path.with_suffix('.png'))
            logger.info("SHAP Analyzer: Successfully generated multi-sample SHAP decision plot.")

        except Exception as e:
            logger.error("SHAP Analyzer: Failed to generate multi-sample SHAP decision plot.", exc_info=True)
            results["shap_multi_decision_plot_path"] = None


        try:
            logger.info("SHAP Analyzer: Generating SHAP decision plot without Normalized Depth...")


            if depth_feature_name in sorted_features:
                depth_idx = sorted_features.index(depth_feature_name)


                features_no_depth = sorted_features[:depth_idx] + sorted_features[depth_idx+1:]
                shap_values_no_depth = np.delete(shap_values_sorted, depth_idx, axis=1)
                X_test_no_depth_df = X_test_sorted.drop(columns=[depth_feature_name])


                n_samples_decision_nd = min(plot_config.get("decision_plot_samples", 50), shap_values_no_depth.shape[0])
                n_features_decision_nd = min(plot_config.get("decision_plot_features", 10), len(features_no_depth))


                decision_plot_save_path_no_depth = plots_shap_dir / "shap_decision_plot_no_depth"
                plot_shap_decision(
                    base_value=expected_value_scalar,
                    shap_values=shap_values_no_depth[:n_samples_decision_nd, :n_features_decision_nd],
                    features=X_test_no_depth_df.iloc[:n_samples_decision_nd, :n_features_decision_nd],
                    feature_names=features_no_depth[:n_features_decision_nd],
                    save_path=decision_plot_save_path_no_depth,
                    show_plot=False,
                    title=f"SHAP decision plot (no depth) - top {n_features_decision_nd}",
                    figsize=(8.3, 6.0),
                    plot_config=plot_config,
                )
                results["shap_decision_plot_no_depth_path"] = str(decision_plot_save_path_no_depth.with_suffix('.png'))
                logger.info("SHAP Analyzer: Successfully generated SHAP decision plot without depth.")
            else:
                logger.warning(f"SHAP Analyzer: '{depth_feature_name}' not found in features, skipping no-depth decision plot.")
                results["shap_decision_plot_no_depth_path"] = None

        except Exception as e:
            logger.error("SHAP Analyzer: Failed to generate SHAP decision plot without depth.", exc_info=True)
            results["shap_decision_plot_no_depth_path"] = None


        logger.info("SHAP Analyzer: Skipping SHAP force plots for now due to unified input complexity.")
        results["shap_force_plot_paths"] = []

    except Exception as e:
        logger.error(f"SHAP Analyzer: An unexpected error occurred during SHAP analysis: {e}", exc_info=True)
        results["error_message_shap"] = str(e)

        results.setdefault("shap_summary_plot_path", None)
        results.setdefault("shap_decision_plot_path", None)
        results.setdefault("feature_importance_csv_path", None)
        results.setdefault("shap_heatmap_plot_path", None)
        results.setdefault("shap_heatmap_no_depth_plot_path", None)
        results.setdefault("feature_clustering_plot_path", None)
        results.setdefault("shap_multi_decision_plot_path", None)
        results.setdefault("shap_decision_plot_no_depth_path", None)
        results.setdefault("shap_force_plot_paths", [])

        results.setdefault("shap_interaction_heatmap_path", None)
        results.setdefault("pdp_ice_like_plot_paths", [])
        results.setdefault("detailed_feature_importance_plot_path", None)


    results.setdefault("shap_summary_grouped_bar_path", None)
    results.setdefault("shap_decision_plot_grouped_path", None)
    results.setdefault("shap_multi_decision_plot_grouped_path", None)
    results.setdefault("feature_clustering_grouped_plot_path", None)
    results.setdefault("feature_groups_json_path", None)

    logger.info("SHAP Analyzer: SHAP analysis finished.")
    return results

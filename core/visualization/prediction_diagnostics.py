import os
import json
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from core.utils.logger_config import logger


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _vertical_axis_label(vertical_dim: str) -> Tuple[str, str]:
    if vertical_dim in ("depth_mbsf", "depth"):
        return "Depth (mbsf)", "m"
    if vertical_dim == "age_ka":
        return "Age", "ka"
    return vertical_dim, ""


def compute_prediction_level_diagnostics(
    predicted_ds,
    *,
    vertical_dim: str,
    pred_var_name: str = "predicted_rey",
    pred_std_var_name: str = "predicted_rey_std",
) -> Dict:
    levels = np.asarray(predicted_ds[vertical_dim].values, dtype=float)
    pred = np.asarray(predicted_ds[pred_var_name].values)
    valid = np.isfinite(pred)

    in_domain = None
    if "is_in_training_age_domain" in predicted_ds:
        in_domain = np.asarray(predicted_ds["is_in_training_age_domain"].values) > 0

    pred_std = None
    if pred_std_var_name in predicted_ds:
        pred_std = np.asarray(predicted_ds[pred_std_var_name].values)

    rows = []
    for i in range(levels.shape[0]):
        m = valid[i]
        n_valid = int(np.sum(m))
        if n_valid == 0:
            rows.append(
                {
                    "level": float(levels[i]),
                    "n_valid": 0,
                    "pred_p50": None,
                    "pred_p05": None,
                    "pred_p95": None,
                    "pred_mean": None,
                    "pred_std": None,
                    "uncertainty_mean": None,
                    "in_domain_fraction": None,
                }
            )
            continue

        p = pred[i][m]
        row = {
            "level": float(levels[i]),
            "n_valid": n_valid,
            "pred_p50": float(np.nanpercentile(p, 50)),
            "pred_p05": float(np.nanpercentile(p, 5)),
            "pred_p95": float(np.nanpercentile(p, 95)),
            "pred_mean": float(np.nanmean(p)),
            "pred_std": float(np.nanstd(p)),
        }

        if pred_std is not None:
            u = pred_std[i][m]
            row["uncertainty_mean"] = float(np.nanmean(u))

        if in_domain is not None:
            row["in_domain_fraction"] = float(np.sum(in_domain[i] & m) / max(1, n_valid))

        rows.append(row)

    axis_name, axis_unit = _vertical_axis_label(vertical_dim)
    meta = {
        "vertical_dim": vertical_dim,
        "vertical_axis_name": axis_name,
        "vertical_axis_unit": axis_unit,
        "training_age_domain_ka": float(predicted_ds.attrs.get("training_age_domain_ka")) if "training_age_domain_ka" in predicted_ds.attrs else None,
        "depth_feature_type_used": predicted_ds.attrs.get("depth_feature_type_used"),
        "vertical_axis_mode": predicted_ds.attrs.get("vertical_axis_mode"),
    }
    return {"_meta": meta, "levels": rows}


def save_prediction_level_diagnostics(
    predicted_ds,
    *,
    vertical_dim: str,
    output_dir: str,
    pred_var_name: str = "predicted_rey",
    pred_std_var_name: str = "predicted_rey_std",
) -> str:
    _ensure_dir(output_dir)
    out = compute_prediction_level_diagnostics(
        predicted_ds,
        vertical_dim=vertical_dim,
        pred_var_name=pred_var_name,
        pred_std_var_name=pred_std_var_name,
    )
    path = os.path.join(output_dir, "prediction_level_diagnostics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    logger.info(f"Prediction diagnostics: wrote {path}")
    return path


def plot_domain_fraction_by_level(
    predicted_ds,
    *,
    vertical_dim: str,
    output_dir: str,
    pred_var_name: str = "predicted_rey",
) -> Optional[str]:
    if "is_in_training_age_domain" not in predicted_ds:
        return None

    _ensure_dir(output_dir)
    levels = np.asarray(predicted_ds[vertical_dim].values, dtype=float)
    pred = np.asarray(predicted_ds[pred_var_name].values)
    valid = np.isfinite(pred)
    in_domain = np.asarray(predicted_ds["is_in_training_age_domain"].values) > 0

    frac = []
    n_valid = []
    for i in range(levels.shape[0]):
        m = valid[i]
        n = int(np.sum(m))
        n_valid.append(n)
        if n == 0:
            frac.append(np.nan)
        else:
            frac.append(float(np.sum(in_domain[i] & m) / n))

    axis_name, axis_unit = _vertical_axis_label(vertical_dim)
    plt.figure(figsize=(10, 4))
    plt.plot(levels, frac, marker="o", linewidth=1.5)
    plt.ylim(0, 1.0)
    plt.xlabel(f"{axis_name} ({axis_unit})".strip())
    plt.ylabel("In-domain fraction")
    plt.title("Prediction domain coverage by level")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "domain_fraction_by_level.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    logger.info(f"Prediction diagnostics: saved {out_path}")
    return out_path


def plot_vertical_transform_summary(predicted_ds, *, vertical_dim: str, output_dir: str) -> Optional[str]:
    """
    Summarize how depth<->age mapping varies spatially for each requested level.
    - If predicting by depth: show age_ka_proxy distribution by depth.
    - If predicting by age: show depth_mbsf distribution by age.
    """
    _ensure_dir(output_dir)
    axis_name, axis_unit = _vertical_axis_label(vertical_dim)
    levels = np.asarray(predicted_ds[vertical_dim].values, dtype=float)

    if vertical_dim in ("depth_mbsf", "depth") and "age_ka_proxy" in predicted_ds:
        y = np.asarray(predicted_ds["age_ka_proxy"].values)
        y_label = "Age proxy (ka)"
        fname = "age_proxy_distribution_by_depth.png"
        x_label = f"{axis_name} ({axis_unit})".strip()
    elif vertical_dim == "age_ka" and "depth_mbsf" in predicted_ds:
        y = np.asarray(predicted_ds["depth_mbsf"].values)
        y_label = "Depth (mbsf, m)"
        fname = "depth_distribution_by_age.png"
        x_label = f"{axis_name} ({axis_unit})".strip()
    else:
        return None

    p05 = []
    p50 = []
    p95 = []
    for i in range(levels.shape[0]):
        yy = y[i]
        yy = yy[np.isfinite(yy)]
        if yy.size == 0:
            p05.append(np.nan)
            p50.append(np.nan)
            p95.append(np.nan)
        else:
            p05.append(float(np.nanpercentile(yy, 5)))
            p50.append(float(np.nanpercentile(yy, 50)))
            p95.append(float(np.nanpercentile(yy, 95)))

    plt.figure(figsize=(10, 4))
    plt.plot(levels, p50, color="C0", linewidth=1.8, label="p50")
    plt.fill_between(levels, p05, p95, color="C0", alpha=0.25, label="p05–p95")
    if "training_age_domain_ka" in predicted_ds.attrs and vertical_dim in ("depth_mbsf", "depth"):
        try:
            t = float(predicted_ds.attrs["training_age_domain_ka"])
            plt.axhline(t, color="C3", linestyle="--", linewidth=1.5, label="Training age domain")
        except Exception:
            pass
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title("Vertical mapping variability across the grid")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    out_path = os.path.join(output_dir, fname)
    plt.savefig(out_path, dpi=200)
    plt.close()
    logger.info(f"Prediction diagnostics: saved {out_path}")
    return out_path

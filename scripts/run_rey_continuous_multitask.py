from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xarray as xr
from core.utils.netcdf_io import open_dataset
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, WeightedRandomSampler

from core.datasets.rey_dataset import REYPointDataset
from core.models.rey_multitask import (
    MechanismAwareREYRegressor,
    ordinal_monotonic_penalty,
    student_t_nll,
)
from core.training.pipeline import _stratified_group_train_val_test_split
from core.utils.logger_config import setup_logger
from core.utils.serialization_utils import NumpyJSONEncoder


MECHANISM3_CLASS_NAMES = [
    "carbonate_background",
    "class1_enriched",
    "class2_slow_sedimentation",
]


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, cls=NumpyJSONEncoder, indent=2, ensure_ascii=False)


def _feature_names(env_nc: Path, explicit: str | None = None) -> list[str]:
    if explicit:
        names = [part.strip() for part in str(explicit).split(",") if part.strip()]
        if names:
            return names
    with open_dataset(env_nc) as ds:
        return [str(name) for name in ds.data_vars if str(name) != "land_mask"]


def _prepare_points(points_path: Path, output_dir: Path) -> Path:
    df = pd.read_parquet(points_path).copy()
    if "rey" not in df.columns:
        raise KeyError("Point table must contain continuous REY column 'rey'.")
    if "mechanism_class3" not in df.columns:
        if "mech_class_id" in df.columns:
            mech = pd.to_numeric(df["mech_class_id"], errors="coerce")
            df["mechanism_class3"] = mech.clip(lower=0, upper=2).astype("Int64")
        elif "exploration_class_id" in df.columns:
            exp = pd.to_numeric(df["exploration_class_id"], errors="coerce")
            mech = np.where(exp == 0, 0, np.where(exp == 1, 1, 2))
            df["mechanism_class3"] = pd.Series(mech, index=df.index).astype("Int64")
        else:
            raise KeyError("Point table requires 'mech_class_id' or 'exploration_class_id' to build mechanism_class3.")
    if df["mechanism_class3"].isna().any():
        raise ValueError("mechanism_class3 contains missing labels after preparation.")
    df["mechanism_class3"] = pd.to_numeric(df["mechanism_class3"], errors="coerce").astype(int)
    out = output_dir / "inputs" / "rey_points_mechanism3.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


class REYMultitaskDataset(Dataset):
    def __init__(self, base: REYPointDataset, thresholds: list[float]) -> None:
        self.base = base
        self.thresholds = np.asarray([float(x) for x in thresholds], dtype=np.float32)
        if "mechanism_class3" not in self.base.rey_data.columns:
            raise KeyError("base.rey_data must contain mechanism_class3.")
        core_max = self.base.rey_data.groupby("core_id", dropna=False)["rey"].transform("max")
        core_target = self.base._transform_rey_target(core_max.to_numpy(dtype=float))
        self.core_peak_norm = ((core_target - float(self.base.rey_mean)) / float(self.base.rey_std)).astype(np.float32)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        item = self.base[idx]
        patch, y_norm, sample_weight = item[:3]
        row = self.base.rey_data.iloc[int(idx)]
        raw_rey = float(row["rey"])
        mechanism = int(row["mechanism_class3"])
        threshold_labels = (self.thresholds <= raw_rey).astype(np.float32)
        return (
            patch,
            y_norm.float().view(1),
            sample_weight.float().view(1),
            torch.tensor(mechanism, dtype=torch.long),
            torch.tensor(threshold_labels, dtype=torch.float32),
            torch.tensor(raw_rey, dtype=torch.float32),
            torch.tensor(float(self.core_peak_norm[int(idx)]), dtype=torch.float32),
            torch.tensor(int(idx), dtype=torch.long),
        )


def _make_base_dataset(
    *,
    points_path: Path,
    env_nc: Path,
    feature_names: list[str],
    patch_size: int,
    sed_rate_nc: Path | None,
    depth_feature_type: str,
    target_transform_name: str = "log1p",
    target_transform_scale: float = 1000.0,
    target_transform_hinge_beta: float = 0.25,
    norm_params_override: dict[str, Any] | None,
    use_target_weights: bool,
    max_weight: float,
    augment: bool,
    shift_pixels: int,
    add_coord_channels: bool = False,
) -> REYPointDataset:
    return REYPointDataset(
        str(points_path),
        str(env_nc),
        feature_names,
        patch_size=(int(patch_size), int(patch_size)),
        task_type="regression",
        rey_target_transform_name=str(target_transform_name),
        rey_target_transform_scale=float(target_transform_scale),
        rey_target_transform_hinge_beta=float(target_transform_hinge_beta),
        sed_rate_nc_path=str(sed_rate_nc) if sed_rate_nc else None,
        depth_feature_type_override=str(depth_feature_type),
        norm_params_override=norm_params_override,
        use_weighted_loss=bool(use_target_weights),
        weighting_strategy="log",
        max_weight=float(max_weight),
        augment_data=bool(augment),
        shift_augmentation_pixels=int(shift_pixels),
        add_coord_channels=bool(add_coord_channels),
    )


def _build_split(
    df: pd.DataFrame,
    *,
    group_col: str,
    seed: int,
    test_ratio: float,
    val_ratio: float,
    n_bins: int,
) -> tuple[list[int], list[int], list[int]]:
    if group_col not in df.columns:
        raise KeyError(f"Missing group column '{group_col}' for grouped split.")
    groups = df[group_col].astype(str).fillna("missing").to_numpy()
    targets = np.log1p(pd.to_numeric(df["rey"], errors="coerce").fillna(0.0).to_numpy(dtype=float))
    return _stratified_group_train_val_test_split(
        dataset_size=len(df),
        groups=groups,
        targets=targets,
        test_split_ratio=float(test_ratio),
        validation_split_ratio=float(val_ratio),
        random_state=int(seed),
        n_bins=int(n_bins),
        statistic="max",
    )


def _save_split(output_dir: Path, df: pd.DataFrame, train_idx: list[int], val_idx: list[int], test_idx: list[int]) -> None:
    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    np.save(split_dir / "train_indices.npy", np.asarray(train_idx, dtype=int))
    np.save(split_dir / "val_indices.npy", np.asarray(val_idx, dtype=int))
    np.save(split_dir / "test_indices.npy", np.asarray(test_idx, dtype=int))
    rows = []
    for split, indices in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        for idx in indices:
            row = {"dataset_index": int(idx), "split": split}
            for col in ("core_id", "site_group", "mechanism_class3", "exploration_class_id", "rey"):
                if col in df.columns:
                    row[col] = df.iloc[int(idx)][col]
            rows.append(row)
    pd.DataFrame(rows).to_csv(split_dir / "split_assignments.csv", index=False)


def _load_split_indices(path: Path, dataset_size: int) -> tuple[list[int], list[int], list[int]]:
    required = {
        "train": path / "train_indices.npy",
        "val": path / "val_indices.npy",
        "test": path / "test_indices.npy",
    }
    missing = [str(file) for file in required.values() if not file.exists()]
    if missing:
        raise FileNotFoundError(f"Missing external split index files: {missing}")
    loaded = {
        name: np.load(file).astype(int).tolist()
        for name, file in required.items()
    }
    all_indices: list[int] = []
    for split_name, indices in loaded.items():
        bad = [idx for idx in indices if idx < 0 or idx >= int(dataset_size)]
        if bad:
            raise ValueError(f"External {split_name} split contains out-of-range indices: {bad[:10]}")
        all_indices.extend(indices)
    counts = pd.Series(all_indices).value_counts()
    duplicated = counts[counts > 1]
    if not duplicated.empty:
        raise ValueError(f"External split indices overlap across train/val/test: {duplicated.index[:10].tolist()}")
    return loaded["train"], loaded["val"], loaded["test"]


def _build_train_sampler(
    dataset: REYMultitaskDataset,
    train_idx: list[int],
    args: argparse.Namespace,
) -> tuple[SubsetRandomSampler | WeightedRandomSampler, dict[str, Any]]:
    train_indices = [int(idx) for idx in train_idx]
    core_power = float(args.core_balanced_sampler_power)
    high_factor = float(args.high_rey_sampler_weight)
    use_weighted = core_power > 0.0 or high_factor > 1.0
    if not use_weighted:
        return SubsetRandomSampler(train_indices), {
            "type": "subset_random",
            "n_train": int(len(train_indices)),
        }

    weights = np.zeros(len(dataset), dtype=np.float64)
    train_df = dataset.base.rey_data.iloc[train_indices].copy()
    base_weights = pd.Series(1.0, index=train_df.index, dtype=float)

    core_summary: dict[str, Any] = {}
    if core_power > 0.0:
        group_col = str(args.group_col)
        if group_col not in train_df.columns:
            raise KeyError(f"Cannot use core-balanced sampler because group column '{group_col}' is missing.")
        group_values = train_df[group_col].astype(str).fillna("missing")
        counts = group_values.value_counts()
        median_count = float(counts.median()) if len(counts) else 1.0
        factors = (median_count / group_values.map(counts).astype(float)).pow(core_power)
        factors = factors.clip(
            lower=1.0 / max(float(args.core_balanced_sampler_max_factor), 1.0),
            upper=max(float(args.core_balanced_sampler_max_factor), 1.0),
        )
        base_weights *= factors.to_numpy(dtype=float)
        core_summary = {
            "group_col": group_col,
            "power": core_power,
            "max_factor": float(args.core_balanced_sampler_max_factor),
            "n_groups": int(counts.size),
            "median_group_samples": median_count,
        }

    high_summary: dict[str, Any] = {}
    if high_factor > 1.0:
        rey = pd.to_numeric(train_df["rey"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        high_mask = rey >= float(args.high_rey_sampler_threshold)
        base_weights *= np.where(high_mask, high_factor, 1.0)
        high_summary = {
            "threshold": float(args.high_rey_sampler_threshold),
            "factor": high_factor,
            "n_high_samples": int(high_mask.sum()),
        }

    mean_weight = float(np.mean(base_weights.to_numpy(dtype=float))) if len(base_weights) else 1.0
    if mean_weight > 0:
        base_weights = base_weights / mean_weight
    for dataset_index, weight in zip(train_indices, base_weights.to_numpy(dtype=float), strict=False):
        weights[int(dataset_index)] = max(float(weight), 1e-8)

    num_samples = max(1, int(round(len(train_indices) * float(args.sampler_num_samples_multiplier))))
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=num_samples,
        replacement=True,
    )
    summary = {
        "type": "weighted_random",
        "n_train": int(len(train_indices)),
        "num_samples": int(num_samples),
        "weight_min": float(np.min(base_weights.to_numpy(dtype=float))) if len(base_weights) else float("nan"),
        "weight_max": float(np.max(base_weights.to_numpy(dtype=float))) if len(base_weights) else float("nan"),
        "weight_mean": float(np.mean(base_weights.to_numpy(dtype=float))) if len(base_weights) else float("nan"),
        "core_balanced": core_summary,
        "high_rey": high_summary,
    }
    return sampler, summary


def _safe_corr(x: np.ndarray, y: np.ndarray, *, rank: bool = False) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    xs = pd.Series(x).rank().to_numpy(dtype=float) if rank else np.asarray(x, dtype=float)
    ys = pd.Series(y).rank().to_numpy(dtype=float) if rank else np.asarray(y, dtype=float)
    if np.nanstd(xs) < 1e-12 or np.nanstd(ys) < 1e-12:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def _regression_metrics(true_rey: np.ndarray, pred_rey: np.ndarray) -> dict[str, float]:
    true_rey = np.asarray(true_rey, dtype=float)
    pred_rey = np.asarray(pred_rey, dtype=float)
    finite = np.isfinite(true_rey) & np.isfinite(pred_rey)
    true_rey = true_rey[finite]
    pred_rey = pred_rey[finite]
    log_true = np.log1p(np.clip(true_rey, 0.0, None))
    log_pred = np.log1p(np.clip(pred_rey, 0.0, None))
    out = {
        "n": int(true_rey.size),
        "log_mae": float(mean_absolute_error(log_true, log_pred)) if true_rey.size else float("nan"),
        "log_rmse": float(math.sqrt(mean_squared_error(log_true, log_pred))) if true_rey.size else float("nan"),
        "mae_rey": float(mean_absolute_error(true_rey, pred_rey)) if true_rey.size else float("nan"),
        "rmse_rey": float(math.sqrt(mean_squared_error(true_rey, pred_rey))) if true_rey.size else float("nan"),
        "spearman_log": _safe_corr(log_true, log_pred, rank=True),
        "pearson_log": _safe_corr(log_true, log_pred, rank=False),
    }
    out["r2_log"] = float(r2_score(log_true, log_pred)) if true_rey.size >= 2 else float("nan")
    return out


def _transform_rey_values(values: np.ndarray, *, transform: str, scale: float, hinge_beta: float = 0.25) -> np.ndarray:
    arr = np.clip(np.asarray(values, dtype=float), 0.0, None)
    if transform == "log1p":
        return np.log1p(arr)
    if transform == "asinh":
        return np.arcsinh(arr / max(float(scale), 1e-6))
    if transform == "hinged_log":
        threshold = max(float(scale), 1e-6)
        beta = max(float(hinge_beta), 0.0)
        base = np.log1p(arr)
        hinge_start = np.log1p(threshold)
        return base + beta * np.maximum(0.0, base - hinge_start)
    raise ValueError(f"Unsupported REY target transform: {transform}")


def _normalize_target_value(raw_rey: float, norm_params: dict[str, Any]) -> float:
    rey_params = norm_params.get("rey", {}) if isinstance(norm_params, dict) else {}
    transformed = _transform_rey_values(
        np.asarray([float(raw_rey)], dtype=float),
        transform=str(rey_params.get("transform", "log1p")),
        scale=float(rey_params.get("scale", 1000.0)),
        hinge_beta=float(rey_params.get("hinge_beta", 0.25)),
    )[0]
    mean = float(rey_params.get("mean", 0.0))
    std = float(rey_params.get("std", 1.0))
    if std < 1e-6:
        std = 1.0
    return float((transformed - mean) / std)


def _normalize_target_edges(raw_edges: str, norm_params: dict[str, Any]) -> list[float]:
    edges = [float(part.strip()) for part in str(raw_edges).split(",") if part.strip()]
    return [_normalize_target_value(edge, norm_params) for edge in edges]


def _regression_loss(
    y_true: torch.Tensor,
    out: dict[str, torch.Tensor],
    *,
    loss_name: str,
    student_t_nu: float,
    huber_delta: float,
    mse_weight: float,
    asymmetric_high_threshold_norm: float,
    asymmetric_low_threshold_norm: float,
    asymmetric_under_weight: float,
    asymmetric_over_weight: float,
    bin_balance_edges_norm: list[float] | None = None,
    bin_balance_max_weight: float = 2.5,
    student_t_anchor_weight: float = 0.0,
) -> torch.Tensor:
    y = y_true.view_as(out["mu"])
    mu = out["mu"]
    name = str(loss_name)
    err = mu - y
    if name == "student_t":
        return student_t_nll(y, mu, out["log_sigma"], nu=float(student_t_nu))
    if name == "student_t_mae":
        base = student_t_nll(y, mu, out["log_sigma"], nu=float(student_t_nu))
        anchor = torch.abs(err).squeeze(1)
        return base + float(student_t_anchor_weight) * anchor
    if name == "huber":
        return F.huber_loss(mu, y, delta=float(huber_delta), reduction="none").squeeze(1)
    if name == "log_cosh":
        return (err + F.softplus(-2.0 * err) - math.log(2.0)).squeeze(1)
    if name == "mae_mse_blend":
        err = torch.abs(mu - y)
        mse = (mu - y) ** 2
        return ((1.0 - float(mse_weight)) * err + float(mse_weight) * mse).squeeze(1)
    if name in ("asymmetric_huber", "balanced_asymmetric_huber"):
        raw = F.huber_loss(mu, y, delta=float(huber_delta), reduction="none").squeeze(1)
        y_flat = y.view(-1)
        err_flat = err.view(-1)
        asym_weight = torch.ones_like(raw)
        high_under = (y_flat >= float(asymmetric_high_threshold_norm)) & (err_flat < 0.0)
        low_over = (y_flat <= float(asymmetric_low_threshold_norm)) & (err_flat > 0.0)
        asym_weight = torch.where(
            high_under,
            torch.full_like(asym_weight, float(asymmetric_under_weight)),
            asym_weight,
        )
        asym_weight = torch.where(
            low_over,
            torch.full_like(asym_weight, float(asymmetric_over_weight)),
            asym_weight,
        )
        if name == "balanced_asymmetric_huber" and bin_balance_edges_norm:
            edges = torch.as_tensor(bin_balance_edges_norm, dtype=y_flat.dtype, device=y_flat.device)
            bin_ids = torch.bucketize(y_flat.detach(), edges)
            bin_weight = torch.ones_like(raw)
            unique_bins, counts = torch.unique(bin_ids, return_counts=True)
            expected = float(y_flat.numel()) / max(float(unique_bins.numel()), 1.0)
            for bin_id, count in zip(unique_bins, counts, strict=False):
                factor = expected / max(float(count.item()), 1.0)
                factor = min(max(factor, 1.0 / max(float(bin_balance_max_weight), 1.0)), float(bin_balance_max_weight))
                bin_weight = torch.where(bin_ids == bin_id, torch.full_like(bin_weight, factor), bin_weight)
            asym_weight = asym_weight * bin_weight
        return raw * asym_weight
    raise ValueError(f"Unsupported regression loss: {loss_name}")


def _underprediction_loss(y_true: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    under = torch.relu(y_true.view_as(mu) - mu)
    return torch.mean(under * under) if under.numel() else mu.new_tensor(0.0)


def _high_value_underprediction_loss(
    y_true: torch.Tensor,
    mu: torch.Tensor,
    rey_raw: torch.Tensor,
    *,
    threshold: float,
) -> torch.Tensor:
    mask = rey_raw.view(-1) >= float(threshold)
    if not torch.any(mask):
        return mu.new_tensor(0.0)
    return _underprediction_loss(y_true.view(-1)[mask], mu.view(-1)[mask])


def _low_value_overprediction_loss(
    y_true: torch.Tensor,
    mu: torch.Tensor,
    rey_raw: torch.Tensor,
    *,
    threshold: float,
) -> torch.Tensor:
    mask = rey_raw.view(-1) <= float(threshold)
    if not torch.any(mask):
        return mu.new_tensor(0.0)
    over = torch.relu(mu.view(-1)[mask] - y_true.view(-1)[mask])
    return torch.mean(over * over) if over.numel() else mu.new_tensor(0.0)


def _tail_quantile_underprediction_loss(
    y_true: torch.Tensor,
    mu: torch.Tensor,
    rey_raw: torch.Tensor,
    *,
    quantile: float,
    min_samples: int,
) -> torch.Tensor:
    if int(rey_raw.numel()) < int(min_samples):
        return mu.new_tensor(0.0)
    cutoff = torch.quantile(rey_raw.view(-1).float(), float(quantile))
    mask = rey_raw.view(-1) >= cutoff
    if int(mask.sum().item()) < int(min_samples):
        return mu.new_tensor(0.0)
    return _underprediction_loss(y_true.view(-1)[mask], mu.view(-1)[mask])


def _batch_core_tail_underprediction_loss(
    y_true: torch.Tensor,
    mu: torch.Tensor,
    rey_raw: torch.Tensor,
    indices: torch.Tensor,
    core_codes: torch.Tensor,
    *,
    min_samples: int,
    high_threshold: float,
) -> torch.Tensor:
    if indices.numel() == 0:
        return mu.new_tensor(0.0)
    batch_codes = core_codes[indices.view(-1).long()].to(mu.device)
    y_flat = y_true.view(-1)
    mu_flat = mu.view(-1)
    rey_flat = rey_raw.view(-1)
    losses = []
    for code in torch.unique(batch_codes):
        mask = batch_codes == code
        if int(mask.sum().item()) < int(min_samples):
            continue
        if torch.max(rey_flat[mask]) < float(high_threshold):
            continue
        true_peak = torch.max(y_flat[mask])
        pred_peak = torch.max(mu_flat[mask])
        losses.append(torch.relu(true_peak - pred_peak) ** 2)
    if not losses:
        return mu.new_tensor(0.0)
    return torch.stack(losses).mean()


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    labels = [0, 1, 2]
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)) if y_true.size else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if np.unique(y_true).size > 1 else float("nan"),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if y_true.size else float("nan"),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist() if y_true.size else [],
    }
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    for idx, name in enumerate(MECHANISM3_CLASS_NAMES):
        out[f"{name}_precision"] = float(precision[idx])
        out[f"{name}_recall"] = float(recall[idx])
        out[f"{name}_f1"] = float(f1[idx])
        out[f"{name}_support"] = int(support[idx])
    return out


def _threshold_metrics(df: pd.DataFrame, thresholds: list[float]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for i, threshold in enumerate(thresholds):
        y_true = (df["rey_true"].to_numpy(dtype=float) >= float(threshold)).astype(int)
        p = df[f"prob_rey_ge_{int(threshold)}"].to_numpy(dtype=float)
        y_pred = (p >= 0.5).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[1], zero_division=0)
        out[f"rey_ge_{int(threshold)}_positive_support"] = int(y_true.sum())
        out[f"rey_ge_{int(threshold)}_negative_support"] = int((1 - y_true).sum())
        out[f"rey_ge_{int(threshold)}_ap"] = (
            float(average_precision_score(y_true, p)) if np.unique(y_true).size > 1 else float("nan")
        )
        out[f"rey_ge_{int(threshold)}_roc_auc"] = (
            float(roc_auc_score(y_true, p)) if np.unique(y_true).size > 1 else float("nan")
        )
        out[f"rey_ge_{int(threshold)}_brier"] = float(brier_score_loss(y_true, p)) if y_true.size else float("nan")
        out[f"rey_ge_{int(threshold)}_precision_at_0_5"] = float(precision[0])
        out[f"rey_ge_{int(threshold)}_recall_at_0_5"] = float(recall[0])
        out[f"rey_ge_{int(threshold)}_f1_at_0_5"] = float(f1[0])
    return out


def _best_binary_threshold(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    if y_true.size == 0 or np.unique(score).size <= 1:
        return float(score[0]) if score.size else 0.0
    best = None
    for thr in np.unique(score):
        pred = (score >= thr).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        acc = accuracy_score(y_true, pred)
        key = (f1, acc, float(thr))
        if best is None or key > best[0]:
            best = (key, float(thr))
    return float(best[1])


def _binary_metrics_at_threshold(y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    y_pred = (score >= float(threshold)).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[1], zero_division=0)
    return {
        "positive_support": int(y_true.sum()),
        "negative_support": int((1 - y_true).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)) if y_true.size else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if np.unique(y_true).size > 1 else float("nan"),
        "precision": float(precision[0]),
        "recall": float(recall[0]),
        "f1": float(f1[0]),
    }


def _sample_threshold_calibration_metrics(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    thresholds: list[float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rey_threshold in thresholds:
        score_col = f"prob_rey_ge_{int(rey_threshold)}"
        y_val = (val_df["rey_true"].to_numpy(dtype=float) >= float(rey_threshold)).astype(int)
        y_test = (test_df["rey_true"].to_numpy(dtype=float) >= float(rey_threshold)).astype(int)
        if y_val.size == 0 or np.unique(y_val).size < 2:
            rows.append(
                {
                    "rey_threshold": float(rey_threshold),
                    "score": score_col,
                    "status": "insufficient_validation_support",
                    "val_positive_samples": int(y_val.sum()),
                    "test_positive_samples": int(y_test.sum()),
                }
            )
            continue
        selected = _best_binary_threshold(y_val, val_df[score_col].to_numpy(dtype=float))
        for split_name, table, labels in (("val", val_df, y_val), ("test", test_df, y_test)):
            metrics = _binary_metrics_at_threshold(labels, table[score_col].to_numpy(dtype=float), selected)
            rows.append(
                {
                    "rey_threshold": float(rey_threshold),
                    "score": score_col,
                    "status": "completed",
                    "selected_probability_threshold": float(selected),
                    "split": split_name,
                    "n_samples": int(len(table)),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def _parse_float_list(value: str | None, *, expected: int, default: float) -> list[float]:
    if value is None or not str(value).strip():
        return [float(default)] * int(expected)
    parsed = [float(part.strip()) for part in str(value).split(",") if part.strip()]
    if len(parsed) == 1:
        return parsed * int(expected)
    if len(parsed) != int(expected):
        raise ValueError(f"Expected {expected} comma-separated values, got {len(parsed)}: {value!r}")
    return parsed


def _mechanism_class_weights(
    df: pd.DataFrame,
    indices: list[int],
    *,
    mode: str,
    n_classes: int = 3,
) -> torch.Tensor | None:
    if mode == "none":
        return None
    labels = pd.to_numeric(df.iloc[indices]["mechanism_class3"], errors="coerce").to_numpy(dtype=int)
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    counts = np.maximum(counts, 1.0)
    if mode == "inverse":
        weights = counts.sum() / (n_classes * counts)
    elif mode == "sqrt":
        weights = np.sqrt(counts.sum() / (n_classes * counts))
    else:
        raise ValueError(f"Unsupported mechanism class weighting mode: {mode}")
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


def _threshold_pos_weights(df: pd.DataFrame, indices: list[int], thresholds: list[float], *, cap: float) -> torch.Tensor:
    rey = pd.to_numeric(df.iloc[indices]["rey"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    weights = []
    for threshold in thresholds:
        pos = int(np.sum(rey >= float(threshold)))
        neg = int(rey.size - pos)
        if pos <= 0:
            weights.append(1.0)
        else:
            weights.append(float(min(float(cap), max(1.0, neg / pos))))
    return torch.tensor(weights, dtype=torch.float32)


def _pairwise_rank_loss(mu: torch.Tensor, y_true: torch.Tensor, *, margin: float) -> torch.Tensor:
    y = y_true.view(-1)
    pred = mu.view(-1)
    if y.numel() < 2:
        return pred.new_tensor(0.0)
    diff_y = y[:, None] - y[None, :]
    mask = torch.triu(torch.abs(diff_y) > float(margin), diagonal=1)
    if not torch.any(mask):
        return pred.new_tensor(0.0)
    sign = torch.sign(diff_y[mask])
    diff_pred = pred[:, None] - pred[None, :]
    return F.softplus(-sign * diff_pred[mask]).mean()


def _profile_pairwise_rank_loss(
    mu: torch.Tensor,
    y_true: torch.Tensor,
    indices: torch.Tensor,
    core_codes: torch.Tensor,
    *,
    margin: float,
    min_pairs: int,
) -> torch.Tensor:
    if indices.numel() < 2:
        return mu.new_tensor(0.0)
    batch_codes = core_codes[indices.view(-1).long()].to(mu.device)
    y = y_true.view(-1)
    pred = mu.view(-1)
    losses = []
    for code in torch.unique(batch_codes):
        mask = batch_codes == code
        if int(mask.sum().item()) < 2:
            continue
        yy = y[mask]
        pp = pred[mask]
        diff_y = yy[:, None] - yy[None, :]
        pair_mask = torch.triu(torch.abs(diff_y) > float(margin), diagonal=1)
        if int(pair_mask.sum().item()) < int(min_pairs):
            continue
        sign = torch.sign(diff_y[pair_mask])
        diff_pred = pp[:, None] - pp[None, :]
        losses.append(F.softplus(-sign * diff_pred[pair_mask]).mean())
    if not losses:
        return mu.new_tensor(0.0)
    return torch.stack(losses).mean()


def _core_cvar_loss(
    per_sample_loss: torch.Tensor,
    indices: torch.Tensor,
    core_codes: torch.Tensor,
    *,
    top_fraction: float,
    min_samples: int,
) -> torch.Tensor:
    if indices.numel() == 0:
        return per_sample_loss.new_tensor(0.0)
    batch_codes = core_codes[indices.view(-1).long()].to(per_sample_loss.device)
    core_losses = []
    for code in torch.unique(batch_codes):
        mask = batch_codes == code
        if int(mask.sum().item()) < int(min_samples):
            continue
        core_losses.append(torch.mean(per_sample_loss.view(-1)[mask]))
    if not core_losses:
        return per_sample_loss.new_tensor(0.0)
    values = torch.stack(core_losses)
    k = max(1, int(math.ceil(values.numel() * max(min(float(top_fraction), 1.0), 0.0))))
    return torch.topk(values, k=k, largest=True).values.mean()


def _focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    pos_weight: torch.Tensor | None,
    gamma: float,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
    if gamma <= 0:
        return bce
    prob = torch.sigmoid(logits)
    p_t = (prob * targets) + ((1.0 - prob) * (1.0 - targets))
    return bce * torch.pow(1.0 - p_t, float(gamma))


def _selection_score(metrics: dict[str, Any], *, thresholds: list[float], args: argparse.Namespace) -> float:
    if str(args.selection_metric) == "log_mae":
        return float(metrics["regression"]["log_mae"])
    main_threshold = 1000.0 if 1000.0 in [float(x) for x in thresholds] else float(thresholds[-1])
    threshold_key = f"rey_ge_{int(main_threshold)}_f1_at_0_5"
    log_mae = float(metrics["regression"]["log_mae"])
    spearman = float(metrics["regression"].get("spearman_log", 0.0) or 0.0)
    threshold_f1 = float(metrics["thresholds"].get(threshold_key, 0.0) or 0.0)
    mechanism_macro_f1 = float(metrics["mechanism3"].get("macro_f1", 0.0) or 0.0)
    class1_recall = float(metrics["mechanism3"].get("class1_enriched_recall", 0.0) or 0.0)
    return (
        log_mae
        - float(args.selection_spearman_weight) * spearman
        - float(args.selection_threshold_f1_weight) * threshold_f1
        - float(args.selection_mechanism_f1_weight) * mechanism_macro_f1
        - float(args.selection_class1_recall_weight) * class1_recall
    )


def _core_presence_metrics(val_df: pd.DataFrame, test_df: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rey_threshold in thresholds:
        for agg in ("max", "p95", "mean"):
            def group_table(df: pd.DataFrame) -> pd.DataFrame:
                grouped = []
                for core_id, sub in df.groupby("core_id", dropna=False):
                    pred = sub["rey_pred"].to_numpy(dtype=float)
                    true = sub["rey_true"].to_numpy(dtype=float)
                    if agg == "max":
                        score = float(np.max(pred))
                    elif agg == "p95":
                        score = float(np.percentile(pred, 95))
                    else:
                        score = float(np.mean(pred))
                    grouped.append(
                        {
                            "core_id": str(core_id),
                            "presence_label": int(np.any(true >= float(rey_threshold))),
                            "score": score,
                            "n_samples": int(len(sub)),
                        }
                    )
                return pd.DataFrame(grouped)

            gv = group_table(val_df)
            gt = group_table(test_df)
            if gv.empty or gt.empty or gv["presence_label"].nunique() < 2:
                rows.append(
                    {
                        "rey_threshold": float(rey_threshold),
                        "aggregator": agg,
                        "status": "insufficient_validation_support",
                        "val_n_cores": int(len(gv)),
                        "val_positive_cores": int(gv["presence_label"].sum()) if len(gv) else 0,
                        "test_n_cores": int(len(gt)),
                        "test_positive_cores": int(gt["presence_label"].sum()) if len(gt) else 0,
                    }
                )
                continue
            thr = _best_binary_threshold(gv["presence_label"].to_numpy(), gv["score"].to_numpy())
            val_score = gv["score"].to_numpy(dtype=float)
            test_score = gt["score"].to_numpy(dtype=float)
            val_y = gv["presence_label"].to_numpy(dtype=int)
            test_y = gt["presence_label"].to_numpy(dtype=int)
            val_ap = float(average_precision_score(val_y, val_score)) if np.unique(val_y).size > 1 else float("nan")
            test_ap = float(average_precision_score(test_y, test_score)) if np.unique(test_y).size > 1 else float("nan")
            val_auc = float(roc_auc_score(val_y, val_score)) if np.unique(val_y).size > 1 else float("nan")
            test_auc = float(roc_auc_score(test_y, test_score)) if np.unique(test_y).size > 1 else float("nan")
            for split_name, table in (("val", gv), ("test", gt)):
                y = table["presence_label"].to_numpy(dtype=int)
                pred = (table["score"].to_numpy(dtype=float) >= thr).astype(int)
                rows.append(
                    {
                        "rey_threshold": float(rey_threshold),
                        "aggregator": agg,
                        "status": "completed",
                        "selected_score_threshold": float(thr),
                        "split": split_name,
                        "n_cores": int(len(table)),
                        "positive_cores": int(y.sum()),
                        "ap": val_ap if split_name == "val" else test_ap,
                        "roc_auc": val_auc if split_name == "val" else test_auc,
                        "accuracy": float(accuracy_score(y, pred)) if len(y) else float("nan"),
                        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if np.unique(y).size > 1 else float("nan"),
                        "f1": float(f1_score(y, pred, zero_division=0)),
                    }
                )
    return pd.DataFrame(rows)


@torch.no_grad()
def _predict(
    model: MechanismAwareREYRegressor,
    loader: DataLoader,
    *,
    device: torch.device,
    dataset: REYPointDataset,
    thresholds: list[float],
    spatial_context_mode: str = "full",
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, Any]] = []
    mean = float(dataset.rey_mean)
    std = float(dataset.rey_std)
    for batch in loader:
        patch, y_norm, _weight, mech, _thr_labels, rey_raw, _core_peak_norm, indices = batch
        patch = _apply_spatial_context(patch.to(device), spatial_context_mode)
        out = model(patch)
        mu_tensor = torch.nan_to_num(out["mu"], nan=0.0, posinf=8.0, neginf=-8.0).clamp(-8.0, 8.0)
        sigma_tensor = torch.exp(torch.nan_to_num(out["log_sigma"], nan=0.0, posinf=3.0, neginf=-4.0).clamp(-4.0, 3.0))
        mechanism_logits = torch.nan_to_num(out["mechanism_logits"], nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        threshold_logits = torch.nan_to_num(out["threshold_logits"], nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        mu = mu_tensor.detach().cpu().view(-1).numpy()
        sigma = sigma_tensor.detach().cpu().view(-1).numpy()
        mech_prob = torch.softmax(mechanism_logits, dim=1).detach().cpu().numpy()
        thr_prob = torch.sigmoid(threshold_logits).detach().cpu().numpy()
        thr_prob = np.minimum.accumulate(thr_prob, axis=1)
        pred_transformed = np.nan_to_num((mu * std) + mean, nan=mean, posinf=11.5, neginf=0.0)
        if str(getattr(dataset, "rey_target_transform_name", "log1p")) in ("log1p", "hinged_log"):
            pred_transformed = np.clip(pred_transformed, 0.0, 11.5)
        pred_rey = dataset._inverse_transform_rey_target(pred_transformed)
        pred_rey = np.clip(pred_rey, 0.0, None)
        pred_log = np.log1p(pred_rey)
        for j, idx in enumerate(indices.detach().cpu().numpy().astype(int).tolist()):
            meta = dataset.rey_data.iloc[idx]
            row = {
                "dataset_index": int(idx),
                "core_id": meta.get("core_id", ""),
                "site_group": meta.get("site_group", ""),
                "mechanism_true": int(mech[j].item()),
                "mechanism_pred": int(np.argmax(mech_prob[j])),
                "rey_true": float(rey_raw[j].item()),
                "rey_pred": float(pred_rey[j]),
                "log_rey_pred": float(pred_log[j]),
                "pred_sigma_norm": float(sigma[j]),
            }
            for k, name in enumerate(MECHANISM3_CLASS_NAMES):
                row[f"prob_{name}"] = float(mech_prob[j, k])
            for k, threshold in enumerate(thresholds):
                row[f"prob_rey_ge_{int(threshold)}"] = float(thr_prob[j, k])
            rows.append(row)
    return pd.DataFrame(rows)


def _apply_spatial_context(patch: torch.Tensor, mode: str) -> torch.Tensor:
    """Keep tensor shape fixed while removing neighbourhood variation for OFAT ablation."""
    mode = str(mode or "full")
    if mode == "full":
        return patch
    if mode == "center_broadcast":
        center = patch[:, :, patch.shape[-2] // 2, patch.shape[-1] // 2]
        return center.unsqueeze(-1).unsqueeze(-1).expand_as(patch)
    raise ValueError(f"Unsupported spatial_context_mode: {mode}")


def _evaluate_predictions(df: pd.DataFrame, thresholds: list[float]) -> dict[str, Any]:
    return {
        "regression": _regression_metrics(df["rey_true"].to_numpy(), df["rey_pred"].to_numpy()),
        "mechanism3": _classification_metrics(
            df["mechanism_true"].to_numpy(dtype=int),
            df["mechanism_pred"].to_numpy(dtype=int),
        ),
        "thresholds": _threshold_metrics(df, thresholds),
    }


def _train_one(args: argparse.Namespace) -> dict[str, Any]:
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    points_path = _prepare_points(Path(args.points), output_dir)
    feature_names = _feature_names(Path(args.env_nc), args.feature_names)
    thresholds = [float(x) for x in str(args.thresholds).split(",") if str(x).strip()]

    sed_rate_nc = Path(args.sed_rate_nc) if args.sed_rate_nc else None
    probe_dataset = _make_base_dataset(
        points_path=points_path,
        env_nc=Path(args.env_nc),
        feature_names=feature_names,
        patch_size=int(args.patch_size),
        sed_rate_nc=sed_rate_nc,
        depth_feature_type=str(args.depth_feature_type),
        target_transform_name=str(args.target_transform),
        target_transform_scale=float(args.target_transform_scale),
        target_transform_hinge_beta=float(args.target_transform_hinge_beta),
        norm_params_override=None,
        use_target_weights=bool(args.use_target_weights),
        max_weight=float(args.max_weight),
        augment=False,
        shift_pixels=0,
        add_coord_channels=bool(args.add_coord_channels),
    )
    if args.split_indices_dir:
        train_idx, val_idx, test_idx = _load_split_indices(Path(args.split_indices_dir), len(probe_dataset.rey_data))
    else:
        train_idx, val_idx, test_idx = _build_split(
            probe_dataset.rey_data,
            group_col=str(args.group_col),
            seed=int(args.split_seed) if args.split_seed is not None else seed,
            test_ratio=float(args.test_split_ratio),
            val_ratio=float(args.validation_split_ratio),
            n_bins=int(args.group_split_stratify_bins),
        )
    _save_split(output_dir, probe_dataset.rey_data, train_idx, val_idx, test_idx)

    train_target = _transform_rey_values(
        probe_dataset.rey_data.iloc[train_idx]["rey"].astype(float).to_numpy(),
        transform=str(args.target_transform),
        scale=float(args.target_transform_scale),
        hinge_beta=float(args.target_transform_hinge_beta),
    )
    train_mean = float(np.mean(train_target))
    train_std = float(np.std(train_target))
    if train_std < 1e-6:
        train_std = 1.0
    norm_params = json.loads(json.dumps(probe_dataset.norm_params, cls=NumpyJSONEncoder))
    norm_params["rey"] = {
        "mean": train_mean,
        "std": train_std,
        "transform": str(args.target_transform),
        "scale": float(args.target_transform_scale),
        "hinge_beta": float(args.target_transform_hinge_beta),
        "task_type": "mechanism_aware_rey_regression",
    }

    base_dataset = _make_base_dataset(
        points_path=points_path,
        env_nc=Path(args.env_nc),
        feature_names=feature_names,
        patch_size=int(args.patch_size),
        sed_rate_nc=sed_rate_nc,
        depth_feature_type=str(args.depth_feature_type),
        target_transform_name=str(args.target_transform),
        target_transform_scale=float(args.target_transform_scale),
        target_transform_hinge_beta=float(args.target_transform_hinge_beta),
        norm_params_override=norm_params,
        use_target_weights=bool(args.use_target_weights),
        max_weight=float(args.max_weight),
        augment=bool(args.augment),
        shift_pixels=int(args.shift_pixels),
        add_coord_channels=bool(args.add_coord_channels),
    )
    dataset = REYMultitaskDataset(base_dataset, thresholds)

    batch_size = int(args.batch_size)
    num_workers = int(args.num_workers)
    train_sampler, sampler_summary = _build_train_sampler(dataset, train_idx, args)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        drop_last=True,
    )
    eval_batch = int(args.eval_batch_size or batch_size)
    val_loader = DataLoader(dataset, batch_size=eval_batch, sampler=SubsetRandomSampler(val_idx), num_workers=num_workers)
    test_loader = DataLoader(dataset, batch_size=eval_batch, sampler=SubsetRandomSampler(test_idx), num_workers=num_workers)
    train_eval_loader = DataLoader(dataset, batch_size=eval_batch, sampler=SubsetRandomSampler(train_idx), num_workers=num_workers)

    input_channels = int(base_dataset[0][0].shape[0])
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = MechanismAwareREYRegressor(
        input_channels=input_channels,
        n_mechanisms=3,
        n_thresholds=len(thresholds),
        cnn_filters=[int(x) for x in str(args.cnn_filters).split(",") if x.strip()],
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
        use_multiscale=not bool(args.no_multiscale),
        use_core_aux_head=float(args.core_peak_loss_weight) > 0.0,
        use_tail_residual_head=bool(args.use_tail_residual_head),
        n_domain_experts=int(args.n_domain_experts),
        regression_head_mode=str(args.regression_head_mode),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    mechanism_weight = _mechanism_class_weights(
        base_dataset.rey_data,
        train_idx,
        mode=str(args.mechanism_class_weighting),
        n_classes=3,
    )
    if mechanism_weight is not None:
        mechanism_weight = mechanism_weight.to(device)
    threshold_pos_weight = None
    if bool(args.use_threshold_pos_weight):
        threshold_pos_weight = _threshold_pos_weights(
            base_dataset.rey_data,
            train_idx,
            thresholds,
            cap=float(args.threshold_pos_weight_cap),
        ).to(device)
    threshold_loss_multipliers = torch.tensor(
        _parse_float_list(args.threshold_loss_multipliers, expected=len(thresholds), default=1.0),
        dtype=torch.float32,
        device=device,
    )
    core_code_values = pd.Categorical(base_dataset.rey_data[str(args.group_col)].astype(str)).codes.astype(int)
    core_codes = torch.tensor(core_code_values, dtype=torch.long, device=device)
    high_rey_threshold = float(args.high_rey_loss_threshold)
    high_rey_weight = float(args.high_rey_loss_weight)
    asymmetric_high_threshold_norm = _normalize_target_value(float(args.asymmetric_high_threshold), norm_params)
    asymmetric_low_threshold_norm = _normalize_target_value(float(args.asymmetric_low_threshold), norm_params)
    bin_balance_edges_norm = _normalize_target_edges(str(args.bin_balance_edges), norm_params)

    best_val = float("inf")
    best_val_log_mae = float("nan")
    best_epoch = -1
    epochs_without_improve = 0
    history: list[dict[str, Any]] = []
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    best_path = model_dir / "mechanism_aware_rey_regressor_best.pth"

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses = []
        for batch in train_loader:
            patch, y_norm, weight, mech, threshold_labels, rey_raw, core_peak_norm, indices = batch
            patch = _apply_spatial_context(patch.to(device), str(args.spatial_context_mode))
            y_norm = y_norm.to(device)
            weight = weight.to(device).view(-1)
            mech = mech.to(device)
            threshold_labels = threshold_labels.to(device)
            rey_raw = rey_raw.to(device)
            core_peak_norm = core_peak_norm.to(device).view(-1, 1)
            indices = indices.to(device)

            optimizer.zero_grad(set_to_none=True)
            out = model(patch)
            reg = _regression_loss(
                y_norm,
                out,
                loss_name=str(args.regression_loss),
                student_t_nu=float(args.student_t_nu),
                huber_delta=float(args.huber_delta),
                mse_weight=float(args.mae_mse_blend_mse_weight),
                asymmetric_high_threshold_norm=asymmetric_high_threshold_norm,
                asymmetric_low_threshold_norm=asymmetric_low_threshold_norm,
                asymmetric_under_weight=float(args.asymmetric_under_weight),
                asymmetric_over_weight=float(args.asymmetric_over_weight),
                bin_balance_edges_norm=bin_balance_edges_norm,
                bin_balance_max_weight=float(args.bin_balance_max_weight),
                student_t_anchor_weight=float(args.student_t_anchor_weight),
            )
            reg_weight = weight * torch.where(
                rey_raw >= high_rey_threshold,
                torch.full_like(weight, high_rey_weight),
                torch.ones_like(weight),
            )
            reg_weighted = reg * reg_weight
            thr_raw = _focal_bce_with_logits(
                out["threshold_logits"],
                threshold_labels,
                pos_weight=threshold_pos_weight,
                gamma=float(args.threshold_focal_gamma),
            )
            thr = torch.sum(thr_raw * threshold_loss_multipliers.view(1, -1), dim=1) / torch.sum(threshold_loss_multipliers)
            cls = F.cross_entropy(out["mechanism_logits"], mech, weight=mechanism_weight, reduction="none")
            cls_mean = torch.mean(cls * weight)
            mono = ordinal_monotonic_penalty(out["threshold_logits"])
            gate_mean = torch.softmax(out["mechanism_logits"], dim=1).mean(dim=0)
            gate_load = torch.mean((gate_mean - (1.0 / 3.0)) ** 2)
            if "domain_prob" in out:
                domain_mean = out["domain_prob"].mean(dim=0)
                domain_balance = torch.mean((domain_mean - (1.0 / float(out["domain_prob"].shape[1]))) ** 2)
                domain_entropy = -torch.mean(torch.sum(out["domain_prob"] * torch.log(out["domain_prob"].clamp_min(1e-8)), dim=1))
                domain_residual_l2 = torch.mean(out["domain_residuals"] ** 2)
            else:
                domain_balance = out["mu"].new_tensor(0.0)
                domain_entropy = out["mu"].new_tensor(0.0)
                domain_residual_l2 = out["mu"].new_tensor(0.0)
            rank = _pairwise_rank_loss(out["mu"], y_norm, margin=float(args.rank_loss_margin))
            profile_rank = _profile_pairwise_rank_loss(
                out["mu"],
                y_norm,
                indices,
                core_codes,
                margin=float(args.profile_rank_loss_margin),
                min_pairs=int(args.profile_rank_min_pairs),
            )
            core_cvar = _core_cvar_loss(
                reg_weighted.detach() if bool(args.core_cvar_detach_base) else reg_weighted,
                indices,
                core_codes,
                top_fraction=float(args.core_cvar_top_fraction),
                min_samples=int(args.core_cvar_min_samples),
            )
            high_under = _high_value_underprediction_loss(
                y_norm,
                out["mu"],
                rey_raw,
                threshold=float(args.high_underprediction_threshold),
            )
            low_over = _low_value_overprediction_loss(
                y_norm,
                out["mu"],
                rey_raw,
                threshold=float(args.low_overprediction_threshold),
            )
            tail_under = _tail_quantile_underprediction_loss(
                y_norm,
                out["mu"],
                rey_raw,
                quantile=float(args.tail_underprediction_quantile),
                min_samples=int(args.tail_underprediction_min_samples),
            )
            core_tail = _batch_core_tail_underprediction_loss(
                y_norm,
                out["mu"],
                rey_raw,
                indices,
                core_codes,
                min_samples=int(args.core_tail_min_samples),
                high_threshold=float(args.core_tail_threshold),
            )
            if "core_peak_mu" in out:
                core_peak_raw = F.huber_loss(
                    out["core_peak_mu"],
                    core_peak_norm,
                    delta=float(args.core_peak_huber_delta),
                    reduction="none",
                ).view(-1)
                core_peak_loss = torch.mean(core_peak_raw * weight)
            else:
                core_peak_loss = out["mu"].new_tensor(0.0)
            if "tail_gate_logit" in out:
                tail_labels = (rey_raw.view(-1, 1) >= float(args.tail_residual_threshold)).float()
                tail_gate_loss = F.binary_cross_entropy_with_logits(
                    out["tail_gate_logit"],
                    tail_labels,
                    reduction="none",
                ).view(-1)
                tail_gate_loss = torch.mean(tail_gate_loss * weight)
                tail_residual_l2 = torch.mean(out["tail_residual"].view(-1) ** 2)
            else:
                tail_gate_loss = out["mu"].new_tensor(0.0)
                tail_residual_l2 = out["mu"].new_tensor(0.0)
            mechanism_guard = torch.relu(cls_mean - torch.as_tensor(float(args.mechanism_guard_max_loss), device=device))
            loss = (
                torch.mean(reg_weighted)
                + float(args.threshold_loss_weight) * torch.mean(thr * weight)
                + float(args.mechanism_loss_weight) * cls_mean
                + float(args.rank_loss_weight) * rank
                + float(args.profile_rank_loss_weight) * profile_rank
                + float(args.core_cvar_loss_weight) * core_cvar
                + float(args.monotonic_loss_weight) * mono
                + float(args.gate_balance_weight) * gate_load
                + float(args.high_underprediction_loss_weight) * high_under
                + float(args.low_overprediction_loss_weight) * low_over
                + float(args.tail_underprediction_loss_weight) * tail_under
                + float(args.core_tail_loss_weight) * core_tail
                + float(args.core_peak_loss_weight) * core_peak_loss
                + float(args.mechanism_guard_weight) * mechanism_guard
                + float(args.tail_gate_loss_weight) * tail_gate_loss
                + float(args.tail_residual_l2_weight) * tail_residual_l2
                + float(args.domain_balance_weight) * domain_balance
                + float(args.domain_entropy_weight) * domain_entropy
                + float(args.domain_residual_l2_weight) * domain_residual_l2
            )
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            grads_finite = True
            for param in model.parameters():
                if param.grad is not None and not torch.all(torch.isfinite(param.grad)):
                    grads_finite = False
                    break
            if not grads_finite:
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.step()
            with torch.no_grad():
                for param in model.parameters():
                    param.data = torch.nan_to_num(param.data, nan=0.0, posinf=10.0, neginf=-10.0)
            losses.append(float(loss.detach().cpu()))

        val_pred = _predict(
            model,
            val_loader,
            device=device,
            dataset=base_dataset,
            thresholds=thresholds,
            spatial_context_mode=str(args.spatial_context_mode),
        )
        val_metrics = _evaluate_predictions(val_pred, thresholds)
        val_loss_key = _selection_score(val_metrics, thresholds=thresholds, args=args)
        row = {
            "epoch": int(epoch),
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "val_log_mae": val_metrics["regression"]["log_mae"],
            "val_log_rmse": val_metrics["regression"]["log_rmse"],
            "val_spearman_log": val_metrics["regression"]["spearman_log"],
            "val_mechanism_macro_f1": val_metrics["mechanism3"]["macro_f1"],
            "val_class1_recall": val_metrics["mechanism3"]["class1_enriched_recall"],
            "val_selection_score": float(val_loss_key),
        }
        history.append(row)
        if val_loss_key < best_val - 1e-5:
            best_val = val_loss_key
            best_val_log_mae = float(val_metrics["regression"]["log_mae"])
            best_epoch = epoch
            epochs_without_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": {
                        "input_channels": input_channels,
                        "cnn_filters": [int(x) for x in str(args.cnn_filters).split(",") if x.strip()],
                        "hidden_dim": int(args.hidden_dim),
                        "dropout": float(args.dropout),
                        "use_multiscale": not bool(args.no_multiscale),
                        "regression_head_mode": str(args.regression_head_mode),
                        "spatial_context_mode": str(args.spatial_context_mode),
                        "use_core_aux_head": float(args.core_peak_loss_weight) > 0.0,
                        "use_tail_residual_head": bool(args.use_tail_residual_head),
                        "n_domain_experts": int(args.n_domain_experts),
                        "thresholds": thresholds,
                        "mechanism_class_names": MECHANISM3_CLASS_NAMES,
                        "selection_metric": str(args.selection_metric),
                        "regression_loss": str(args.regression_loss),
                        "target_transform": str(args.target_transform),
                        "target_transform_scale": float(args.target_transform_scale),
                        "high_underprediction_loss_weight": float(args.high_underprediction_loss_weight),
                        "low_overprediction_loss_weight": float(args.low_overprediction_loss_weight),
                        "low_overprediction_threshold": float(args.low_overprediction_threshold),
                        "asymmetric_high_threshold": float(args.asymmetric_high_threshold),
                        "asymmetric_low_threshold": float(args.asymmetric_low_threshold),
                        "asymmetric_under_weight": float(args.asymmetric_under_weight),
                        "asymmetric_over_weight": float(args.asymmetric_over_weight),
                        "bin_balance_edges": str(args.bin_balance_edges),
                        "bin_balance_max_weight": float(args.bin_balance_max_weight),
                        "student_t_anchor_weight": float(args.student_t_anchor_weight),
                        "profile_rank_loss_weight": float(args.profile_rank_loss_weight),
                        "profile_rank_loss_margin": float(args.profile_rank_loss_margin),
                        "profile_rank_min_pairs": int(args.profile_rank_min_pairs),
                        "core_cvar_loss_weight": float(args.core_cvar_loss_weight),
                        "core_cvar_top_fraction": float(args.core_cvar_top_fraction),
                        "core_cvar_min_samples": int(args.core_cvar_min_samples),
                        "core_cvar_detach_base": bool(args.core_cvar_detach_base),
                        "tail_underprediction_loss_weight": float(args.tail_underprediction_loss_weight),
                        "core_tail_loss_weight": float(args.core_tail_loss_weight),
                        "core_peak_loss_weight": float(args.core_peak_loss_weight),
                        "mechanism_guard_weight": float(args.mechanism_guard_weight),
                        "tail_gate_loss_weight": float(args.tail_gate_loss_weight),
                        "tail_residual_l2_weight": float(args.tail_residual_l2_weight),
                        "domain_balance_weight": float(args.domain_balance_weight),
                        "domain_entropy_weight": float(args.domain_entropy_weight),
                        "domain_residual_l2_weight": float(args.domain_residual_l2_weight),
                    },
                    "norm_params": norm_params,
                    "feature_names": feature_names,
                    "best_epoch": int(best_epoch),
                    "best_val_log_mae": float(best_val_log_mae),
                    "best_val_selection_score": float(best_val),
                },
                best_path,
            )
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= int(args.patience):
                break

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    train_pred = _predict(
        model, train_eval_loader, device=device, dataset=base_dataset, thresholds=thresholds,
        spatial_context_mode=str(args.spatial_context_mode),
    )
    val_pred = _predict(
        model, val_loader, device=device, dataset=base_dataset, thresholds=thresholds,
        spatial_context_mode=str(args.spatial_context_mode),
    )
    test_pred = _predict(
        model, test_loader, device=device, dataset=base_dataset, thresholds=thresholds,
        spatial_context_mode=str(args.spatial_context_mode),
    )

    table_dir = output_dir / "tables"
    for split_name, pred in (("train", train_pred), ("val", val_pred), ("test", test_pred)):
        split_dir = table_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        pred.to_csv(split_dir / "mechanism_aware_rey_predictions.csv", index=False)
        _write_json(split_dir / "metrics.json", _evaluate_predictions(pred, thresholds))
    core_metrics = _core_presence_metrics(val_pred, test_pred, thresholds)
    core_metrics.to_csv(table_dir / "core_presence_val_selected_metrics.csv", index=False)
    sample_calibration = _sample_threshold_calibration_metrics(val_pred, test_pred, thresholds)
    sample_calibration.to_csv(table_dir / "sample_threshold_val_selected_metrics.csv", index=False)

    summary = {
        "status": "completed",
        "run_id": output_dir.name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "best_epoch": int(best_epoch),
        "best_val_log_mae": float(best_val_log_mae),
        "best_val_selection_score": float(best_val),
        "config": vars(args),
        "feature_count": len(feature_names),
        "input_channels": int(input_channels),
        "splits": {
            "train_n": int(len(train_idx)),
            "val_n": int(len(val_idx)),
            "test_n": int(len(test_idx)),
            "train_cores": int(base_dataset.rey_data.iloc[train_idx][str(args.group_col)].astype(str).nunique()),
            "val_cores": int(base_dataset.rey_data.iloc[val_idx][str(args.group_col)].astype(str).nunique()),
            "test_cores": int(base_dataset.rey_data.iloc[test_idx][str(args.group_col)].astype(str).nunique()),
        },
        "sampler": sampler_summary,
        "train_metrics": _evaluate_predictions(train_pred, thresholds),
        "val_metrics": _evaluate_predictions(val_pred, thresholds),
        "test_metrics": _evaluate_predictions(test_pred, thresholds),
        "artifacts": {
            "best_model": str(best_path),
            "predictions_train": str(table_dir / "train" / "mechanism_aware_rey_predictions.csv"),
            "predictions_val": str(table_dir / "val" / "mechanism_aware_rey_predictions.csv"),
            "predictions_test": str(table_dir / "test" / "mechanism_aware_rey_predictions.csv"),
            "core_presence_metrics": str(table_dir / "core_presence_val_selected_metrics.csv"),
            "sample_threshold_calibration_metrics": str(table_dir / "sample_threshold_val_selected_metrics.csv"),
        },
    }
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    _write_json(output_dir / "run_summary.json", summary)
    return summary


def main() -> None:
    setup_logger(level="INFO")
    parser = argparse.ArgumentParser(description="Train mechanism-aware continuous REY multitask model.")
    parser.add_argument("--env-nc", required=True, type=Path)
    parser.add_argument("--points", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--feature-names", default=None, help="Comma-separated env features. Defaults to all env vars except land_mask.")
    parser.add_argument("--sed-rate-nc", default="data/Average sedimentation rate.nc")
    parser.add_argument("--depth-feature-type", choices=["age_ka_proxy", "depth_mbsf", "zero"], default="age_ka_proxy")
    parser.add_argument("--group-col", default="core_id")
    parser.add_argument("--seed", type=int, default=310)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Grouped split seed. Defaults to --seed for backward compatibility; set a fixed value for seed ensembles.",
    )
    parser.add_argument("--test-split-ratio", type=float, default=0.15)
    parser.add_argument("--validation-split-ratio", type=float, default=0.1765)
    parser.add_argument("--group-split-stratify-bins", type=int, default=6)
    parser.add_argument(
        "--split-indices-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory containing train_indices.npy, val_indices.npy, "
            "and test_indices.npy. Used for explicit core-fold OOF training."
        ),
    )
    parser.add_argument("--patch-size", type=int, default=11)
    parser.add_argument("--thresholds", default="600,800,1000,1500")
    parser.add_argument("--target-transform", choices=["log1p", "asinh", "hinged_log"], default="log1p")
    parser.add_argument("--target-transform-scale", type=float, default=1000.0)
    parser.add_argument(
        "--target-transform-hinge-beta",
        type=float,
        default=0.25,
        help="Extra tail slope for target-transform=hinged_log above --target-transform-scale ppm.",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--cnn-filters", default="64,128,128")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--no-multiscale", action="store_true")
    parser.add_argument(
        "--regression-head-mode",
        choices=["mechanism_gated", "global"],
        default="mechanism_gated",
        help="Use the mechanism-gated residual mean or the global regression mean only.",
    )
    parser.add_argument(
        "--spatial-context-mode",
        choices=["full", "center_broadcast"],
        default="full",
        help="center_broadcast removes patch neighbourhood variation without changing tensor shape or parameter count.",
    )
    parser.add_argument(
        "--regression-loss",
        choices=[
            "student_t",
            "student_t_mae",
            "huber",
            "log_cosh",
            "mae_mse_blend",
            "asymmetric_huber",
            "balanced_asymmetric_huber",
        ],
        default="student_t",
    )
    parser.add_argument("--student-t-nu", type=float, default=4.0)
    parser.add_argument("--student-t-anchor-weight", type=float, default=0.0)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--mae-mse-blend-mse-weight", type=float, default=0.25)
    parser.add_argument("--asymmetric-high-threshold", type=float, default=1000.0)
    parser.add_argument("--asymmetric-low-threshold", type=float, default=300.0)
    parser.add_argument("--asymmetric-under-weight", type=float, default=1.0)
    parser.add_argument("--asymmetric-over-weight", type=float, default=1.0)
    parser.add_argument("--bin-balance-edges", default="300,600,1000,1500,2000")
    parser.add_argument("--bin-balance-max-weight", type=float, default=2.5)
    parser.add_argument("--threshold-loss-weight", type=float, default=0.35)
    parser.add_argument("--mechanism-loss-weight", type=float, default=0.2)
    parser.add_argument(
        "--mechanism-class-weighting",
        choices=["none", "sqrt", "inverse"],
        default="sqrt",
        help="Class weighting for the auxiliary 3-class mechanism head.",
    )
    parser.add_argument("--use-threshold-pos-weight", action="store_true")
    parser.add_argument("--threshold-pos-weight-cap", type=float, default=8.0)
    parser.add_argument(
        "--threshold-loss-multipliers",
        default="0.8,1.0,1.4,1.8",
        help="Comma-separated per-threshold BCE multipliers.",
    )
    parser.add_argument("--threshold-focal-gamma", type=float, default=1.0)
    parser.add_argument("--high-rey-loss-threshold", type=float, default=1000.0)
    parser.add_argument("--high-rey-loss-weight", type=float, default=1.75)
    parser.add_argument("--high-rey-sampler-threshold", type=float, default=1000.0)
    parser.add_argument("--high-rey-sampler-weight", type=float, default=1.0)
    parser.add_argument(
        "--core-balanced-sampler-power",
        type=float,
        default=0.0,
        help=(
            "If >0, use weighted replacement sampling so large cores do not dominate "
            "training batches. 1.0 approximates inverse core-size sampling."
        ),
    )
    parser.add_argument("--core-balanced-sampler-max-factor", type=float, default=4.0)
    parser.add_argument("--sampler-num-samples-multiplier", type=float, default=1.0)
    parser.add_argument("--high-underprediction-loss-weight", type=float, default=0.0)
    parser.add_argument("--high-underprediction-threshold", type=float, default=1000.0)
    parser.add_argument("--low-overprediction-loss-weight", type=float, default=0.0)
    parser.add_argument("--low-overprediction-threshold", type=float, default=300.0)
    parser.add_argument("--tail-underprediction-loss-weight", type=float, default=0.0)
    parser.add_argument("--tail-underprediction-quantile", type=float, default=0.90)
    parser.add_argument("--tail-underprediction-min-samples", type=int, default=8)
    parser.add_argument("--core-tail-loss-weight", type=float, default=0.0)
    parser.add_argument("--core-tail-threshold", type=float, default=1000.0)
    parser.add_argument("--core-tail-min-samples", type=int, default=3)
    parser.add_argument("--core-peak-loss-weight", type=float, default=0.0)
    parser.add_argument("--core-peak-huber-delta", type=float, default=1.0)
    parser.add_argument("--mechanism-guard-weight", type=float, default=0.0)
    parser.add_argument("--mechanism-guard-max-loss", type=float, default=0.75)
    parser.add_argument("--use-tail-residual-head", action="store_true")
    parser.add_argument("--tail-residual-threshold", type=float, default=1000.0)
    parser.add_argument("--tail-gate-loss-weight", type=float, default=0.0)
    parser.add_argument("--tail-residual-l2-weight", type=float, default=0.0)
    parser.add_argument("--n-domain-experts", type=int, default=0)
    parser.add_argument("--domain-balance-weight", type=float, default=0.0)
    parser.add_argument("--domain-entropy-weight", type=float, default=0.0)
    parser.add_argument("--domain-residual-l2-weight", type=float, default=0.0)
    parser.add_argument("--rank-loss-weight", type=float, default=0.08)
    parser.add_argument("--rank-loss-margin", type=float, default=0.25)
    parser.add_argument("--profile-rank-loss-weight", type=float, default=0.0)
    parser.add_argument("--profile-rank-loss-margin", type=float, default=0.20)
    parser.add_argument("--profile-rank-min-pairs", type=int, default=1)
    parser.add_argument("--core-cvar-loss-weight", type=float, default=0.0)
    parser.add_argument("--core-cvar-top-fraction", type=float, default=0.25)
    parser.add_argument("--core-cvar-min-samples", type=int, default=2)
    parser.add_argument("--core-cvar-detach-base", action="store_true")
    parser.add_argument(
        "--selection-metric",
        choices=["log_mae", "composite"],
        default="composite",
        help="Validation checkpoint metric.",
    )
    parser.add_argument("--selection-spearman-weight", type=float, default=0.10)
    parser.add_argument("--selection-threshold-f1-weight", type=float, default=0.12)
    parser.add_argument("--selection-mechanism-f1-weight", type=float, default=0.05)
    parser.add_argument("--selection-class1-recall-weight", type=float, default=0.04)
    parser.add_argument("--monotonic-loss-weight", type=float, default=0.05)
    parser.add_argument("--gate-balance-weight", type=float, default=0.02)
    parser.add_argument("--use-target-weights", action="store_true")
    parser.add_argument("--max-weight", type=float, default=5.0)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--shift-pixels", type=int, default=0)
    parser.add_argument(
        "--add-coord-channels",
        action="store_true",
        help="Append sin/cos latitude/longitude channels to each patch; prediction-time legal spatial context.",
    )
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    summary = _train_one(args)
    print(json.dumps({"status": summary["status"], "output_dir": summary["output_dir"], "best_epoch": summary["best_epoch"]}, indent=2))


if __name__ == "__main__":
    main()

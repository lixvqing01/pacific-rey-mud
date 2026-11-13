from __future__ import annotations

import argparse
import json
import math
import tempfile
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr
from core.utils.netcdf_io import open_dataset
from torch.utils.data import DataLoader

from core.datasets.rey_dataset import REYPointDataset
from core.models.rey_multitask import MechanismAwareREYRegressor
from core.utils.logger_config import setup_logger
from core.utils.serialization_utils import NumpyJSONEncoder


DEFAULT_MODEL_RUNS = [
    "models/seed714",
]
DEFAULT_ENV_NC = "data/env_gridded.nc"
DEFAULT_SED_RATE_NC = "data/Average sedimentation rate.nc"
DEFAULT_AGE_LAYERS_KA = ",".join(str(x) for x in [*range(0, 2501, 250), 2580])
MECHANISM_CLASS_NAMES = [
    "carbonate_background",
    "class1_enriched",
    "class2_slow_sedimentation",
]


def _emit_progress(**payload: Any) -> None:
    print(f"PROGRESS_JSON {json.dumps(payload, cls=NumpyJSONEncoder, ensure_ascii=False)}", flush=True)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, cls=NumpyJSONEncoder, indent=2, ensure_ascii=False)


def _coord_names(ds: xr.Dataset) -> tuple[str, str]:
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    if lat_name not in ds.coords or lon_name not in ds.coords:
        raise KeyError(f"Could not find latitude/longitude coordinates in env grid. Coords={list(ds.coords)}")
    return lat_name, lon_name


def _load_checkpoint(run_dir: Path) -> dict[str, Any]:
    ckpt_path = run_dir / "models" / "mechanism_aware_rey_regressor_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {ckpt_path}")
    return torch.load(ckpt_path, map_location="cpu", weights_only=False)


def _build_model(ckpt: dict[str, Any], device: torch.device) -> MechanismAwareREYRegressor:
    cfg = dict(ckpt.get("model_config", {}) or {})
    model = MechanismAwareREYRegressor(
        input_channels=int(cfg["input_channels"]),
        n_mechanisms=len(cfg.get("mechanism_class_names", MECHANISM_CLASS_NAMES)),
        n_thresholds=len(cfg.get("thresholds", [600, 800, 1000, 1500])),
        cnn_filters=[int(x) for x in cfg.get("cnn_filters", [64, 128, 128])],
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        dropout=float(cfg.get("dropout", 0.25)),
        use_multiscale=bool(cfg.get("use_multiscale", True)),
        use_core_aux_head=bool(cfg.get("use_core_aux_head", False)),
        use_tail_residual_head=bool(cfg.get("use_tail_residual_head", False)),
        n_domain_experts=int(cfg.get("n_domain_experts", 0) or 0),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _interpolate_sed_rate(sed_rate_nc: Path, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    with open_dataset(sed_rate_nc) as ds:
        var = "Average sedimentation rate"
        if var not in ds.data_vars:
            var = str(next(iter(ds.data_vars)))
        field = ds[var]
        lon_name = "lon" if "lon" in field.coords else "longitude"
        lat_name = "lat" if "lat" in field.coords else "latitude"
        lon_interp = ((np.asarray(lons, dtype=float) + 180.0) % 360.0) - 180.0
        vals = field.interp(
            {lon_name: ("points", lon_interp), lat_name: ("points", np.asarray(lats, dtype=float))},
            method="linear",
        ).values
        if np.isnan(vals).any():
            bad = np.isnan(vals)
            nearest = field.interp(
                {lon_name: ("points", lon_interp[bad]), lat_name: ("points", np.asarray(lats, dtype=float)[bad])},
                method="nearest",
            ).values
            vals[bad] = nearest
    vals = np.asarray(vals, dtype=np.float64)
    vals[~np.isfinite(vals)] = 1e-3
    return np.clip(vals, 1e-3, None)


def _make_grid_points(
    *,
    env_nc: Path,
    sed_rate_nc: Path,
    age_ka: float,
    stride: int,
    max_points: int | None,
    output_dir: Path,
) -> tuple[Path, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with open_dataset(env_nc) as ds:
        lat_name, lon_name = _coord_names(ds)
        lats = ds[lat_name].values.astype(float)
        lons = ds[lon_name].values.astype(float)
    lat_idx = np.arange(0, len(lats), max(1, int(stride)), dtype=int)
    lon_idx = np.arange(0, len(lons), max(1, int(stride)), dtype=int)
    lat_grid, lon_grid = np.meshgrid(lats[lat_idx], lons[lon_idx], indexing="ij")
    flat_lat = lat_grid.reshape(-1)
    flat_lon = lon_grid.reshape(-1)
    regular_grid = True
    if max_points is not None and max_points > 0 and flat_lat.size > max_points:
        keep = np.linspace(0, flat_lat.size - 1, int(max_points)).round().astype(int)
        flat_lat = flat_lat[keep]
        flat_lon = flat_lon[keep]
        regular_grid = False
    sed_rate = _interpolate_sed_rate(sed_rate_nc, flat_lon, flat_lat)
    physical_depth = np.asarray(age_ka, dtype=float) * sed_rate / 100.0


    df = pd.DataFrame(
        {
            "lon": flat_lon,
            "lat": flat_lat,
            "depth": float(age_ka),
            "rey": 0.0,
            "mechanism_class3": 0,
            "core_id": "grid",
            "site_group": "grid",
            "grid_row": np.arange(flat_lat.size, dtype=int),
        }
    )
    path = output_dir / "inputs" / f"grid_points_age_{int(round(age_ka))}ka.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    grid_shape = np.asarray([len(lat_idx), len(lon_idx)], dtype=int) if regular_grid else np.asarray([0, 0], dtype=int)
    return path, flat_lat, flat_lon, physical_depth, grid_shape


def _make_grid_geometry(
    *,
    env_nc: Path,
    sed_rate_nc: Path,
    initial_age_ka: float,
    stride: int,
    target_resolution_deg: float | None,
    max_points: int | None,
    output_dir: Path,
) -> tuple[Path, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with open_dataset(env_nc) as ds:
        lat_name, lon_name = _coord_names(ds)
        lats = ds[lat_name].values.astype(float)
        lons = ds[lon_name].values.astype(float)
    if target_resolution_deg is not None and float(target_resolution_deg) > 0:
        lat_idx = _indices_for_target_resolution(lats, float(target_resolution_deg))
        lon_idx = _indices_for_target_resolution(lons, float(target_resolution_deg))
    else:
        lat_idx = np.arange(0, len(lats), max(1, int(stride)), dtype=np.int64)
        lon_idx = np.arange(0, len(lons), max(1, int(stride)), dtype=np.int64)
    row_grid, col_grid = np.meshgrid(lat_idx, lon_idx, indexing="ij")
    lat_grid, lon_grid = np.meshgrid(lats[lat_idx], lons[lon_idx], indexing="ij")
    flat_rows = row_grid.reshape(-1)
    flat_cols = col_grid.reshape(-1)
    flat_lat = lat_grid.reshape(-1)
    flat_lon = lon_grid.reshape(-1)
    regular_grid = True
    if max_points is not None and max_points > 0 and flat_lat.size > max_points:
        keep = np.linspace(0, flat_lat.size - 1, int(max_points)).round().astype(int)
        flat_rows = flat_rows[keep]
        flat_cols = flat_cols[keep]
        flat_lat = flat_lat[keep]
        flat_lon = flat_lon[keep]
        regular_grid = False
    sed_rate = _interpolate_sed_rate(sed_rate_nc, flat_lon, flat_lat)
    df = pd.DataFrame(
        {
            "lon": flat_lon,
            "lat": flat_lat,
            "depth": float(initial_age_ka),
            "rey": 0.0,
            "mechanism_class3": 0,
            "core_id": "grid",
            "site_group": "grid",
            "grid_row": np.arange(flat_lat.size, dtype=int),
        }
    )
    path = output_dir / "inputs" / "grid_points_prediction.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    grid_shape = np.asarray([len(lat_idx), len(lon_idx)], dtype=int) if regular_grid else np.asarray([0, 0], dtype=int)
    return path, flat_lat, flat_lon, flat_rows, flat_cols, sed_rate, grid_shape


def _indices_for_target_resolution(coords: np.ndarray, target_resolution_deg: float) -> np.ndarray:
    coord = np.asarray(coords, dtype=float)
    if coord.size <= 1:
        return np.arange(coord.size, dtype=np.int64)
    resolution = float(target_resolution_deg)
    if (not np.isfinite(resolution)) or resolution <= 0:
        return np.arange(coord.size, dtype=np.int64)
    ascending = bool(coord[-1] >= coord[0])
    work = coord if ascending else coord[::-1]
    targets = np.arange(float(work[0]), float(work[-1]) + resolution * 0.5, resolution, dtype=float)
    pos = np.searchsorted(work, targets, side="left")
    pos = np.clip(pos, 0, work.size - 1)
    left = np.clip(pos - 1, 0, work.size - 1)
    right = pos
    choose_right = np.abs(work[right] - targets) <= np.abs(targets - work[left])
    idx_work = np.where(choose_right, right, left).astype(np.int64)
    idx = idx_work if ascending else (work.size - 1 - idx_work)
    return np.unique(np.sort(idx)).astype(np.int64)


def _inverse_rey(mu_norm: np.ndarray, dataset: REYPointDataset) -> tuple[np.ndarray, np.ndarray]:
    pred_transformed = (np.asarray(mu_norm, dtype=float) * float(dataset.rey_std)) + float(dataset.rey_mean)
    if str(getattr(dataset, "rey_target_transform_name", "log1p")) in ("log1p", "hinged_log"):
        pred_transformed = np.clip(pred_transformed, 0.0, 11.5)
    pred_rey = dataset._inverse_transform_rey_target(pred_transformed)
    pred_rey = np.clip(pred_rey, 0.0, None)
    return np.log1p(pred_rey), pred_rey


def _build_prediction_dataset(*, first: dict[str, Any], env_nc: Path, points_path: Path) -> REYPointDataset:
    return REYPointDataset(
        str(points_path),
        str(env_nc),
        list(first["feature_names"]),
        patch_size=(11, 11),
        task_type="regression",
        sed_rate_nc_path=None,
        depth_feature_type_override="depth_mbsf",
        norm_params_override=dict(first["norm_params"]),
        augment_data=False,
    )


def _set_prediction_dataset_age(dataset: REYPointDataset, age_ka: float) -> None:
    dataset.rey_data.loc[:, "depth"] = float(age_ka)


def _ffill_axis(arr: np.ndarray, axis: int) -> np.ndarray:
    arr = np.moveaxis(arr, axis, -1)
    mask = np.isnan(arr)
    if not bool(mask.any()):
        return np.moveaxis(arr, -1, axis)
    valid = ~mask
    idx = np.where(valid, np.arange(arr.shape[-1], dtype=np.int64), 0)
    any_valid = valid.any(axis=-1)
    first_valid = np.argmax(valid, axis=-1)
    leading_nan = mask[..., 0] & any_valid
    idx[..., 0] = np.where(leading_nan, first_valid, idx[..., 0])
    np.maximum.accumulate(idx, axis=-1, out=idx)
    filled = np.take_along_axis(arr, idx, axis=-1)
    return np.moveaxis(filled, -1, axis)


def _bfill_axis(arr: np.ndarray, axis: int) -> np.ndarray:
    arr = np.moveaxis(arr, axis, -1)
    mask = np.isnan(arr)
    if not bool(mask.any()):
        return np.moveaxis(arr, -1, axis)
    valid = ~mask
    idx = np.where(valid, np.arange(arr.shape[-1], dtype=np.int64), arr.shape[-1] - 1)
    any_valid = valid.any(axis=-1)
    last_valid = arr.shape[-1] - 1 - np.argmax(valid[..., ::-1], axis=-1)
    trailing_nan = mask[..., -1] & any_valid
    idx[..., -1] = np.where(trailing_nan, last_valid, idx[..., -1])
    idx = np.minimum.accumulate(idx[..., ::-1], axis=-1)[..., ::-1]
    filled = np.take_along_axis(arr, idx, axis=-1)
    return np.moveaxis(filled, -1, axis)


def _ffill_bfill_patch_batch(patches: np.ndarray) -> np.ndarray:


    filled = _ffill_axis(patches, axis=3)
    filled = _ffill_axis(filled, axis=2)
    filled = _bfill_axis(filled, axis=3)
    filled = _bfill_axis(filled, axis=2)
    return filled


def _depth_norm_for_age(dataset: REYPointDataset, age_ka: float) -> float:
    transformed_depth_val = float(age_ka)
    if dataset.depth_transform_method == "log1p":
        transformed_depth_val = float(np.log1p(transformed_depth_val))
    elif dataset.depth_transform_method == "sqrt":
        transformed_depth_val = float(np.sqrt(transformed_depth_val))
    depth_channel_idx = len(dataset.feature_names)
    if dataset.env_normalization_method == "zscore":
        return float(
            (transformed_depth_val - dataset.channel_means[depth_channel_idx])
            / dataset.channel_stds[depth_channel_idx]
        )
    if dataset.env_normalization_method == "minmax":
        depth_range = dataset.channel_maxs[depth_channel_idx] - dataset.channel_mins[depth_channel_idx]
        if depth_range > 1e-6:
            return float((transformed_depth_val - dataset.channel_mins[depth_channel_idx]) / depth_range)
    return 0.0


def _extract_grid_patch_batch(
    *,
    dataset: REYPointDataset,
    padded_env: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    age_ka: float,
) -> torch.Tensor:
    static_patch, coord_patch = _extract_grid_static_patch_batch(
        dataset=dataset,
        padded_env=padded_env,
        rows=rows,
        cols=cols,
        lats=lats,
        lons=lons,
    )
    return _combine_static_patch_with_depth(
        dataset=dataset,
        static_patch=static_patch,
        coord_patch=coord_patch,
        age_ka=float(age_ka),
    )


def _extract_grid_static_patch_batch(
    *,
    dataset: REYPointDataset,
    padded_env: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:
    ph, pw = dataset.patch_size
    rr = rows[:, None, None] + np.arange(ph, dtype=np.int64)[None, :, None]
    cc = cols[:, None, None] + np.arange(pw, dtype=np.int64)[None, None, :]
    patches = padded_env[:, rr, cc].transpose(1, 0, 2, 3).astype(np.float32, copy=False)

    env_missingness_np = None
    if dataset.add_env_missingness_channels:
        env_missingness_np = np.isnan(patches).astype(np.float32)

    if np.isnan(patches).any():
        patches = _ffill_bfill_patch_batch(patches)
        if np.isnan(patches).any():
            if dataset.env_normalization_method == "zscore":
                fill_values = np.asarray(dataset.channel_means[: len(dataset.feature_names)], dtype=np.float32)
            else:
                fill_values = np.asarray(dataset.channel_mins[: len(dataset.feature_names)], dtype=np.float32)
            patches = np.where(np.isnan(patches), fill_values[None, :, None, None], patches)

    if dataset.env_normalization_method == "zscore":
        means = np.asarray(dataset.channel_means[: len(dataset.feature_names)], dtype=np.float32)
        stds = np.asarray(dataset.channel_stds[: len(dataset.feature_names)], dtype=np.float32)
        patches = (patches - means[None, :, None, None]) / stds[None, :, None, None]
    elif dataset.env_normalization_method == "minmax":
        mins = np.asarray(dataset.channel_mins[: len(dataset.feature_names)], dtype=np.float32)
        maxs = np.asarray(dataset.channel_maxs[: len(dataset.feature_names)], dtype=np.float32)
        ranges = np.where((maxs - mins) > 1e-6, maxs - mins, 1.0).astype(np.float32)
        patches = (patches - mins[None, :, None, None]) / ranges[None, :, None, None]
    else:
        patches = patches.astype(np.float32, copy=False)

    parts = [patches]
    if env_missingness_np is not None:
        parts.append(env_missingness_np)
    static_patch = np.concatenate(parts, axis=1).astype(np.float32, copy=False) if len(parts) > 1 else patches
    coord_patch = None
    if dataset.add_coord_channels:
        lon_norm = ((lons.astype(np.float64) + 180.0) % 360.0) - 180.0
        lon_rad = np.deg2rad(lon_norm)
        lat_rad = np.deg2rad(lats.astype(np.float64))
        coord_vals = np.stack(
            [np.sin(lat_rad), np.cos(lat_rad), np.sin(lon_rad), np.cos(lon_rad)],
            axis=1,
        ).astype(np.float32)
        coord_patch = np.broadcast_to(coord_vals[:, :, None, None], (rows.size, 4, ph, pw)).copy()
    return np.ascontiguousarray(static_patch), coord_patch


def _combine_static_patch_with_depth(
    *,
    dataset: REYPointDataset,
    static_patch: np.ndarray,
    coord_patch: np.ndarray | None,
    age_ka: float,
) -> torch.Tensor:
    ph, pw = dataset.patch_size
    n = int(static_patch.shape[0])
    depth_value = _depth_norm_for_age(dataset, float(age_ka))
    parts = [static_patch, np.full((n, 1, ph, pw), depth_value, dtype=np.float32)]
    if coord_patch is not None:
        parts.append(coord_patch)
    combined = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
    return torch.from_numpy(np.ascontiguousarray(combined))


def _predict_patch_outputs(
    *,
    patch: torch.Tensor,
    models: list[MechanismAwareREYRegressor],
    device: torch.device,
    use_amp: bool,
    channels_last: bool,
) -> dict[str, np.ndarray]:
    patch = patch.to(device, non_blocking=(device.type == "cuda"))
    if bool(channels_last) and device.type == "cuda":
        patch = patch.contiguous(memory_format=torch.channels_last)
    mu_models: list[torch.Tensor] = []
    sigma_models: list[torch.Tensor] = []
    mech_models: list[torch.Tensor] = []
    thr_models: list[torch.Tensor] = []
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda" and bool(use_amp)
        else nullcontext()
    )
    with autocast_context:
        for model in models:
            out = model(patch)
            mu = torch.nan_to_num(out["mu"], nan=0.0, posinf=8.0, neginf=-8.0).clamp(-8.0, 8.0).view(-1).float()
            sigma = torch.exp(torch.nan_to_num(out["log_sigma"], nan=0.0, posinf=3.0, neginf=-4.0).clamp(-4.0, 3.0)).view(-1).float()
            mech_logits = torch.nan_to_num(out["mechanism_logits"], nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0).float()
            thr_logits = torch.nan_to_num(out["threshold_logits"], nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0).float()
            mu_models.append(mu)
            sigma_models.append(sigma)
            mech_models.append(torch.softmax(mech_logits, dim=1))
            thr_models.append(torch.sigmoid(thr_logits))
    mu_stack = torch.stack(mu_models, dim=0)
    sigma_stack = torch.stack(sigma_models, dim=0)
    mech_mean = torch.stack(mech_models, dim=0).mean(dim=0)
    thr_mean = torch.stack(thr_models, dim=0).mean(dim=0)
    thr_mean = torch.cummin(thr_mean, dim=1).values
    return {
        "mu": mu_stack.mean(dim=0).detach().cpu().numpy(),
        "sigma": sigma_stack.mean(dim=0).detach().cpu().numpy(),
        "ensemble_std": mu_stack.std(dim=0, unbiased=False).detach().cpu().numpy(),
        "mech": mech_mean.detach().cpu().numpy(),
        "thr": thr_mean.detach().cpu().numpy(),
    }


def _append_model_outputs(
    *,
    patch: torch.Tensor,
    models: list[MechanismAwareREYRegressor],
    device: torch.device,
    use_amp: bool,
    channels_last: bool,
    mu_mean_parts: list[np.ndarray],
    sigma_mean_parts: list[np.ndarray],
    ensemble_std_parts: list[np.ndarray],
    mech_mean_parts: list[np.ndarray],
    thr_mean_parts: list[np.ndarray],
) -> None:
    out = _predict_patch_outputs(
        patch=patch,
        models=models,
        device=device,
        use_amp=use_amp,
        channels_last=channels_last,
    )
    mu_mean_parts.append(out["mu"])
    sigma_mean_parts.append(out["sigma"])
    ensemble_std_parts.append(out["ensemble_std"])
    mech_mean_parts.append(out["mech"])
    thr_mean_parts.append(out["thr"])


@torch.no_grad()
def _predict_age_layer(
    *,
    models: list[MechanismAwareREYRegressor],
    ckpts: list[dict[str, Any]],
    dataset: REYPointDataset,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    device: torch.device,
    use_amp: bool,
    channels_last: bool,
    age_ka: float,
    age_index: int,
    n_age_layers: int,
) -> dict[str, np.ndarray]:
    first = ckpts[0]
    cfg = dict(first.get("model_config", {}) or {})
    thresholds = [float(x) for x in cfg.get("thresholds", [600, 800, 1000, 1500])]
    loader_kwargs: dict[str, Any] = {
        "batch_size": int(batch_size),
        "shuffle": False,
        "num_workers": max(0, int(num_workers)),
        "pin_memory": (device.type == "cuda"),
        "persistent_workers": bool(int(num_workers) > 0),
    }
    if int(num_workers) > 0:
        loader_kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
    loader = DataLoader(dataset, **loader_kwargs)
    n_points = len(dataset)
    n_batches = int(math.ceil(max(n_points, 1) / max(int(batch_size), 1)))
    _emit_progress(
        stage="predict_age_layer",
        event="start",
        age_ka=float(age_ka),
        age_index=int(age_index),
        n_age_layers=int(n_age_layers),
        n_points=int(n_points),
        n_batches=int(n_batches),
        percent=float(((age_index - 1) / max(n_age_layers, 1)) * 0.96),
    )
    mu_mean_parts: list[np.ndarray] = []
    sigma_mean_parts: list[np.ndarray] = []
    ensemble_std_parts: list[np.ndarray] = []
    mech_mean_parts: list[np.ndarray] = []
    thr_mean_parts: list[np.ndarray] = []
    for batch_idx, batch in enumerate(loader, start=1):
        _append_model_outputs(
            patch=batch[0],
            models=models,
            device=device,
            use_amp=use_amp,
            channels_last=channels_last,
            mu_mean_parts=mu_mean_parts,
            sigma_mean_parts=sigma_mean_parts,
            ensemble_std_parts=ensemble_std_parts,
            mech_mean_parts=mech_mean_parts,
            thr_mean_parts=thr_mean_parts,
        )
        completed_points = min(int(batch_idx) * int(batch_size), int(n_points))
        if batch_idx == 1 or batch_idx == n_batches or batch_idx % 5 == 0:
            layer_fraction = completed_points / max(int(n_points), 1)
            overall_fraction = ((int(age_index) - 1) + layer_fraction) / max(int(n_age_layers), 1)
            _emit_progress(
                stage="predict_age_layer",
                event="batch",
                age_ka=float(age_ka),
                age_index=int(age_index),
                n_age_layers=int(n_age_layers),
                batch_index=int(batch_idx),
                n_batches=int(n_batches),
                completed_points=int(completed_points),
                n_points=int(n_points),
                percent=float(overall_fraction * 0.96),
            )
    mu_mean = np.concatenate(mu_mean_parts)
    pred_log, pred_rey = _inverse_rey(mu_mean, dataset)
    mech_mean = np.concatenate(mech_mean_parts, axis=0)
    thr_mean = np.minimum.accumulate(np.concatenate(thr_mean_parts, axis=0), axis=1)
    sigma_mean = np.concatenate(sigma_mean_parts)
    ensemble_std = np.concatenate(ensemble_std_parts)
    entropy = -np.sum(np.clip(mech_mean, 1e-8, 1.0) * np.log(np.clip(mech_mean, 1e-8, 1.0)), axis=1)
    confidence = 1.0 - entropy / max(math.log(mech_mean.shape[1]), 1e-8)
    _emit_progress(
        stage="predict_age_layer",
        event="done",
        age_ka=float(age_ka),
        age_index=int(age_index),
        n_age_layers=int(n_age_layers),
        n_points=int(n_points),
        percent=float((age_index / max(n_age_layers, 1)) * 0.96),
    )
    return {
        "thresholds": np.asarray(thresholds, dtype=float),
        "pred_logrey": pred_log,
        "pred_rey_ppm": pred_rey,
        "pred_sigma_norm": sigma_mean,
        "ensemble_std_logrey_norm": ensemble_std,
        "mechanism_prob": mech_mean,
        "threshold_prob": thr_mean,
        "pred_mechanism_class_id": np.argmax(mech_mean, axis=1).astype(np.int16),
        "mechanism_entropy": entropy,
        "prediction_confidence": confidence,
    }


@torch.no_grad()
def _predict_age_layer_fast_grid(
    *,
    models: list[MechanismAwareREYRegressor],
    ckpts: list[dict[str, Any]],
    dataset: REYPointDataset,
    padded_env: np.ndarray,
    row_indices: np.ndarray,
    col_indices: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
    channels_last: bool,
    age_ka: float,
    age_index: int,
    n_age_layers: int,
) -> dict[str, np.ndarray]:
    first = ckpts[0]
    cfg = dict(first.get("model_config", {}) or {})
    thresholds = [float(x) for x in cfg.get("thresholds", [600, 800, 1000, 1500])]
    n_points = int(row_indices.size)
    n_batches = int(math.ceil(max(n_points, 1) / max(int(batch_size), 1)))
    _emit_progress(
        stage="predict_age_layer_fast_grid",
        event="start",
        age_ka=float(age_ka),
        age_index=int(age_index),
        n_age_layers=int(n_age_layers),
        n_points=int(n_points),
        n_batches=int(n_batches),
        percent=float(((age_index - 1) / max(n_age_layers, 1)) * 0.96),
    )
    mu_mean_parts: list[np.ndarray] = []
    sigma_mean_parts: list[np.ndarray] = []
    ensemble_std_parts: list[np.ndarray] = []
    mech_mean_parts: list[np.ndarray] = []
    thr_mean_parts: list[np.ndarray] = []
    for batch_idx, start in enumerate(range(0, n_points, int(batch_size)), start=1):
        end = min(start + int(batch_size), n_points)
        patch = _extract_grid_patch_batch(
            dataset=dataset,
            padded_env=padded_env,
            rows=row_indices[start:end],
            cols=col_indices[start:end],
            lats=lats[start:end],
            lons=lons[start:end],
            age_ka=float(age_ka),
        )
        _append_model_outputs(
            patch=patch,
            models=models,
            device=device,
            use_amp=use_amp,
            channels_last=channels_last,
            mu_mean_parts=mu_mean_parts,
            sigma_mean_parts=sigma_mean_parts,
            ensemble_std_parts=ensemble_std_parts,
            mech_mean_parts=mech_mean_parts,
            thr_mean_parts=thr_mean_parts,
        )
        if device.type == "cuda":
            del patch
        completed_points = int(end)
        if batch_idx == 1 or batch_idx == n_batches or batch_idx % 5 == 0:
            layer_fraction = completed_points / max(int(n_points), 1)
            overall_fraction = ((int(age_index) - 1) + layer_fraction) / max(int(n_age_layers), 1)
            _emit_progress(
                stage="predict_age_layer_fast_grid",
                event="batch",
                age_ka=float(age_ka),
                age_index=int(age_index),
                n_age_layers=int(n_age_layers),
                batch_index=int(batch_idx),
                n_batches=int(n_batches),
                completed_points=int(completed_points),
                n_points=int(n_points),
                percent=float(overall_fraction * 0.96),
            )
    mu_mean = np.concatenate(mu_mean_parts)
    pred_log, pred_rey = _inverse_rey(mu_mean, dataset)
    mech_mean = np.concatenate(mech_mean_parts, axis=0)
    thr_mean = np.minimum.accumulate(np.concatenate(thr_mean_parts, axis=0), axis=1)
    sigma_mean = np.concatenate(sigma_mean_parts)
    ensemble_std = np.concatenate(ensemble_std_parts)
    entropy = -np.sum(np.clip(mech_mean, 1e-8, 1.0) * np.log(np.clip(mech_mean, 1e-8, 1.0)), axis=1)
    confidence = 1.0 - entropy / max(math.log(mech_mean.shape[1]), 1e-8)
    _emit_progress(
        stage="predict_age_layer_fast_grid",
        event="done",
        age_ka=float(age_ka),
        age_index=int(age_index),
        n_age_layers=int(n_age_layers),
        n_points=int(n_points),
        percent=float((age_index / max(n_age_layers, 1)) * 0.96),
    )
    return {
        "thresholds": np.asarray(thresholds, dtype=float),
        "pred_logrey": pred_log,
        "pred_rey_ppm": pred_rey,
        "pred_sigma_norm": sigma_mean,
        "ensemble_std_logrey_norm": ensemble_std,
        "mechanism_prob": mech_mean,
        "threshold_prob": thr_mean,
        "pred_mechanism_class_id": np.argmax(mech_mean, axis=1).astype(np.int16),
        "mechanism_entropy": entropy,
        "prediction_confidence": confidence,
    }


def _finalize_age_output(
    *,
    thresholds: list[float],
    dataset: REYPointDataset,
    mu_mean_parts: list[np.ndarray],
    sigma_mean_parts: list[np.ndarray],
    ensemble_std_parts: list[np.ndarray],
    mech_mean_parts: list[np.ndarray],
    thr_mean_parts: list[np.ndarray],
) -> dict[str, np.ndarray]:
    mu_mean = np.concatenate(mu_mean_parts)
    pred_log, pred_rey = _inverse_rey(mu_mean, dataset)
    mech_mean = np.concatenate(mech_mean_parts, axis=0)
    thr_mean = np.minimum.accumulate(np.concatenate(thr_mean_parts, axis=0), axis=1)
    sigma_mean = np.concatenate(sigma_mean_parts)
    ensemble_std = np.concatenate(ensemble_std_parts)
    entropy = -np.sum(np.clip(mech_mean, 1e-8, 1.0) * np.log(np.clip(mech_mean, 1e-8, 1.0)), axis=1)
    confidence = 1.0 - entropy / max(math.log(mech_mean.shape[1]), 1e-8)
    return {
        "thresholds": np.asarray(thresholds, dtype=float),
        "pred_logrey": pred_log,
        "pred_rey_ppm": pred_rey,
        "pred_sigma_norm": sigma_mean,
        "ensemble_std_logrey_norm": ensemble_std,
        "mechanism_prob": mech_mean,
        "threshold_prob": thr_mean,
        "pred_mechanism_class_id": np.argmax(mech_mean, axis=1).astype(np.int16),
        "mechanism_entropy": entropy,
        "prediction_confidence": confidence,
    }


@torch.no_grad()
def _predict_fast_grid_all_ages(
    *,
    models: list[MechanismAwareREYRegressor],
    ckpts: list[dict[str, Any]],
    dataset: REYPointDataset,
    padded_env: np.ndarray,
    row_indices: np.ndarray,
    col_indices: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    ages: list[float],
    batch_size: int,
    age_batch_size: int,
    device: torch.device,
    use_amp: bool,
    channels_last: bool,
) -> list[dict[str, np.ndarray]]:
    first = ckpts[0]
    cfg = dict(first.get("model_config", {}) or {})
    thresholds = [float(x) for x in cfg.get("thresholds", [600, 800, 1000, 1500])]
    n_points = int(row_indices.size)
    n_batches = int(math.ceil(max(n_points, 1) / max(int(batch_size), 1)))
    n_age_layers = len(ages)
    age_batch = max(1, min(int(age_batch_size), max(int(n_age_layers), 1)))
    _emit_progress(
        stage="predict_fast_grid_spatial_batches",
        event="start",
        n_age_layers=int(n_age_layers),
        age_batch_size=int(age_batch),
        n_points=int(n_points),
        n_batches=int(n_batches),
        percent=0.0,
    )
    age_parts: list[dict[str, list[np.ndarray]]] = [
        {
            "mu": [],
            "sigma": [],
            "ensemble_std": [],
            "mech": [],
            "thr": [],
        }
        for _ in ages
    ]
    for batch_idx, start in enumerate(range(0, n_points, int(batch_size)), start=1):
        end = min(start + int(batch_size), n_points)
        static_patch, coord_patch = _extract_grid_static_patch_batch(
            dataset=dataset,
            padded_env=padded_env,
            rows=row_indices[start:end],
            cols=col_indices[start:end],
            lats=lats[start:end],
            lons=lons[start:end],
        )
        current_batch_size = int(end - start)
        for age_start in range(0, n_age_layers, age_batch):
            age_end = min(age_start + age_batch, n_age_layers)
            patch = torch.cat(
                [
                    _combine_static_patch_with_depth(
                        dataset=dataset,
                        static_patch=static_patch,
                        coord_patch=coord_patch,
                        age_ka=float(ages[age_idx]),
                    )
                    for age_idx in range(age_start, age_end)
                ],
                dim=0,
            )
            out = _predict_patch_outputs(
                patch=patch,
                models=models,
                device=device,
                use_amp=use_amp,
                channels_last=channels_last,
            )
            if device.type == "cuda":
                del patch
            for offset, age_idx in enumerate(range(age_start, age_end)):
                sl = slice(offset * current_batch_size, (offset + 1) * current_batch_size)
                parts = age_parts[age_idx]
                parts["mu"].append(out["mu"][sl])
                parts["sigma"].append(out["sigma"][sl])
                parts["ensemble_std"].append(out["ensemble_std"][sl])
                parts["mech"].append(out["mech"][sl])
                parts["thr"].append(out["thr"][sl])
        completed_points = int(end)
        if batch_idx == 1 or batch_idx == n_batches or batch_idx % 5 == 0:
            overall_fraction = completed_points / max(int(n_points), 1)
            _emit_progress(
                stage="predict_fast_grid_spatial_batches",
                event="batch",
                batch_index=int(batch_idx),
                n_batches=int(n_batches),
                completed_points=int(completed_points),
                n_points=int(n_points),
                n_age_layers=int(n_age_layers),
                age_batch_size=int(age_batch),
                percent=float(overall_fraction * 0.96),
            )
    outputs: list[dict[str, np.ndarray]] = []
    for age_idx, parts in enumerate(age_parts, start=1):
        outputs.append(
            _finalize_age_output(
                thresholds=thresholds,
                dataset=dataset,
                mu_mean_parts=parts["mu"],
                sigma_mean_parts=parts["sigma"],
                ensemble_std_parts=parts["ensemble_std"],
                mech_mean_parts=parts["mech"],
                thr_mean_parts=parts["thr"],
            )
        )
        _emit_progress(
            stage="predict_fast_grid_spatial_batches",
            event="finalize_age",
            age_ka=float(ages[age_idx - 1]),
            age_index=int(age_idx),
            n_age_layers=int(n_age_layers),
            percent=float(0.96 + 0.005 * age_idx / max(n_age_layers, 1)),
        )
    _emit_progress(stage="predict_fast_grid_spatial_batches", event="done", percent=0.96)
    return outputs


def _parse_ages(value: str) -> list[float]:
    ages = [float(part.strip()) for part in str(value).split(",") if part.strip()]
    if not ages:
        raise ValueError("At least one age layer is required.")
    return ages


def _plot_summary_maps(ds2: xr.Dataset, output_dir: Path) -> dict[str, str]:
    map_dir = output_dir / "maps"
    map_dir.mkdir(parents=True, exist_ok=True)
    variables = [
        ("max_pred_rey_ppm_0_2580ka", "Peak predicted REY ppm"),
        ("p95_pred_rey_ppm_0_2580ka", "P95 predicted REY ppm"),
        ("max_prob_rey_ge_1000ppm_0_2580ka", "Max P(REY >= 1000 ppm)"),
        ("max_prob_rey_ge_1500ppm_0_2580ka", "Max P(REY >= 1500 ppm)"),
        ("is_prob_rey_ge_1000ppm_0_2580ka", "High-value mask: P(REY >= 1000 ppm) >= 0.5"),
        ("is_prob_rey_ge_1500ppm_0_2580ka", "High-value mask: P(REY >= 1500 ppm) >= 0.5"),
        ("screening_score_ge1000", "Screening score >=1000 ppm"),
        ("screening_score_ge1500", "Screening score >=1500 ppm"),
        ("peak_age_ka_pred_rey", "Peak REY age ka"),
        ("dominant_mechanism_class_id_0_2580ka", "Dominant mechanism class"),
        ("dominant_mechanism_confidence_0_2580ka", "Dominant mechanism confidence"),
        ("prob_carbonate_background_mean_0_2580ka", "Mean P(carbonate background)"),
        ("prob_class1_enriched_mean_0_2580ka", "Mean P(class1 enriched)"),
        ("prob_class2_slow_sedimentation_mean_0_2580ka", "Mean P(class2 slow sedimentation)"),
        ("prediction_confidence_mean_0_2580ka", "Mean prediction confidence"),
    ]
    artifacts: dict[str, str] = {}
    regular_grid = "latitude" in ds2.coords and "longitude" in ds2.coords
    for var_name, title in variables:
        if var_name not in ds2:
            continue
        values = np.asarray(ds2[var_name].values, dtype=float)
        fig, ax = plt.subplots(figsize=(10, 5.2), dpi=180)
        if regular_grid and values.ndim == 2:
            lats = np.asarray(ds2["latitude"].values, dtype=float)
            lons = np.asarray(ds2["longitude"].values, dtype=float)
            extent = [float(np.nanmin(lons)), float(np.nanmax(lons)), float(np.nanmin(lats)), float(np.nanmax(lats))]
            image = ax.imshow(values, origin="lower", extent=extent, aspect="auto", cmap="viridis")
        else:
            lats = np.asarray(ds2["latitude"].values, dtype=float)
            lons = np.asarray(ds2["longitude"].values, dtype=float)
            image = ax.scatter(lons, lats, c=values, s=5, cmap="viridis", linewidths=0)
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        cbar = fig.colorbar(image, ax=ax, shrink=0.86)
        cbar.ax.set_ylabel(var_name)
        fig.tight_layout()
        out_path = map_dir / f"{var_name}.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        artifacts[var_name] = str(out_path)
    return artifacts


def main() -> None:
    setup_logger(level="INFO")
    parser = argparse.ArgumentParser(description="Predict continuous REY and mechanism/threshold maps on an env grid.")
    parser.add_argument("--model-run-dirs", nargs="+", default=DEFAULT_MODEL_RUNS)
    parser.add_argument("--env-nc", type=Path, default=Path(DEFAULT_ENV_NC))
    parser.add_argument("--sed-rate-nc", type=Path, default=Path(DEFAULT_SED_RATE_NC))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--age-layers-ka", default=DEFAULT_AGE_LAYERS_KA)
    parser.add_argument("--stride", type=int, default=10, help="Grid stride for reproducible preview/full-run control; 10 is about 1 degree, 1 keeps the native 0.1-degree env grid.")
    parser.add_argument(
        "--target-resolution-deg",
        type=float,
        default=0.0,
        help=(
            "Optional target horizontal resolution in degrees. When >0, the "
            "script samples the nearest native env-grid cells to this spacing "
            "and ignores --stride for grid selection."
        ),
    )
    parser.add_argument("--max-points-per-layer", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--age-batch-size",
        type=int,
        default=1,
        help=(
            "Number of age/depth layers concatenated into one CUDA forward pass "
            "inside each spatial batch when using fast_grid. Increase to 2-4 if "
            "GPU utilization is low; reduce to 1 on CUDA OOM."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers for CPU patch extraction. Use 0 if Windows multiprocessing causes issues.")
    parser.add_argument("--prefetch-factor", type=int, default=4, help="Batches prefetched per DataLoader worker when num_workers > 0.")
    parser.add_argument(
        "--patch-mode",
        choices=["auto", "fast_grid", "legacy_dataloader"],
        default="auto",
        help=(
            "Patch extraction backend. auto uses vectorised regular-grid extraction "
            "and falls back to the legacy DataLoader path when needed."
        ),
    )
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA automatic mixed precision inference.")
    parser.add_argument("--no-channels-last", action="store_true", help="Disable channels-last CUDA inference layout.")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    ckpts = [_load_checkpoint(Path(path)) for path in args.model_run_dirs]
    models = [_build_model(ckpt, device) for ckpt in ckpts]
    channels_last = bool(device.type == "cuda" and not bool(args.no_channels_last))
    if channels_last:
        models = [model.to(memory_format=torch.channels_last) for model in models]
    ages = _parse_ages(args.age_layers_ka)
    max_points = int(args.max_points_per_layer) if int(args.max_points_per_layer) > 0 else None
    _emit_progress(
        stage="setup",
        event="ready",
        n_models=len(models),
        n_age_layers=len(ages),
        stride=int(args.stride),
        target_resolution_deg=float(args.target_resolution_deg or 0.0),
        batch_size=int(args.batch_size),
        age_batch_size=int(args.age_batch_size),
        num_workers=int(args.num_workers),
        prefetch_factor=int(args.prefetch_factor),
        patch_mode=str(args.patch_mode),
        device=str(device),
        channels_last=bool(channels_last),
        percent=0.0,
    )

    age_outputs: list[dict[str, np.ndarray]] = []
    lat_ref: np.ndarray | None = None
    lon_ref: np.ndarray | None = None
    grid_shape_ref: np.ndarray | None = None
    depth_ref: list[np.ndarray] = []
    prediction_dataset: REYPointDataset | None = None
    _emit_progress(stage="make_grid_points", event="start", n_age_layers=len(ages), percent=0.0)
    points_path, flat_lat, flat_lon, flat_rows, flat_cols, sed_rate, grid_shape = _make_grid_geometry(
        env_nc=args.env_nc,
        sed_rate_nc=args.sed_rate_nc,
        initial_age_ka=float(ages[0]),
        stride=int(args.stride),
        target_resolution_deg=(float(args.target_resolution_deg) if float(args.target_resolution_deg or 0.0) > 0 else None),
        max_points=max_points,
        output_dir=output_dir,
    )
    lat_ref = flat_lat
    lon_ref = flat_lon
    grid_shape_ref = grid_shape
    prediction_dataset = _build_prediction_dataset(
        first=ckpts[0],
        env_nc=args.env_nc,
        points_path=points_path,
    )
    use_fast_grid = str(args.patch_mode) in ("auto", "fast_grid")
    padded_env: np.ndarray | None = None
    if use_fast_grid:
        padded_env = np.pad(
            prediction_dataset.env_data_np,
            ((0, 0), (prediction_dataset.pad_rows, prediction_dataset.pad_rows), (prediction_dataset.pad_cols, prediction_dataset.pad_cols)),
            mode="edge",
        )
    _emit_progress(
        stage="make_grid_points",
        event="done",
        n_points=int(lat_ref.size),
        regular_grid=bool(int(grid_shape_ref[0]) > 0 and int(grid_shape_ref[1]) > 0),
        patch_mode=("fast_grid" if use_fast_grid else "legacy_dataloader"),
        percent=0.01,
    )
    if use_fast_grid and padded_env is not None:
        depth_ref = [np.asarray(float(age) * sed_rate / 100.0, dtype=np.float64) for age in ages]
        age_outputs = _predict_fast_grid_all_ages(
            models=models,
            ckpts=ckpts,
            dataset=prediction_dataset,
            padded_env=padded_env,
            row_indices=flat_rows,
            col_indices=flat_cols,
            lats=lat_ref,
            lons=lon_ref,
            ages=[float(age) for age in ages],
            batch_size=int(args.batch_size),
            age_batch_size=int(args.age_batch_size),
            device=device,
            use_amp=not bool(args.no_amp),
            channels_last=bool(channels_last),
        )
    else:
        for age_index, age in enumerate(ages, start=1):
            depth = np.asarray(float(age) * sed_rate / 100.0, dtype=np.float64)
            _set_prediction_dataset_age(prediction_dataset, float(age))
            depth_ref.append(depth)
            age_outputs.append(
                _predict_age_layer(
                    models=models,
                    ckpts=ckpts,
                    dataset=prediction_dataset,
                    batch_size=int(args.batch_size),
                    num_workers=int(args.num_workers),
                    prefetch_factor=int(args.prefetch_factor),
                    device=device,
                    use_amp=not bool(args.no_amp),
                    channels_last=bool(channels_last),
                    age_ka=float(age),
                    age_index=int(age_index),
                    n_age_layers=len(ages),
                )
            )

    assert lat_ref is not None and lon_ref is not None
    age_coord = np.asarray(ages, dtype=float)
    point_coord = np.arange(lat_ref.size, dtype=int)
    thresholds = age_outputs[0]["thresholds"]
    regular_grid = (
        grid_shape_ref is not None
        and int(grid_shape_ref[0]) > 0
        and int(grid_shape_ref[1]) > 0
        and int(grid_shape_ref[0] * grid_shape_ref[1]) == int(point_coord.size)
    )

    def stack_layer(name: str, dtype=np.float32) -> np.ndarray:
        arr = np.vstack([x[name] for x in age_outputs]).astype(dtype)
        if regular_grid:
            return arr.reshape(len(age_coord), int(grid_shape_ref[0]), int(grid_shape_ref[1]))
        return arr

    if regular_grid:
        nlat, nlon = int(grid_shape_ref[0]), int(grid_shape_ref[1])
        y_dim: tuple[str, ...] = ("age_ka", "latitude", "longitude")
        xy_dim: tuple[str, ...] = ("latitude", "longitude")
        coords: dict[str, np.ndarray] = {
            "age_ka": age_coord,
            "latitude": lat_ref.reshape(nlat, nlon)[:, 0].astype(np.float32),
            "longitude": lon_ref.reshape(nlat, nlon)[0, :].astype(np.float32),
        }
        depth_arr = np.vstack(depth_ref).reshape(len(age_coord), nlat, nlon).astype(np.float32)
    else:
        y_dim = ("age_ka", "point")
        xy_dim = ("point",)
        coords = {"age_ka": age_coord, "point": point_coord}
        depth_arr = np.vstack(depth_ref).astype(np.float32)

    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
        "depth_mbsf": (y_dim, depth_arr),
        "pred_logrey": (y_dim, stack_layer("pred_logrey")),
        "pred_rey_ppm": (y_dim, stack_layer("pred_rey_ppm")),
        "pred_sigma_norm": (y_dim, stack_layer("pred_sigma_norm")),
        "ensemble_std_logrey_norm": (y_dim, stack_layer("ensemble_std_logrey_norm")),
        "pred_mechanism_class_id": (y_dim, stack_layer("pred_mechanism_class_id", np.int16)),
        "mechanism_entropy": (y_dim, stack_layer("mechanism_entropy")),
        "prediction_confidence": (y_dim, stack_layer("prediction_confidence")),
    }
    if not regular_grid:
        data_vars["latitude"] = (("point",), lat_ref.astype(np.float32))
        data_vars["longitude"] = (("point",), lon_ref.astype(np.float32))
    mech_stack = np.stack([x["mechanism_prob"] for x in age_outputs], axis=0)
    if regular_grid:
        mech_stack = mech_stack.reshape(len(age_coord), int(grid_shape_ref[0]), int(grid_shape_ref[1]), len(MECHANISM_CLASS_NAMES))
    for idx, name in enumerate(MECHANISM_CLASS_NAMES):
        data_vars[f"prob_{name}"] = (y_dim, mech_stack[..., idx].astype(np.float32))
    thr_stack = np.stack([x["threshold_prob"] for x in age_outputs], axis=0)
    if regular_grid:
        thr_stack = thr_stack.reshape(len(age_coord), int(grid_shape_ref[0]), int(grid_shape_ref[1]), len(thresholds))
    for idx, threshold in enumerate(thresholds):
        data_vars[f"prob_rey_ge_{int(threshold)}ppm"] = (y_dim, thr_stack[..., idx].astype(np.float32))
    prob1000_3d = thr_stack[..., list(thresholds).index(1000.0)] if 1000.0 in list(thresholds) else np.zeros_like(stack_layer("pred_rey_ppm"))
    prob1500_3d = thr_stack[..., list(thresholds).index(1500.0)] if 1500.0 in list(thresholds) else np.zeros_like(stack_layer("pred_rey_ppm"))
    pred_rey_3d = stack_layer("pred_rey_ppm")
    data_vars["is_prob_rey_ge_1000ppm"] = (y_dim, (prob1000_3d >= 0.5).astype(np.int16))
    data_vars["is_prob_rey_ge_1500ppm"] = (y_dim, (prob1500_3d >= 0.5).astype(np.int16))
    data_vars["is_pred_rey_ge_1000ppm"] = (y_dim, (pred_rey_3d >= 1000.0).astype(np.int16))
    data_vars["is_pred_rey_ge_1500ppm"] = (y_dim, (pred_rey_3d >= 1500.0).astype(np.int16))
    data_vars["high_value_tier_id"] = (
        y_dim,
        np.where(prob1500_3d >= 0.5, 2, np.where(prob1000_3d >= 0.5, 1, 0)).astype(np.int16),
    )

    ds3 = xr.Dataset(data_vars=data_vars, coords=coords)
    ds3.attrs.update(
        {
            "workflow": "mechanism-aware continuous REY regression",
            "model_run_dirs": json.dumps([str(x) for x in args.model_run_dirs]),
            "env_nc": str(args.env_nc),
            "sed_rate_nc": str(args.sed_rate_nc),
            "grid_stride": int(args.stride),
            "target_resolution_deg": float(args.target_resolution_deg or 0.0),
            "batch_size": int(args.batch_size),
            "age_batch_size": int(args.age_batch_size),
            "prefetch_factor": int(args.prefetch_factor),
            "patch_mode": "fast_grid" if use_fast_grid else "legacy_dataloader",
            "cuda_amp_enabled": int(bool(device.type == "cuda" and not bool(args.no_amp))),
            "channels_last_enabled": int(bool(channels_last)),
            "prediction_dataset_reused": 1,
            "regular_lat_lon_grid": int(bool(regular_grid)),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    nc3 = output_dir / "continuous_rey_mechanism_threshold_predictions_3d.nc"
    _emit_progress(stage="write_netcdf", event="3d_start", percent=0.965)
    ds3.to_netcdf(nc3, engine='h5netcdf')

    pred_rey = ds3["pred_rey_ppm"].values
    prob1000 = ds3["prob_rey_ge_1000ppm"].values if "prob_rey_ge_1000ppm" in ds3 else np.zeros_like(pred_rey)
    prob1500 = ds3["prob_rey_ge_1500ppm"].values if "prob_rey_ge_1500ppm" in ds3 else np.zeros_like(pred_rey)
    peak_idx = np.nanargmax(pred_rey, axis=0)
    prob1500_peak_idx = np.nanargmax(prob1500, axis=0)
    prob_mech_mean = np.stack(
        [
            np.nanmean(ds3["prob_carbonate_background"].values, axis=0),
            np.nanmean(ds3["prob_class1_enriched"].values, axis=0),
            np.nanmean(ds3["prob_class2_slow_sedimentation"].values, axis=0),
        ],
        axis=0,
    )
    dominant_mech = np.nanargmax(prob_mech_mean, axis=0).astype(np.int16)
    dominant_mech_conf = np.nanmax(prob_mech_mean, axis=0).astype(np.float32)
    max_pred_rey = np.nanmax(pred_rey, axis=0)
    max_prob1000 = np.nanmax(prob1000, axis=0)
    max_prob1500 = np.nanmax(prob1500, axis=0)
    summary_vars = {
        "max_pred_rey_ppm_0_2580ka": (xy_dim, max_pred_rey.astype(np.float32)),
        "p95_pred_rey_ppm_0_2580ka": (xy_dim, np.nanpercentile(pred_rey, 95, axis=0).astype(np.float32)),
        "mean_pred_rey_ppm_0_2580ka": (xy_dim, np.nanmean(pred_rey, axis=0).astype(np.float32)),
        "max_prob_rey_ge_1000ppm_0_2580ka": (xy_dim, max_prob1000.astype(np.float32)),
        "max_prob_rey_ge_1500ppm_0_2580ka": (xy_dim, max_prob1500.astype(np.float32)),
        "is_prob_rey_ge_1000ppm_0_2580ka": (xy_dim, (max_prob1000 >= 0.5).astype(np.int16)),
        "is_prob_rey_ge_1500ppm_0_2580ka": (xy_dim, (max_prob1500 >= 0.5).astype(np.int16)),
        "is_peak_pred_rey_ge_1000ppm_0_2580ka": (xy_dim, (max_pred_rey >= 1000.0).astype(np.int16)),
        "is_peak_pred_rey_ge_1500ppm_0_2580ka": (xy_dim, (max_pred_rey >= 1500.0).astype(np.int16)),
        "high_value_tier_id_0_2580ka": (xy_dim, np.where(max_prob1500 >= 0.5, 2, np.where(max_prob1000 >= 0.5, 1, 0)).astype(np.int16)),
        "peak_age_ka_pred_rey": (xy_dim, age_coord[peak_idx].astype(np.float32)),
        "peak_age_ka_prob_ge_1500ppm": (xy_dim, age_coord[prob1500_peak_idx].astype(np.float32)),
        "dominant_mechanism_class_id_0_2580ka": (xy_dim, dominant_mech),
        "dominant_mechanism_confidence_0_2580ka": (xy_dim, dominant_mech_conf),
        "prob_carbonate_background_mean_0_2580ka": (xy_dim, prob_mech_mean[0].astype(np.float32)),
        "prob_class1_enriched_mean_0_2580ka": (xy_dim, np.nanmean(ds3["prob_class1_enriched"].values, axis=0).astype(np.float32)),
        "prob_class2_slow_sedimentation_mean_0_2580ka": (xy_dim, np.nanmean(ds3["prob_class2_slow_sedimentation"].values, axis=0).astype(np.float32)),
        "prediction_confidence_mean_0_2580ka": (xy_dim, np.nanmean(ds3["prediction_confidence"].values, axis=0).astype(np.float32)),
        "screening_score_ge1000": (xy_dim, (max_prob1000 * np.nanmean(ds3["prediction_confidence"].values, axis=0)).astype(np.float32)),
        "screening_score_ge1500": (xy_dim, (max_prob1500 * np.nanmean(ds3["prediction_confidence"].values, axis=0)).astype(np.float32)),
    }
    if not regular_grid:
        summary_vars["latitude"] = (("point",), lat_ref.astype(np.float32))
        summary_vars["longitude"] = (("point",), lon_ref.astype(np.float32))
    summary_coords = (
        {
            "latitude": coords["latitude"],
            "longitude": coords["longitude"],
        }
        if regular_grid
        else {"point": point_coord}
    )
    ds2 = xr.Dataset(data_vars=summary_vars, coords=summary_coords)
    ds2.attrs.update(ds3.attrs)
    ds2.attrs["screening_score_definition"] = "max threshold probability over age layers multiplied by mean prediction confidence."
    nc2 = output_dir / "continuous_rey_paper_summary_2d.nc"
    _emit_progress(stage="write_netcdf", event="2d_start", percent=0.975)
    ds2.to_netcdf(nc2, engine='h5netcdf')
    _emit_progress(stage="plot_maps", event="start", percent=0.98)
    map_artifacts = _plot_summary_maps(ds2, output_dir)
    _emit_progress(stage="plot_maps", event="done", percent=0.99, n_maps=len(map_artifacts))

    summary = {
        "status": "completed",
        "output_dir": str(output_dir),
        "prediction_3d_nc": str(nc3),
        "summary_2d_nc": str(nc2),
        "n_models": len(models),
        "n_age_layers": len(ages),
        "n_points": int(point_coord.size),
        "n_model_evaluations": int(point_coord.size * len(ages) * len(models)),
        "regular_lat_lon_grid": bool(regular_grid),
        "age_layers_ka": ages,
        "thresholds": thresholds.tolist(),
        "batch_size": int(args.batch_size),
        "age_batch_size": int(args.age_batch_size),
        "target_resolution_deg": float(args.target_resolution_deg or 0.0),
        "num_workers": int(args.num_workers),
        "prefetch_factor": int(args.prefetch_factor),
        "patch_mode": "fast_grid" if use_fast_grid else "legacy_dataloader",
        "cuda_amp_enabled": bool(device.type == "cuda" and not bool(args.no_amp)),
        "channels_last_enabled": bool(channels_last),
        "prediction_dataset_reused": True,
        "device": str(device),
        "map_pngs": map_artifacts,
    }
    _write_json(output_dir / "continuous_rey_prediction_summary.json", summary)
    _emit_progress(stage="completed", event="done", percent=1.0, output_dir=str(output_dir))
    print(json.dumps(summary, cls=NumpyJSONEncoder, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

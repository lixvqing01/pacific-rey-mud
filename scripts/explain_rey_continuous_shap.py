from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import numpy as np
import pandas as pd
import shap
import torch
import xarray as xr
from torch.utils.data import DataLoader, Subset

from core.datasets.rey_dataset import REYPointDataset
from core.models.rey_multitask import MechanismAwareREYRegressor
from core.utils.logger_config import setup_logger
from core.utils.serialization_utils import NumpyJSONEncoder
from core.visualization.shap_plots import plot_shap_summary_optimized
from scripts.run_rey_continuous_multitask import _make_base_dataset, _prepare_points


DEFAULT_RUN_DIR = Path("models/seed310")
DEFAULT_ENV_NC = Path("data/env_gridded.nc")
DEFAULT_POINTS = Path("data/geochem_class_table.parquet")
DEFAULT_SED_RATE_NC = Path("data/Average sedimentation rate.nc")


PROCESS_COLORS = {
    "Tectono-Geomorphic Setting": "#D6B85A",
    "Biogeochemical Environment": "#E38A8A",
    "Ocean Circulation & Transport Dynamics": "#7FA6D6",
    "Sedimentary Burial & Stratigraphic Context": "#7CBF9E",
    "Benthic Ecological Response": "#9C8DC2",
    "Other": "#C7C7C7",
}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, cls=NumpyJSONEncoder, indent=2, ensure_ascii=False)


def _feature_category(feature_name: str) -> str:
    name = str(feature_name)
    if "__cat_" in name:
        name = name.split("__cat_", 1)[0]
    key = name.strip().lower()
    if "geomorphic features" in key or "seafloor lithology" in key:
        return "Tectono-Geomorphic Setting"
    if (
        "water depth" in key
        or "gravity anomaly" in key
        or "magnetic anomaly" in key
        or "vertical gravity gradient" in key
        or "distance_to_seamount" in key
        or "distance_to_guyot" in key
        or "distance_to_trench" in key
        or "distance_to_abyss" in key
        or "distance_to_spreading_ridge" in key
        or "distance_to_basin" in key
    ):
        return "Tectono-Geomorphic Setting"
    if "age_ka_proxy" in key or "average sedimentation rate" in key or "sediment thickness" in key:
        return "Sedimentary Burial & Stratigraphic Context"
    if "bottom current speed sd" in key or "del3he" in key:
        return "Ocean Circulation & Transport Dynamics"
    if (
        "phosphate" in key
        or "total organic carbon" in key
        or "toc" in key
        or "summer primary productivity" in key
        or "bottom water dissolved oxygen" in key
        or "oxygen" in key
    ):
        return "Biogeochemical Environment"
    if "macrobenthic biomass" in key:
        return "Benthic Ecological Response"
    return "Other"


class LogREYOutputWrapper(torch.nn.Module):
    def __init__(self, model: MechanismAwareREYRegressor, norm_params: dict[str, Any]) -> None:
        super().__init__()
        self.model = model
        rey_cfg = norm_params.get("rey", {}) if isinstance(norm_params.get("rey"), dict) else {}
        self.mean = float(rey_cfg.get("mean", 0.0))
        self.std = float(rey_cfg.get("std", 1.0) or 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        mu = torch.nan_to_num(out["mu"], nan=0.0, posinf=8.0, neginf=-8.0).clamp(-8.0, 8.0)
        return (mu * self.std) + self.mean


def _load_checkpoint(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "models" / "mechanism_aware_rey_regressor_best.pth"
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def _build_model(ckpt: dict[str, Any], device: torch.device) -> MechanismAwareREYRegressor:
    cfg = dict(ckpt.get("model_config", {}) or {})
    model = MechanismAwareREYRegressor(
        input_channels=int(cfg["input_channels"]),
        n_mechanisms=len(cfg.get("mechanism_class_names", ["carbonate_background", "class1_enriched", "class2_slow_sedimentation"])),
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


def _load_split_indices(run_dir: Path, split: str, n: int) -> np.ndarray:
    path = run_dir / "splits" / f"{split}_indices.npy"
    if not path.exists():
        return np.arange(n, dtype=int)
    idx = np.load(path).astype(int)
    return idx[(idx >= 0) & (idx < n)]


def _select_indices(indices: np.ndarray, n_select: int, seed: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=int)
    if indices.size <= int(n_select):
        return indices
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(indices, size=int(n_select), replace=False))


def _make_loader(dataset: REYPointDataset, indices: np.ndarray, batch_size: int) -> DataLoader:
    subset = Subset(dataset, [int(i) for i in indices])
    return DataLoader(subset, batch_size=int(batch_size), shuffle=False, num_workers=0)


def _collect_patches(loader: DataLoader, max_batches: int | None = None) -> torch.Tensor:
    patches = []
    for batch_i, batch in enumerate(loader):
        patches.append(batch[0].float())
        if max_batches is not None and batch_i + 1 >= int(max_batches):
            break
    if not patches:
        raise ValueError("No patches available for SHAP.")
    return torch.cat(patches, dim=0)


def main() -> None:
    setup_logger(level="INFO")
    parser = argparse.ArgumentParser(description="Run SHAP explanation for the continuous REY baseline.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--env-nc", type=Path, default=DEFAULT_ENV_NC)
    parser.add_argument("--points", type=Path, default=DEFAULT_POINTS)
    parser.add_argument("--sed-rate-nc", type=Path, default=DEFAULT_SED_RATE_NC)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--background-samples", type=int, default=48)
    parser.add_argument("--explain-samples", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=310)
    parser.add_argument("--top-n", type=int, default=14)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = _load_checkpoint(Path(args.run_dir))
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = _build_model(ckpt, device)
    wrapped = LogREYOutputWrapper(model, dict(ckpt["norm_params"])).to(device).eval()

    points_prepared = _prepare_points(Path(args.points), output_dir)
    feature_names = [str(x) for x in ckpt["feature_names"]]
    dataset = _make_base_dataset(
        points_path=points_prepared,
        env_nc=Path(args.env_nc),
        feature_names=feature_names,
        patch_size=11,
        sed_rate_nc=Path(args.sed_rate_nc),
        depth_feature_type="age_ka_proxy",
        target_transform_name="log1p",
        norm_params_override=dict(ckpt["norm_params"]),
        use_target_weights=False,
        max_weight=5.0,
        augment=False,
        shift_pixels=0,
        add_coord_channels=False,
    )

    train_idx = _select_indices(_load_split_indices(Path(args.run_dir), "train", len(dataset)), int(args.background_samples), int(args.seed))
    test_idx = _select_indices(_load_split_indices(Path(args.run_dir), "test", len(dataset)), int(args.explain_samples), int(args.seed) + 1)
    background = _collect_patches(_make_loader(dataset, train_idx, int(args.batch_size))).to(device)
    explain = _collect_patches(_make_loader(dataset, test_idx, int(args.batch_size))).to(device)

    explainer = shap.GradientExplainer(wrapped, background)
    explain.requires_grad_(True)
    shap_values = explainer.shap_values(explain)
    if isinstance(shap_values, (list, tuple)):
        shap_values = shap_values[0]
    shap_arr = np.asarray(shap_values)
    if shap_arr.ndim == 5 and shap_arr.shape[-1] == 1:
        shap_arr = np.squeeze(shap_arr, axis=-1)
    if shap_arr.ndim != 4:
        raise ValueError(f"Unexpected SHAP shape: {shap_arr.shape}")
    shap_agg = np.nanmean(shap_arr, axis=(2, 3))
    x_agg = explain.detach().cpu().numpy().mean(axis=(2, 3))
    all_feature_names = feature_names + ["age_ka_proxy"]
    if shap_agg.shape[1] != len(all_feature_names):
        all_feature_names = [f"ch_{i}" for i in range(shap_agg.shape[1])]

    shap_df = pd.DataFrame(shap_agg, columns=all_feature_names)
    x_df = pd.DataFrame(x_agg, columns=all_feature_names)
    importance = (
        pd.DataFrame({"feature": all_feature_names, "mean_abs_shap": np.nanmean(np.abs(shap_agg), axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance["process_group"] = importance["feature"].map(_feature_category)
    importance.to_csv(output_dir / "continuous_rey_shap_feature_importance.csv", index=False)
    shap_df.to_csv(output_dir / "continuous_rey_shap_values_aggregated.csv", index=False)
    x_df.to_csv(output_dir / "continuous_rey_shap_feature_values_aggregated.csv", index=False)

    category_map = {name: _feature_category(name) for name in all_feature_names}
    for name in all_feature_names:
        if "__cat_" in name:
            category_map.setdefault(name.split("__cat_", 1)[0].strip(), _feature_category(name))

    plot_out = plot_shap_summary_optimized(
        X_df=x_df,
        shap_values_matrix=shap_df.to_numpy(dtype=float),
        category_map=category_map,
        colors_map=PROCESS_COLORS,
        figsize=(12, 7),
        save_path=output_dir / "continuous_rey_shap_summary_optimized",
        dpi=300,
        top_n=int(args.top_n),
        plot_config={"export_formats": ["png", "svg"], "dpi": 300},
        title="Environmental controls on predicted log(REY)",
    )

    group_summary = (
        importance.groupby("process_group", dropna=False)["mean_abs_shap"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    total = float(group_summary["mean_abs_shap"].sum())
    group_summary["fraction"] = group_summary["mean_abs_shap"] / total if total > 0 else np.nan
    group_summary.to_csv(output_dir / "continuous_rey_shap_process_group_importance.csv", index=False)

    summary = {
        "status": "completed",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "run_dir": str(args.run_dir),
        "target": "normalized logREY transformed back to log1p(REY) space",
        "device": str(device),
        "n_background": int(background.shape[0]),
        "n_explain": int(explain.shape[0]),
        "n_features": int(shap_agg.shape[1]),
        "artifacts": {
            "optimized_shap_summary": plot_out.get("primary_path"),
            "optimized_shap_summary_all": plot_out.get("plot_paths", []),
            "feature_importance_csv": str(output_dir / "continuous_rey_shap_feature_importance.csv"),
            "process_group_importance_csv": str(output_dir / "continuous_rey_shap_process_group_importance.csv"),
            "shap_values_csv": str(output_dir / "continuous_rey_shap_values_aggregated.csv"),
            "feature_values_csv": str(output_dir / "continuous_rey_shap_feature_values_aggregated.csv"),
        },
    }
    _write_json(output_dir / "continuous_rey_shap_summary.json", summary)
    print(json.dumps(summary, cls=NumpyJSONEncoder, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

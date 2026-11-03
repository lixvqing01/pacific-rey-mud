from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
)

from core.utils.serialization_utils import NumpyJSONEncoder


MECHANISM_CLASS_NAMES = [
    "carbonate_background",
    "class1_enriched",
    "class2_slow_sedimentation",
]
FACIES_LABELS = ["Facies I", "Facies II", "Facies III"]
MECHANISM_COLORS = {
    0: "#8F969E",
    1: "#D65F5F",
    2: "#4C8FC4",
}
PANEL_LABELS = ["a", "b", "c", "d"]
THRESHOLDS = [600.0, 800.0, 1000.0, 1500.0]
DEFAULT_REPEATED_EVAL_DIR = Path("outputs/balanced_repeated_evaluation")
DEFAULT_REPEATED_SPLIT_DIR = Path("outputs/balanced_repeated_core_splits")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, cls=NumpyJSONEncoder, indent=2, ensure_ascii=False)


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12.5,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 11.5,
            "legend.fontsize": 10.8,
            "axes.linewidth": 0.9,
            "axes.edgecolor": "#333333",
            "axes.titlepad": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.dpi": 300,
        }
    )


def _style_axis(ax: plt.Axes, *, grid_axis: str = "both") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.tick_params(axis="both", colors="#222222", length=3.5, width=0.8)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color="#D8D8D8", linewidth=0.55, alpha=0.58)
    ax.set_axisbelow(True)


def _add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
        color="#111111",
    )


def _facies_label(cls: int | float | str) -> str:
    try:
        idx = int(cls)
    except Exception:
        return str(cls)
    return FACIES_LABELS[idx] if 0 <= idx < len(FACIES_LABELS) else f"Facies {idx}"


def _save_figure(fig: plt.Figure, stem: str, output_dir: Path) -> dict[str, str]:
    png = output_dir / f"{stem}.png"
    svg = output_dir / f"{stem}.svg"
    fig.savefig(png, bbox_inches="tight", dpi=300)
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png), "svg": str(svg)}


def _load_test_predictions(ensemble_dir: Path) -> pd.DataFrame:
    path = ensemble_dir / "ensemble_test_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing test prediction table: {path}")
    df = pd.read_csv(path)
    if "rey_true" not in df.columns or "rey_pred" not in df.columns:
        raise ValueError(f"{path} must contain rey_true and rey_pred columns.")
    if "mechanism_true" not in df.columns or "mechanism_pred" not in df.columns:
        raise ValueError(f"{path} must contain mechanism_true and mechanism_pred columns.")
    return df


def _load_seed_summary(ensemble_dir: Path) -> pd.DataFrame:
    path = ensemble_dir / "single_seed_test_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing single-seed summary: {path}")
    return pd.read_csv(path)


def _load_ensemble_metrics(ensemble_dir: Path) -> dict[str, Any]:
    path = ensemble_dir / "ensemble_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing ensemble metrics: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_repeated_metrics(repeated_eval_dir: Path) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    metrics_path = repeated_eval_dir / "repeated_test_metrics.csv"
    stats_path = repeated_eval_dir / "repeated_test_metric_stats.csv"
    if not metrics_path.exists() or not stats_path.exists():
        return None, None
    return pd.read_csv(metrics_path), pd.read_csv(stats_path)


def _ensure_repeated_eval(
    repeated_eval_dir: Path,
    repeated_split_dir: Path,
    ensemble_dir: Path,
) -> Path | None:
    metrics_path = repeated_eval_dir / "repeated_test_metrics.csv"
    if metrics_path.exists():
        return repeated_eval_dir
    if not repeated_split_dir.exists():
        return None
    from scripts.evaluate_rey_repeated_core_splits import evaluate_repeated_splits

    repeated_eval_dir.mkdir(parents=True, exist_ok=True)
    args = argparse.Namespace(
        ensemble_dir=ensemble_dir,
        repeated_split_dir=repeated_split_dir,
        output_dir=repeated_eval_dir,
        thresholds="600,800,1000,1500",
    )
    evaluate_repeated_splits(args)
    return repeated_eval_dir


def _prepare_test_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_rey_true"] = np.log1p(np.clip(out["rey_true"].to_numpy(dtype=float), 0.0, None))
    out["log_rey_pred"] = np.log1p(np.clip(out["rey_pred"].to_numpy(dtype=float), 0.0, None))
    out["residual_logrey"] = out["log_rey_pred"] - out["log_rey_true"]
    out["abs_residual_logrey"] = np.abs(out["residual_logrey"])
    out["rey_bin"] = pd.cut(
        out["rey_true"],
        bins=[0, 600, 800, 1000, 1500, np.inf],
        labels=["<600", "600-800", "800-1000", "1000-1500", ">=1500"],
        include_lowest=True,
        right=False,
    )
    out["high_1000"] = (out["rey_true"] >= 1000.0).astype(int)
    out["high_1500"] = (out["rey_true"] >= 1500.0).astype(int)
    out["facies_true_label"] = out["mechanism_true"].map(_facies_label)
    out["facies_pred_label"] = out["mechanism_pred"].map(_facies_label)
    return out


def _threshold_arrays(df: pd.DataFrame, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    y_true = (df["rey_true"].to_numpy(dtype=float) >= threshold).astype(int)
    y_score = df[f"prob_rey_ge_{int(threshold)}"].to_numpy(dtype=float)
    return y_true, y_score


def _plot_regression_and_residuals(df: pd.DataFrame) -> plt.Figure:
    _set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 10.8), dpi=180, constrained_layout=True)
    ax = axes[0, 0]
    for cls in sorted(df["mechanism_true"].dropna().unique()):
        sub = df[df["mechanism_true"] == cls]
        ax.scatter(
            sub["log_rey_true"],
            sub["log_rey_pred"],
            s=20,
            alpha=0.64,
            color=MECHANISM_COLORS.get(int(cls), "#666666"),
            label=_facies_label(cls),
            edgecolors="none",
            rasterized=True,
        )
    lims = [
        float(np.nanmin([df["log_rey_true"].min(), df["log_rey_pred"].min()])),
        float(np.nanmax([df["log_rey_true"].max(), df["log_rey_pred"].max()])),
    ]
    ax.plot(lims, lims, color="#1A1A1A", lw=1.05, ls="--", label="1:1")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Observed log1p(REY)")
    ax.set_ylabel("Predicted log1p(REY)")
    ax.set_title("Observed vs. predicted REY")
    ax.legend(frameon=False, loc="lower right", handletextpad=0.35, borderaxespad=0.3)
    _style_axis(ax)
    _add_panel_label(ax, PANEL_LABELS[0])
    ax.text(
        0.02,
        0.98,
        f"n={len(df)}\nMAE={df['residual_logrey'].abs().mean():.3f}\n"
        f"Spearman={df[['log_rey_true','log_rey_pred']].corr(method='spearman').iloc[0,1]:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#BBBBBB", alpha=0.94),
    )

    ax = axes[0, 1]
    for cls in sorted(df["mechanism_true"].dropna().unique()):
        sub = df[df["mechanism_true"] == cls]
        ax.scatter(
            sub["log_rey_true"],
            sub["residual_logrey"],
            s=18,
            alpha=0.44,
            color=MECHANISM_COLORS.get(int(cls), "#666666"),
            edgecolors="none",
            rasterized=True,
        )
    ax.axhline(0.0, color="#1A1A1A", lw=1.0)
    ax.set_xlabel("Observed log1p(REY)")
    ax.set_ylabel("Residual (pred - obs)")
    ax.set_title("Residual by grade")
    _style_axis(ax)
    _add_panel_label(ax, PANEL_LABELS[1])

    ax = axes[1, 0]
    bins = np.linspace(df["residual_logrey"].min(), df["residual_logrey"].max(), 28)
    for cls in sorted(df["mechanism_true"].dropna().unique()):
        sub = df[df["mechanism_true"] == cls]
        ax.hist(
            sub["residual_logrey"],
            bins=bins,
            alpha=0.36,
            density=True,
            color=MECHANISM_COLORS.get(int(cls), "#666666"),
            label=_facies_label(cls),
        )
    ax.axvline(0.0, color="#1A1A1A", lw=1.0)
    ax.set_xlabel("Residual (pred - obs)")
    ax.set_ylabel("Density")
    ax.set_title("Residual distribution by facies")
    ax.legend(frameon=False, loc="upper left", handlelength=1.3)
    _style_axis(ax, grid_axis="y")
    _add_panel_label(ax, PANEL_LABELS[2])

    ax = axes[1, 1]
    grouped = df.groupby("rey_bin", observed=True)["residual_logrey"].agg(["mean", "median", "count"]).reset_index()
    x = np.arange(len(grouped))
    ax.bar(x, grouped["mean"], color="#8FB9B6", alpha=0.82, label="Mean")
    ax.plot(x, grouped["median"], color="#C85C2E", marker="o", markersize=5, lw=1.7, label="Median")
    ax.axhline(0.0, color="#1A1A1A", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(grouped["rey_bin"].astype(str), rotation=18, ha="right")
    ax.set_ylabel("Residual")
    ax.set_title("Mean / median residual by REY band")
    _style_axis(ax, grid_axis="y")
    _add_panel_label(ax, PANEL_LABELS[3])
    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin
    offset = 0.035 * yrange
    for xi, mean_value, cnt in zip(x, grouped["mean"], grouped["count"]):
        if mean_value >= 0:
            y_text = mean_value + offset
            va = "bottom"
        else:
            y_text = mean_value - offset
            va = "top"
        ax.text(
            xi,
            y_text,
            f"n={int(cnt)}",
            ha="center",
            va=va,
            fontsize=9.5,
            rotation=0,
            color="#333333",
        )
    ymin2 = min(ymin, float(grouped["mean"].min()) - offset * 2.2)
    ymax2 = max(ymax, float(grouped["mean"].max()) + offset * 2.2)
    ax.set_ylim(ymin2, ymax2)
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index("Mean"), labels.index("Median")] if "Mean" in labels and "Median" in labels else range(len(labels))
    ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        frameon=False,
        loc="upper right",
        ncol=2,
        handlelength=1.4,
        columnspacing=1.0,
    )

    return fig


def _plot_threshold_screening(df: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame]:
    _set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 10.0), dpi=180, constrained_layout=True)
    rows = []
    for idx, threshold in enumerate([1000.0, 1500.0]):
        y_true, y_score = _threshold_arrays(df, threshold)
        ap = average_precision_score(y_true, y_score)
        brier = brier_score_loss(y_true, y_score)
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        frac_pos, mean_pred = calibration_curve(y_true, y_score, n_bins=10, strategy="quantile")
        row = {
            "threshold": threshold,
            "ap": float(ap),
            "brier": float(brier),
            "positive_support": int(y_true.sum()),
            "negative_support": int((1 - y_true).sum()),
        }
        rows.append(row)

        ax = axes[0, idx]
        ax.plot(recall, precision, color="#35699A", lw=2.4)
        ax.axhline(y_true.mean(), color="#777777", lw=1.1, ls="--")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"REY >= {int(threshold)} ppm: precision-recall")
        ax.text(
            0.04,
            0.07,
            f"AP = {ap:.3f}\nPositive n = {int(y_true.sum())}",
            transform=ax.transAxes,
            fontsize=10.5,
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#C7C7C7", alpha=0.94),
        )
        _style_axis(ax)
        _add_panel_label(ax, PANEL_LABELS[idx])

        ax = axes[1, idx]
        ax.plot([0, 1], [0, 1], color="#222222", lw=1.0, ls="--")
        ax.plot(mean_pred, frac_pos, marker="o", markersize=5.2, color="#C85C2E", lw=2.2)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"REY >= {int(threshold)} ppm: calibration")
        ax.text(
            0.04,
            0.07,
            f"Brier = {brier:.3f}",
            transform=ax.transAxes,
            fontsize=10.5,
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#C7C7C7", alpha=0.94),
        )
        _style_axis(ax)
        _add_panel_label(ax, PANEL_LABELS[idx + 2])

    return fig, pd.DataFrame(rows)


def _plot_mechanism_diagnostics(df: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame]:
    _set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.3), dpi=180, constrained_layout=True)
    labels = list(range(len(MECHANISM_CLASS_NAMES)))
    cm = confusion_matrix(df["mechanism_true"], df["mechanism_pred"], labels=labels)
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    im = axes[0].imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    axes[0].set_xticks(labels)
    axes[0].set_yticks(labels)
    axes[0].set_xticklabels(FACIES_LABELS)
    axes[0].set_yticklabels(FACIES_LABELS)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Observed")
    axes[0].set_title("Facies confusion matrix")
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            color = "white" if cm_norm[i, j] > 0.55 else "#1A1A1A"
            axes[0].text(j, i, f"{cm_norm[i, j]:.2f}\nn={cm[i, j]}", ha="center", va="center", fontsize=11, color=color)
    cbar = fig.colorbar(im, ax=axes[0], shrink=0.84, pad=0.025)
    cbar.set_label("Row-normalized fraction", fontsize=11.5)
    axes[0].tick_params(length=0)
    _add_panel_label(axes[0], "a")

    rows = []
    width = 0.25
    x = np.arange(len(MECHANISM_CLASS_NAMES))
    for cls in labels:
        y_true = (df["mechanism_true"].to_numpy(dtype=int) == cls).astype(int)
        y_pred = (df["mechanism_pred"].to_numpy(dtype=int) == cls).astype(int)
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        rows.append(
            {
                "mechanism_class_id": cls,
                "mechanism_class_name": MECHANISM_CLASS_NAMES[cls],
                "facies_label": FACIES_LABELS[cls],
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": int(y_true.sum()),
            }
        )
    metric_df = pd.DataFrame(rows)
    axes[1].bar(x - width, metric_df["precision"], width, label="Precision", color="#5B83B4")
    axes[1].bar(x, metric_df["recall"], width, label="Recall", color="#D98B47")
    axes[1].bar(x + width, metric_df["f1"], width, label="F1", color="#69A66D")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(FACIES_LABELS)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Score")
    axes[1].set_title("Facies classification metrics")
    axes[1].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, borderaxespad=0.0)
    _style_axis(axes[1], grid_axis="y")
    _add_panel_label(axes[1], "b")
    return fig, metric_df


def _plot_stability(seed_summary: pd.DataFrame, repeated_metrics: pd.DataFrame | None, repeated_stats: pd.DataFrame | None) -> plt.Figure:
    _set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 10.2), dpi=180, constrained_layout=True)
    seed_metric_cols = ["log_mae", "rey_ge_1000_ap", "rey_ge_1500_ap", "mechanism_macro_f1"]
    repeat_metric_cols = ["log_mae", "ap1000", "ap1500", "mechanism_macro_f1"]
    titles = ["logREY MAE", "AP1000", "AP1500", "facies macro-F1"]
    for panel_idx, (ax, seed_metric, repeat_metric, title) in enumerate(zip(axes.ravel(), seed_metric_cols, repeat_metric_cols, titles)):
        if seed_metric not in seed_summary.columns:
            ax.set_visible(False)
            continue
        y = seed_summary[seed_metric].to_numpy(dtype=float)
        x = np.arange(len(y))
        ax.plot(x, y, marker="o", markersize=5.2, color="#35699A", lw=1.8)
        x_labels = [str(x) for x in seed_summary["run_id"].str.extract(r"seed(\d+)")[0].fillna(seed_summary["run_id"]).tolist()]
        if repeated_metrics is not None and repeat_metric in repeated_metrics.columns:
            repeat_values = repeated_metrics[repeat_metric].to_numpy(dtype=float)
            box_pos = len(y) + 0.8
            ax.boxplot(
                [repeat_values],
                positions=[box_pos],
                widths=0.45,
                patch_artist=True,
                showmeans=True,
                boxprops={"facecolor": "#E6EEF7", "edgecolor": "#35699A", "linewidth": 1.1},
                medianprops={"color": "#C85C2E", "linewidth": 1.5},
                whiskerprops={"color": "#555555", "linewidth": 1.0},
                capprops={"color": "#555555", "linewidth": 1.0},
                meanprops={"marker": "o", "markerfacecolor": "#69A66D", "markeredgecolor": "#69A66D", "markersize": 4.5},
            )
            ax.scatter(
                np.full_like(repeat_values, box_pos, dtype=float) + np.linspace(-0.12, 0.12, len(repeat_values)),
                repeat_values,
                s=9,
                alpha=0.45,
                color="#35699A",
                edgecolors="none",
                rasterized=True,
            )
            x = np.append(x, box_pos)
            x_labels.append("repeats")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=18, ha="right")
        ax.set_title(title)
        ax.set_xlabel("Seed / repeated splits")
        _style_axis(ax, grid_axis="y")
        _add_panel_label(ax, PANEL_LABELS[panel_idx])
    axes[0, 0].set_ylabel("Lower is better")
    axes[0, 1].set_ylabel("Higher is better")
    axes[1, 0].set_ylabel("Higher is better")
    axes[1, 1].set_ylabel("Higher is better")

    if repeated_stats is not None and not repeated_stats.empty:
        mean_row = repeated_stats.loc[repeated_stats["stat"] == "mean"]
        if not mean_row.empty:
            mean_row = mean_row.iloc[0]
            axes[1, 1].text(
                0.03,
                0.05,
                f"Repeated mean\nlogMAE={mean_row['log_mae']:.3f}\nAP1000={mean_row['ap1000']:.3f}\nAP1500={mean_row['ap1500']:.3f}\nmacro-F1={mean_row['mechanism_macro_f1']:.3f}",
                transform=axes[1, 1].transAxes,
                va="bottom",
                fontsize=10.2,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#BBBBBB", alpha=0.94),
            )
    return fig


def _plot_core_audit(df: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame]:
    core = (
        df.groupby("core_id", dropna=False)
        .agg(
            n_samples=("rey_true", "size"),
            max_rey_true=("rey_true", "max"),
            max_rey_pred=("rey_pred", "max"),
            max_prob1000=("prob_rey_ge_1000", "max"),
            max_prob1500=("prob_rey_ge_1500", "max"),
            mean_residual=("residual_logrey", "mean"),
            median_residual=("residual_logrey", "median"),
        )
        .reset_index()
    )
    core["observed_1000"] = (core["max_rey_true"] >= 1000.0).astype(int)
    core["observed_1500"] = (core["max_rey_true"] >= 1500.0).astype(int)
    _set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.9), dpi=180, constrained_layout=True)
    ax = axes[0]
    ax.scatter(
        core["max_rey_true"],
        core["max_rey_pred"],
        s=30,
        alpha=0.78,
        c=core["observed_1000"],
        cmap=ListedColormap(["#5B83B4", "#C85C2E"]),
        edgecolors="white",
        linewidths=0.25,
        rasterized=True,
    )
    lims = [0.0, float(max(core["max_rey_true"].max(), core["max_rey_pred"].max()))]
    ax.plot(lims, lims, color="#1A1A1A", lw=1.0, ls="--")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Observed core maximum REY (ppm)")
    ax.set_ylabel("Predicted core maximum REY (ppm)")
    ax.set_title("Core-level max REY")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    _style_axis(ax)
    _add_panel_label(ax, "a")

    ax = axes[1]
    ax.scatter(
        core["max_prob1000"],
        core["max_rey_true"],
        s=30,
        alpha=0.78,
        c=core["observed_1500"],
        cmap=ListedColormap(["#8FB9B6", "#C85C2E"]),
        edgecolors="white",
        linewidths=0.25,
        rasterized=True,
    )
    ax.axhline(1000.0, color="#35699A", lw=1.0, ls="--")
    ax.axhline(1500.0, color="#B53B36", lw=1.0, ls="--")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Core max P(REY >= 1000)")
    ax.set_ylabel("Observed core maximum REY (ppm)")
    ax.set_title("Core screening support")
    _style_axis(ax)
    _add_panel_label(ax, "b")
    return fig, core


def build_performance_bundle(
    *,
    ensemble_dir: Path,
    output_dir: Path,
    repeated_eval_dir: Path | None = None,
    repeated_split_dir: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    test_df = _prepare_test_table(_load_test_predictions(ensemble_dir))
    seed_summary = _load_seed_summary(ensemble_dir)
    ensemble_metrics = _load_ensemble_metrics(ensemble_dir)

    resolved_repeated_dir = None
    repeated_metrics = None
    repeated_stats = None
    if repeated_eval_dir is not None:
        resolved_repeated_dir = repeated_eval_dir
        if not (resolved_repeated_dir / "repeated_test_metrics.csv").exists() and repeated_split_dir is not None:
            resolved_repeated_dir = _ensure_repeated_eval(resolved_repeated_dir, repeated_split_dir, ensemble_dir)
        repeated_metrics, repeated_stats = _load_repeated_metrics(resolved_repeated_dir) if resolved_repeated_dir else (None, None)

    threshold_fig, threshold_metrics = _plot_threshold_screening(test_df)
    regression_fig = _plot_regression_and_residuals(test_df)
    mechanism_fig, mechanism_metrics = _plot_mechanism_diagnostics(test_df)
    stability_fig = _plot_stability(seed_summary, repeated_metrics, repeated_stats)
    core_fig, core_summary = _plot_core_audit(test_df)

    figures = {}
    figures["fig_01_regression"] = _save_figure(regression_fig, "fig_01_fixed_test_regression", output_dir)
    figures["fig_02_threshold"] = _save_figure(threshold_fig, "fig_02_threshold_screening", output_dir)
    figures["fig_03_mechanism"] = _save_figure(mechanism_fig, "fig_03_mechanism_diagnostics", output_dir)
    figures["fig_04_stability"] = _save_figure(stability_fig, "fig_04_seed_and_repeated_stability", output_dir)
    figures["fig_05_core"] = _save_figure(core_fig, "fig_05_core_audit", output_dir)

    test_table_path = output_dir / "test_predictions_with_residuals.csv"
    test_df.to_csv(test_table_path, index=False)
    threshold_metrics_path = output_dir / "threshold_metrics.csv"
    threshold_metrics.to_csv(threshold_metrics_path, index=False)
    threshold_bins_path = output_dir / "threshold_calibration_bins.csv"
    threshold_bin_rows = []
    for threshold in [1000.0, 1500.0]:
        y_true, y_score = _threshold_arrays(test_df, threshold)
        frac_pos, mean_pred = calibration_curve(y_true, y_score, n_bins=10, strategy="quantile")
        for idx, (mp, fp) in enumerate(zip(mean_pred, frac_pos), start=1):
            threshold_bin_rows.append(
                {
                    "threshold": threshold,
                    "bin_index": idx,
                    "mean_predicted_probability": float(mp),
                    "observed_frequency": float(fp),
                }
            )
    pd.DataFrame(threshold_bin_rows).to_csv(threshold_bins_path, index=False)
    mechanism_metrics_path = output_dir / "mechanism_metrics.csv"
    mechanism_metrics.to_csv(mechanism_metrics_path, index=False)
    seed_summary_path = output_dir / "seed_comparison.csv"
    seed_summary.to_csv(seed_summary_path, index=False)
    core_summary_path = output_dir / "core_summary.csv"
    core_summary.to_csv(core_summary_path, index=False)

    repeated_payload: dict[str, str] | None = None
    if repeated_metrics is not None and repeated_stats is not None and resolved_repeated_dir is not None:
        repeated_metrics_path = output_dir / "repeated_test_metrics.csv"
        repeated_stats_path = output_dir / "repeated_test_metric_stats.csv"
        repeated_metrics.to_csv(repeated_metrics_path, index=False)
        repeated_stats.to_csv(repeated_stats_path, index=False)
        repeated_payload = {
            "repeated_eval_dir": str(resolved_repeated_dir),
            "repeated_test_metrics_csv": str(repeated_metrics_path),
            "repeated_test_metric_stats_csv": str(repeated_stats_path),
        }

    summary = {
        "status": "completed",
        "ensemble_dir": str(ensemble_dir),
        "output_dir": str(output_dir),
        "test_metrics": {
            "n": int(len(test_df)),
            "log_mae": float(np.mean(np.abs(test_df["residual_logrey"]))),
            "spearman_log": float(test_df[["log_rey_true", "log_rey_pred"]].corr(method="spearman").iloc[0, 1]),
            "r2_log": float(ensemble_metrics.get("test_metrics", {}).get("regression", {}).get("r2_log", float("nan"))),
            "ap1000": float(average_precision_score((test_df["rey_true"] >= 1000.0).astype(int), test_df["prob_rey_ge_1000"])),
            "ap1500": float(average_precision_score((test_df["rey_true"] >= 1500.0).astype(int), test_df["prob_rey_ge_1500"])),
            "mechanism_macro_f1": float(ensemble_metrics.get("test_metrics", {}).get("mechanism3", {}).get("macro_f1", float("nan"))),
        },
        "artifacts": {
            "figures": figures,
            "test_predictions_with_residuals_csv": str(test_table_path),
            "threshold_metrics_csv": str(threshold_metrics_path),
            "threshold_calibration_bins_csv": str(threshold_bins_path),
            "mechanism_metrics_csv": str(mechanism_metrics_path),
            "seed_comparison_csv": str(seed_summary_path),
            "core_summary_csv": str(core_summary_path),
        },
        "repeated": repeated_payload,
    }
    _write_json(output_dir / "performance_bundle_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export paper-facing REY performance figures and tables.")
    parser.add_argument("--ensemble-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeated-eval-dir", type=Path, default=DEFAULT_REPEATED_EVAL_DIR)
    parser.add_argument("--repeated-split-dir", type=Path, default=DEFAULT_REPEATED_SPLIT_DIR)
    args = parser.parse_args()
    summary = build_performance_bundle(
        ensemble_dir=args.ensemble_dir,
        output_dir=args.output_dir,
        repeated_eval_dir=args.repeated_eval_dir,
        repeated_split_dir=args.repeated_split_dir,
    )
    print(json.dumps(summary, cls=NumpyJSONEncoder, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import xarray as xr
from core.utils.netcdf_io import open_dataset

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from core.utils.logger_config import logger
from core.visualization.publication_style import publication_style, save_figure


def _ensure_dir(path: str | Path) -> str:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def _safe_class_names(class_names: Optional[list[str]], n_classes: int) -> list[str]:
    if class_names and len(class_names) == n_classes:
        return [str(x) for x in class_names]
    return [f"class_{i}" for i in range(int(n_classes))]


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] == 0:
        return np.zeros((0, int(z.shape[1]) if z.ndim == 2 else 0), dtype=np.float64)
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(np.clip(z, -60.0, 60.0))
    return ez / np.clip(np.sum(ez, axis=1, keepdims=True), 1e-12, None)


def save_classification_tables(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prob: np.ndarray | None,
    class_names: list[str] | None,
    output_dir: str,
) -> dict[str, str]:
    """
    Saves JSON/CSV tables for a multiclass classification evaluation.

    Returns a dict of written file paths.
    """
    out_dir = _ensure_dir(output_dir)

    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=int).reshape(-1)
    n_classes = int(prob.shape[1]) if isinstance(prob, np.ndarray) and prob.ndim == 2 else int(max(y_true.max(initial=0), y_pred.max(initial=0)) + 1)
    names = _safe_class_names(class_names, n_classes)

    paths: dict[str, str] = {}


    try:
        from sklearn.metrics import classification_report

        report = classification_report(
            y_true,
            y_pred,
            labels=list(range(n_classes)),
            target_names=names,
            output_dict=True,
            zero_division=0,
        )
        json_path = os.path.join(out_dir, "classification_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        paths["classification_report_json"] = json_path

        df = pd.DataFrame(report).T
        csv_path = os.path.join(out_dir, "classification_report.csv")
        df.to_csv(csv_path, index=True)
        paths["classification_report_csv"] = csv_path
    except Exception as e:
        logger.warning(f"Classification tables: failed to write classification report: {e}")


    try:
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
        cm_df = pd.DataFrame(cm, index=names, columns=names)
        cm_path = os.path.join(out_dir, "confusion_matrix.csv")
        cm_df.to_csv(cm_path, index=True)
        paths["confusion_matrix_csv"] = cm_path
    except Exception as e:
        logger.warning(f"Classification tables: failed to write confusion matrix: {e}")

    if isinstance(prob, np.ndarray) and prob.ndim == 2 and prob.size > 0:
        try:
            pmax = np.max(prob, axis=1)
            entropy = -np.sum(np.clip(prob, 1e-12, 1.0) * np.log(np.clip(prob, 1e-12, 1.0)), axis=1)
            summary = {
                "n": int(prob.shape[0]),
                "pmax_mean": float(np.mean(pmax)),
                "pmax_p50": float(np.percentile(pmax, 50)),
                "pmax_p10": float(np.percentile(pmax, 10)),
                "pmax_p90": float(np.percentile(pmax, 90)),
                "entropy_mean": float(np.mean(entropy)),
                "entropy_p50": float(np.percentile(entropy, 50)),
            }
            p_path = os.path.join(out_dir, "probability_summary.json")
            with open(p_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            paths["probability_summary_json"] = p_path
        except Exception as e:
            logger.warning(f"Classification tables: failed to write probability summary: {e}")

    return paths


def plot_confusion_matrices(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None,
    output_dir: str,
    plot_config: Optional[dict[str, Any]] = None,
) -> dict[str, list[str]]:
    """
    Saves confusion matrix plots (counts and row-normalized).

    Returns dict of plot basenames -> written file paths.
    """
    out_dir = _ensure_dir(output_dir)
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=int).reshape(-1)
    n_classes = int(max(y_true.max(initial=0), y_pred.max(initial=0)) + 1)
    names = _safe_class_names(class_names, n_classes)

    try:
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes))).astype(float)
    except Exception as e:
        logger.warning(f"Classification plots: failed to compute confusion matrix: {e}")
        return {}

    plots: dict[str, list[str]] = {}

    with publication_style(plot_config):

        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion matrix (counts)")
        ax.set_xticks(range(n_classes), labels=names, rotation=30, ha="right")
        ax.set_yticks(range(n_classes), labels=names)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for (ii, jj), v in np.ndenumerate(cm):
            ax.text(jj, ii, f"{int(v)}", ha="center", va="center", fontsize=8)
        fig.tight_layout()
        plots["confusion_counts"] = save_figure(fig, path_no_ext=os.path.join(out_dir, "confusion_counts"), plot_config=plot_config)
        plt.close(fig)


        row_sum = np.clip(cm.sum(axis=1, keepdims=True), 1e-12, None)
        cm_norm = cm / row_sum
        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion matrix (row-normalized)")
        ax.set_xticks(range(n_classes), labels=names, rotation=30, ha="right")
        ax.set_yticks(range(n_classes), labels=names)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for (ii, jj), v in np.ndenumerate(cm_norm):
            ax.text(jj, ii, f"{v:.2f}", ha="center", va="center", fontsize=8)
        fig.tight_layout()
        plots["confusion_row_norm"] = save_figure(fig, path_no_ext=os.path.join(out_dir, "confusion_row_norm"), plot_config=plot_config)
        plt.close(fig)

    return plots


def plot_confidence_histograms(
    *,
    prob: np.ndarray,
    output_dir: str,
    plot_config: Optional[dict[str, Any]] = None,
) -> dict[str, list[str]]:
    """
    Saves p(max) and entropy histograms as uncertainty proxies.
    """
    out_dir = _ensure_dir(output_dir)
    prob = np.asarray(prob, dtype=float)
    if prob.ndim != 2 or prob.size == 0:
        return {}

    pmax = np.max(prob, axis=1)
    entropy = -np.sum(np.clip(prob, 1e-12, 1.0) * np.log(np.clip(prob, 1e-12, 1.0)), axis=1)

    plots: dict[str, list[str]] = {}
    with publication_style(plot_config):
        fig, ax = plt.subplots(figsize=(6.0, 4.2))
        ax.hist(pmax, bins=20, alpha=0.9, edgecolor="white")
        ax.set_xlabel("max class probability")
        ax.set_ylabel("Count")
        ax.set_title("Prediction confidence (pmax)")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        plots["pmax_hist"] = save_figure(fig, path_no_ext=os.path.join(out_dir, "pmax_hist"), plot_config=plot_config)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.0, 4.2))
        ax.hist(entropy, bins=20, alpha=0.9, edgecolor="white")
        ax.set_xlabel("entropy (nats)")
        ax.set_ylabel("Count")
        ax.set_title("Prediction uncertainty (entropy)")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        plots["entropy_hist"] = save_figure(fig, path_no_ext=os.path.join(out_dir, "entropy_hist"), plot_config=plot_config)
        plt.close(fig)

    return plots


def plot_top1_reliability(
    *,
    y_true: np.ndarray,
    prob: np.ndarray,
    output_dir: str,
    n_bins: int = 10,
    plot_config: Optional[dict[str, Any]] = None,
) -> dict[str, list[str]]:
    """
    Reliability diagram using the top-1 predicted probability vs. empirical accuracy.

    This is a simple, interpretable calibration diagnostic for multiclass models.
    """
    out_dir = _ensure_dir(output_dir)
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    prob = np.asarray(prob, dtype=float)
    if prob.ndim != 2 or prob.shape[0] != y_true.shape[0] or prob.size == 0:
        return {}

    y_hat = np.argmax(prob, axis=1)
    conf = np.max(prob, axis=1)
    acc = (y_hat == y_true).astype(float)


    n_bins = max(5, int(n_bins))
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(conf, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    xs, ys, ns = [], [], []
    for b in range(n_bins):
        m = bin_ids == b
        if not np.any(m):
            continue
        xs.append(float(np.mean(conf[m])))
        ys.append(float(np.mean(acc[m])))
        ns.append(int(np.sum(m)))

    if not xs:
        return {}

    plots: dict[str, list[str]] = {}
    with publication_style(plot_config):
        fig, ax = plt.subplots(figsize=(5.8, 4.4))
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Ideal")
        ax.plot(xs, ys, marker="o", label="Top-1 reliability")
        for x, y, n in zip(xs, ys, ns):
            ax.annotate(str(n), (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
        ax.set_xlabel("Mean predicted confidence")
        ax.set_ylabel("Empirical accuracy")
        ax.set_title("Reliability diagram (top-1)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower right")
        fig.tight_layout()
        plots["top1_reliability"] = save_figure(fig, path_no_ext=os.path.join(out_dir, "top1_reliability"), plot_config=plot_config)
        plt.close(fig)

    return plots


def plot_one_vs_rest_pr_curves(
    *,
    y_true: np.ndarray,
    prob: np.ndarray,
    class_names: list[str] | None,
    output_dir: str,
    target_class: int | None = None,
    plot_config: Optional[dict[str, Any]] = None,
) -> dict[str, list[str]]:
    """
    Precision-Recall curves (one-vs-rest) for multiclass models.

    This is especially useful when a target class is rare: PR curves are
    more informative than ROC curves under heavy imbalance.
    """
    out_dir = _ensure_dir(output_dir)
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    prob = np.asarray(prob, dtype=float)
    if prob.ndim != 2 or prob.shape[0] != y_true.shape[0] or prob.size == 0:
        return {}

    n_classes = int(prob.shape[1])
    names = _safe_class_names(class_names, n_classes)

    try:
        from sklearn.metrics import average_precision_score, precision_recall_curve
    except Exception as e:
        logger.warning(f"PR curves: scikit-learn required but not available: {e}")
        return {}

    classes_to_plot = list(range(n_classes))
    if target_class is not None and 0 <= int(target_class) < n_classes:

        classes_to_plot = [int(target_class)] + [c for c in classes_to_plot if c != int(target_class)]

    plots: dict[str, list[str]] = {}
    with publication_style(plot_config):
        fig, ax = plt.subplots(figsize=(6.2, 4.8))
        for k in classes_to_plot:
            yk = (y_true == k).astype(int)
            if int(np.sum(yk)) == 0:
                continue
            pk = prob[:, k]
            prec, rec, _thr = precision_recall_curve(yk, pk)
            ap = average_precision_score(yk, pk)
            label = f"{names[k]} (AP={ap:.3f}, n={int(np.sum(yk))})"
            if target_class is not None and k == int(target_class):
                ax.plot(rec, prec, linewidth=2.4, label=label)
            else:
                ax.plot(rec, prec, linewidth=1.4, alpha=0.9, label=label)

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("One-vs-rest Precision–Recall")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower left", fontsize=8, frameon=True)
        fig.tight_layout()
        plots["pr_curves_ovr"] = save_figure(fig, path_no_ext=os.path.join(out_dir, "pr_curves_ovr"), plot_config=plot_config)
        plt.close(fig)

    return plots


def save_target_class_threshold_sweep(
    *,
    y_true: np.ndarray,
    prob: np.ndarray,
    class_names: list[str] | None,
    output_dir: str,
    target_class: int,
    betas: tuple[float, ...] = (0.5, 1.0, 2.0),
) -> dict[str, str]:

    """
    Save a threshold sweep (one-vs-rest) for a target class.

    This is useful when you care about *precision* (few false positives) for a rare class.

    Writes:
    - target_class_threshold_sweep.csv
    - target_class_threshold_recommendations.json
    """
    out_dir = _ensure_dir(output_dir)
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    prob = np.asarray(prob, dtype=float)
    if prob.ndim != 2 or prob.shape[0] != y_true.shape[0] or prob.size == 0:
        return {}

    n_classes = int(prob.shape[1])
    if not (0 <= int(target_class) < n_classes):
        return {}

    names = _safe_class_names(class_names, n_classes)

    try:
        from sklearn.metrics import precision_recall_curve, average_precision_score
    except Exception as e:
        logger.warning(f"Threshold sweep: scikit-learn required but not available: {e}")
        return {}

    yk = (y_true == int(target_class)).astype(int)
    if int(np.sum(yk)) == 0:
        return {}

    pk = prob[:, int(target_class)]
    prec, rec, thr = precision_recall_curve(yk, pk)

    thr_aligned = np.concatenate([thr, [1.0]])
    eps = 1e-12

    rows = []
    for p, r, t in zip(prec.tolist(), rec.tolist(), thr_aligned.tolist()):
        rows.append({"threshold": float(t), "precision": float(p), "recall": float(r)})
    df = pd.DataFrame(rows)

    for beta in betas:
        b2 = float(beta) ** 2
        df[f"f{beta:g}"] = (1 + b2) * (df["precision"] * df["recall"]) / np.clip((b2 * df["precision"] + df["recall"]), eps, None)

    paths: dict[str, str] = {}
    csv_path = os.path.join(out_dir, f"target_class{int(target_class)}_threshold_sweep.csv")
    df.to_csv(csv_path, index=False)
    paths["threshold_sweep_csv"] = csv_path


    recs = {
        "target_class": int(target_class),
        "target_name": names[int(target_class)],
        "n_pos": int(np.sum(yk)),
        "average_precision": float(average_precision_score(yk, pk)),
        "best": {},
    }
    for beta in betas:
        col = f"f{beta:g}"
        j = int(df[col].to_numpy().argmax())
        recs["best"][f"F{beta:g}"] = {
            "threshold": float(df.loc[j, "threshold"]),
            "precision": float(df.loc[j, "precision"]),
            "recall": float(df.loc[j, "recall"]),
            "score": float(df.loc[j, col]),
        }

    json_path = os.path.join(out_dir, f"target_class{int(target_class)}_threshold_recommendations.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(recs, f, ensure_ascii=False, indent=2)
    paths["threshold_recommendations_json"] = json_path

    return paths


def infer_high_grade_classes(class_names: list[str] | None, n_classes: int) -> list[int]:
    """Infer high-grade class ids from names, falling back to the 5-class convention."""

    names = _safe_class_names(class_names, n_classes)
    out = [i for i, name in enumerate(names) if "high" in str(name).lower()]
    if not out and int(n_classes) >= 5:
        out = [2, 4]
    return [int(i) for i in out if 0 <= int(i) < int(n_classes)]


def compute_high_grade_metrics(
    *,
    y_true: np.ndarray,
    prob: np.ndarray,
    class_names: list[str] | None,
    high_grade_classes: list[int] | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Binary one-vs-rest diagnostics for any high-grade class."""

    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    prob = np.asarray(prob, dtype=float)
    if prob.ndim != 2 or prob.shape[0] != y_true.shape[0] or prob.size == 0:
        return {}

    n_classes = int(prob.shape[1])
    high_classes = (
        [int(x) for x in high_grade_classes]
        if high_grade_classes is not None
        else infer_high_grade_classes(class_names, n_classes)
    )
    high_classes = [i for i in high_classes if 0 <= i < n_classes]
    if not high_classes:
        return {}

    try:
        from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
    except Exception as e:
        logger.warning(f"High-grade metrics: scikit-learn required but unavailable: {e}")
        return {}

    y_bin = np.isin(y_true, high_classes).astype(int)


    p_high = np.clip(np.sum(prob[:, high_classes], axis=1), 0.0, 1.0)
    y_hat = (p_high >= float(threshold)).astype(int)

    out: dict[str, Any] = {
        "high_grade_classes": high_classes,
        "threshold": float(threshold),
        "n": int(y_true.size),
        "n_positive": int(np.sum(y_bin)),
        "positive_rate": float(np.mean(y_bin)) if y_bin.size else float("nan"),
    }
    try:
        pr, re, f1, _ = precision_recall_fscore_support(y_bin, y_hat, average="binary", zero_division=0)
        out.update({"precision": float(pr), "recall": float(re), "f1": float(f1)})
    except Exception:
        pass
    try:
        if int(np.unique(y_bin).size) >= 2:
            out["average_precision"] = float(average_precision_score(y_bin, p_high))
            out["roc_auc"] = float(roc_auc_score(y_bin, p_high))
    except Exception:
        pass
    return out


def save_high_grade_threshold_sweep(
    *,
    y_true: np.ndarray,
    prob: np.ndarray,
    class_names: list[str] | None,
    output_dir: str,
    high_grade_classes: list[int] | None = None,
    betas: tuple[float, ...] = (0.5, 1.0, 2.0),
) -> dict[str, str]:
    """
    Save a threshold sweep for P(any high-grade).

    This is the threshold counterpart to compute_high_grade_metrics and is used
    by cascade runs where high-grade screening may be more useful than argmax
    hard labels.
    """

    out_dir = _ensure_dir(output_dir)
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    prob = np.asarray(prob, dtype=float)
    if prob.ndim != 2 or prob.shape[0] != y_true.shape[0] or prob.size == 0:
        return {}

    n_classes = int(prob.shape[1])
    high_classes = (
        [int(x) for x in high_grade_classes]
        if high_grade_classes is not None
        else infer_high_grade_classes(class_names, n_classes)
    )
    high_classes = [i for i in high_classes if 0 <= i < n_classes]
    if not high_classes:
        return {}

    try:
        from sklearn.metrics import average_precision_score, precision_recall_curve
    except Exception as e:
        logger.warning(f"High-grade threshold sweep: scikit-learn required but unavailable: {e}")
        return {}

    y_bin = np.isin(y_true, high_classes).astype(int)
    if int(np.sum(y_bin)) == 0:
        return {}
    p_high = np.clip(np.sum(prob[:, high_classes], axis=1), 0.0, 1.0)
    prec, rec, thr = precision_recall_curve(y_bin, p_high)
    thr_aligned = np.concatenate([thr, [1.0]])
    eps = 1e-12

    rows = []
    for p, r, t in zip(prec.tolist(), rec.tolist(), thr_aligned.tolist()):
        rows.append({"threshold": float(t), "precision": float(p), "recall": float(r)})
    df = pd.DataFrame(rows)
    for beta in betas:
        b2 = float(beta) ** 2
        df[f"f{beta:g}"] = (1 + b2) * (df["precision"] * df["recall"]) / np.clip(
            (b2 * df["precision"] + df["recall"]),
            eps,
            None,
        )

    paths: dict[str, str] = {}
    csv_path = os.path.join(out_dir, "high_grade_threshold_sweep.csv")
    df.to_csv(csv_path, index=False)
    paths["high_grade_threshold_sweep_csv"] = csv_path

    recs = {
        "target": "any_high_grade",
        "high_grade_classes": [int(x) for x in high_classes],
        "high_grade_class_names": [_safe_class_names(class_names, n_classes)[int(x)] for x in high_classes],
        "n_pos": int(np.sum(y_bin)),
        "average_precision": float(average_precision_score(y_bin, p_high)),
        "best": {},
    }
    for beta in betas:
        col = f"f{beta:g}"
        j = int(df[col].to_numpy().argmax())
        recs["best"][f"F{beta:g}"] = {
            "threshold": float(df.loc[j, "threshold"]),
            "precision": float(df.loc[j, "precision"]),
            "recall": float(df.loc[j, "recall"]),
            "score": float(df.loc[j, col]),
        }

    json_path = os.path.join(out_dir, "high_grade_threshold_recommendations.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(recs, f, ensure_ascii=False, indent=2)
    paths["high_grade_threshold_recommendations_json"] = json_path
    return paths


def save_high_grade_fixed_threshold_metrics(
    *,
    y_true: np.ndarray,
    prob: np.ndarray,
    class_names: list[str] | None,
    output_dir: str,
    thresholds: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.30, 0.50),
    high_grade_classes: list[int] | None = None,
) -> dict[str, str]:
    """Save any-high metrics at fixed thresholds for split-stability comparison."""

    out_dir = _ensure_dir(output_dir)
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    prob = np.asarray(prob, dtype=float)
    if prob.ndim != 2 or prob.shape[0] != y_true.shape[0] or prob.size == 0:
        return {}

    n_classes = int(prob.shape[1])
    high_classes = (
        [int(x) for x in high_grade_classes]
        if high_grade_classes is not None
        else infer_high_grade_classes(class_names, n_classes)
    )
    high_classes = [i for i in high_classes if 0 <= i < n_classes]
    if not high_classes:
        return {}

    try:
        from sklearn.metrics import average_precision_score, precision_recall_fscore_support
    except Exception as e:
        logger.warning(f"High-grade fixed thresholds: scikit-learn required but unavailable: {e}")
        return {}

    y_bin = np.isin(y_true, high_classes).astype(int)
    p_high = np.clip(np.sum(prob[:, high_classes], axis=1), 0.0, 1.0)
    ap = float(average_precision_score(y_bin, p_high)) if int(np.unique(y_bin).size) >= 2 else np.nan
    rows: list[dict[str, Any]] = []
    eps = 1e-12
    for threshold in thresholds:
        t = float(np.clip(float(threshold), 0.0, 1.0))
        y_hat = (p_high >= t).astype(int)
        pr, re, f1, _ = precision_recall_fscore_support(y_bin, y_hat, average="binary", zero_division=0)
        f2 = (5.0 * float(pr) * float(re)) / max((4.0 * float(pr)) + float(re), eps)
        rows.append(
            {
                "threshold": t,
                "precision": float(pr),
                "recall": float(re),
                "f1": float(f1),
                "f2": float(f2),
                "n": int(y_bin.size),
                "n_positive": int(np.sum(y_bin)),
                "n_pred_positive": int(np.sum(y_hat)),
                "pred_positive_rate": float(np.mean(y_hat)) if y_hat.size else np.nan,
                "average_precision": ap,
                "high_grade_classes": ",".join(str(int(x)) for x in high_classes),
            }
        )

    csv_path = os.path.join(out_dir, "high_grade_fixed_threshold_metrics.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return {"high_grade_fixed_threshold_metrics_csv": csv_path}


def _summarize_numeric_by_group(df: pd.DataFrame, group_col: str, value_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for col in value_cols:
        vals = pd.to_numeric(df[col], errors="coerce")
        for group_value, idx in df.groupby(group_col, dropna=False).groups.items():
            v = vals.loc[idx].dropna().to_numpy(dtype=float)
            rows.append(
                {
                    "feature": col,
                    "group": group_value,
                    "n": int(v.size),
                    "mean": float(np.mean(v)) if v.size else np.nan,
                    "std": float(np.std(v, ddof=1)) if v.size > 1 else np.nan,
                    "median": float(np.median(v)) if v.size else np.nan,
                    "p10": float(np.percentile(v, 10)) if v.size else np.nan,
                    "p90": float(np.percentile(v, 90)) if v.size else np.nan,
                }
            )
    return pd.DataFrame(rows)


def save_env_patch_missingness_diagnostics(
    *,
    points_df: pd.DataFrame,
    env_grid_path: str,
    feature_names: list[str],
    patch_size: tuple[int, int],
    split_indices: dict[str, list[int] | np.ndarray],
    output_dir: str,
    label_col: str = "grade_geochem_class_id",
    pred_tables: dict[str, pd.DataFrame] | None = None,
) -> dict[str, str]:
    """Measure raw NaN support in the environmental patches before dataset filling."""

    out_dir = Path(_ensure_dir(output_dir))
    if points_df is None or points_df.empty or not env_grid_path or not os.path.exists(env_grid_path):
        return {}
    if "lon" not in points_df.columns or "lat" not in points_df.columns:
        return {}

    feature_names = [str(x) for x in feature_names if str(x) != "land_mask"]
    if not feature_names:
        return {}

    paths: dict[str, str] = {}
    try:
        with open_dataset(env_grid_path) as ds:
            features = [f for f in feature_names if f in ds.data_vars]
            if not features:
                return {}
            env_ds = ds[features]
            if "longitude" in env_ds.coords:
                lon_coord = env_ds["longitude"]
                if float(lon_coord.min(skipna=True).item()) < 0:
                    env_ds = env_ds.assign_coords(
                        longitude=xr.where(env_ds["longitude"] < 0, env_ds["longitude"] + 360, env_ds["longitude"])
                    ).sortby("longitude")
                elif float(lon_coord.max(skipna=True).item()) >= 360:
                    env_ds = env_ds.assign_coords(longitude=env_ds["longitude"] % 360).sortby("longitude")
            env_ds = env_ds.load()
            lat_vals = np.asarray(env_ds["latitude"].values, dtype=float)
            lon_vals = np.asarray(env_ds["longitude"].values, dtype=float)
            env_arr = np.stack([np.asarray(env_ds[f].values, dtype=np.float32) for f in features], axis=0)
    except Exception as e:
        logger.warning(f"Env missingness diagnostics: failed to load env grid: {e}")
        return {}

    lat_res = abs(float(lat_vals[1] - lat_vals[0])) if lat_vals.size > 1 else 0.0
    lon_res = abs(float(lon_vals[1] - lon_vals[0])) if lon_vals.size > 1 else 0.0
    pad_rows = int(patch_size[0]) // 2
    pad_cols = int(patch_size[1]) // 2

    def _patch_stats(row: pd.Series) -> dict[str, Any]:
        lon = float(row["lon"])
        lat = float(row["lat"])
        lon = lon % 360.0
        lat_idx = int(np.clip(np.searchsorted(lat_vals, lat - lat_res / 2.0), 0, len(lat_vals) - 1))
        lon_idx = int(np.clip(np.searchsorted(lon_vals, lon - lon_res / 2.0), 0, len(lon_vals) - 1))
        row_start = lat_idx - pad_rows
        row_end = lat_idx + pad_rows + 1
        col_start = lon_idx - pad_cols
        col_end = lon_idx + pad_cols + 1
        actual_row_start = max(0, row_start)
        actual_row_end = min(len(lat_vals), row_end)
        actual_col_start = max(0, col_start)
        actual_col_end = min(len(lon_vals), col_end)
        patch = env_arr[:, actual_row_start:actual_row_end, actual_col_start:actual_col_end]
        pad_top = max(0, -row_start)
        pad_bottom = max(0, row_end - len(lat_vals))
        pad_left = max(0, -col_start)
        pad_right = max(0, col_end - len(lon_vals))
        if any([pad_top > 0, pad_bottom > 0, pad_left > 0, pad_right > 0]):
            patch = np.pad(patch, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode="edge")
        if patch.shape[1:] != tuple(patch_size):
            fixed = np.full((len(features), int(patch_size[0]), int(patch_size[1])), np.nan, dtype=np.float32)
            copy_h = min(fixed.shape[1], patch.shape[1])
            copy_w = min(fixed.shape[2], patch.shape[2])
            fixed[:, :copy_h, :copy_w] = patch[:, :copy_h, :copy_w]
            patch = fixed
        missing = np.isnan(patch)
        by_channel = missing.reshape(len(features), -1)
        return {
            "env_patch_missing_fraction": float(np.mean(missing)),
            "env_patch_n_missing": int(np.sum(missing)),
            "env_patch_n_total": int(missing.size),
            "env_patch_channels_any_missing": int(np.sum(np.any(by_channel, axis=1))),
            "env_patch_channels_all_missing": int(np.sum(np.all(by_channel, axis=1))),
        }

    pred_tables = pred_tables or {}
    all_rows: list[pd.DataFrame] = []
    for split_name, indices in split_indices.items():
        idx = np.asarray(indices, dtype=int)
        idx = idx[(idx >= 0) & (idx < len(points_df))]
        if idx.size == 0:
            continue
        split_df = points_df.iloc[idx].copy().reset_index(drop=False).rename(columns={"index": "source_index"})
        stat_rows = [_patch_stats(row) for _, row in split_df.iterrows()]
        stat_df = pd.concat([split_df.reset_index(drop=True), pd.DataFrame(stat_rows)], axis=1)
        stat_df["split"] = str(split_name)
        pred_df = pred_tables.get(split_name)
        if isinstance(pred_df, pd.DataFrame) and "cascade_global_index" in pred_df.columns and "cascade_global_index" in stat_df.columns:
            pred_cols = [
                c
                for c in ["cascade_global_index", "cascade_y_pred", "y_pred", "pmax", "entropy"]
                if c in pred_df.columns
            ]
            stat_df = stat_df.merge(pred_df[pred_cols], on="cascade_global_index", how="left")
        sample_path = out_dir / str(split_name) / "env_patch_missingness_by_sample.csv"
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        stat_df.to_csv(sample_path, index=False)
        paths[f"{split_name}_env_patch_missingness_by_sample_csv"] = str(sample_path)
        all_rows.append(stat_df)

        group_specs = [("true_class", label_col), ("pred_class", "cascade_y_pred"), ("site_group", "site_group")]
        for suffix, group_col in group_specs:
            if group_col not in stat_df.columns:
                continue
            summary = _summarize_numeric_by_group(
                stat_df,
                group_col,
                [
                    "env_patch_missing_fraction",
                    "env_patch_n_missing",
                    "env_patch_channels_any_missing",
                    "env_patch_channels_all_missing",
                ],
            )
            path = out_dir / str(split_name) / f"env_patch_missingness_summary_by_{suffix}.csv"
            summary.to_csv(path, index=False)
            paths[f"{split_name}_env_patch_missingness_summary_by_{suffix}_csv"] = str(path)

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        combined_path = out_dir / "env_patch_missingness_all_splits.csv"
        combined.to_csv(combined_path, index=False)
        paths["env_patch_missingness_all_splits_csv"] = str(combined_path)
    return paths


def save_b2_separability_diagnostics(
    *,
    points_df: pd.DataFrame,
    output_dir: str,
    env_grid_path: str | None = None,
    feature_names: list[str] | None = None,
) -> dict[str, str]:
    """Write lightweight class2 low/high support and separability diagnostics."""

    out_dir = Path(_ensure_dir(output_dir))
    if points_df is None or points_df.empty or "mech_class_id" not in points_df.columns:
        return {}

    df = points_df.copy()
    df["mech_class_id"] = pd.to_numeric(df["mech_class_id"], errors="coerce")
    b2 = df.loc[df["mech_class_id"] == 2].copy()
    if b2.empty:
        return {}
    if "class2_grade_label" not in b2.columns:
        if "class2_grade_subclass_id" in b2.columns:
            b2["class2_grade_label"] = pd.to_numeric(b2["class2_grade_subclass_id"], errors="coerce")
        elif "grade_geochem_class_id" in b2.columns:
            gid = pd.to_numeric(b2["grade_geochem_class_id"], errors="coerce")
            b2["class2_grade_label"] = np.where(gid.isin([3, 4]), gid - 3, np.nan)
    b2["class2_grade_label"] = pd.to_numeric(b2["class2_grade_label"], errors="coerce")
    b2 = b2.loc[b2["class2_grade_label"].isin([0, 1])].copy()
    if b2.empty:
        return {}

    paths: dict[str, str] = {}
    agg_spec: dict[str, tuple[str, Any]] = {"n": ("class2_grade_label", "size")}
    if "site_group" in b2.columns:
        agg_spec["n_site_group"] = ("site_group", lambda s: int(s.astype(str).nunique()))
    if "rey" in b2.columns:
        agg_spec["rey_median"] = ("rey", "median")
        agg_spec["rey_p90"] = ("rey", lambda s: float(np.nanpercentile(pd.to_numeric(s, errors="coerce"), 90)))
    support = b2.groupby("class2_grade_label", dropna=False).agg(**agg_spec).reset_index()
    support_path = out_dir / "class2_group_support.csv"
    support.to_csv(support_path, index=False)
    paths["class2_group_support_csv"] = str(support_path)

    if "site_group" in b2.columns:
        site_counts = (
            b2.assign(site_group=b2["site_group"].astype(str))
            .pivot_table(index="site_group", columns="class2_grade_label", values="mech_class_id", aggfunc="size", fill_value=0)
            .reset_index()
        )
        site_counts.columns = [str(c) for c in site_counts.columns]
        site_path = out_dir / "class2_label_distribution_by_site_group.csv"
        site_counts.to_csv(site_path, index=False)
        paths["class2_label_distribution_by_site_group_csv"] = str(site_path)


    env_cols: list[str] = []
    if env_grid_path and feature_names and "lon" in b2.columns and "lat" in b2.columns and os.path.exists(env_grid_path):
        try:
            with open_dataset(env_grid_path) as ds:
                features = [f for f in feature_names if f in ds.data_vars and f != "land_mask"]
                env_ds = ds[features]
                if "longitude" in env_ds.coords and float(env_ds["longitude"].min(skipna=True).item()) < 0:
                    env_ds = env_ds.assign_coords(
                        longitude=xr.where(env_ds["longitude"] < 0, env_ds["longitude"] + 360, env_ds["longitude"])
                    ).sortby("longitude")
                env_ds = env_ds.load()
                lon_vals = np.asarray(env_ds["longitude"].values, dtype=float)
                lat_vals = np.asarray(env_ds["latitude"].values, dtype=float)
                for feature in features:
                    arr = np.asarray(env_ds[feature].values, dtype=float)
                    values = []
                    for _, row in b2.iterrows():
                        lat_idx = int(np.clip(np.searchsorted(lat_vals, float(row["lat"])), 0, len(lat_vals) - 1))
                        lon_idx = int(np.clip(np.searchsorted(lon_vals, float(row["lon"]) % 360.0), 0, len(lon_vals) - 1))
                        values.append(arr[lat_idx, lon_idx])
                    col = f"env__{feature}"
                    b2[col] = values
                    env_cols.append(col)
        except Exception as e:
            logger.warning(f"B2 separability diagnostics: failed to sample env center features: {e}")

    exclude = {
        "mech_class_id",
        "mech_subclass_id",
        "class1_grade_subclass_id",
        "class2_grade_subclass_id",
        "class1_grade_label",
        "class2_grade_label",
        "grade_geochem_class_id",
        "cascade_global_index",
        "cascade_source_index",
    }
    numeric_cols = [
        c
        for c in b2.columns
        if c not in exclude and (str(c).startswith("env__") or pd.api.types.is_numeric_dtype(b2[c]))
    ]
    keep_cols = []
    for col in numeric_cols:
        vals = pd.to_numeric(b2[col], errors="coerce")
        if int(vals.notna().sum()) >= 6 and float(vals.nunique(dropna=True)) > 1:
            keep_cols.append(col)
    if keep_cols:
        summary = _summarize_numeric_by_group(b2, "class2_grade_label", keep_cols)
        summary_path = out_dir / "class2_numeric_feature_summary.csv"
        summary.to_csv(summary_path, index=False)
        paths["class2_numeric_feature_summary_csv"] = str(summary_path)

        effect_rows: list[dict[str, Any]] = []
        y = b2["class2_grade_label"].to_numpy(dtype=float)
        for col in keep_cols:
            vals = pd.to_numeric(b2[col], errors="coerce").to_numpy(dtype=float)
            lo = vals[y == 0]
            hi = vals[y == 1]
            lo = lo[np.isfinite(lo)]
            hi = hi[np.isfinite(hi)]
            if lo.size < 2 or hi.size < 2:
                continue
            pooled = np.sqrt(((lo.size - 1) * np.var(lo, ddof=1) + (hi.size - 1) * np.var(hi, ddof=1)) / max(lo.size + hi.size - 2, 1))
            smd = (float(np.mean(hi)) - float(np.mean(lo))) / float(pooled) if pooled > 1e-12 else np.nan
            row = {
                "feature": col,
                "n_low": int(lo.size),
                "n_high": int(hi.size),
                "mean_low": float(np.mean(lo)),
                "mean_high": float(np.mean(hi)),
                "standardized_mean_difference_high_minus_low": float(smd),
                "abs_smd": float(abs(smd)) if np.isfinite(smd) else np.nan,
                "is_env_feature": bool(col in env_cols or str(col).startswith("env__")),
            }
            try:
                from sklearn.metrics import roc_auc_score

                valid = np.isfinite(vals) & np.isin(y, [0, 1])
                if int(np.unique(y[valid]).size) >= 2:
                    row["univariate_auc_high"] = float(roc_auc_score(y[valid].astype(int), vals[valid]))
            except Exception:
                pass
            effect_rows.append(row)
        effect_df = pd.DataFrame(effect_rows).sort_values("abs_smd", ascending=False) if effect_rows else pd.DataFrame()
        effect_path = out_dir / "class2_numeric_feature_effect_sizes.csv"
        effect_df.to_csv(effect_path, index=False)
        paths["class2_numeric_feature_effect_sizes_csv"] = str(effect_path)
    return paths


def save_grade_class_diagnostics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prob: np.ndarray,
    class_names: list[str] | None,
    output_dir: str,
    rey_values: np.ndarray | None = None,
    pred_prefix: str = "pred",
) -> dict[str, str]:
    """
    Save grade-aware class diagnostics:
    - per true/predicted class REY summaries;
    - high-grade one-vs-rest metrics;
    - per-class AP values.
    """

    out_dir = _ensure_dir(output_dir)
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=int).reshape(-1)
    prob = np.asarray(prob, dtype=float)
    if prob.ndim != 2 or prob.shape[0] != y_true.shape[0] or y_pred.shape[0] != y_true.shape[0]:
        return {}

    n_classes = int(prob.shape[1])
    names = _safe_class_names(class_names, n_classes)
    rey = None
    if rey_values is not None:
        rey = pd.to_numeric(pd.Series(rey_values), errors="coerce").to_numpy(dtype=float)
        if rey.shape[0] != y_true.shape[0]:
            rey = None

    paths: dict[str, str] = {}
    rows = []
    for axis_name, labels in [("true", y_true), (str(pred_prefix), y_pred)]:
        for class_id in range(n_classes):
            m = labels == class_id
            vals = rey[m] if rey is not None else np.asarray([], dtype=float)
            vals = vals[np.isfinite(vals)] if vals.size else vals
            rows.append(
                {
                    "axis": axis_name,
                    "class_id": int(class_id),
                    "class_name": names[class_id],
                    "n": int(np.sum(m)),
                    "rey_median": float(np.nanmedian(vals)) if vals.size else np.nan,
                    "rey_p90": float(np.nanpercentile(vals, 90)) if vals.size else np.nan,
                    "p_rey_ge_1000": float(np.mean(vals >= 1000.0)) if vals.size else np.nan,
                    "p_rey_ge_1500": float(np.mean(vals >= 1500.0)) if vals.size else np.nan,
                }
            )
    summary_path = os.path.join(out_dir, "grade_class_rey_summary.csv")
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    paths["grade_class_rey_summary_csv"] = summary_path

    try:
        from sklearn.metrics import average_precision_score, precision_recall_fscore_support

        metric_rows = []
        for class_id in range(n_classes):
            yk = (y_true == class_id).astype(int)
            yhk = (y_pred == class_id).astype(int)
            pr, re, f1, _ = precision_recall_fscore_support(yk, yhk, average="binary", zero_division=0)
            row = {
                "class_id": int(class_id),
                "class_name": names[class_id],
                "n_true": int(np.sum(yk)),
                "n_pred": int(np.sum(yhk)),
                "precision": float(pr),
                "recall": float(re),
                "f1": float(f1),
            }
            if int(np.unique(yk).size) >= 2:
                row["average_precision"] = float(average_precision_score(yk, prob[:, class_id]))
            metric_rows.append(row)
        per_class_path = os.path.join(out_dir, "per_class_probability_metrics.csv")
        pd.DataFrame(metric_rows).to_csv(per_class_path, index=False)
        paths["per_class_probability_metrics_csv"] = per_class_path
    except Exception as e:
        logger.warning(f"Grade diagnostics: failed to write per-class metrics: {e}")

    high_metrics = compute_high_grade_metrics(y_true=y_true, prob=prob, class_names=class_names)
    if high_metrics:
        high_path = os.path.join(out_dir, "high_grade_metrics.json")
        with open(high_path, "w", encoding="utf-8") as f:
            json.dump(high_metrics, f, ensure_ascii=False, indent=2)
        paths["high_grade_metrics_json"] = high_path
        paths.update(
            save_high_grade_threshold_sweep(
                y_true=y_true,
                prob=prob,
                class_names=class_names,
                output_dir=out_dir,
                high_grade_classes=high_metrics.get("high_grade_classes"),
            )
            or {}
        )
        paths.update(
            save_high_grade_fixed_threshold_metrics(
                y_true=y_true,
                prob=prob,
                class_names=class_names,
                output_dir=out_dir,
                high_grade_classes=high_metrics.get("high_grade_classes"),
            )
            or {}
        )

    return paths


def save_hierarchical_classification_diagnostics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prob: np.ndarray,
    class_names: list[str] | None,
    output_dir: str,
) -> dict[str, str]:
    """
    Save staged diagnostics for the 5-class grade geochem workflow.

    Levels:
    - A: carbonate vs noncarbonate
    - B: within noncarbonate, class1 vs class2
    - C1: within class1, low vs high
    - C2: within class2, low vs high
    """

    out_dir = _ensure_dir(output_dir)
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=int).reshape(-1)
    prob = np.asarray(prob, dtype=float)
    if prob.ndim != 2 or prob.shape[0] != y_true.shape[0] or y_pred.shape[0] != y_true.shape[0]:
        return {}

    n_classes = int(prob.shape[1])
    if n_classes < 5:
        return {}

    paths: dict[str, str] = {}
    summary_rows: list[dict[str, Any]] = []

    try:
        from sklearn.metrics import (
            average_precision_score,
            balanced_accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
            precision_recall_fscore_support,
        )
    except Exception as e:
        logger.warning(f"Hierarchical diagnostics: scikit-learn required but unavailable: {e}")
        return {}

    def _write_stage(
        *,
        stage_name: str,
        labels_true: np.ndarray,
        labels_pred: np.ndarray,
        labels_prob: np.ndarray | None,
        stage_class_names: list[str],
    ) -> None:
        stage_dir = Path(out_dir) / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)

        labels_true = np.asarray(labels_true, dtype=int).reshape(-1)
        labels_pred = np.asarray(labels_pred, dtype=int).reshape(-1)
        if labels_true.size == 0 or labels_pred.size != labels_true.size:
            return

        label_ids = list(range(len(stage_class_names)))
        cm = confusion_matrix(labels_true, labels_pred, labels=label_ids)
        pd.DataFrame(cm, index=stage_class_names, columns=stage_class_names).to_csv(
            stage_dir / "confusion_matrix.csv",
            index=True,
        )

        report = classification_report(
            labels_true,
            labels_pred,
            labels=label_ids,
            target_names=stage_class_names,
            output_dict=True,
            zero_division=0,
        )
        with open(stage_dir / "classification_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        pd.DataFrame(report).T.to_csv(stage_dir / "classification_report.csv", index=True)

        try:
            macro_f1 = float(f1_score(labels_true, labels_pred, average="macro", zero_division=0))
        except Exception:
            macro_f1 = float("nan")
        try:
            bal_acc = float(balanced_accuracy_score(labels_true, labels_pred))
        except Exception:
            bal_acc = float("nan")

        summary_row: dict[str, Any] = {
            "stage": stage_name,
            "n": int(labels_true.size),
            "macro_f1": macro_f1,
            "balanced_accuracy": bal_acc,
        }

        if labels_prob is not None and isinstance(labels_prob, np.ndarray) and labels_prob.ndim == 2 and labels_prob.shape[0] == labels_true.size:
            prob_rows: list[dict[str, Any]] = []
            for class_id, class_name in enumerate(stage_class_names):
                yk = (labels_true == class_id).astype(int)
                yhk = (labels_pred == class_id).astype(int)
                pr, re, f1, _ = precision_recall_fscore_support(yk, yhk, average="binary", zero_division=0)
                row = {
                    "class_id": int(class_id),
                    "class_name": class_name,
                    "n_true": int(np.sum(yk)),
                    "n_pred": int(np.sum(yhk)),
                    "precision": float(pr),
                    "recall": float(re),
                    "f1": float(f1),
                    "average_precision": np.nan,
                }
                if int(np.unique(yk).size) >= 2:
                    try:
                        row["average_precision"] = float(average_precision_score(yk, labels_prob[:, class_id]))
                    except Exception:
                        pass
                prob_rows.append(row)
            pd.DataFrame(prob_rows).to_csv(stage_dir / "per_class_probability_metrics.csv", index=False)
            if len(stage_class_names) == 2:
                try:
                    summary_row["positive_average_precision"] = float(
                        average_precision_score((labels_true == 1).astype(int), labels_prob[:, 1])
                    )
                except Exception:
                    pass

        summary_rows.append(summary_row)

    prob = np.clip(prob, 1e-12, 1.0)

    level_a_true = (y_true > 0).astype(int)
    level_a_pred = (y_pred > 0).astype(int)
    level_a_prob = np.column_stack([prob[:, 0], np.sum(prob[:, 1:], axis=1)])
    _write_stage(
        stage_name="level_a_carbonate_vs_noncarbonate",
        labels_true=level_a_true,
        labels_pred=level_a_pred,
        labels_prob=level_a_prob,
        stage_class_names=["carbonate_background", "noncarbonate"],
    )

    non_mask = y_true > 0
    if np.any(non_mask):
        y_true_non = y_true[non_mask]
        y_pred_non = y_pred[non_mask]
        prob_non = prob[non_mask]

        level_b_true = ((y_true_non - 1) // 2).astype(int)
        level_b_pred = ((np.clip(y_pred_non, 1, 4) - 1) // 2).astype(int)
        level_b_prob = np.column_stack(
            [
                prob_non[:, 1] + prob_non[:, 2],
                prob_non[:, 3] + prob_non[:, 4],
            ]
        )
        _write_stage(
            stage_name="level_b_class1_vs_class2_within_noncarbonate",
            labels_true=level_b_true,
            labels_pred=level_b_pred,
            labels_prob=level_b_prob,
            stage_class_names=["class1", "class2"],
        )

        class1_mask = level_b_true == 0
        if np.any(class1_mask):
            c1_true = (y_true_non[class1_mask] - 1).astype(int)
            c1_pred = np.where(level_b_pred[class1_mask] == 0, np.clip(y_pred_non[class1_mask] - 1, 0, 1), 0).astype(int)
            c1_prob_raw = prob_non[class1_mask][:, 1:3]
            c1_prob_sum = np.clip(np.sum(c1_prob_raw, axis=1, keepdims=True), 1e-12, None)
            c1_prob = c1_prob_raw / c1_prob_sum
            _write_stage(
                stage_name="level_c1_low_vs_high_within_true_class1",
                labels_true=c1_true,
                labels_pred=c1_pred,
                labels_prob=c1_prob,
                stage_class_names=["class1_low_grade", "class1_high_grade"],
            )

        class2_mask = level_b_true == 1
        if np.any(class2_mask):
            c2_true = (y_true_non[class2_mask] - 3).astype(int)
            c2_pred = np.where(level_b_pred[class2_mask] == 1, np.clip(y_pred_non[class2_mask] - 3, 0, 1), 0).astype(int)
            c2_prob_raw = prob_non[class2_mask][:, 3:5]
            c2_prob_sum = np.clip(np.sum(c2_prob_raw, axis=1, keepdims=True), 1e-12, None)
            c2_prob = c2_prob_raw / c2_prob_sum
            _write_stage(
                stage_name="level_c2_low_vs_high_within_true_class2",
                labels_true=c2_true,
                labels_pred=c2_pred,
                labels_prob=c2_prob,
                stage_class_names=["class2_low_grade", "class2_high_grade"],
            )

    summary_path = os.path.join(out_dir, "hierarchical_stage_metrics.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    paths["hierarchical_stage_metrics_csv"] = summary_path
    return paths


def save_split_group_diagnostics(
    *,
    rey_df: pd.DataFrame,
    split_indices: dict[str, np.ndarray | list[int]],
    output_dir: str,
    label_col: str | None = None,
    group_col: str | None = None,
    class_names: list[str] | None = None,
    relaxed_overlap_col: str | None = None,
) -> dict[str, str]:
    """Save leakage diagnostics for group-based train/val/test splits."""

    out_dir = _ensure_dir(output_dir)
    if not isinstance(rey_df, pd.DataFrame) or rey_df.empty:
        return {}

    if group_col is None:
        for cand in ("site_group", "core_id", "site_hole"):
            if cand in rey_df.columns:
                group_col = cand
                break
    if group_col is None or group_col not in rey_df.columns:
        return {}

    n_classes = 0
    labels = None
    if label_col and label_col in rey_df.columns:
        labels = pd.to_numeric(rey_df[label_col], errors="coerce").to_numpy(dtype=float)
        finite = labels[np.isfinite(labels)]
        n_classes = int(np.nanmax(finite)) + 1 if finite.size else 0
    names = _safe_class_names(class_names, n_classes) if n_classes > 0 else []

    assign_rows = []
    group_to_splits: dict[str, set[str]] = {}
    summary_rows = []
    for split_name, indices in split_indices.items():
        idx = np.asarray(indices, dtype=int)
        idx = idx[(idx >= 0) & (idx < len(rey_df))]
        if idx.size == 0:
            summary_rows.append({"split": split_name, "n_samples": 0, "n_groups": 0})
            continue
        g = rey_df.iloc[idx][group_col].astype(str).to_numpy()
        for ii, gg in zip(idx.tolist(), g.tolist()):
            assign_rows.append({"dataset_index": int(ii), "split": split_name, "group": gg})
            group_to_splits.setdefault(gg, set()).add(split_name)
        row: dict[str, Any] = {
            "split": split_name,
            "n_samples": int(idx.size),
            "n_groups": int(pd.Series(g).nunique()),
        }
        if labels is not None and n_classes > 0:
            y = labels[idx]
            for class_id in range(n_classes):
                count = int(np.sum(y == class_id))
                row[f"class{class_id}_n"] = count
                if class_id < len(names):
                    row[f"{names[class_id]}_n"] = count
        summary_rows.append(row)

    paths: dict[str, str] = {}
    summary_path = os.path.join(out_dir, "split_group_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    paths["split_group_summary_csv"] = summary_path

    assignments_path = os.path.join(out_dir, "split_group_assignments.csv")
    pd.DataFrame(assign_rows).to_csv(assignments_path, index=False)
    paths["split_group_assignments_csv"] = assignments_path

    leakage_rows = [
        {"group": g, "splits": ",".join(sorted(s)), "n_splits": int(len(s))}
        for g, s in sorted(group_to_splits.items())
        if len(s) > 1
    ]
    leakage_path = os.path.join(out_dir, "split_group_leakage.csv")
    pd.DataFrame(leakage_rows, columns=["group", "splits", "n_splits"]).to_csv(leakage_path, index=False)
    paths["split_group_leakage_csv"] = leakage_path

    relaxed_overlap_rows = []
    relaxed_overlap_summary: dict[str, Any] | None = None
    if relaxed_overlap_col and relaxed_overlap_col in rey_df.columns and relaxed_overlap_col != group_col:
        split_to_overlap_groups: dict[str, set[str]] = {}
        overlap_to_splits: dict[str, set[str]] = {}
        split_sample_counts: dict[str, int] = {}
        split_overlap_sample_counts: dict[str, int] = {}
        for split_name, indices in split_indices.items():
            idx = np.asarray(indices, dtype=int)
            idx = idx[(idx >= 0) & (idx < len(rey_df))]
            split_sample_counts[split_name] = int(idx.size)
            vals = rey_df.iloc[idx][relaxed_overlap_col].astype(str).str.strip()
            vals = vals[~vals.isin(["", "nan", "None", "<NA>"])]
            split_to_overlap_groups[split_name] = set(vals.tolist())
            for value in vals.tolist():
                overlap_to_splits.setdefault(value, set()).add(split_name)

        cross_split_groups = {
            group: splits for group, splits in overlap_to_splits.items() if len(splits) > 1
        }
        for split_name, indices in split_indices.items():
            idx = np.asarray(indices, dtype=int)
            idx = idx[(idx >= 0) & (idx < len(rey_df))]
            vals = rey_df.iloc[idx][relaxed_overlap_col].astype(str).str.strip()
            in_overlap = vals.isin(cross_split_groups.keys())
            split_overlap_sample_counts[split_name] = int(in_overlap.sum())

        for group, splits in sorted(cross_split_groups.items()):
            split_names = sorted(splits)
            row: dict[str, Any] = {
                relaxed_overlap_col: group,
                "splits": ",".join(split_names),
                "n_splits": int(len(split_names)),
            }
            for split_name, indices in split_indices.items():
                idx = np.asarray(indices, dtype=int)
                idx = idx[(idx >= 0) & (idx < len(rey_df))]
                mask = rey_df.iloc[idx][relaxed_overlap_col].astype(str).str.strip() == group
                row[f"{split_name}_n_samples"] = int(mask.sum())
                if group_col in rey_df.columns:
                    row[f"{split_name}_n_{group_col}"] = int(
                        rey_df.iloc[idx][mask.to_numpy()][group_col].astype(str).nunique()
                    )
            relaxed_overlap_rows.append(row)

        relaxed_overlap_path = os.path.join(out_dir, "split_site_group_overlap_audit.csv")
        pd.DataFrame(relaxed_overlap_rows).to_csv(relaxed_overlap_path, index=False)
        paths["split_site_group_overlap_audit_csv"] = relaxed_overlap_path
        relaxed_overlap_summary = {
            "relaxed_overlap_col": relaxed_overlap_col,
            "primary_group_col": group_col,
            "n_cross_split_groups": int(len(cross_split_groups)),
            "cross_split_group_rate": float(
                len(cross_split_groups) / max(1, len(overlap_to_splits))
            ),
            "split_overlap_sample_counts": split_overlap_sample_counts,
            "split_overlap_sample_rates": {
                split_name: float(split_overlap_sample_counts.get(split_name, 0) / max(1, count))
                for split_name, count in split_sample_counts.items()
            },
            "interpretation": (
                "Relaxed core_id split audit: primary groups remain disjoint, "
                "but site_group overlap is allowed and reported."
            ),
        }

    meta = {
        "group_col": group_col,
        "label_col": label_col,
        "n_leaking_groups": int(len(leakage_rows)),
        "leakage_detected": bool(len(leakage_rows) > 0),
        "relaxed_overlap": relaxed_overlap_summary,
    }
    meta_path = os.path.join(out_dir, "split_group_diagnostics.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    paths["split_group_diagnostics_json"] = meta_path

    return paths

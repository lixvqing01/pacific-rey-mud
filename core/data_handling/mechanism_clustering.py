from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


@dataclass(frozen=True)
class MechanismClusteringResult:
    algorithm: str
    label_col: str
    class_names: list[str]
    feature_cols: list[str]
    order_by_col: str
    n_classes: int
    random_state: int
    cluster_order_by_median: list[float]
    kmeans_inertia: float


@dataclass(frozen=True)
class TwoStageMechanismClusteringResult:
    algorithm: str
    label_col: str
    class_names: list[str]
    n_classes: int
    random_state: int
    stage1: dict[str, Any]
    stage2: dict[str, Any]


GRADE_GEOCHEM_CLASS_NAMES = [
    "carbonate_background",
    "class1_low_grade",
    "class1_high_grade",
    "class2_low_grade",
    "class2_high_grade",
]

EXPLORATION_4CLASS_NAMES = [
    "carbonate_background",
    "class1_enriched",
    "class2_low_grade",
    "class2_high_grade",
]

REE_Y_FEATURE_COLS = [
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Y",
]


def _coerce_numeric_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.loc[:, cols].copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _safe_json_float(value: Any) -> float | None:
    try:
        v = float(value)
    except Exception:
        return None
    return v if np.isfinite(v) else None


def _rey_summary_by_class(
    df: pd.DataFrame,
    *,
    label_col: str,
    class_names: list[str],
    rey_col: str = "rey",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    labels = pd.to_numeric(df[label_col], errors="coerce") if label_col in df.columns else pd.Series([], dtype=float)
    rey = pd.to_numeric(df[rey_col], errors="coerce") if rey_col in df.columns else pd.Series(np.nan, index=df.index)
    for class_id, class_name in enumerate(class_names):
        m = labels == int(class_id)
        vals = rey.loc[m].to_numpy(dtype=np.float64, copy=False)
        vals = vals[np.isfinite(vals)]
        rows.append(
            {
                "class_id": int(class_id),
                "class_name": str(class_name),
                "n": int(np.sum(m)),
                "rey_median": _safe_json_float(np.nanmedian(vals)) if vals.size else None,
                "rey_p90": _safe_json_float(np.nanpercentile(vals, 90)) if vals.size else None,
                "p_ge_1000": _safe_json_float(np.mean(vals >= 1000.0)) if vals.size else None,
                "p_ge_1500": _safe_json_float(np.mean(vals >= 1500.0)) if vals.size else None,
            }
        )
    return rows


def _fit_predict_grade_subclusters(
    df: pd.DataFrame,
    *,
    mask: np.ndarray,
    feature_cols: list[str],
    rey_col: str,
    random_state: int,
    n_init: int,
    max_iter: int,
    init: str,
    scaler_kind: str,
    softmax_temperature: float | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    idx = np.where(np.asarray(mask, dtype=bool))[0]
    if idx.size < 6:
        raise ValueError(f"Too few samples for grade k=2 clustering: n={idx.size}")

    x_raw = _coerce_numeric_frame(df.iloc[idx].copy(), feature_cols).to_numpy(dtype=np.float64, copy=True)
    x_raw = np.where(np.isfinite(x_raw) & (x_raw >= 0.0), np.log1p(x_raw), np.nan)
    imputer = SimpleImputer(strategy="median")
    x_imp = imputer.fit_transform(x_raw)

    sk = str(scaler_kind or "robust").strip().lower()
    if sk == "standard":
        scaler = StandardScaler(with_mean=True, with_std=True)
    else:
        sk = "robust"
        scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))
    x = scaler.fit_transform(x_imp)

    km = KMeans(
        n_clusters=2,
        random_state=int(random_state),
        n_init=int(n_init),
        max_iter=int(max_iter),
        init=str(init or "k-means++"),
    )
    labels_raw = km.fit_predict(x).astype(int)

    rey = pd.to_numeric(df.iloc[idx][rey_col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    med = []
    for k in range(2):
        m = np.nanmedian(rey[labels_raw == k])
        med.append(float(m) if np.isfinite(m) else float("nan"))

    order = np.argsort(np.array([np.inf if not np.isfinite(m) else m for m in med]))
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    labels = inv[labels_raw].astype(int)

    centers = np.asarray(km.cluster_centers_, dtype=np.float64)
    d2 = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    if softmax_temperature is None or not np.isfinite(float(softmax_temperature)) or float(softmax_temperature) <= 0:
        t = float(np.nanmedian(d2)) if np.isfinite(np.nanmedian(d2)) else 1.0
        t = max(t, 1e-6)
    else:
        t = float(softmax_temperature)
    logits = -d2 / t
    logits = logits - np.max(logits, axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / np.clip(np.sum(probs, axis=1, keepdims=True), 1e-12, None)
    probs = probs[:, order]

    artifacts = {
        "features": list(feature_cols),
        "feature_transform": "log1p(element)",
        "ordering_rule": "cluster_ordered_by_median_rey_ascending",
        "cluster_order_by_rey_median": [med[int(i)] for i in order.tolist()],
        "fit_mask": {"n_total": int(len(df)), "n_fit": int(idx.size)},
        "missing_rate": _missing_rate_summary(df.iloc[idx].copy(), feature_cols),
        "imputer": {"strategy": "median", "statistics_": imputer.statistics_.tolist()},
        "scaler": {
            "kind": sk,
            "center_": scaler.center_.tolist() if hasattr(scaler, "center_") else None,
            "scale_": scaler.scale_.tolist() if hasattr(scaler, "scale_") else None,
            "quantile_range": list(getattr(scaler, "quantile_range", (25.0, 75.0))),
        },
        "kmeans": {
            "n_clusters": 2,
            "n_init": int(getattr(km, "n_init", int(n_init))),
            "max_iter": int(getattr(km, "max_iter", 300)),
            "inertia_": float(getattr(km, "inertia_", float("nan"))),
            "cluster_centers_": np.asarray(getattr(km, "cluster_centers_", np.empty((0, 0))), dtype=float).tolist(),
        },
        "soft_assignment": {"method": "softmax(-squared_euclidean / temperature)", "temperature": float(t)},
    }
    return labels, probs, artifacts


def _missing_rate_summary(df: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        summary[str(c)] = float(np.mean(~np.isfinite(s.to_numpy(dtype=np.float64))))
    return summary


def fit_predict_mechanism_kmeans(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    n_classes: int = 3,
    label_col: str = "mech_class_id",
    prob_prefix: str = "mech_class_prob_",
    order_by_col: str = "rey",
    class_names: list[str] | None = None,
    random_state: int = 42,
    n_init: int = 20,
    max_iter: int = 300,
    init: str = "k-means++",
    log1p_transform: bool = True,
    scaler_kind: str = "robust",
    softmax_temperature: float | None = None,
) -> tuple[pd.DataFrame, MechanismClusteringResult, dict[str, Any]]:
    """
    Assigns mechanism-style classes via an unsupervised KMeans on geochemical features.

    Notes:
    - This is intentionally "mechanism-oriented": it clusters on geochemical signatures (optionally including ΣREY/rey),
      and then orders class indices by the median of `order_by_col` for stable interpretation across runs.
    - Returned labels are integers in [0, n_classes).
    """
    if n_classes < 2:
        raise ValueError("n_classes must be >= 2")
    if not feature_cols:
        raise ValueError("feature_cols must be non-empty")
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    if order_by_col not in df.columns:
        raise ValueError(f"order_by_col '{order_by_col}' not found in DataFrame")

    x_raw = _coerce_numeric_frame(df, feature_cols).to_numpy(dtype=np.float64, copy=True)
    order_raw = pd.to_numeric(df[order_by_col], errors="coerce").to_numpy(dtype=np.float64, copy=True)

    x_raw = np.where(np.isfinite(x_raw), x_raw, np.nan)
    if log1p_transform:

        x_raw = np.where(x_raw >= 0, np.log1p(x_raw), np.nan)

    imputer = SimpleImputer(strategy="median")
    x_imp = imputer.fit_transform(x_raw)

    scaler_kind_norm = str(scaler_kind or "robust").strip().lower()
    if scaler_kind_norm == "standard":
        scaler = StandardScaler(with_mean=True, with_std=True)
    else:
        scaler_kind_norm = "robust"
        scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))
    x = scaler.fit_transform(x_imp)

    km = KMeans(
        n_clusters=int(n_classes),
        random_state=int(random_state),
        n_init=int(n_init),
        max_iter=int(max_iter),
        init=str(init or "k-means++"),
    )
    labels = km.fit_predict(x).astype(int)

    # Soft assignment (pseudo-probabilities) from distances to centroids for diagnostics/visualization.
    centers = np.asarray(km.cluster_centers_, dtype=np.float64)

    d2 = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    if softmax_temperature is None or not np.isfinite(float(softmax_temperature)) or float(softmax_temperature) <= 0:

        t = float(np.nanmedian(d2)) if np.isfinite(np.nanmedian(d2)) else 1.0
        t = max(t, 1e-6)
    else:
        t = float(softmax_temperature)
    logits = -d2 / t
    logits = logits - np.max(logits, axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / np.clip(np.sum(probs, axis=1, keepdims=True), 1e-12, None)


    medians = []
    for k in range(int(n_classes)):
        m = np.nanmedian(order_raw[labels == k])
        medians.append(float(m) if np.isfinite(m) else float("nan"))
    order = np.argsort(np.array([np.inf if not np.isfinite(m) else m for m in medians]))
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    labels_ord = inv[labels]
    probs_ord = probs[:, order]

    if class_names is None:
        class_names = [f"class_{i}" for i in range(int(n_classes))]
    if len(class_names) != int(n_classes):
        raise ValueError(f"class_names length must equal n_classes ({n_classes})")

    out = df.copy()
    out[label_col] = labels_ord.astype(int)
    for c in range(int(n_classes)):
        out[f"{prob_prefix}{c}"] = probs_ord[:, c].astype(np.float32)

    result = MechanismClusteringResult(
        algorithm="kmeans",
        label_col=label_col,
        class_names=list(class_names),
        feature_cols=list(feature_cols),
        order_by_col=order_by_col,
        n_classes=int(n_classes),
        random_state=int(random_state),
        cluster_order_by_median=[medians[int(i)] for i in order.tolist()],
        kmeans_inertia=float(getattr(km, "inertia_", float("nan"))),
    )

    artifacts: dict[str, Any] = {
        "result": asdict(result),
        "settings": {
            "log1p_transform": bool(log1p_transform),
            "scaler_kind": str(scaler_kind_norm),
            "init": str(init or "k-means++"),
            "n_init": int(n_init),
            "max_iter": int(max_iter),
        },
        "missing_rate": _missing_rate_summary(df, feature_cols),
        "imputer": {
            "strategy": "median",
            "statistics_": imputer.statistics_.tolist(),
        },
        "scaler": {
            "center_": scaler.center_.tolist() if hasattr(scaler, "center_") else None,
            "scale_": scaler.scale_.tolist() if hasattr(scaler, "scale_") else None,
            "quantile_range": list(getattr(scaler, "quantile_range", (25.0, 75.0))),
        },
        "kmeans": {
            "n_clusters": int(getattr(km, "n_clusters", int(n_classes))),
            "n_init": int(getattr(km, "n_init", int(n_init))),
            "max_iter": int(getattr(km, "max_iter", 300)),
            "inertia_": float(getattr(km, "inertia_", float("nan"))),
            "cluster_centers_": np.asarray(getattr(km, "cluster_centers_", np.empty((0, 0))), dtype=float).tolist(),
        },
        "soft_assignment": {
            "method": "softmax(-squared_euclidean / temperature)",
            "temperature": float(t),
        },
    }


    try:
        from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

        metrics: dict[str, float] = {}
        if x.shape[0] >= 3 and int(n_classes) >= 2 and int(n_classes) < x.shape[0]:
            metrics["silhouette"] = float(silhouette_score(x, labels))
            metrics["davies_bouldin"] = float(davies_bouldin_score(x, labels))
            metrics["calinski_harabasz"] = float(calinski_harabasz_score(x, labels))
        artifacts["internal_metrics"] = metrics
    except Exception:
        artifacts["internal_metrics"] = {}
    return out, result, artifacts


def fit_predict_two_stage_mechanism_kmeans(
    df: pd.DataFrame,
    *,
    stage1_feature_cols: list[str],
    stage2_feature_cols: list[str],
    stage2_k: int = 2,
    label_col: str = "mech_class_id",
    prob_prefix: str = "mech_class_prob_",
    endmember_col: str = "mech_endmember_id",
    subclass_col: str = "mech_subclass_id",
    carbonate_order_by_col: str = "ratio_cao_al",
    stage2_order_by_col: str = "proxy_metal_index",
    class_names: list[str] | None = None,
    random_state: int = 42,
    n_init: int = 20,
    max_iter: int = 300,
    init: str = "k-means++",
    scaler_kind: str = "robust",
    softmax_temperature: float | None = None,
    stage1_fit_mask: np.ndarray | None = None,
    stage2_fit_mask: np.ndarray | None = None,
) -> tuple[pd.DataFrame, TwoStageMechanismClusteringResult, dict[str, Any]]:
    """
    Two-stage KMeans mechanism workflow:
      Stage-1: carbonate vs non-carbonate endmember using (log) ratios (k=2)
      Stage-2: within non-carbonate, split enrichment mechanism into k=stage2_k

    Notes
    - Stage-1 cluster is mapped to `endmember_col` with 0=carbonate, 1=non-carbonate using `carbonate_order_by_col`.
    - Stage-2 subclasses are ordered by median of `stage2_order_by_col` (descending) for stable interpretation.
    - Final `label_col` is 0 for carbonate, and 1..k for Stage-2 subclasses.
    - Returned prob columns are pseudo-probabilities derived from softmax over centroid distances.
    """
    if stage2_k < 2:
        raise ValueError("stage2_k must be >= 2")
    if not stage1_feature_cols or not stage2_feature_cols:
        raise ValueError("stage1_feature_cols and stage2_feature_cols must be non-empty")

    missing1 = [c for c in stage1_feature_cols if c not in df.columns]
    missing2 = [c for c in stage2_feature_cols if c not in df.columns]
    if missing1:
        raise ValueError(f"Missing stage1_feature_cols: {missing1}")
    if missing2:
        raise ValueError(f"Missing stage2_feature_cols: {missing2}")
    if carbonate_order_by_col not in df.columns:
        raise ValueError(f"carbonate_order_by_col '{carbonate_order_by_col}' not found in DataFrame")
    if stage2_order_by_col not in df.columns:
        raise ValueError(f"stage2_order_by_col '{stage2_order_by_col}' not found in DataFrame")

    def _prep_x(cols: list[str]) -> tuple[np.ndarray, SimpleImputer, Any]:
        x_raw = _coerce_numeric_frame(df, cols).to_numpy(dtype=np.float64, copy=True)
        x_raw = np.where(np.isfinite(x_raw), x_raw, np.nan)
        imp = SimpleImputer(strategy="median")
        x_imp = imp.fit_transform(x_raw)
        sk = str(scaler_kind or "robust").strip().lower()
        if sk == "standard":
            scaler = StandardScaler(with_mean=True, with_std=True)
        else:
            sk = "robust"
            scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))
        x = scaler.fit_transform(x_imp)
        return x, imp, scaler


    stage1_fit_mask_arr = None
    if stage1_fit_mask is not None:
        m = np.asarray(stage1_fit_mask, dtype=bool).reshape(-1)
        if len(m) != len(df):
            raise ValueError(
                f"stage1_fit_mask length mismatch: got {len(m)} expected {len(df)}"
            )
        stage1_fit_mask_arr = m

    x1_all, imp1, sc1 = _prep_x(stage1_feature_cols)
    if stage1_fit_mask_arr is None:
        x1_fit = x1_all
    else:
        if int(np.sum(stage1_fit_mask_arr)) < 6:
            raise ValueError(
                "Too few samples for stage1 fitting after applying stage1_fit_mask: "
                f"n_fit={int(np.sum(stage1_fit_mask_arr))}"
            )
        x1_fit = x1_all[stage1_fit_mask_arr]

    km1 = KMeans(
        n_clusters=2,
        random_state=int(random_state),
        n_init=int(n_init),
        max_iter=int(max_iter),
        init=str(init or "k-means++"),
    )
    km1.fit(x1_fit)
    lab1_raw = km1.predict(x1_all).astype(int)


    ratio = pd.to_numeric(df[carbonate_order_by_col], errors="coerce").to_numpy(dtype=np.float64, copy=True)
    ratio_fit = ratio if stage1_fit_mask_arr is None else ratio[stage1_fit_mask_arr]
    lab1_fit = lab1_raw if stage1_fit_mask_arr is None else lab1_raw[stage1_fit_mask_arr]

    med1 = []
    for k in range(2):
        m1 = np.nanmedian(ratio_fit[lab1_fit == k])
        med1.append(float(m1) if np.isfinite(m1) else float("nan"))
    carbonate_raw = int(np.nanargmax(np.array([(-np.inf if not np.isfinite(m1) else m1) for m1 in med1])))


    endmember = np.where(lab1_raw == carbonate_raw, 0, 1).astype(int)

    centers1 = np.asarray(km1.cluster_centers_, dtype=np.float64)
    d2_1 = np.sum((x1_all[:, None, :] - centers1[None, :, :]) ** 2, axis=2)
    if softmax_temperature is None or not np.isfinite(float(softmax_temperature)) or float(softmax_temperature) <= 0:
        t1 = float(np.nanmedian(d2_1)) if np.isfinite(np.nanmedian(d2_1)) else 1.0
        t1 = max(t1, 1e-6)
    else:
        t1 = float(softmax_temperature)
    logits1 = -d2_1 / t1
    logits1 = logits1 - np.max(logits1, axis=1, keepdims=True)
    p1 = np.exp(logits1)
    p1 = p1 / np.clip(np.sum(p1, axis=1, keepdims=True), 1e-12, None)

    p_end = np.zeros_like(p1)
    p_end[:, 0] = p1[:, carbonate_raw]
    p_end[:, 1] = p1[:, 1 - carbonate_raw]


    mask_nc = endmember == 1
    fit_mask_nc = mask_nc
    if stage2_fit_mask is not None:
        m = np.asarray(stage2_fit_mask, dtype=bool).reshape(-1)
        if len(m) != len(df):
            raise ValueError(
                f"stage2_fit_mask length mismatch: got {len(m)} expected {len(df)}"
            )
        fit_mask_nc = fit_mask_nc & m

    if int(np.sum(fit_mask_nc)) < int(stage2_k) * 3:
        raise ValueError(
            "Too few non-carbonate samples for stage2 fitting after applying stage2_fit_mask: "
            f"n_fit={int(np.sum(fit_mask_nc))} stage2_k={int(stage2_k)}"
        )

    x2_fit_raw = _coerce_numeric_frame(df.loc[fit_mask_nc].copy(), stage2_feature_cols).to_numpy(dtype=np.float64, copy=True)
    x2_fit_raw = np.where(np.isfinite(x2_fit_raw), x2_fit_raw, np.nan)
    imp2 = SimpleImputer(strategy="median")
    x2_fit_imp = imp2.fit_transform(x2_fit_raw)
    sk2 = str(scaler_kind or "robust").strip().lower()
    if sk2 == "standard":
        sc2 = StandardScaler(with_mean=True, with_std=True)
    else:
        sk2 = "robust"
        sc2 = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))
    x2_fit = sc2.fit_transform(x2_fit_imp)

    km2 = KMeans(
        n_clusters=int(stage2_k),
        random_state=int(random_state),
        n_init=int(n_init),
        max_iter=int(max_iter),
        init=str(init or "k-means++"),
    )
    lab2_fit_raw = km2.fit_predict(x2_fit).astype(int)


    order2_raw = pd.to_numeric(df.loc[fit_mask_nc, stage2_order_by_col], errors="coerce").to_numpy(dtype=np.float64, copy=True)
    med2 = []
    for k in range(int(stage2_k)):
        m2 = np.nanmedian(order2_raw[lab2_fit_raw == k])
        med2.append(float(m2) if np.isfinite(m2) else float("nan"))
    order2 = np.argsort(np.array([np.inf if not np.isfinite(m2) else -m2 for m2 in med2]))
    inv2 = np.empty_like(order2)
    inv2[order2] = np.arange(order2.size)


    x2_all_raw = _coerce_numeric_frame(df.loc[mask_nc].copy(), stage2_feature_cols).to_numpy(dtype=np.float64, copy=True)
    x2_all_raw = np.where(np.isfinite(x2_all_raw), x2_all_raw, np.nan)
    x2_all_imp = imp2.transform(x2_all_raw)
    x2_all = sc2.transform(x2_all_imp)
    lab2_all_raw = km2.predict(x2_all).astype(int)
    lab2_all = inv2[lab2_all_raw]

    centers2 = np.asarray(km2.cluster_centers_, dtype=np.float64)
    d2_2 = np.sum((x2_all[:, None, :] - centers2[None, :, :]) ** 2, axis=2)
    if softmax_temperature is None or not np.isfinite(float(softmax_temperature)) or float(softmax_temperature) <= 0:
        t2 = float(np.nanmedian(d2_2)) if np.isfinite(np.nanmedian(d2_2)) else 1.0
        t2 = max(t2, 1e-6)
    else:
        t2 = float(softmax_temperature)
    logits2 = -d2_2 / t2
    logits2 = logits2 - np.max(logits2, axis=1, keepdims=True)
    p2 = np.exp(logits2)
    p2 = p2 / np.clip(np.sum(p2, axis=1, keepdims=True), 1e-12, None)
    p2_ord = p2[:, order2]


    out = df.copy()
    out[endmember_col] = endmember.astype(int)

    subclass_full = np.full(shape=(len(out),), fill_value=-1, dtype=int)
    if mask_nc.any():
        subclass_full[mask_nc] = lab2_all.astype(int)
    out[subclass_col] = subclass_full.astype(int)

    final_labels = np.full(shape=(len(out),), fill_value=0, dtype=int)
    if mask_nc.any():
        final_labels[mask_nc] = 1 + lab2_all.astype(int)
    out[label_col] = final_labels.astype(int)

    n_classes = 1 + int(stage2_k)
    if class_names is None:
        class_names = ["carbonate_background"] + [f"noncarbonate_mode_{i+1}" for i in range(int(stage2_k))]
    if len(class_names) != n_classes:
        raise ValueError(f"class_names length must equal n_classes ({n_classes}), got {len(class_names)}")


    probs_final = np.zeros((len(out), n_classes), dtype=np.float64)
    probs_final[:, 0] = p_end[:, 0]
    if mask_nc.any():
        probs_final[mask_nc, 1:] = (p_end[mask_nc, 1:2] * p2_ord).astype(np.float64)
    for c in range(n_classes):
        out[f"{prob_prefix}{c}"] = probs_final[:, c].astype(np.float32)

    result = TwoStageMechanismClusteringResult(
        algorithm="two_stage_kmeans",
        label_col=label_col,
        class_names=list(class_names),
        n_classes=int(n_classes),
        random_state=int(random_state),
        stage1={
            "feature_cols": list(stage1_feature_cols),
            "carbonate_order_by_col": str(carbonate_order_by_col),
            "carbonate_raw_label": int(carbonate_raw),
            "cluster_order_by_median": med1,
            "kmeans_inertia": float(getattr(km1, "inertia_", float("nan"))),
        },
        stage2={
            "feature_cols": list(stage2_feature_cols),
            "n_clusters": int(stage2_k),
            "order_by_col": str(stage2_order_by_col),
            "cluster_order_by_median": [med2[int(i)] for i in order2.tolist()],
            "kmeans_inertia": float(getattr(km2, "inertia_", float("nan"))),
        },
    )

    artifacts: dict[str, Any] = {
        "result": {
            "algorithm": result.algorithm,
            "label_col": result.label_col,
            "class_names": result.class_names,
            "n_classes": result.n_classes,
            # For backward-compatible visualization code that expects a flat list.
            "feature_cols": list(stage1_feature_cols) + list(stage2_feature_cols),
        },
        "settings": {
            "scaler_kind": str(str(scaler_kind or "robust").strip().lower()),
            "init": str(init or "k-means++"),
            "n_init": int(n_init),
            "max_iter": int(max_iter),
        },
        "stage1": {
            "feature_cols": list(stage1_feature_cols),
            "carbonate_order_by_col": str(carbonate_order_by_col),
            "carbonate_raw_label": int(carbonate_raw),
            "missing_rate": _missing_rate_summary(
                df if stage1_fit_mask_arr is None else df.loc[stage1_fit_mask_arr].copy(),
                stage1_feature_cols,
            ),
            "fit_mask": {
                "n_total": int(len(df)),
                "n_fit": int(len(df)) if stage1_fit_mask_arr is None else int(np.sum(stage1_fit_mask_arr)),
                "fit_fraction": None if stage1_fit_mask_arr is None else float(np.sum(stage1_fit_mask_arr)) / float(len(df)) if len(df) > 0 else None,
            },
            "imputer": {"strategy": "median", "statistics_": imp1.statistics_.tolist()},
            "scaler": {
                "center_": sc1.center_.tolist() if hasattr(sc1, "center_") else None,
                "scale_": sc1.scale_.tolist() if hasattr(sc1, "scale_") else None,
                "quantile_range": list(getattr(sc1, "quantile_range", (25.0, 75.0))),
            },
            "kmeans": {
                "n_clusters": 2,
                "inertia_": float(getattr(km1, "inertia_", float("nan"))),
                "cluster_centers_": np.asarray(getattr(km1, "cluster_centers_", np.empty((0, 0))), dtype=float).tolist(),
            },
            "soft_assignment": {"method": "softmax(-d2/t)", "temperature": float(t1)},
        },
        "stage2": {
            "feature_cols": list(stage2_feature_cols),
            "n_clusters": int(stage2_k),
            "order_by_col": str(stage2_order_by_col),
            "missing_rate": _missing_rate_summary(df.loc[fit_mask_nc].copy(), stage2_feature_cols),
            "fit_mask": {
                "n_noncarbonate_total": int(np.sum(mask_nc)),
                "n_fit": int(np.sum(fit_mask_nc)),
                "fit_fraction_of_noncarbonate": float(np.sum(fit_mask_nc)) / float(np.sum(mask_nc)) if int(np.sum(mask_nc)) > 0 else None,
            },
            "imputer": {"strategy": "median", "statistics_": imp2.statistics_.tolist()},
            "scaler": {
                "center_": sc2.center_.tolist() if hasattr(sc2, "center_") else None,
                "scale_": sc2.scale_.tolist() if hasattr(sc2, "scale_") else None,
                "quantile_range": list(getattr(sc2, "quantile_range", (25.0, 75.0))),
            },
            "kmeans": {
                "n_clusters": int(stage2_k),
                "inertia_": float(getattr(km2, "inertia_", float("nan"))),
                "cluster_centers_": np.asarray(getattr(km2, "cluster_centers_", np.empty((0, 0))), dtype=float).tolist(),
            },
            "soft_assignment": {"method": "softmax(-d2/t)", "temperature": float(t2)},
            "cluster_order_by_median": [med2[int(i)] for i in order2.tolist()],
        },
        "internal_metrics": {},
    }


    try:
        metrics: dict[str, Any] = {}
        if x1_fit.shape[0] >= 3:

            metrics["stage1_silhouette"] = float(silhouette_score(x1_fit, lab1_fit))
            metrics["stage1_davies_bouldin"] = float(davies_bouldin_score(x1_fit, lab1_fit))
            metrics["stage1_calinski_harabasz"] = float(calinski_harabasz_score(x1_fit, lab1_fit))
        if x2_fit.shape[0] >= 3 and int(stage2_k) < x2_fit.shape[0]:
            metrics["stage2_silhouette"] = float(silhouette_score(x2_fit, lab2_fit_raw))
            metrics["stage2_davies_bouldin"] = float(davies_bouldin_score(x2_fit, lab2_fit_raw))
            metrics["stage2_calinski_harabasz"] = float(calinski_harabasz_score(x2_fit, lab2_fit_raw))
        artifacts["internal_metrics"] = metrics
    except Exception:
        artifacts["internal_metrics"] = {}

    return out, result, artifacts


def fit_predict_grade_geochem_classification(
    df: pd.DataFrame,
    *,
    stage1_feature_cols: list[str],
    stage2_feature_cols: list[str],
    grade_feature_cols: list[str] | None = None,
    mech_label_col: str = "mech_class_id",
    mech_prob_prefix: str = "mech_class_prob_",
    grade_label_col: str = "grade_geochem_class_id",
    grade_name_col: str = "grade_geochem_class_name",
    grade_prob_prefix: str = "grade_geochem_class_prob_",
    endmember_col: str = "mech_endmember_id",
    subclass_col: str = "mech_subclass_id",
    class1_grade_subclass_col: str = "class1_grade_subclass_id",
    class2_grade_subclass_col: str = "class2_grade_subclass_id",
    rey_col: str = "rey",
    random_state: int = 42,
    n_init: int = 20,
    max_iter: int = 300,
    init: str = "k-means++",
    scaler_kind: str = "robust",
    softmax_temperature: float | None = None,
    stage1_fit_mask: np.ndarray | None = None,
    stage2_fit_mask: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Build the default 5-class grade-aware geochemical labels.

    Main classes are the existing two-stage mechanism classes:
    0 carbonate_background, 1 noncarbonate_metal_enriched, 2 noncarbonate_detrital_background.
    Within class 1 and class 2, separate low/high grade with k=2 using log1p(La..Lu+Y).
    """
    if rey_col not in df.columns:
        raise ValueError(f"rey_col '{rey_col}' not found in DataFrame")

    grade_feature_cols = list(grade_feature_cols or REE_Y_FEATURE_COLS)
    missing_grade = [c for c in grade_feature_cols if c not in df.columns]
    if missing_grade:
        raise ValueError(
            "Grade geochem classification requires REE+Y element columns. "
            f"Missing: {missing_grade}"
        )

    mech_names = [
        "carbonate_background",
        "noncarbonate_metal_enriched",
        "noncarbonate_detrital_background",
    ]
    out, _, mech_artifacts = fit_predict_two_stage_mechanism_kmeans(
        df,
        stage1_feature_cols=stage1_feature_cols,
        stage2_feature_cols=stage2_feature_cols,
        stage2_k=2,
        label_col=mech_label_col,
        prob_prefix=mech_prob_prefix,
        endmember_col=endmember_col,
        subclass_col=subclass_col,
        carbonate_order_by_col="ratio_cao_al",
        stage2_order_by_col=("proxy_metal_index" if "proxy_metal_index" in df.columns else stage2_feature_cols[0]),
        class_names=mech_names,
        random_state=int(random_state),
        n_init=int(n_init),
        max_iter=int(max_iter),
        init=str(init or "k-means++"),
        scaler_kind=str(scaler_kind or "robust"),
        softmax_temperature=softmax_temperature,
        stage1_fit_mask=stage1_fit_mask,
        stage2_fit_mask=stage2_fit_mask,
    )

    mech = pd.to_numeric(out[mech_label_col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    class1_mask = mech == 1
    class2_mask = mech == 2

    class1_labels, class1_probs, class1_artifacts = _fit_predict_grade_subclusters(
        out,
        mask=class1_mask,
        feature_cols=grade_feature_cols,
        rey_col=rey_col,
        random_state=int(random_state) + 101,
        n_init=int(n_init),
        max_iter=int(max_iter),
        init=str(init or "k-means++"),
        scaler_kind=str(scaler_kind or "robust"),
        softmax_temperature=softmax_temperature,
    )
    class2_labels, class2_probs, class2_artifacts = _fit_predict_grade_subclusters(
        out,
        mask=class2_mask,
        feature_cols=grade_feature_cols,
        rey_col=rey_col,
        random_state=int(random_state) + 202,
        n_init=int(n_init),
        max_iter=int(max_iter),
        init=str(init or "k-means++"),
        scaler_kind=str(scaler_kind or "robust"),
        softmax_temperature=softmax_temperature,
    )

    out[class1_grade_subclass_col] = -1
    out[class2_grade_subclass_col] = -1
    out.loc[class1_mask, class1_grade_subclass_col] = class1_labels.astype(int)
    out.loc[class2_mask, class2_grade_subclass_col] = class2_labels.astype(int)

    grade_label = np.zeros(len(out), dtype=np.int16)
    grade_label[class1_mask] = 1 + class1_labels.astype(np.int16)
    grade_label[class2_mask] = 3 + class2_labels.astype(np.int16)
    out[grade_label_col] = grade_label.astype(int)
    out[grade_name_col] = [GRADE_GEOCHEM_CLASS_NAMES[int(i)] for i in grade_label.tolist()]

    exploration_label = np.zeros(len(out), dtype=np.int16)
    exploration_label[class1_mask] = 1
    exploration_label[class2_mask] = 2 + class2_labels.astype(np.int16)
    out["exploration_class_id"] = exploration_label.astype(int)
    out["exploration_class_name"] = [EXPLORATION_4CLASS_NAMES[int(i)] for i in exploration_label.tolist()]

    grade_probs = np.zeros((len(out), 5), dtype=np.float64)


    p0 = pd.to_numeric(out.get(f"{mech_prob_prefix}0", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    p1 = pd.to_numeric(out.get(f"{mech_prob_prefix}1", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    p2 = pd.to_numeric(out.get(f"{mech_prob_prefix}2", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    grade_probs[:, 0] = p0
    grade_probs[np.where(class1_mask)[0], 1:3] = p1[class1_mask][:, None] * class1_probs
    grade_probs[np.where(class2_mask)[0], 3:5] = p2[class2_mask][:, None] * class2_probs
    denom = np.clip(np.sum(grade_probs, axis=1, keepdims=True), 1e-12, None)
    grade_probs = grade_probs / denom
    for c in range(5):
        out[f"{grade_prob_prefix}{c}"] = grade_probs[:, c].astype(np.float32)

    exploration_probs = np.zeros((len(out), 4), dtype=np.float64)
    exploration_probs[:, 0] = p0
    exploration_probs[:, 1] = p1
    exploration_probs[np.where(class2_mask)[0], 2:4] = p2[class2_mask][:, None] * class2_probs
    denom4 = np.clip(np.sum(exploration_probs, axis=1, keepdims=True), 1e-12, None)
    exploration_probs = exploration_probs / denom4
    for c in range(4):
        out[f"exploration_class_prob_{c}"] = exploration_probs[:, c].astype(np.float32)

    counts = out[grade_label_col].value_counts().sort_index()
    exploration_counts = out["exploration_class_id"].value_counts().sort_index()
    artifacts = {
        "algorithm": "two_stage_mechanism_plus_grade_kmeans",
        "fit_domain": "all",
        "label_col": grade_label_col,
        "class_names": list(GRADE_GEOCHEM_CLASS_NAMES),
        "exploration_label_col": "exploration_class_id",
        "exploration_class_names": list(EXPLORATION_4CLASS_NAMES),
        "class_name_mapping": {str(i): name for i, name in enumerate(GRADE_GEOCHEM_CLASS_NAMES)},
        "exploration_class_name_mapping": {str(i): name for i, name in enumerate(EXPLORATION_4CLASS_NAMES)},
        "class_counts": {str(int(k)): int(v) for k, v in counts.items()},
        "exploration_class_counts": {str(int(k)): int(v) for k, v in exploration_counts.items()},
        "intermediate_columns": {
            "mech_label_col": mech_label_col,
            "mech_subclass_col": subclass_col,
            "class1_grade_subclass_col": class1_grade_subclass_col,
            "class2_grade_subclass_col": class2_grade_subclass_col,
            "exploration_label_col": "exploration_class_id",
        },
        "stage1": mech_artifacts.get("stage1", {}),
        "stage2": mech_artifacts.get("stage2", {}),
        "mechanism_artifacts": {
            "result": mech_artifacts.get("result", {}),
            "feature_selection": mech_artifacts.get("feature_selection", {}),
            "internal_metrics": mech_artifacts.get("internal_metrics", {}),
        },
        "class1_grade_kmeans": class1_artifacts,
        "class2_grade_kmeans": class2_artifacts,
        "post_hoc_rey_summary_per_class": _rey_summary_by_class(
            out,
            label_col=grade_label_col,
            class_names=list(GRADE_GEOCHEM_CLASS_NAMES),
            rey_col=rey_col,
        ),
        "post_hoc_rey_summary_per_exploration_class": _rey_summary_by_class(
            out,
            label_col="exploration_class_id",
            class_names=list(EXPLORATION_4CLASS_NAMES),
            rey_col=rey_col,
        ),
    }
    return out, artifacts


# Backward-compatible API name (older code calls *_gmm).
def fit_predict_mechanism_gmm(*args, **kwargs):
    return fit_predict_mechanism_kmeans(*args, **kwargs)

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class MechanismClusterSummary:
    label_col: str
    class_ids: list[int]
    class_names: list[str]
    counts: dict[int, int]
    median_table: pd.DataFrame
    mean_table: pd.DataFrame


def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def summarize_clusters(
    df: pd.DataFrame,
    *,
    label_col: str = "mech_class_id",
    feature_cols: Iterable[str],
    class_names: list[str] | None = None,
) -> MechanismClusterSummary:
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")
    feature_cols = [c for c in feature_cols if c in df.columns]
    if not feature_cols:
        raise ValueError("No valid feature_cols found in DataFrame")

    labels = _coerce_numeric_series(df[label_col]).astype("Int64")
    valid_mask = labels.notna()
    dfv = df.loc[valid_mask, [label_col, *feature_cols]].copy()
    dfv[label_col] = _coerce_numeric_series(dfv[label_col]).astype(int)

    for c in feature_cols:
        dfv[c] = _coerce_numeric_series(dfv[c])

    class_ids = sorted(dfv[label_col].unique().tolist())
    if class_names is None:
        class_names = [f"class_{i}" for i in class_ids]
    else:
        if len(class_names) <= max(class_ids):
            padded = list(class_names) + [f"class_{i}" for i in range(len(class_names), max(class_ids) + 1)]
            class_names = padded
        class_names = [class_names[i] for i in class_ids]

    counts = dfv[label_col].value_counts().to_dict()
    med = dfv.groupby(label_col)[feature_cols].median(numeric_only=True)
    mean = dfv.groupby(label_col)[feature_cols].mean(numeric_only=True)

    med.index.name = "class_id"
    mean.index.name = "class_id"
    return MechanismClusterSummary(
        label_col=label_col,
        class_ids=class_ids,
        class_names=class_names,
        counts={int(k): int(v) for k, v in counts.items()},
        median_table=med,
        mean_table=mean,
    )


def plot_class_counts(summary: MechanismClusterSummary):
    import matplotlib.pyplot as plt

    class_ids = summary.class_ids
    names = summary.class_names
    values = [summary.counts.get(int(i), 0) for i in class_ids]

    fig, ax = plt.subplots(figsize=(8, 3.2), constrained_layout=True)
    ax.bar(range(len(class_ids)), values, color="#4C78A8")
    ax.set_xticks(range(len(class_ids)))
    ax.set_xticklabels([f"{n}\n(id={i})" for n, i in zip(names, class_ids)], rotation=0)
    ax.set_ylabel("n")
    ax.set_title("Mechanism class counts")
    for i, v in enumerate(values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    return fig


def plot_feature_boxplot(
    df: pd.DataFrame,
    *,
    label_col: str,
    feature: str,
    class_ids: list[int] | None = None,
    class_names: list[str] | None = None,
    log1p: bool = False,
):
    import matplotlib.pyplot as plt

    if label_col not in df.columns or feature not in df.columns:
        raise ValueError("Missing label_col/feature")
    labels = _coerce_numeric_series(df[label_col]).astype("Int64")
    values = _coerce_numeric_series(df[feature])
    mask = labels.notna() & values.notna()
    d = pd.DataFrame({label_col: labels[mask].astype(int), feature: values[mask]})
    if log1p:
        d[feature] = np.log1p(d[feature].clip(lower=0))

    if class_ids is None:
        class_ids = sorted(d[label_col].unique().tolist())
    if class_names is None:
        class_names = [f"class_{i}" for i in class_ids]
    else:
        if len(class_names) <= max(class_ids):
            padded = list(class_names) + [f"class_{i}" for i in range(len(class_names), max(class_ids) + 1)]
            class_names = padded
        class_names = [class_names[i] for i in class_ids]

    data = [d.loc[d[label_col] == i, feature].to_numpy() for i in class_ids]
    fig, ax = plt.subplots(figsize=(8, 3.2), constrained_layout=True)
    ax.boxplot(data, labels=[f"{n}\n(id={i})" for n, i in zip(class_names, class_ids)], showfliers=False)
    ax.set_title(f"{feature} by mechanism class" + (" (log1p)" if log1p else ""))
    ax.set_ylabel(feature + (" (log1p)" if log1p else ""))
    return fig


def plot_class_feature_heatmap(
    table: pd.DataFrame,
    *,
    title: str = "Class-wise feature medians (z-scored per feature)",
    row_labels: list[str] | None = None,
):
    import matplotlib.pyplot as plt

    if table.empty:
        raise ValueError("Empty table")
    vals = table.to_numpy(dtype=float, copy=True)

    mu = np.nanmean(vals, axis=0, keepdims=True)
    sd = np.nanstd(vals, axis=0, keepdims=True)
    sd = np.where(sd > 0, sd, 1.0)
    z = (vals - mu) / sd

    fig, ax = plt.subplots(figsize=(max(8, 0.28 * table.shape[1]), 2.6), constrained_layout=True)
    im = ax.imshow(z, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    ax.set_yticks(range(table.shape[0]))
    if row_labels and len(row_labels) == table.shape[0]:
        ax.set_yticklabels(row_labels)
    else:
        ax.set_yticklabels([str(i) for i in table.index.tolist()])
    ax.set_xticks(range(table.shape[1]))
    ax.set_xticklabels(list(table.columns), rotation=60, ha="right", fontsize=8)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("z-score")
    return fig


def plot_pca_scatter(
    df: pd.DataFrame,
    *,
    label_col: str,
    feature_cols: Iterable[str],
    log1p_transform: bool = True,
):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import RobustScaler

    feature_cols = [c for c in feature_cols if c in df.columns]
    if label_col not in df.columns or not feature_cols:
        raise ValueError("Missing label_col or empty feature_cols")
    labels = _coerce_numeric_series(df[label_col]).astype("Int64")
    x = df.loc[labels.notna(), feature_cols].copy()
    y = labels[labels.notna()].astype(int).to_numpy()
    for c in feature_cols:
        x[c] = _coerce_numeric_series(x[c])
    xr = x.to_numpy(dtype=np.float64, copy=True)
    xr = np.where(np.isfinite(xr), xr, np.nan)
    if log1p_transform:
        xr = np.where(xr >= 0, np.log1p(xr), np.nan)
    xr = SimpleImputer(strategy="median").fit_transform(xr)
    xr = RobustScaler(with_centering=True, with_scaling=True).fit_transform(xr)
    p = PCA(n_components=2, random_state=0).fit_transform(xr)

    fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    sc = ax.scatter(p[:, 0], p[:, 1], c=y, cmap="tab10", s=14, alpha=0.7)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(
        "PCA of geochemical features ("
        + ("log1p + " if log1p_transform else "")
        + "robust scale)"
    )
    cb = fig.colorbar(sc, ax=ax, shrink=0.85)
    cb.set_label("class_id")
    return fig


def plot_kmeans_elbow_curve(
    df: pd.DataFrame,
    *,
    feature_cols: Iterable[str],
    k_values: Iterable[int] = (2, 3, 4, 5, 6, 7, 8),
    random_state: int = 42,
    n_init: int = 20,
    log1p_transform: bool = True,
):
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import RobustScaler

    feature_cols = [c for c in feature_cols if c in df.columns]
    if not feature_cols:
        raise ValueError("No valid feature_cols found for elbow curve")

    x = df.loc[:, feature_cols].copy()
    for c in feature_cols:
        x[c] = _coerce_numeric_series(x[c])
    xr = x.to_numpy(dtype=np.float64, copy=True)
    xr = np.where(np.isfinite(xr), xr, np.nan)
    if log1p_transform:
        xr = np.where(xr >= 0, np.log1p(xr), np.nan)
    xr = SimpleImputer(strategy="median").fit_transform(xr)
    xr = RobustScaler(with_centering=True, with_scaling=True).fit_transform(xr)

    ks = [int(k) for k in k_values if int(k) >= 2]
    inertias: list[float] = []
    for k in ks:
        km = KMeans(n_clusters=int(k), random_state=int(random_state), n_init=int(n_init))
        km.fit(xr)
        inertias.append(float(getattr(km, "inertia_", float("nan"))))

    fig, ax = plt.subplots(figsize=(6.2, 3.2), constrained_layout=True)
    ax.plot(ks, inertias, marker="o")
    ax.set_xlabel("K (n_clusters)")
    ax.set_ylabel("Inertia (within-cluster SSE)")
    ax.set_title("KMeans model selection (elbow)")
    ax.grid(True, alpha=0.3)
    return fig


def plot_endmember_scatter(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    label_col: str,
    title: str = "Stage-1 endmember separation",
):
    if x_col not in df.columns or y_col not in df.columns or label_col not in df.columns:
        raise ValueError("Missing x_col/y_col/label_col")
    x = _coerce_numeric_series(df[x_col])
    y = _coerce_numeric_series(df[y_col])
    lab = _coerce_numeric_series(df[label_col]).astype("Int64")
    m = x.notna() & y.notna() & lab.notna()
    x = x[m].to_numpy(dtype=float)
    y = y[m].to_numpy(dtype=float)
    labv = lab[m].astype(int).to_numpy()

    fig, ax = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
    sc = ax.scatter(x, y, c=labv, cmap="tab10", s=14, alpha=0.65)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    cb = fig.colorbar(sc, ax=ax, shrink=0.85)
    cb.set_label(label_col)
    return fig


def plot_kmeans_k_selection_metrics(
    df: pd.DataFrame,
    *,
    feature_cols: Iterable[str],
    k_values: Iterable[int] = (2, 3, 4, 5, 6, 7, 8),
    random_state: int = 42,
    n_init: int = 20,
    max_iter: int = 300,
    init: str = "k-means++",
    log1p_transform: bool = True,
    scaler_kind: str = "robust",
):
    """
    Plot internal KMeans metrics vs K (for quick clustering diagnostics).

    These are *unsupervised* feature-space metrics; they are not the final model evaluation.
    """
    from sklearn.cluster import KMeans
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
    from sklearn.preprocessing import RobustScaler, StandardScaler

    feature_cols = [c for c in feature_cols if c in df.columns]
    if not feature_cols:
        raise ValueError("No valid feature_cols found for KMeans metrics")

    x = df.loc[:, feature_cols].copy()
    for c in feature_cols:
        x[c] = _coerce_numeric_series(x[c])
    xr = x.to_numpy(dtype=np.float64, copy=True)
    xr = np.where(np.isfinite(xr), xr, np.nan)
    if log1p_transform:
        xr = np.where(xr >= 0, np.log1p(xr), np.nan)
    xr = SimpleImputer(strategy="median").fit_transform(xr)

    sk = str(scaler_kind or "robust").strip().lower()
    if sk == "standard":
        scaler = StandardScaler(with_mean=True, with_std=True)
    else:
        sk = "robust"
        scaler = RobustScaler(with_centering=True, with_scaling=True)
    xr = scaler.fit_transform(xr)

    ks = [int(k) for k in k_values if int(k) >= 2 and int(k) < xr.shape[0]]
    if not ks:
        raise ValueError("No valid K values to evaluate")

    inertia: list[float] = []
    sil: list[float] = []
    db: list[float] = []
    ch: list[float] = []

    for k in ks:
        km = KMeans(
            n_clusters=int(k),
            random_state=int(random_state),
            n_init=int(n_init),
            max_iter=int(max_iter),
            init=str(init or "k-means++"),
        )
        labels = km.fit_predict(xr)
        inertia.append(float(getattr(km, "inertia_", float("nan"))))
        try:
            sil.append(float(silhouette_score(xr, labels)))
        except Exception:
            sil.append(float("nan"))
        try:
            db.append(float(davies_bouldin_score(xr, labels)))
        except Exception:
            db.append(float("nan"))
        try:
            ch.append(float(calinski_harabasz_score(xr, labels)))
        except Exception:
            ch.append(float("nan"))

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(ks, inertia, marker="o")
    ax.set_title("Inertia (lower is tighter)")
    ax.set_xlabel("K")
    ax.set_ylabel("Inertia")
    ax.grid(True, alpha=0.25)

    ax = axes[0, 1]
    ax.plot(ks, sil, marker="o", color="#2c7fb8")
    ax.set_title("Silhouette (higher is better)")
    ax.set_xlabel("K")
    ax.set_ylabel("Silhouette")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 0]
    ax.plot(ks, db, marker="o", color="#d95f0e")
    ax.set_title("Davies–Bouldin (lower is better)")
    ax.set_xlabel("K")
    ax.set_ylabel("DB index")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    ax.plot(ks, ch, marker="o", color="#31a354")
    ax.set_title("Calinski–Harabasz (higher is better)")
    ax.set_xlabel("K")
    ax.set_ylabel("CH score")
    ax.grid(True, alpha=0.25)

    fig.suptitle(f"KMeans internal metrics (log1p={bool(log1p_transform)}, scaler={sk})", y=1.02)
    return fig

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _safe_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    aa = _num(a)
    bb = _num(b)
    return aa / bb.replace(0, np.nan)


def _geom_mean(a: pd.Series, b: pd.Series) -> pd.Series:
    aa = _num(a)
    bb = _num(b)
    m = (aa > 0) & (bb > 0)
    out = pd.Series(np.nan, index=aa.index, dtype="float64")
    out.loc[m] = np.sqrt((aa.loc[m].astype(float) * bb.loc[m].astype(float)).to_numpy())
    return out


def add_mechanism_proxy_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a small set of mechanism-oriented proxy features derived from available geochemistry columns.

    Notes
    - This function does NOT use ΣREY / 'rey' for proxy calculation.
    - Proxies are computed in "raw concentration space" (ppm / wt%); downstream clustering may apply log1p+scaling.
    """
    out = df.copy()


    if "Y" in out.columns and "Ho" in out.columns:
        out["proxy_y_ho"] = _safe_ratio(out["Y"], out["Ho"])
    if "U" in out.columns and "Th" in out.columns:
        out["proxy_u_th"] = _safe_ratio(out["U"], out["Th"])
        out["proxy_th_u"] = _safe_ratio(out["Th"], out["U"])


    if "La" in out.columns and "Yb" in out.columns:
        out["proxy_la_yb"] = _safe_ratio(out["La"], out["Yb"])
    elif "La" in out.columns and "Lu" in out.columns:
        out["proxy_la_lu"] = _safe_ratio(out["La"], out["Lu"])


    if "Dy" in out.columns and "Yb" in out.columns:
        out["proxy_dy_yb"] = _safe_ratio(out["Dy"], out["Yb"])
    elif "Gd" in out.columns and "Yb" in out.columns:
        out["proxy_gd_yb"] = _safe_ratio(out["Gd"], out["Yb"])


    if "Ce" in out.columns:
        ce = _num(out["Ce"])
        ce_star = None
        if "La" in out.columns and "Pr" in out.columns:
            ce_star = _geom_mean(out["La"], out["Pr"])
        elif "La" in out.columns and "Nd" in out.columns:
            ce_star = _geom_mean(out["La"], out["Nd"])
        if ce_star is not None:
            out["proxy_ce_anom"] = ce / ce_star.replace(0, np.nan)


    if "Eu" in out.columns:
        eu = _num(out["Eu"])
        eu_star = None
        if "Sm" in out.columns and "Gd" in out.columns:
            eu_star = _geom_mean(out["Sm"], out["Gd"])
        elif "Sm" in out.columns and "Tb" in out.columns:
            eu_star = _geom_mean(out["Sm"], out["Tb"])
        if eu_star is not None:
            out["proxy_eu_anom"] = eu / eu_star.replace(0, np.nan)

    return out


def _safe_log_pos(x: pd.Series) -> pd.Series:
    xx = _num(x)
    xx = xx.where(np.isfinite(xx), np.nan)
    xx = xx.where(xx > 0, np.nan)
    return np.log(xx.astype("float64"))


def add_two_stage_mechanism_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features used by the two-stage mechanism workflow:
    - Stage-1 endmember features: log(CaO/Al2O3), log(TiO2/Al2O3)
    - Stage-2 mechanism features within non-carbonate: log(P2O5/Al2O3), log(MnO/Al2O3), log(Fe2O3/Al2O3),
      log(Mo/Al2O3), log(V/Al2O3), log(Zn/Al2O3) plus a composite metal index.

    Notes
    - Ratios are computed in raw space, then log-transformed with natural log. Non-positive ratios become NaN
      and can be imputed downstream.
    - This function does not use ΣREY/'rey' at all.
    """
    out = df.copy()


    if "CaO" in out.columns and "Al2O3" in out.columns:
        out["ratio_cao_al"] = _safe_ratio(out["CaO"], out["Al2O3"])
        out["log_ratio_cao_al"] = _safe_log_pos(out["ratio_cao_al"])
    if "TiO2" in out.columns and "Al2O3" in out.columns:
        out["ratio_tio2_al"] = _safe_ratio(out["TiO2"], out["Al2O3"])
        out["log_ratio_tio2_al"] = _safe_log_pos(out["ratio_tio2_al"])

    if "P2O5" in out.columns and "Al2O3" in out.columns:
        out["ratio_p_al"] = _safe_ratio(out["P2O5"], out["Al2O3"])
        out["log_ratio_p_al"] = _safe_log_pos(out["ratio_p_al"])
    if "MnO" in out.columns and "Al2O3" in out.columns:
        out["ratio_mn_al"] = _safe_ratio(out["MnO"], out["Al2O3"])
        out["log_ratio_mn_al"] = _safe_log_pos(out["ratio_mn_al"])
    if "Fe2O3" in out.columns and "Al2O3" in out.columns:
        out["ratio_fe_al"] = _safe_ratio(out["Fe2O3"], out["Al2O3"])
        out["log_ratio_fe_al"] = _safe_log_pos(out["ratio_fe_al"])
    if "Mo" in out.columns and "Al2O3" in out.columns:
        out["ratio_mo_al"] = _safe_ratio(out["Mo"], out["Al2O3"])
        out["log_ratio_mo_al"] = _safe_log_pos(out["ratio_mo_al"])
    if "V" in out.columns and "Al2O3" in out.columns:
        out["ratio_v_al"] = _safe_ratio(out["V"], out["Al2O3"])
        out["log_ratio_v_al"] = _safe_log_pos(out["ratio_v_al"])
    if "Zn" in out.columns and "Al2O3" in out.columns:
        out["ratio_zn_al"] = _safe_ratio(out["Zn"], out["Al2O3"])
        out["log_ratio_zn_al"] = _safe_log_pos(out["ratio_zn_al"])


    metal_terms = [
        c
        for c in [
            "log_ratio_mo_al",
            "log_ratio_v_al",
            "log_ratio_mn_al",
            "log_ratio_fe_al",
            "log_ratio_zn_al",
        ]
        if c in out.columns
    ]
    if metal_terms:
        out["proxy_metal_index"] = out[metal_terms].mean(axis=1, skipna=True)

    return out


@dataclass(frozen=True)
class FeatureSelectionResult:
    selected: list[str]
    dropped_missing: list[str]
    dropped_correlated: list[str]


def select_mechanism_feature_cols(
    df: pd.DataFrame,
    *,
    candidates: Iterable[str],
    max_missing_rate: float = 0.30,
    max_abs_corr: float = 0.95,
) -> FeatureSelectionResult:
    """
    Filter candidate clustering variables by:
    - missingness threshold
    - redundancy removal via correlation threshold (greedy)
    """
    candidates = [str(c) for c in candidates if str(c) in df.columns]
    if not candidates:
        return FeatureSelectionResult(selected=[], dropped_missing=[], dropped_correlated=[])

    x = df.loc[:, candidates].copy()
    for c in candidates:
        x[c] = _num(x[c])

    miss = x.isna().mean(axis=0)
    keep = [c for c in candidates if float(miss.get(c, 1.0)) <= float(max_missing_rate)]
    dropped_missing = [c for c in candidates if c not in keep]
    if len(keep) <= 1:
        return FeatureSelectionResult(selected=keep, dropped_missing=dropped_missing, dropped_correlated=[])


    corr = x.loc[:, keep].corr(method="spearman").abs()
    selected: list[str] = []
    dropped_correlated: list[str] = []
    for c in keep:
        if not selected:
            selected.append(c)
            continue
        too_close = False
        for s in selected:
            v = float(corr.loc[c, s]) if (c in corr.index and s in corr.columns) else 0.0
            if np.isfinite(v) and v >= float(max_abs_corr):
                too_close = True
                break
        if too_close:
            dropped_correlated.append(c)
        else:
            selected.append(c)

    return FeatureSelectionResult(
        selected=selected,
        dropped_missing=dropped_missing,
        dropped_correlated=dropped_correlated,
    )

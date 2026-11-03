import os
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def detect_columns(df: pd.DataFrame) -> Tuple[str, str, Optional[str]]:
    """
    Try to detect depth, REY and core ID columns from the raw REY.xlsx.
    - depth: any column containing 'depth' (case-insensitive)
    - REY: any column containing 'rey' (case-insensitive)
    - core_id: priority order: 'core', 'borehole', 'hole', 'site', 'station', 'id'
    """
    cols_lower = {c: c.lower() for c in df.columns}


    depth_candidates = [c for c, lc in cols_lower.items() if "depth" in lc]
    if not depth_candidates:
        raise ValueError(f"Could not find a depth column in columns: {list(df.columns)}")
    depth_col = depth_candidates[0]


    rey_candidates = [c for c, lc in cols_lower.items() if "rey" in lc]
    if not rey_candidates:
        raise ValueError(f"Could not find a REY column in columns: {list(df.columns)}")
    rey_col = rey_candidates[0]


    core_keywords = ["core", "borehole", "hole", "site", "station", "id"]
    core_col: Optional[str] = None
    for key in core_keywords:
        for c, lc in cols_lower.items():
            if key in lc:
                core_col = c
                break
        if core_col is not None:
            break

    return depth_col, rey_col, core_col


def plot_rey_vertical_profiles(
    excel_path: Path,
    output_path: Path,
    sheet_name: Optional[str] = None,
    max_depth_m: Optional[float] = None,
) -> None:
    print(f"[plot_rey_vertical_profiles] Reading REY data from: {excel_path}")
    if not excel_path.exists():
        raise FileNotFoundError(f"REY Excel file not found at {excel_path}")

    if sheet_name is None:
        xls = pd.ExcelFile(excel_path)
        if not xls.sheet_names:
            raise ValueError("No sheets found in REY Excel file.")
        sheet_name = xls.sheet_names[0]
    else:
        xls = pd.ExcelFile(excel_path)
        if sheet_name not in xls.sheet_names:
            raise ValueError(f"Sheet '{sheet_name}' not found. Available: {xls.sheet_names}")

    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    print(f"[plot_rey_vertical_profiles] Loaded sheet '{sheet_name}' with {len(df)} rows.")
    print("[plot_rey_vertical_profiles] Columns:", list(df.columns))

    depth_col, rey_col, core_col = detect_columns(df)
    print(f"[plot_rey_vertical_profiles] Using depth column: '{depth_col}'")
    print(f"[plot_rey_vertical_profiles] Using REY column: '{rey_col}'")
    print(f"[plot_rey_vertical_profiles] Using core ID column: '{core_col}'")


    df_clean = df.dropna(subset=[depth_col, rey_col]).copy()
    if core_col is not None:
        df_clean = df_clean.dropna(subset=[core_col])


    if max_depth_m is not None:
        df_clean = df_clean[df_clean[depth_col] <= max_depth_m]
        print(f"[plot_rey_vertical_profiles] After depth <= {max_depth_m} m filter: {len(df_clean)} rows.")
    else:
        print(f"[plot_rey_vertical_profiles] After dropping NaNs: {len(df_clean)} rows.")


    if core_col is not None:
        unique_cores = df_clean[core_col].astype(str).unique()
        core_to_idx = {core_id: idx for idx, core_id in enumerate(sorted(unique_cores))}
        df_clean["_core_idx"] = df_clean[core_col].astype(str).map(core_to_idx)
    else:

        df_clean["_core_idx"] = 0
        core_to_idx = {"core_0": 0}


    depths = df_clean[depth_col].to_numpy(dtype=float)
    rey_values = df_clean[rey_col].to_numpy(dtype=float)
    core_indices = df_clean["_core_idx"].to_numpy(dtype=float)


    finite_rey = rey_values[np.isfinite(rey_values)]
    if finite_rey.size == 0:
        raise ValueError("No finite REY values found.")

    vmin = np.percentile(finite_rey, 2)
    vmax = np.percentile(finite_rey, 98)
    print(f"[plot_rey_vertical_profiles] REY value range (2-98th percentile): [{vmin:.3f}, {vmax:.3f}]")


    output_path.parent.mkdir(parents=True, exist_ok=True)


    plt.figure(figsize=(10, 8))
    sc = plt.scatter(
        core_indices,
        depths,
        c=rey_values,
        s=15,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        edgecolors="none",
        alpha=0.9,
    )
    plt.gca().invert_yaxis()
    plt.colorbar(sc, label=f"{rey_col} (absolute)")
    plt.xlabel("Core index")
    plt.ylabel(f"{depth_col} (sub-bottom)")
    plt.title("Vertical REY distribution across all sediment cores")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[plot_rey_vertical_profiles] Saved figure to: {output_path}")


def main():
    excel_path = Path("data/raw/REY.xlsx")
    output_dir = Path("outputs/vertical_profiles")
    output_path_all = output_dir / "rey_vertical_profiles.png"
    output_path_15m = output_dir / "rey_vertical_profiles_max15m.png"


    plot_rey_vertical_profiles(excel_path, output_path_all)

    plot_rey_vertical_profiles(excel_path, output_path_15m, max_depth_m=15.0)


if __name__ == "__main__":
    main()

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

import matplotlib as mpl


def _export_formats_from_config(plot_config: Optional[dict]) -> list[str]:
    if not plot_config:
        return ["png"]
    fmts = (
        plot_config.get("export_formats")
        or plot_config.get("publication_formats")
        or plot_config.get("plot_export_formats")
        or None
    )
    if not fmts:
        return ["png"]
    if isinstance(fmts, str):
        fmts = [fmts]
    out: list[str] = []
    for f in fmts:
        if not f:
            continue
        s = str(f).lower().lstrip(".")
        if s in {"png", "pdf", "svg"}:
            out.append(s)
    return out or ["png"]


def _dpi_from_config(plot_config: Optional[dict]) -> int:
    if not plot_config:
        return 300
    dpi = (
        plot_config.get("publication_dpi")
        or plot_config.get("dpi")
        or plot_config.get("plot_dpi")
        or 300
    )
    try:
        return int(dpi)
    except Exception:
        return 300


@contextmanager
def publication_style(plot_config: Optional[dict] = None) -> Iterator[None]:
    """
    Context manager that applies a consistent, journal-friendly plotting style.

    We avoid depending on seaborn styles to keep outputs stable across environments.
    """
    base_font = 10
    if plot_config:
        try:
            base_font = int(plot_config.get("base_fontsize", base_font))
        except Exception:
            base_font = 10

    rc = {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.size": base_font,
        "axes.titlesize": base_font + 2,
        "axes.labelsize": base_font + 1,
        "xtick.labelsize": base_font,
        "ytick.labelsize": base_font,
        "legend.fontsize": base_font - 1,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "lines.linewidth": 1.8,
        "lines.markersize": 5.0,
    }

    with mpl.rc_context(rc=rc):
        yield


def save_figure(
    fig,
    *,
    path_no_ext: str | Path,
    plot_config: Optional[dict] = None,
    dpi: Optional[int] = None,
    formats: Optional[Sequence[str]] = None,
) -> list[str]:
    """
    Save a figure in one or more formats (png/pdf/svg) with consistent settings.

    Returns list of written file paths.
    """
    out_paths: list[str] = []
    p = Path(path_no_ext)
    p.parent.mkdir(parents=True, exist_ok=True)

    fmts = list(formats) if formats is not None else _export_formats_from_config(plot_config)
    dpi_eff = int(dpi) if dpi is not None else _dpi_from_config(plot_config)

    for fmt in fmts:
        ext = str(fmt).lower().lstrip(".")
        out = p.with_suffix(f".{ext}")
        if ext in {"pdf", "svg"}:
            fig.savefig(out, bbox_inches="tight", facecolor="white")
        else:
            fig.savefig(out, dpi=dpi_eff, bbox_inches="tight", facecolor="white")
        out_paths.append(str(out))

    return out_paths


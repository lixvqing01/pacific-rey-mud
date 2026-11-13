from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import xarray as xr
from core.utils.netcdf_io import open_dataset

from core.utils.serialization_utils import NumpyJSONEncoder
from scripts.predict_rey_continuous_grid import _plot_summary_maps


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, cls=NumpyJSONEncoder, indent=2, ensure_ascii=False)


def replot_rey_continuous_maps(
    *,
    prediction_dir: Path,
    summary_2d_nc: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    prediction_dir = Path(prediction_dir)
    summary_2d_nc = Path(summary_2d_nc) if summary_2d_nc else prediction_dir / "continuous_rey_paper_summary_2d.nc"
    output_dir = Path(output_dir) if output_dir else prediction_dir
    if not summary_2d_nc.exists():
        raise FileNotFoundError(f"Missing 2D prediction summary NetCDF: {summary_2d_nc}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open_dataset(summary_2d_nc) as ds:
        ds_loaded = ds.load()
    map_pngs = _plot_summary_maps(ds_loaded, output_dir)

    summary_path = output_dir / "continuous_rey_prediction_summary.json"
    existing: dict[str, Any] = {}
    if summary_path.exists():
        try:
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    result = {
        **existing,
        "status": "maps_replotted",
        "prediction_dir": str(prediction_dir),
        "summary_2d_nc": str(summary_2d_nc),
        "map_pngs": map_pngs,
    }
    _write_json(summary_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate continuous REY map PNGs from an existing 2D prediction NetCDF.")
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--summary-2d-nc", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    result = replot_rey_continuous_maps(
        prediction_dir=args.prediction_dir,
        summary_2d_nc=args.summary_2d_nc,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, cls=NumpyJSONEncoder, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

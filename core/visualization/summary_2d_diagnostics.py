import json
import os
from typing import Any, Dict, Optional

import numpy as np

from core.utils.logger_config import logger


def save_summary_2d_json(summary_ds, *, output_dir: str, meta: Dict[str, Any]) -> str:
    os.makedirs(output_dir, exist_ok=True)
    payload: Dict[str, Any] = {"_meta": meta, "variables": {}, "comparisons": {}}

    for name, da in summary_ds.data_vars.items():
        arr = np.asarray(da.values)
        valid = np.isfinite(arr)
        payload["variables"][name] = {
            "shape": list(arr.shape),
            "n_valid": int(np.sum(valid)),
            "min": float(np.nanmin(arr)) if np.any(valid) else None,
            "p50": float(np.nanpercentile(arr, 50)) if np.any(valid) else None,
            "p95": float(np.nanpercentile(arr, 95)) if np.any(valid) else None,
            "max": float(np.nanmax(arr)) if np.any(valid) else None,
        }


    try:
        windows = meta.get("windows", []) if isinstance(meta, dict) else []
        tags = [w.get("tag") for w in windows if isinstance(w, dict) and w.get("tag")]
        tags = [t for t in tags if isinstance(t, str)]

        def _pairwise(prefix: str) -> Dict[str, Any]:
            out: Dict[str, Any] = {}
            for i in range(len(tags)):
                for j in range(i + 1, len(tags)):
                    a_name = f"{prefix}{tags[i]}"
                    b_name = f"{prefix}{tags[j]}"
                    if a_name not in summary_ds.data_vars or b_name not in summary_ds.data_vars:
                        continue
                    a = np.asarray(summary_ds[a_name].values)
                    b = np.asarray(summary_ds[b_name].values)
                    m = np.isfinite(a) & np.isfinite(b)
                    if not np.any(m):
                        continue
                    av = a[m].ravel()
                    bv = b[m].ravel()
                    corr = float(np.corrcoef(av, bv)[0, 1]) if av.size > 1 else None
                    diff = av - bv
                    out[f"{tags[i]}_vs_{tags[j]}"] = {
                        "n": int(m.sum()),
                        "corr": corr,
                        "diff_mean": float(diff.mean()),
                        "diff_abs_mean": float(np.mean(np.abs(diff))),
                        "diff_p95_abs": float(np.percentile(np.abs(diff), 95)),
                    }
            return out

        payload["comparisons"]["predicted_rey_age_mean"] = _pairwise("predicted_rey_age_mean_0_")

        q = meta.get("quantile")
        if isinstance(q, (float, int)):
            payload["comparisons"][f"predicted_rey_age_p{int(float(q)*100)}"] = _pairwise(
                f"predicted_rey_age_p{int(float(q)*100)}_0_"
            )
    except Exception as e:
        logger.debug(f"2D summary diagnostics: failed to compute comparisons: {e}")

    path = os.path.join(output_dir, "summary_2d_diagnostics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info(f"2D summary diagnostics: wrote {path}")
    return path

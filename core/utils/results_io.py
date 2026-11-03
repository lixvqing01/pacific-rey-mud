from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from core.utils.serialization_utils import NumpyJSONEncoder


RESULTS_SCHEMA_VERSION = "1.0"


def write_json(path: str | Path, payload: Any) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, cls=NumpyJSONEncoder, ensure_ascii=False)
    return str(path)


def write_text(path: str | Path, text: str) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return str(path)


def snapshot_environment(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "results_schema_version": RESULTS_SCHEMA_VERSION,
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "conda": {
            "default_env": os.getenv("CONDA_DEFAULT_ENV"),
            "prefix": os.getenv("CONDA_PREFIX"),
        },
        "packages": {},
    }

    try:
        from importlib.metadata import version as _version

        pkg_names = [
            "numpy",
            "pandas",
            "xarray",
            "matplotlib",
            "scikit-learn",
            "torch",
            "shap",
            "streamlit",
        ]
        for name in pkg_names:
            try:
                info["packages"][name] = _version(name)
            except Exception:
                pass
    except Exception:
        pass


    try:
        import torch  # type: ignore

        info["torch"] = {
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": getattr(torch.version, "cuda", None),
        }
    except Exception:
        pass

    if extra:
        info["extra"] = extra
    return info


def build_paths_manifest(run_dir: str | Path, run_summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a compact, journal-friendly manifest from run_summary["artifacts"]["paths"] plus core files.
    Avoids hashing or walking huge directories; focuses on reproducibility and navigation.
    """
    run_dir = Path(run_dir)

    def _exists(p: Any) -> bool:
        try:
            return p is not None and Path(str(p)).exists()
        except Exception:
            return False

    manifest: Dict[str, Any] = {
        "results_schema_version": RESULTS_SCHEMA_VERSION,
        "run_id": run_summary.get("run_id"),
        "run_type": run_summary.get("run_type"),
        "run_dir": str(run_dir),
        "core_files": {
            "run_summary": str(run_dir / "run_summary.json"),
            "pipeline_config": str(run_dir / "pipeline_config.json"),
            "environment_snapshot": str(run_dir / "environment_snapshot.json"),
            "dataset_summary": str(run_dir / "dataset_summary.json"),
            "run_report": str(run_dir / "RUN_REPORT.md"),
        },
        "artifacts_paths": run_summary.get("artifacts", {}).get("paths", {}),
        "existing_core_files": {},
    }

    for k, v in manifest["core_files"].items():
        manifest["existing_core_files"][k] = _exists(v)

    return manifest


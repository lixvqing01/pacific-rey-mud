import json
import os
from pathlib import Path
from .serialization_utils import NumpyJSONEncoder

def save_run_summary(run_id, summary_data):
    """
    Saves the run summary dictionary to a JSON file.
    Uses a custom JSON encoder to handle NumPy data types.

    Args:
        run_id (str): The unique identifier for the run.
        summary_data (dict): The dictionary containing the run results.
    """

    run_dir = None
    try:
        run_dir = (
            (summary_data.get("artifacts", {}) or {}).get("paths", {}) or {}
        ).get("base")
    except Exception:
        run_dir = None

    if not run_dir:
        try:
            run_dir = ((summary_data.get("config", {}) or {}).get("paths", {}) or {}).get("run_base_dir")
        except Exception:
            run_dir = None

    if not run_dir:
        from core.utils.constants import RESULTS_GUI_DIR
        run_dir = str(Path(RESULTS_GUI_DIR) / run_id)

    run_dir = Path(str(run_dir))
    summary_file_path = run_dir / "run_summary.json"

    if not run_dir.exists():
        print(f"Warning: Run directory {run_dir} does not exist. Cannot save summary.")
        return

    try:
        with open(summary_file_path, 'w', encoding="utf-8") as f:
            json.dump(summary_data, f, cls=NumpyJSONEncoder, indent=4)
        print(f"Run summary saved to {summary_file_path}")
    except Exception as e:
        print(f"Error saving run summary for run {run_id}: {e}")

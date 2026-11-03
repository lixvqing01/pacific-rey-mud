import numpy as np
import json
from pathlib import Path

class NumpyJSONEncoder(json.JSONEncoder):
    """
    A custom JSON encoder that handles NumPy data types.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, Path):
            return str(obj)

        return json.JSONEncoder.default(self, obj)

def convert_numpy_to_native(obj):
    """
    Recursively converts NumPy data types in a dictionary or list to native Python types
    to ensure JSON serializability.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_native(elem) for elem in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj
# utils.py

import numpy as np
from typing import Any

def convert_numpy_to_python(obj: Any) -> Any:
    """
    Recursively convert NumPy data types to native Python data types.

    Parameters:
    - obj: The object to convert.

    Returns:
    - The converted object.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    else:
        return obj

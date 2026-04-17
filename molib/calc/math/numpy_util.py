from typing import Any

import numpy as np
from numpy import dtype, ndarray, generic


def to_up_vec3() -> np.ndarray[Any, dtype[Any]] | ndarray[Any, dtype[generic]]:
    return np.array([0.0, 0.0, 1.0], dtype=np.float32)


def get_np_array(p1: ndarray) -> np.ndarray:
    """get as numpy array"""
    return np.asarray(p1, dtype=np.float32)

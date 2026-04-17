from typing import Any

import numpy as np
from numpy import dtype, ndarray, generic


def to_up_vec3() -> np.ndarray[Any, dtype[Any]] | ndarray[Any, dtype[generic]]:
    return np.array([0.0, 0.0, 1.0], dtype=np.float32)


def get_np_array(p1: ndarray) -> np.ndarray:
    """get as numpy array"""
    return np.asarray(p1, dtype=np.float32)


def generate_colors_from_positions(positions: np.ndarray,
                                   r: float, g: float, b: float) -> np.ndarray:
    """generate colors from positions"""
    return np.tile([r, g, b], (len(positions), 1)).astype(np.float32)

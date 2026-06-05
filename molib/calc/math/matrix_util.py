import numpy as np
from molib.calc.math.normal import normalize
from numpy import ndarray


def cross_normalize(binormal: float, direction: ndarray) -> float:
    """Calculate normal from cross product"""
    normal = cross(binormal, direction)
    normal = normalize(normal)
    return normal


def cross(binormal: float, direction: ndarray) -> np.ndarray:
    """cross helper"""
    return np.cross(binormal, direction)

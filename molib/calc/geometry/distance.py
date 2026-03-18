"""
Geometry
"""

import numpy as np


def distance(a: np.ndarray, b: np.ndarray) -> np.floating:
    """
    distance

    :param a: np.ndarray [x, y, z]
    :param b: np.ndarray [x, y, z]
    :return: np.floating distance
    """
    return np.linalg.norm(a - b)


def euclidean_distance(
    x1: float, y1: float, z1: float, x2: float, y2: float, z2: float
) -> float:
    """
    euclidean_distance

    :param x1: float
    :param y1: float
    :param z1: float
    :param x2: float
    :param y2: float
    :param z2: float
    :return: float
    """
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

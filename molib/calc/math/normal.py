import numpy as np

from molib.calc.math.vector import Vector3
from molib.core.constants import MoLibConstant
from molib.pdb.molscript.math import v3_cross_product, v3_normalize


def v3_cross_normalize(direction: Vector3, side: Vector3) -> Vector3:
    """cross normalize Vec3"""
    normal = v3_cross_product(direction, side)
    normal = v3_normalize(normal)
    return normal


def normalize(v) -> float:
    """normalize"""
    # Helper function to normalize vectors
    norm = np.linalg.norm(v)
    if norm < MoLibConstant.EPSILON:
        return v
    return v / norm

from molib.calc.math.vector import Vector3
from molib.pdb.molscript.math import v3_cross_product, v3_normalize


def cross_normalize(direction: Vector3, side: Vector3) -> Vector3:
    """cross normalize"""
    normal = v3_cross_product(direction, side)
    normal = v3_normalize(normal)
    return normal

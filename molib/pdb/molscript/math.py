"""
Vector helper functions
"""

import math
from typing import Dict

import numpy as np
from molib.calc.math.vector import Vector3


def v3_difference_inplace(normal: Vector3, p1: Vector3, p2: Vector3):
    """Calculate vector difference."""
    if normal is None or p1 is None or p2 is None:
        raise ValueError("Arguments must be non-None")
    normal.x = p1.x - p2.x
    normal.y = p1.y - p2.y
    normal.z = p1.z - p2.z


def v3_array_extract(attr_name: str):
    return np.array(
        [
            [
                getattr(ss, attr_name).x,
                getattr(ss, attr_name).y,
                getattr(ss, attr_name).z,
            ]
            for ss in self.segments
        ],
        dtype=np.float32,
    )


def v3_normalize_inplace(normal: Vector3):
    """Normalize the vector."""
    if normal is None:
        raise ValueError("Arguments must be non-None")
    length = (normal.x**2 + normal.y**2 + normal.z**2) ** 0.5
    normal.x /= length
    normal.y /= length
    normal.z /= length


def v3_cross_product_inplace(result: Vector3, v1: Vector3, v2: Vector3):
    """Calculate the cross product of two vectors."""
    if result is None or v1 is None or v2 is None:
        raise ValueError("Arguments must be non-None")
    result.x = v1.y * v2.z - v1.z * v2.y
    result.y = v1.z * v2.x - v1.x * v2.z
    result.z = v1.x * v2.y - v1.y * v2.x


def v3_middle_inplace(dest: Vector3, p1: Vector3, p2: Vector3):
    """
    dest = (p1 + p2) / 2
    """
    if dest is None or p1 is None or p2 is None:
        raise ValueError("Arguments must be non-None")
    dest.x = 0.5 * (p1.x + p2.x)
    dest.y = 0.5 * (p1.y + p2.y)
    dest.z = 0.5 * (p1.z + p2.z)


def v3_sum_scaled(
    dest: Dict[str, float], v1: Dict[str, float], s: float, v2: Dict[str, float]
) -> None:
    """
    dest = v1 + s * v2
    """
    if dest is None or v1 is None or v2 is None:
        raise ValueError("Arguments must be non-None")
    dest["x"] = v1["x"] + s * v2["x"]
    dest["y"] = v1["y"] + s * v2["y"]
    dest["z"] = v1["z"] + s * v2["z"]


def v3_add_scaled_inplace(
    dest: Dict[str, float], s: float, v: Dict[str, float]
) -> None:
    """
    dest += s * v
    """
    if dest is None or v is None:
        raise ValueError("Arguments must be non-None")

    dest["x"] += s * v["x"]
    dest["y"] += s * v["y"]
    dest["z"] += s * v["z"]


def v3_length(v: Vector3) -> float:
    """v3_length"""
    if v is None:
        raise ValueError("Arguments must be non-None")
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def v3_difference(v1: Vector3, v2: Vector3) -> Vector3:
    """v3_difference"""
    if v1 is None or v2 is None:
        raise ValueError("Arguments must be non-None")
    return Vector3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z)


def v3_angle(v1: Vector3, v2: Vector3) -> float:
    """
    Angle between v1 and v2 in radians.
    Clamps to [-1, 1] to guard against numerical errors.
    """
    if v1 is None or v2 is None:
        raise ValueError("Arguments must be non-None Vector3.")
    dot_product = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z
    len1 = v3_length(v1)
    len2 = v3_length(v2)
    if len1 <= 0.0 or len2 <= 0.0:
        raise ValueError("Cannot compute angle with zero-length vector.")
    cos_theta = dot_product / (len1 * len2)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.acos(cos_theta)


def v3_angle_points(p1: Vector3, p2: Vector3, p3: Vector3) -> float:
    """
    Angle at p2 formed by points (p1, p2, p3).
    """
    if p1 is None or p2 is None or p3 is None:
        raise ValueError("Points must be non-None Vector3.")
    v1 = v3_difference(p1, p2)
    v2 = v3_difference(p3, p2)
    return v3_angle(v1, v2)


def v3_add_scaled(dest: Vector3, s: float, v: Vector3) -> None:
    """
    dest += s * v
    """
    if dest is None or v is None:
        raise ValueError("Arguments must be non-None")

    dest["x"] += s * v["x"]
    dest["y"] += s * v["y"]
    dest["z"] += s * v["z"]


def v3_cross_product(v1: Vector3, v2: Vector3) -> Vector3:
    """v3_cross_product"""
    if v1 is None or v2 is None:
        raise ValueError("Arguments must be non-None")
    return Vector3(
        v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x
    )


def v3_dot_product(v1: Vector3, v2: Vector3) -> float:
    """v3_dot_product"""
    if v1 is None or v2 is None:
        raise ValueError("Arguments must be non-None")
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z


def v3_magnitude(v: Vector3) -> float:
    """v3_magnitude"""
    if v is None:
        raise ValueError("Arguments must be non-None")
    return math.sqrt(v.x**2 + v.y**2 + v.z**2)


def v3_torsion(p1: Vector3, v1: Vector3, p2: Vector3, v2: Vector3) -> float:
    """
    The torsion angle (in radians) between the vectors v1 and v2
    when viewed along the vector from p1 to p2.
    """
    if p1 is None or v1 is None or p2 is None or v2 is None:
        raise ValueError("Arguments must be non-None")

    dir_vec = v3_difference(p2, p1)
    x1 = v3_cross_product(dir_vec, v1)
    x2 = v3_cross_product(dir_vec, v2)
    angle = v3_angle(x1, x2)
    return angle if v3_dot_product(v1, x2) > 0.0 else -angle


def v3_torsion_points(p1: Vector3, p2: Vector3, p3: Vector3, p4: Vector3) -> float:
    """
    The torsion angle (in radians) formed between the points.
    """
    if p1 is None or p2 is None or p3 is None or p4 is None:
        raise ValueError("Arguments must be non-None")

    v1 = v3_difference(p1, p2)
    v2 = v3_difference(p4, p3)
    return v3_torsion(p2, v1, p3, v2)


def v3_normalize(v: Vector3) -> Vector3:
    """
    Return normalized vector (unit vector in same direction).
    """
    if v is None:
        raise ValueError("Arguments must be non-None")
    length = v3_length(v)
    if length > 1e-8:
        return Vector3(v.x / length, v.y / length, v.z / length)
    else:
        return Vector3(0.0, 0.0, 1.0)  # fallback


def v3_from(obj) -> Vector3:
    """Create a Vector3 from any object with x, y, z attributes."""
    return Vector3(obj.x, obj.y, obj.z)


def v3_lerp(v1: Vector3, v2: Vector3, t: float) -> Vector3:
    """
    Linear interpolation between v1 and v2.
    t=0 -> v1, t=1 -> v2
    """
    return Vector3(
        v1.x * (1.0 - t) + v2.x * t,
        v1.y * (1.0 - t) + v2.y * t,
        v1.z * (1.0 - t) + v2.z * t,
    )


def v3_reverse(v: Vector3) -> Vector3:
    """
    Return the reverse of the vector (same length, opposite direction).
    """
    if v is None:
        raise ValueError("Argument must be non-None")
    return Vector3(-v.x, -v.y, -v.z)


def v3_middle(dest: Vector3, v1: Vector3, v2: Vector3) -> Vector3:
    """
    dest = (v1 + v2) / 2
    """
    return Vector3((v1.x + v2.x) / 2, (v1.y + v2.y) / 2, (v1.z + v2.z) / 2)


def v3_sum(dest: Vector3, v1: Vector3, v2: Vector3) -> Vector3:
    """
    dest = v1 + v2
    """
    if dest is None or v1 is None or v2 is None:
        raise ValueError("Arguments must be non-None Vector3.")
    return Vector3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z)


def v3_add(dest: Vector3, v: Vector3) -> Vector3:
    """
    dest = dest + v
    """
    if dest is None or v is None:
        raise ValueError("Arguments must be non-None Vector3.")
    return Vector3(dest.x + v.x, dest.y + v.y, dest.z + v.z)


def v3_scaled(dest: Vector3, s: float, v: Vector3) -> Vector3:
    """
    dest = s * v
    """
    if dest is None or v is None:
        raise ValueError("Arguments must be non-None Vector3.")
    return Vector3(s * v.x, s * v.y, s * v.z)


def v3_scale(v: Vector3, s: float) -> Vector3:
    """
    dest = s * v (alias of v3_scaled)
    """
    return v3_scaled(Vector3(), s, v)


def v3_reverse(v: Vector3) -> Vector3:
    """
    dest = -v
    """
    return Vector3(-v.x, -v.y, -v.z)


def v3_sum_scaled_vector3(dest: Vector3, v1: Vector3, s: float, v2: Vector3) -> Vector3:
    """
    dest = v1 + s * v2 (Vector3 version)
    """
    if dest is None or v1 is None or v2 is None:
        raise ValueError("Arguments must be non-None Vector3.")
    return Vector3(v1.x + s * v2.x, v1.y + s * v2.y, v1.z + s * v2.z)


def v3_add_scaled_vector3(dest: Vector3, s: float, v: Vector3) -> Vector3:
    """
    dest += s * v (Vector3 version)
    """
    if dest is None or v is None:
        raise ValueError("Arguments must be non-None Vector3.")
    return Vector3(dest.x + s * v.x, dest.y + s * v.y, dest.z + s * v.z)

"""
Hermite parametric cubic 3D curve routines.

Python port of C implementation by Per Kraulis
Copyright (C) 1997-1998 Per Kraulis
     2-Feb-1997  fairly finished
    28-Oct-1997  added tangent routine
    17-Jun-1998  mod's for hgen

Python port: 2024
"""

from molib.calc.math.vector import Vector3
from molib.pdb.molscript.math import v3_scaled, v3_sum

p1 = Vector3()
p2 = Vector3()
v1 = Vector3()
v2 = Vector3()


def hermite_set(
    pos_start: Vector3, pos_finish: Vector3, vec_start: Vector3, vec_finish: Vector3
):
    """
    Set the start and finish points and vectors of the Hermite curve.

    Args:
        pos_start: Starting position of the curve
        pos_finish: Ending position of the curve
        vec_start: Starting tangent vector
        vec_finish: Ending tangent vector
    """
    assert pos_start is not None
    assert pos_finish is not None
    assert vec_start is not None
    assert vec_finish is not None

    global p1, p2, v1, v2
    p1 = pos_start
    p2 = pos_finish
    v1 = vec_start
    v2 = vec_finish


def hermite_get(t: float) -> Vector3:
    """
    Return the point on the Hermite curve corresponding to the
    given parameter value t.

    Args:
        t: Parameter value between 0.0 and 1.0

    Returns:
        Vector3: Point on the curve at parameter t
    """
    assert t >= 0.0
    assert t <= 1.0

    t2 = t * t
    t3 = t2 * t
    tp1 = 2.0 * t3 - 3.0 * t2 + 1.0
    tp2 = -2.0 * t3 + 3.0 * t2
    tv1 = t3 - 2.0 * t2 + t
    tv2 = t3 - t2

    result = Vector3()
    result.x = p1.x * tp1 + p2.x * tp2 + v1.x * tv1 + v2.x * tv2
    result.y = p1.y * tp1 + p2.y * tp2 + v1.y * tv1 + v2.y * tv2
    result.z = p1.z * tp1 + p2.z * tp2 + v1.z * tv1 + v2.z * tv2

    return result


def hermite_get_tangent(t: float) -> Vector3:
    """
    Return the tangent vector of the Hermite curve corresponding
    to the given parameter value t.

    Args:
        t: Parameter value between 0.0 and 1.0

    Returns:
        Vector3: Tangent vector at parameter t
    """
    assert t >= 0.0
    assert t <= 1.0

    t2 = t * t
    tp1 = 6.0 * (t2 - t)
    tp2 = 6.0 * (-t2 + t)
    tv1 = 3.0 * t2 - 4.0 * t + 1.0
    tv2 = 3.0 * t2 - 2.0 * t

    result = Vector3()
    result.x = p1.x * tp1 + p2.x * tp2 + v1.x * tv1 + v2.x * tv2
    result.y = p1.y * tp1 + p2.y * tp2 + v1.y * tv1 + v2.y * tv2
    result.z = p1.z * tp1 + p2.z * tp2 + v1.z * tv1 + v2.z * tv2

    return result


def hermite_generate_points(num_points: int):
    """
    Generate a list of points along the Hermite curve.

    Args:
        num_points: Number of points to generate (must be >= 2)

    Returns:
        list[Vector3]: List of points along the curve
    """
    assert num_points >= 2

    points = []
    for i in range(num_points):
        t = i / (num_points - 1) if num_points > 1 else 0.0
        points.append(hermite_get(t))

    return points


def hermite_generate_tangents(num_points: int):
    """
    Generate a list of tangent vectors along the Hermite curve.

    Args:
        num_points: Number of tangent vectors to generate (must be >= 2)

    Returns:
        list[Vector3]: List of tangent vectors along the curve
    """
    assert num_points >= 2

    tangents = []
    for i in range(num_points):
        t = i / (num_points - 1) if num_points > 1 else 0.0
        tangents.append(hermite_get_tangent(t))

    return tangents


def hermite_get_curve_info():
    """
    Get information about the current Hermite curve.

    Returns:
        dict: Dictionary containing curve parameters
    """
    return {"start_point": p1, "end_point": p2, "start_tangent": v1, "end_tangent": v2}


def hermite_interpolate(p0: Vector3, p1: Vector3, v0: Vector3, v1: Vector3, t: float):
    """Hermite interpolation between two points with tangent vectors."""
    t2 = t * t
    t3 = t2 * t

    # Hermite basis functions
    h0 = 2 * t3 - 3 * t2 + 1
    h1 = -2 * t3 + 3 * t2
    h2 = t3 - 2 * t2 + t
    h3 = t3 - t2

    # Interpolated position
    pos = h0 * p0 + h1 * p1 + h2 * v0 + h3 * v1
    return pos


def hermite_interpolate_vector3(
    p0: Vector3, p1: Vector3, v0: Vector3, v1: Vector3, t: float
):
    """Hermite interpolation between two Vector3 points with tangent vectors."""
    t2 = t * t
    t3 = t2 * t

    # Hermite basis functions
    h0 = 2 * t3 - 3 * t2 + 1
    h1 = -2 * t3 + 3 * t2
    h2 = t3 - 2 * t2 + t
    h3 = t3 - t2

    # Interpolated position using Vector3 operations
    pos = v3_sum(
        Vector3(0, 0, 0),
        v3_sum(
            Vector3(0, 0, 0),
            v3_scaled(Vector3(0, 0, 0), h0, p0),
            v3_scaled(Vector3(0, 0, 0), h1, p1),
        ),
        v3_sum(
            Vector3(0, 0, 0),
            v3_scaled(Vector3(0, 0, 0), h2, v0),
            v3_scaled(Vector3(0, 0, 0), h3, v1),
        ),
    )
    return pos

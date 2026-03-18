"""
Smoothing
"""

from molib.calc.math.vector import Vector3
from molib.pdb.molscript.math import v3_middle_inplace


def priestle_smoothing(points, length, steps):
    if points is None or length is None or steps is None:
        raise ValueError("Points, length, and steps must be non-None")

    # Create a temporary array of Vector3 objects
    ptmp = [Vector3(0.0, 0.0, 0.0) for _ in range(length)]

    for sm in range(steps):
        for slot in range(1, length - 1):
            v3_middle_inplace(ptmp[slot], points[slot - 1], points[slot + 1])
            v3_middle_inplace(ptmp[slot], ptmp[slot], points[slot])

        for slot in range(1, length - 1):
            points[slot] = ptmp[slot]

    return points

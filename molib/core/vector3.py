"""Re-export canonical :class:`~molib.calc.math.vector.Vector3` and tuple helper."""

from __future__ import annotations

from molib.calc.math.vector import Vector3


def _tuple_to_vec(t):
    """Build a Vector3 from a length-3 sequence or x/y/z-like object."""
    return Vector3(*t)

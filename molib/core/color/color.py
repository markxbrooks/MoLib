"""
A module for defining and working with the Color data structure.

This module provides a Color class based on NamedTuple, which is
used to represent a color with specific attributes. Each color
is defined by a specification index and x, y, z coordinates.
"""

from typing import NamedTuple


class Color(NamedTuple):
    """Color"""

    spec: int
    r: float
    g: float
    b: float

    @property
    def x(self) -> float:
        """Backward-compatible alias for red."""
        return self.r

    @property
    def y(self) -> float:
        """Backward-compatible alias for green."""
        return self.g

    @property
    def z(self) -> float:
        """Backward-compatible alias for blue."""
        return self.b
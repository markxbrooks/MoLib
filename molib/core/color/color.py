"""
A module for defining and working with the Color data structure.

This module provides a Color class based on NamedTuple, which is
used to represent a color with specific attributes. Each color
is defined by a specification index and x, y, z coordinates.
"""

from typing import NamedTuple

from PySide6.QtGui import QColor


class Color(NamedTuple):
    """Color"""

    spec: int
    r: float
    g: float
    b: float

    def to_css_tuple(self) -> tuple[int, int, int]:
        """Convert to 0-255 range for CSS"""
        r = int(self.r * 255)
        g = int(self.g * 255)
        b = int(self.b * 255)
        return r, g, b

    def to_qcolor(self) -> QColor:
        """to Qt qcolor"""
        return QColor(
            *self.to_css_tuple()
        )

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
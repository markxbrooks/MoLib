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
    x: float
    y: float
    z: float

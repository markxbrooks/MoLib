"""
Block
"""

from molib.xtal.uglymol.math.marching_cubes import marching_cubes


class Block:
    def __init__(self):
        self._points = None
        self._values = None
        self._size = [0, 0, 0]

    def set(self, points, values, size):
        if size[0] <= 0 or size[1] <= 0 or size[2] <= 0:
            raise ValueError("Grid dimensions are zero along at least one edge")
        length = size[0] * size[1] * size[2]
        if len(values) != length or len(points) != length:
            raise ValueError("isosurface: array size mismatch")

        self._points = points
        self._values = values
        self._size = size

    def clear(self):
        self._points = None
        self._values = None

    def empty(self):
        return self._values is None

    def isosurface(self, iso_level: int, method: str = "marching_cubes"):
        """isosurface"""
        return marching_cubes(self._size, self._values, self._points, iso_level, method)

from __future__ import annotations


class Vector3:
    """Non Dataclass3D point or direction with scalar components (e.g. molib ``Point3D``)."""

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def __getitem__(self, index: int) -> float:
        return [self.x, self.y, self.z][index]

    def __setitem__(self, index: int, value: float):
        if index == 0:
            self.x = value
        elif index == 1:
            self.y = value
        elif index == 2:
            self.z = value
        else:
            raise IndexError(index)


def _tuple_to_vec(t):
    return Vector3(*t)

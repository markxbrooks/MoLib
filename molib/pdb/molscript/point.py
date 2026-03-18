import math

from molib.calc.math.vector import Vector3


class Point3D:
    """3D Point Class"""

    def __init__(self, x: int = 0, y: int = 0, z: int = 0):
        self.x = x
        self.y = y
        self.z = z

    def __sub__(self, other):
        return Point3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def normalize(self):
        mag = math.sqrt(self.x**2 + self.y**2 + self.z**2)
        if mag > 0:
            self.x /= mag
            self.y /= mag
            self.z /= mag
        return self

    def to_vector(p: "Point3D") -> Vector3:
        return Vector3(p.x, p.y, p.z)

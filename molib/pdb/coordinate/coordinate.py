import math


class Coordinates:
    """3D coordinate container with explicit initialization and validation."""

    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float) -> None:
        if not all(isinstance(v, (int, float)) for v in (x, y, z)):
            raise TypeError("Coordinates must be numeric (int or float).")
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    def __repr__(self) -> str:
        return f"Coordinates(x={self.x}, y={self.y}, z={self.z})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Coordinates)
            and self.x == other.x
            and self.y == other.y
            and self.z == other.z
        )

    def as_tuple(self) -> tuple[float, float, float]:
        """Return coordinates as a tuple."""
        return (self.x, self.y, self.z)

    def __add__(self, other: "Coordinates") -> "Coordinates":
        return Coordinates(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Coordinates") -> "Coordinates":
        return Coordinates(self.x - other.x, self.y - other.y, self.z - other.z)

    def scale(self, factor: float) -> "Coordinates":
        """Scale coordinates by a factor."""
        return Coordinates(self.x * factor, self.y * factor, self.z * factor)

    def distance_to(self, other: "Coordinates") -> float:
        """Euclidean distance to another point."""
        dx, dy, dz = self.x - other.x, self.y - other.y, self.z - other.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def normalize(self) -> "Coordinates":
        """Return a normalized (unit-length) vector."""
        length = math.sqrt(self.x**2 + self.y**2 + self.z**2)
        if length == 0:
            return Coordinates(0, 0, 0)
        return self.scale(1.0 / length)

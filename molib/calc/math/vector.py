"""Vector3"""

import numpy as np


class Vector3:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.is_vector3 = True
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __repr__(self) -> str:
        return f"Vector3(x={self.x}, y={self.y}, z={self.z})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector3):
            return False
        return self.x == other.x and self.y == other.y and self.z == other.z

    def set(self, x, y=None, z=None):
        """set"""
        if z is None:
            z = self.z
        self.x = x
        self.y = y
        self.z = z
        return self

    def clone(self):
        """clone"""
        return self.__class__(self.x, self.y, self.z)

    def copy(self, v):
        """copy"""
        self.x = v.x
        self.y = v.y
        self.z = v.z
        return self

    def add(self, v: "Vector3"):
        """add"""
        self.x += v.x
        self.y += v.y
        self.z += v.z
        return self

    def add_vectors(self, a: "Vector3", b: "Vector3"):
        """add_vectors"""
        self.x = a.x + b.x
        self.y = a.y + b.y
        self.z = a.z + b.z
        return self

    def add_scaled_vector(self, v: "Vector3", s: float):
        """add_scaled_vector"""
        self.x += v.x * s
        self.y += v.y * s
        self.z += v.z * s
        return self

    def sub(self, v: "Vector3"):
        """sub"""
        self.x -= v.x
        self.y -= v.y
        self.z -= v.z
        return self

    def sub_vectors(self, a: "Vector3", b: "Vector3"):
        """sub_vectors"""
        self.x = a.x - b.x
        self.y = a.y - b.y
        self.z = a.z - b.z
        return self

    def multiply_scalar(self, scalar):
        """multiply_scalar"""
        self.x *= scalar
        self.y *= scalar
        self.z *= scalar
        return self

    def apply_matrix4(self, m: "Matrix4"):
        """apply_matrix4"""
        x, y, z = self.x, self.y, self.z
        e = m.elements
        w = 1 / (e[3] * x + e[7] * y + e[11] * z + e[15])
        self.x = (e[0] * x + e[4] * y + e[8] * z + e[12]) * w
        self.y = (e[1] * x + e[5] * y + e[9] * z + e[13]) * w
        self.z = (e[2] * x + e[6] * y + e[10] * z + e[14]) * w
        return self

    def apply_quaternion(self, q: "Quaternion"):
        """apply_quaternion"""
        vx, vy, vz = self.x, self.y, self.z
        qx, qy, qz, qw = q.x, q.y, q.z, q.w
        tx = 2 * (qy * vz - qz * vy)
        ty = 2 * (qz * vx - qx * vz)
        tz = 2 * (qx * vy - qy * vx)
        self.x = vx + qw * tx + qy * tz - qz * ty
        self.y = vy + qw * ty + qz * tx - qx * tz
        self.z = vz + qw * tz + qx * ty - qy * tx
        return self

    def unproject(self, camera: "Camera"):
        """unproject"""
        return self.apply_matrix4(camera.projectionMatrixInverse).apply_matrix4(
            camera.matrixWorld
        )

    def transform_direction(self, m: "Matrix4"):
        """transform_direction"""
        x, y, z = self.x, self.y, self.z
        e = m.elements
        self.x = e[0] * x + e[4] * y + e[8] * z
        self.y = e[1] * x + e[5] * y + e[9] * z
        self.z = e[2] * x + e[6] * y + e[10] * z
        return self.normalize()

    def divide_scalar(self, scalar: float):
        """divide_scalar"""
        return self.multiply_scalar(1 / scalar)

    def dot(self, v: "Vector3"):
        """dot"""
        return self.x * v.x + self.y * v.y + self.z * v.z

    def length_sq(self) -> float:
        """length_sq"""
        return self.x * self.x + self.y * self.y + self.z * self.z

    def length(self) -> float:
        """length"""
        return (self.x * self.x + self.y * self.y + self.z * self.z) ** 0.5

    def normalize(self) -> "Vector3":
        """normalize"""
        return self.divide_scalar(self.length() or 1)

    def set_length(self, length: float) -> "Vector3":
        """set_length"""
        return self.normalize().multiply_scalar(length)

    def lerp(self, v: "Vector3", alpha: float) -> "Vector3":
        """lerp"""
        self.x += (v.x - self.x) * alpha
        self.y += (v.y - self.y) * alpha
        self.z += (v.z - self.z) * alpha
        return self

    def cross(self, v: "Vector3") -> "Vector3":
        """cross"""
        return self.cross_vectors(self, v)

    def cross_vectors(self, a: "Vector3", b: "Vector3") -> "Vector3":
        """cross_vectors"""
        ax, ay, az = a.x, a.y, a.z
        bx, by, bz = b.x, b.y, b.z
        self.x = ay * bz - az * by
        self.y = az * bx - ax * bz
        self.z = ax * by - ay * bx
        return self

    def project_on_vector(self, v: "Vector3") -> "Vector3":
        """project_on_vector"""
        denominator = v.length_sq()
        if denominator == 0:
            return self.set(0, 0, 0)
        scalar = v.dot(self) / denominator
        return self.copy(v).multiply_scalar(scalar)

    def project_on_plane(self, plane_normal: "Vector3") -> "Vector3":
        """project_on_plane"""
        _vector.copy(self).project_on_vector(plane_normal)
        return self.sub(_vector)

    def distance_to(self, v: "Vector3") -> float:
        """distance_to"""
        return (self.distance_to_squared(v)) ** 0.5

    def distance_to_squared(self, v: "Vector3") -> float:
        """distance_to_squared"""
        dx = self.x - v.x
        dy = self.y - v.y
        dz = self.z - v.z
        return dx * dx + dy * dy + dz * dz

    def set_from_matrix_position(self, m: "Matrix4") -> "Vector3":
        """set_from_matrix_position"""
        e = m.elements
        self.x = e[12]
        self.y = e[13]
        self.z = e[14]
        return self

    def set_from_matrix_column(self, m: "Matrix4", index: int) -> "Vector3":
        """set_from_matrix_column"""
        return self.from_array(m.elements, index * 4)

    def equals(self, v: "Vector3") -> bool:
        """equals"""
        return v.x == self.x and v.y == self.y and v.z == self.z

    def from_array(self, array: np.ndarray, offset: int = 0) -> "Vector3":
        """fromArray"""
        self.x = array[offset]
        self.y = array[offset + 1]
        self.z = array[offset + 2]
        return self

    def as_np(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


_vector = Vector3()


class Vector4:
    """Vector4"""

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, w: float = 1.0):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def set(self, x: float, y: float, z: float, w: float) -> "Vector4":
        """set"""
        self.x = x
        self.y = y
        self.z = z
        self.w = w
        return self

    def copy(self, v: "Vector4") -> "Vector4":
        """copy"""
        self.x = v.x
        self.y = v.y
        self.z = v.z
        self.w = v.w if hasattr(v, "w") else 1
        return self

    def multiply_scalar(self, scalar: float) -> "Vector4":
        """multiply_scalar"""
        self.x *= scalar
        self.y *= scalar
        self.z *= scalar
        self.w *= scalar
        return self

    def equals(self, v: "Vector4") -> bool:
        """equals"""
        return v.x == self.x and v.y == self.y and v.z == self.z and v.w == self.w

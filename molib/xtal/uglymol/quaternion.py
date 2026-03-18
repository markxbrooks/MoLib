import math


class Quaternion:
    def __init__(self, x=0, y=0, z=0, w=1):
        self._x = x
        self._y = y
        self._z = z
        self._w = w

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    @property
    def w(self):
        return self._w

    def set_from_axis_angle(self, axis, angle):
        """
        http://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToQuaternion/index.htm
        assumes axis is normalized
        """
        half_angle = angle / 2
        s = math.sin(half_angle)

        self._x = axis.x * s
        self._y = axis.y * s
        self._z = axis.z * s
        self._w = math.cos(half_angle)

        return self

    def set_from_rotation_matrix(self, m):
        """
        http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
        assumes the upper 3x3 of m is a pure rotation matrix (i.e, unscaled)
        """
        te = m.elements

        m11 = te[0]
        m12 = te[4]
        m13 = te[8]
        m21 = te[1]
        m22 = te[5]
        m23 = te[9]
        m31 = te[2]
        m32 = te[6]
        m33 = te[10]

        trace = m11 + m22 + m33

        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)

            self._w = 0.25 / s
            self._x = (m32 - m23) * s
            self._y = (m13 - m31) * s
            self._z = (m21 - m12) * s
        elif m11 > m22 and m11 > m33:
            s1 = 2.0 * math.sqrt(1.0 + m11 - m22 - m33)

            self._w = (m32 - m23) / s1
            self._x = 0.25 * s1
            self._y = (m12 + m21) / s1
            self._z = (m13 + m31) / s1
        elif m22 > m33:
            s2 = 2.0 * math.sqrt(1.0 + m22 - m11 - m33)

            self._w = (m13 - m31) / s2
            self._x = (m12 + m21) / s2
            self._y = 0.25 * s2
            self._z = (m23 + m32) / s2
        else:
            s3 = 2.0 * math.sqrt(1.0 + m33 - m11 - m22)

            self._w = (m21 - m12) / s3
            self._x = (m13 + m31) / s3
            self._y = (m23 + m32) / s3
            self._z = 0.25 * s3

        return self

    def set_from_unit_vectors(self, v_from, v_to):
        r = v_from.dot(v_to) + 1
        if r < math.ulp(1.0):
            r = 0
            if abs(v_from.x) > abs(v_from.z):
                self._x = -v_from.y
                self._y = v_from.x
                self._z = 0
                self._w = r
            else:
                self._x = 0
                self._y = -v_from.z
                self._z = v_from.y
                self._w = r
        else:
            self._x = v_from.y * v_to.z - v_from.z * v_to.y
            self._y = v_from.z * v_to.x - v_from.x * v_to.z
            self._z = v_from.x * v_to.y - v_from.y * v_to.x
            self._w = r
        return self.normalize()

    def length(self):
        return math.sqrt(self._x**2 + self._y**2 + self._z**2 + self._w**2)

    def normalize(self):
        l = self.length()
        if l == 0:
            self._x = 0
            self._y = 0
            self._z = 0
            self._w = 1
        else:
            l = 1 / l
            self._x *= l
            self._y *= l
            self._z *= l
            self._w *= l
        return self

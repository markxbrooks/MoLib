"""Matrix 4x4"""

from molib.calc.math.vector import Vector3

_zero = Vector3(0, 0, 0)
_one = Vector3(1, 1, 1)


class Matrix4:
    """Matrix4"""

    def __init__(self):
        self.elements = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

    def set(
        self,
        n11,
        n12,
        n13,
        n14,
        n21,
        n22,
        n23,
        n24,
        n31,
        n32,
        n33,
        n34,
        n41,
        n42,
        n43,
        n44,
    ):
        """set"""
        te = self.elements
        te[0] = n11
        te[4] = n12
        te[8] = n13
        te[12] = n14
        te[1] = n21
        te[5] = n22
        te[9] = n23
        te[13] = n24
        te[2] = n31
        te[6] = n32
        te[10] = n33
        te[14] = n34
        te[3] = n41
        te[7] = n42
        te[11] = n43
        te[15] = n44
        return self

    def copy(self, m):
        """copy"""
        te = self.elements
        me = m.elements

        for i in range(16):
            te[i] = me[i]

        return self

    def make_rotation_from_quaternion(self, q):
        """make_rotation_from_quaternion"""
        return self.compose(_zero, q, _one)

    def compose(self, position, quaternion, scale):
        """compose"""
        te = self.elements

        x, y, z, w = quaternion._x, quaternion._y, quaternion._z, quaternion._w
        x2, y2, z2 = x + x, y + y, z + z
        xx, xy, xz = x * x2, x * y2, x * z2
        yy, yz, zz = y * y2, y * z2, z * z2
        wx, wy, wz = w * x2, w * y2, w * z2

        sx, sy, sz = scale.x, scale.y, scale.z

        te[0] = (1 - (yy + zz)) * sx
        te[1] = (xy + wz) * sx
        te[2] = (xz - wy) * sx
        te[3] = 0

        te[4] = (xy - wz) * sy
        te[5] = (1 - (xx + zz)) * sy
        te[6] = (yz + wx) * sy
        te[7] = 0

        te[8] = (xz + wy) * sz
        te[9] = (yz - wx) * sz
        te[10] = (1 - (xx + yy)) * sz
        te[11] = 0

        te[12] = position.x
        te[13] = position.y
        te[14] = position.z
        te[15] = 1

        return self

    def look_at(self, eye, target, up):
        """look_at"""
        te = self.elements

        # Use local Vector3s instead of global _x, _y, _z
        z = eye.copy().sub(target)
        if z.length_sq() == 0:
            z.z = 1

        z.normalize()
        x = up.copy().cross(z)
        if x.length_sq() == 0:
            # up and z are parallel
            if abs(up.z) == 1:
                z.x += 0.0001
            else:
                z.z += 0.0001
            z.normalize()
            x = up.copy().cross(z)

        x.normalize()
        y = z.copy().cross(x)

        te[0] = x.x
        te[4] = y.x
        te[8] = z.x
        te[1] = x.y
        te[5] = y.y
        te[9] = z.y
        te[2] = x.z
        te[6] = y.z
        te[10] = z.z

        return self

    def multiply_matrices(self, a, b):
        """multiply_matrices"""
        ae = a.elements
        be = b.elements
        te = self.elements

        a11, a12, a13, a14 = ae[0], ae[4], ae[8], ae[12]
        a21, a22, a23, a24 = ae[1], ae[5], ae[9], ae[13]
        a31, a32, a33, a34 = ae[2], ae[6], ae[10], ae[14]
        a41, a42, a43, a44 = ae[3], ae[7], ae[11], ae[15]

        b11, b12, b13, b14 = be[0], be[4], be[8], be[12]
        b21, b22, b23, b24 = be[1], be[5], be[9], be[13]
        b31, b32, b33, b34 = be[2], be[6], be[10], be[14]
        b41, b42, b43, b44 = be[3], be[7], be[11], be[15]

        te[0] = a11 * b11 + a12 * b21 + a13 * b31 + a14 * b41
        te[4] = a11 * b12 + a12 * b22 + a13 * b32 + a14 * b42
        te[8] = a11 * b13 + a12 * b23 + a13 * b33 + a14 * b43
        te[12] = a11 * b14 + a12 * b24 + a13 * b34 + a14 * b44

        te[1] = a21 * b11 + a22 * b21 + a23 * b31 + a24 * b41
        te[5] = a21 * b12 + a22 * b22 + a23 * b32 + a24 * b42
        te[9] = a21 * b13 + a22 * b23 + a23 * b33 + a24 * b43
        te[13] = a21 * b14 + a22 * b24 + a23 * b34 + a24 * b44

        te[2] = a31 * b11 + a32 * b21 + a33 * b31 + a34 * b41
        te[6] = a31 * b12 + a32 * b22 + a33 * b32 + a34 * b42
        te[10] = a31 * b13 + a32 * b23 + a33 * b33 + a34 * b43
        te[14] = a31 * b14 + a32 * b24 + a33 * b34 + a34 * b44

        te[3] = a41 * b11 + a42 * b21 + a43 * b31 + a44 * b41
        te[7] = a41 * b12 + a42 * b22 + a43 * b32 + a44 * b42
        te[11] = a41 * b13 + a42 * b23 + a43 * b33 + a44 * b43
        te[15] = a41 * b14 + a42 * b24 + a43 * b34 + a44 * b44

        return self

    def set_position(self, x, y=None, z=None):
        """set position"""
        te = self.elements

        if hasattr(x, "isVector3"):
            te[12] = x.x
            te[13] = x.y
            te[14] = x.z
        else:
            te[12] = x
            te[13] = y
            te[14] = z

        return self

    def invert(self):
        """invert"""
        te = self.elements
        n11, n21, n31, n41 = te[0], te[1], te[2], te[3]
        n12, n22, n32, n42 = te[4], te[5], te[6], te[7]
        n13, n23, n33, n43 = te[8], te[9], te[10], te[11]
        n14, n24, n34, n44 = te[12], te[13], te[14], te[15]

        t11 = (
            n23 * n34 * n42
            - n24 * n33 * n42
            + n24 * n32 * n43
            - n22 * n34 * n43
            - n23 * n32 * n44
            + n22 * n33 * n44
        )
        t12 = (
            n14 * n33 * n42
            - n13 * n34 * n42
            - n14 * n32 * n43
            + n12 * n34 * n43
            + n13 * n32 * n44
            - n12 * n33 * n44
        )
        t13 = (
            n13 * n24 * n42
            - n14 * n23 * n42
            + n14 * n22 * n43
            - n12 * n24 * n43
            - n13 * n22 * n44
            + n12 * n23 * n44
        )
        t14 = (
            n14 * n23 * n32
            - n13 * n24 * n32
            - n14 * n22 * n33
            + n12 * n24 * n33
            + n13 * n22 * n34
            - n12 * n23 * n34
        )

        det = n11 * t11 + n21 * t12 + n31 * t13 + n41 * t14
        if det == 0:
            raise ValueError("Matrix4 is not invertible")

        det_inv = 1.0 / det

        te[0] = t11 * det_inv
        te[1] = (
            n24 * n33 * n41
            - n23 * n34 * n41
            - n24 * n31 * n43
            + n21 * n34 * n43
            + n23 * n31 * n44
            - n21 * n33 * n44
        ) * det_inv
        te[2] = (
            n22 * n34 * n41
            - n24 * n32 * n41
            + n24 * n31 * n42
            - n21 * n34 * n42
            - n22 * n31 * n44
            + n21 * n32 * n44
        ) * det_inv
        te[3] = (
            n23 * n32 * n41
            - n22 * n33 * n41
            - n23 * n31 * n42
            + n21 * n33 * n42
            + n22 * n31 * n43
            - n21 * n32 * n43
        ) * det_inv

        te[4] = t12 * det_inv
        te[5] = (
            n13 * n34 * n41
            - n14 * n33 * n41
            + n14 * n31 * n43
            - n11 * n34 * n43
            - n13 * n31 * n44
            + n11 * n33 * n44
        ) * det_inv
        te[6] = (
            n14 * n32 * n41
            - n12 * n34 * n41
            - n14 * n31 * n42
            + n11 * n34 * n42
            + n12 * n31 * n44
            - n11 * n32 * n44
        ) * det_inv
        te[7] = (
            n12 * n33 * n41
            - n13 * n32 * n41
            + n13 * n31 * n42
            - n11 * n33 * n42
            - n12 * n31 * n43
            + n11 * n32 * n43
        ) * det_inv

        te[8] = t13 * det_inv
        te[9] = (
            n14 * n23 * n41
            - n13 * n24 * n41
            - n14 * n21 * n43
            + n11 * n24 * n43
            + n13 * n21 * n44
            - n11 * n23 * n44
        ) * det_inv
        te[10] = (
            n12 * n24 * n41
            - n14 * n22 * n41
            + n14 * n21 * n42
            - n11 * n24 * n42
            - n12 * n21 * n44
            + n11 * n22 * n44
        ) * det_inv
        te[11] = (
            n13 * n22 * n41
            - n12 * n23 * n41
            - n13 * n21 * n42
            + n11 * n23 * n42
            + n12 * n21 * n43
            - n11 * n22 * n43
        ) * det_inv

        te[12] = t14 * det_inv
        te[13] = (
            n13 * n24 * n31
            - n14 * n23 * n31
            + n14 * n21 * n33
            - n11 * n24 * n33
            - n13 * n21 * n34
            + n11 * n23 * n34
        ) * det_inv
        te[14] = (
            n14 * n22 * n31
            - n12 * n24 * n31
            - n14 * n21 * n32
            + n11 * n24 * n32
            + n12 * n21 * n34
            - n11 * n22 * n34
        ) * det_inv
        te[15] = (
            n12 * n23 * n31
            - n13 * n22 * n31
            + n13 * n21 * n32
            - n11 * n23 * n32
            - n12 * n21 * n33
            + n11 * n22 * n33
        ) * det_inv

        return self

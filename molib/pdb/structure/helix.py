"""
Code for alpha helices
"""

import numpy as np
from OpenGL.GLU import gluNewQuadric
from OpenGL.raw.GL.VERSION.GL_1_0 import (
    glPopMatrix,
    glPushMatrix,
    glRotatef,
    glTranslatef,
)
from OpenGL.raw.GLU import gluCylinder, gluDeleteQuadric


def draw_cylinder(p1: tuple, p2: tuple, radius: float = 0.2, slices: int = 12):
    """
    draw_cylinder

    :param p1: tuple coordinate_data_main for position_array 1
    :param p2: tuple coordinate_data_main for position_array 2
    :param radius: float
    :param slices: int slices
    :return:
    Draw a cylinder between p1 and p2 using gluCylinder.
    """

    p1 = np.array(p1, dtype=np.float32)
    p2 = np.array(p2, dtype=np.float32)

    direction = p2 - p1
    length = np.linalg.norm(direction)
    if length < 1e-5:
        return  # Degenerate, skip

    direction = direction / length
    z_axis = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z_axis, direction)
    angle = np.degrees(np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0)))

    glPushMatrix()
    try:
        glTranslatef(*p1)
        if np.linalg.norm(axis) > 1e-5 and not np.isnan(angle):
            glRotatef(angle, *axis)

        quad = gluNewQuadric()
        gluCylinder(quad, radius, radius, length, slices, 1)
        gluDeleteQuadric(quad)
    finally:
        glPopMatrix()

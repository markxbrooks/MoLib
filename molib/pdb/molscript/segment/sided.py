"""
Sided Segment
"""

import numpy as np
from molib.core.color import ColorMap
from molib.pdb.molscript.point import Point3D


class SidedSegment:
    def __init__(self, p1, p2, direction, perp1, width, thickness):
        half_width = width / 2

        # Create ribbon corners (quad: p1-p2-p3-p4)
        self.p1 = Point3D(
            p1[0] - perp1[0] * half_width,
            p1[1] - perp1[1] * half_width,
            p1[2] - perp1[2] * half_width,
        )
        self.p2 = Point3D(
            p2[0] - perp1[0] * half_width,
            p2[1] - perp1[1] * half_width,
            p2[2] - perp1[2] * half_width,
        )
        self.p3 = Point3D(
            p2[0] + perp1[0] * half_width,
            p2[1] + perp1[1] * half_width,
            p2[2] + perp1[2] * half_width,
        )
        self.p4 = Point3D(
            p1[0] + perp1[0] * half_width,
            p1[1] + perp1[1] * half_width,
            p1[2] + perp1[2] * half_width,
        )

        # Calculate the top/bottom normal (face normal) for beta sheet hydrogen bonding
        # This normal should be perpendicular to both the strand direction and the carbonyl plane
        n1 = np.cross(direction, perp1)
        n1_length = np.linalg.norm(n1)
        if n1_length > 1e-6:
            n1 /= n1_length
        else:
            # Fallback if direction and perp1 are parallel
            n1 = np.array([0, 0, 1])

        self.n1 = Point3D(*n1)
        self.n2 = Point3D(*n1)
        self.n3 = Point3D(*n1)
        self.n4 = Point3D(*n1)
        self.n = Point3D(*n1)

        # --- NEW: Calculate side normals for better lighting ---
        # Left side (p1->p2 edge)
        edge_left = np.array(
            [self.p2.x - self.p1.x, self.p2.y - self.p1.y, self.p2.z - self.p1.z]
        )
        left_normal = np.cross(edge_left, n1)
        left_normal /= np.linalg.norm(left_normal)

        # Right side (p3->p4 edge)
        edge_right = np.array(
            [self.p4.x - self.p3.x, self.p4.y - self.p3.y, self.p4.z - self.p3.z]
        )
        right_normal = np.cross(edge_right, n1)
        right_normal /= np.linalg.norm(right_normal)

        self.n_side_left = Point3D(*left_normal)
        self.n_side_right = Point3D(*right_normal)

        # Color and midpoint
        self.c = ColorMap.get_ss_colors()[" "]  # Gray for coil
        self.p = Point3D((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, (p1[2] + p2[2]) / 2)

import numpy as np
from decologr import Decologr as log
from molib.core.color import ColorMap
from molib.pdb.molscript.point import Point3D

from elmo.gl.buffers.molecule.secondary_structure.buffers import (
    SecondaryStructureBuffers,
)


def create_helix_ribbon_segment(
    p1: np.ndarray, p2: np.ndarray, direction: np.ndarray, perp1: np.ndarray
):
    """Create a helix ribbon segment with proper helical geometry."""
    try:
        width = 1.5
        thickness = 0.8

        class SimpleSegment:
            def __init__(self, p1, p2, direction, perp1, width, thickness):
                half_width = width / 2

                # Create ribbon corners with helical twist
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

                # Calculate normals
                n1 = np.cross(direction, perp1)
                n1 = n1 / np.linalg.norm(n1)

                self.n1 = Point3D(n1[0], n1[1], n1[2])
                self.n2 = Point3D(n1[0], n1[1], n1[2])
                self.n3 = Point3D(n1[0], n1[1], n1[2])
                self.n4 = Point3D(n1[0], n1[1], n1[2])
                self.n = Point3D(direction[0], direction[1], direction[2])

                self.c = ColorMap.get_ss_colors()["H"]
                self.p = Point3D(
                    (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, (p1[2] + p2[2]) / 2
                )

        return SimpleSegment(p1, p2, direction, perp1, width, thickness)
    except Exception as ex:
        log.message(f"Error creating helix ribbon segment: {ex}")
        return None


def create_helix_segments(
    coords, residues: list["Residue"], secondary_struct: SecondaryStructureBuffers
):
    """Create helix segment geometry with proper helical ribbon structure."""
    log.message(f"  Creating helix segments from {len(coords)} coordinates")

    if len(coords) < 3:
        return 0

    # Create helical ribbon by interpolating between CA atoms
    # and adding helical twist
    for i in range(len(coords) - 1):
        p1 = coords[i]
        p2 = coords[i + 1]

        # Calculate direction vector
        direction = np.array(p2) - np.array(p1)
        length = np.linalg.norm(direction)
        if length < 1e-6:
            continue

        direction = direction / length

        # Create perpendicular vectors for ribbon width
        if abs(direction[0]) < 0.9:
            perp1 = np.cross(direction, [1, 0, 0])
        else:
            perp1 = np.cross(direction, [0, 1, 0])
        perp1 = perp1 / np.linalg.norm(perp1)

        # Add helical twist
        # Calculate rotation angle based on position in helix
        twist_angle = (i * 100.0) * np.pi / 180.0  # 100 degrees per residue

        # Rotate perpendicular vector around direction axis
        cos_a = np.cos(twist_angle)
        sin_a = np.sin(twist_angle)

        # Rodrigues' rotation formula
        rotated_perp = (
            perp1 * cos_a
            + np.cross(direction, perp1) * sin_a
            + direction * np.dot(direction, perp1) * (1 - cos_a)
        )

        # Create helix segment with proper geometry
        segment = create_helix_ribbon_segment(p1, p2, direction, rotated_perp)
        if segment:
            # Add region tracking attributes
            segment.region_start = i == 0  # First segment in region
            segment.region_end = i == len(coords) - 2  # Last segment in region

            secondary_struct.helix_segments.append(segment)
            secondary_struct.helix_segment_count += 1
            log.message(
                f"    Added helix segment {secondary_struct.helix_segment_count}"
            )
        else:
            log.message(f"    Failed to create helix segment {i}")

    log.message(f"  Final helix count: {secondary_struct.helix_segment_count}")
    return secondary_struct.helix_segment_count

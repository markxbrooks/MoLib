import numpy as np
from decologr import Decologr as log
from molib.calc.math.vector import Vector3
from molib.core.color import ColorMap
from molib.entities.residue import Res3D
from molib.pdb.molscript.point import Point3D
from molib.pdb.molscript.smoothing import priestle_smoothing

from elmo.gl.buffers.molecule.secondary_structure.buffers import (
    SecondaryStructureBuffers,
)


def smooth_coordinates_priestle(coords: np.ndarray, smoothing_steps: int = 3):
    """Apply Priestle smoothing to coordinates."""
    if len(coords) < 3:
        return coords

    points = coords_to_vector3(coords)

    # Apply Priestle smoothing
    smoothed_points = priestle_smoothing(points, len(points), smoothing_steps)

    # Convert back to numpy array
    smoothed_coords = np.array([[p.x, p.y, p.z] for p in smoothed_points])

    log.message(
        f"  Applied Priestle smoothing with {smoothing_steps} steps to {len(coords)} coordinates"
    )
    return smoothed_coords


def coords_to_vector3(coords: np.ndarray) -> list[Vector3]:
    """Convert numpy coordinates to Vector3 objects"""
    points = [
        Vector3(float(coord[0]), float(coord[1]), float(coord[2])) for coord in coords
    ]
    return points


def calculate_sheet_plane_normal(coords: np.ndarray) -> np.ndarray:
    """Calculate the sheet plane normal for co-planar beta sheet geometry."""
    if len(coords) < 3:
        # Fallback to arbitrary normal if not enough points
        return np.array([0, 0, 1])

    # Use the first three points to define the sheet plane
    # This ensures all strands in the same sheet will be co-planar
    p1, p2, p3 = coords[0], coords[1], coords[2]

    # Calculate two vectors in the plane
    v1 = p2 - p1
    v2 = p3 - p1

    # Calculate normal using cross product
    normal = np.cross(v1, v2)
    normal_length = np.linalg.norm(normal)

    if normal_length < 1e-6:
        # Fallback if points are collinear
        return np.array([0, 0, 1])

    return normal / normal_length


def calculate_consistent_perpendicular(
    direction: np.ndarray, sheet_normal: np.ndarray
) -> np.ndarray:
    """Calculate a consistent perpendicular vector for co-planar sheet geometry."""
    # First try using the sheet normal
    perp = np.cross(direction, sheet_normal)
    perp_length = np.linalg.norm(perp)

    if perp_length > 1e-6:
        return perp / perp_length

    # If direction is parallel to sheet normal, use a different approach
    # Try with x-axis
    perp = np.cross(direction, [1, 0, 0])
    perp_length = np.linalg.norm(perp)

    if perp_length > 1e-6:
        return perp / perp_length

    # Try with y-axis
    perp = np.cross(direction, [0, 1, 0])
    perp_length = np.linalg.norm(perp)

    if perp_length > 1e-6:
        return perp / perp_length

    # Final fallback
    return np.array([0, 0, 1])


def calculate_carbonyl_plane_normal(
    direction: np.ndarray, sheet_normal: np.ndarray
) -> np.ndarray:
    """Calculate the normal to the carbonyl group plane for proper beta sheet hydrogen bonding.

    In beta sheets, the carbonyl groups (C=O) are positioned perpendicular to the strand direction
    and in the plane of the peptide bond. This function calculates the normal to this plane,
    which should be used to orient the beta strand segments for proper hydrogen bonding.
    """
    # The carbonyl plane normal is perpendicular to both the strand direction and the sheet normal
    # This ensures the segments are coplanar with the carbonyl groups for hydrogen bonding
    carbonyl_plane_normal = np.cross(direction, sheet_normal)
    carbonyl_plane_normal_length = np.linalg.norm(carbonyl_plane_normal)

    if carbonyl_plane_normal_length > 1e-6:
        return carbonyl_plane_normal / carbonyl_plane_normal_length

    # Fallback if direction and sheet_normal are parallel
    # Use a perpendicular to the direction vector
    if abs(direction[0]) < 0.9:  # Not parallel to x-axis
        perp = np.cross(direction, [1, 0, 0])
    else:  # Not parallel to y-axis
        perp = np.cross(direction, [0, 1, 0])

    perp_length = np.linalg.norm(perp)
    if perp_length > 1e-6:
        return perp / perp_length

    # Final fallback
    return np.array([0, 0, 1])


def create_strand_segments(
    coords: np.ndarray,
    residues: list[Res3D],
    secondary_struct: SecondaryStructureBuffers,
):
    """Create strand segment geometry with arrow-like shape."""
    log.message(f"  Creating strand segments from {len(coords)} coordinates")

    if len(coords) < 2:
        return 0

    # Log original strand endpoints
    log.message(f"  Original strand endpoints:")
    log.message(
        f"    Start: ({coords[0][0]:.3f}, {coords[0][1]:.3f}, {coords[0][2]:.3f})"
    )
    log.message(
        f"    End: ({coords[-1][0]:.3f}, {coords[-1][1]:.3f}, {coords[-1][2]:.3f})"
    )
    log.message(
        f"    Overall direction: ({coords[-1][0] - coords[0][0]:.3f}, {coords[-1][1] - coords[0][1]:.3f}, {coords[-1][2] - coords[0][2]:.3f})"
    )

    # Calculate overall strand direction for arrow
    overall_direction = coords[-1] - coords[0]
    overall_direction = overall_direction / np.linalg.norm(overall_direction)
    log.message(f"    Normalized overall direction: {overall_direction}")

    # Calculate sheet plane normal for co-planar beta sheet geometry
    sheet_normal = calculate_sheet_plane_normal(coords)
    log.message(f"    Sheet plane normal: {sheet_normal}")

    # Apply Priestle smoothing to coordinates before creating segments
    coords = smooth_coordinates_priestle(coords, smoothing_steps=3)
    log.message(f"  Using smoothed coordinates for strand segment creation")

    # Log smoothed strand endpoints
    log.message(f"  Smoothed strand endpoints:")
    log.message(
        f"    Start: ({coords[0][0]:.3f}, {coords[0][1]:.3f}, {coords[0][2]:.3f})"
    )
    log.message(
        f"    End: ({coords[-1][0]:.3f}, {coords[-1][1]:.3f}, {coords[-1][2]:.3f})"
    )
    log.message(
        f"    Overall direction: ({coords[-1][0] - coords[0][0]:.3f}, {coords[-1][1] - coords[0][1]:.3f}, {coords[-1][2] - coords[0][2]:.3f})"
    )

    # Create strand segments with arrow-like geometry
    for i in range(len(coords) - 1):
        p1 = coords[i]
        p2 = coords[i + 1]

        # Calculate direction vector
        direction = np.array(p2) - np.array(p1)
        length = np.linalg.norm(direction)
        if length < 1e-6:
            continue

        direction = direction / length

        # Calculate carbonyl plane normal for proper beta sheet hydrogen bonding
        carbonyl_plane_normal = calculate_carbonyl_plane_normal(direction, sheet_normal)

        # Create perpendicular vectors using carbonyl plane normal for co-planar geometry
        perp1 = carbonyl_plane_normal

        # Create arrow-like strand segment
        # Use overall direction for the last segment (arrow), local direction for others
        arrow_direction = overall_direction if i == len(coords) - 2 else direction
        segment = create_strand_arrow_segment(
            p1, p2, arrow_direction, perp1, i == len(coords) - 2, secondary_struct
        )
        if segment:
            # Add region tracking attributes
            segment.region_start = i == 0  # First segment in region
            segment.region_end = i == len(coords) - 2  # Last segment in region

            secondary_struct.strand_segments.append(segment)
            secondary_struct.strand_segment_count += 1
            log.message(
                f"    Added strand segment {secondary_struct.strand_segment_count}"
            )
        else:
            log.message(f"    Failed to create strand segment {i}")

    log.message(f"  Final strand count: {secondary_struct.strand_segment_count}")
    return secondary_struct.strand_segment_count


def create_strand_arrow_segment(
    p1: np.ndarray,
    p2: np.ndarray,
    direction: np.ndarray,
    perp1: np.ndarray,
    is_last: bool,
    secondary_struct=None,
):
    """Create a strand segment with arrow-like geometry."""
    try:
        width = 1.2
        thickness = 0.3

        class SimpleSegment:
            def __init__(
                self,
                p1: np.ndarray,
                p2: np.ndarray,
                direction: np.ndarray,
                perp1: np.ndarray,
                width: float,
                thickness: float,
                is_last: bool,
            ):
                half_width = width / 2

                if is_last:
                    # Create arrow head for last segment
                    arrow_length = 0.8
                    arrow_width = width * 1.5

                    # Arrow base
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

                    # Arrow tip - use the direction parameter directly for consistency
                    # The direction parameter is already the overall direction for the last segment
                    # Rotate the direction by 90 degrees around the strand axis
                    # Calculate local strand direction for rotation axis
                    local_strand_direction = p2 - p1
                    local_strand_direction = local_strand_direction / np.linalg.norm(
                        local_strand_direction
                    )

                    # Calculate perpendicular vector for rotation
                    rotation_axis = np.cross(local_strand_direction, direction)
                    if np.linalg.norm(rotation_axis) > 1e-6:
                        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                        # Rotate direction by 90 degrees around rotation axis
                        rotated_direction = np.cross(rotation_axis, direction)
                        if np.linalg.norm(rotated_direction) > 1e-6:
                            direction = rotated_direction / np.linalg.norm(
                                rotated_direction
                            )

                    # Calculate tip from the arrow base (self.p2), not the original p2
                    tip = (
                        np.array([self.p2.x, self.p2.y, self.p2.z])
                        + direction * arrow_length
                    )
                    self.p3 = Point3D(tip[0], tip[1], tip[2])

                    # Arrow tip is already created above as self.p3
                    # No need for additional extended tip segments

                    # Arrow base other side
                    self.p4 = Point3D(
                        p1[0] + perp1[0] * half_width,
                        p1[1] + perp1[1] * half_width,
                        p1[2] + perp1[2] * half_width,
                    )

                    # Log arrow geometry for debugging
                    log.message(f"    Arrow geometry for last segment:")
                    log.message(f"      Strand endpoints: p1={p1}, p2={p2}")
                    log.message(f"      Direction vector: {direction}")
                    log.message(
                        f"      Arrow base left (p1): ({self.p1.x:.3f}, {self.p1.y:.3f}, {self.p1.z:.3f})"
                    )
                    log.message(
                        f"      Arrow base right (p2): ({self.p2.x:.3f}, {self.p2.y:.3f}, {self.p2.z:.3f})"
                    )
                    log.message(
                        f"      Arrow tip (p3): ({self.p3.x:.3f}, {self.p3.y:.3f}, {self.p3.z:.3f})"
                    )
                    log.message(
                        f"      Arrow base other side (p4): ({self.p4.x:.3f}, {self.p4.y:.3f}, {self.p4.z:.3f})"
                    )

                    # Calculate and log direction verification
                    tip_vector = np.array([self.p3.x, self.p3.y, self.p3.z]) - p2
                    tip_length = np.linalg.norm(tip_vector)
                    log.message(f"      Arrow tip vector from p2: {tip_vector}")
                    log.message(f"      Arrow tip length: {tip_length:.3f}")
                    log.message(f"      Expected arrow length: {arrow_length}")
                else:
                    # Regular strand segment
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

                # Calculate normals using carbonyl plane normal for proper beta sheet hydrogen bonding
                # The normal should be perpendicular to both the strand direction and the carbonyl plane
                n1 = np.cross(direction, perp1)
                n1_length = np.linalg.norm(n1)
                if n1_length > 1e-6:
                    n1 = n1 / n1_length
                else:
                    # Fallback if direction and perp1 are parallel
                    n1 = np.array([0, 0, 1])

                self.n1 = Point3D(n1[0], n1[1], n1[2])
                self.n2 = Point3D(n1[0], n1[1], n1[2])
                self.n3 = Point3D(n1[0], n1[1], n1[2])
                self.n4 = Point3D(n1[0], n1[1], n1[2])
                self.n = Point3D(n1[0], n1[1], n1[2])

                self.c = ColorMap.get_ss_colors()["E"]
                self.p = Point3D(
                    (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, (p1[2] + p2[2]) / 2
                )

        return SimpleSegment(
            p1, p2, direction, perp1, width, thickness, is_last=is_last
        )
    except Exception as ex:
        log.message(f"Error creating strand arrow segment: {ex}")
        return None

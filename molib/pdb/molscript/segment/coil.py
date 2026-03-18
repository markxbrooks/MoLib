import numpy as np
from decologr import Decologr as log
from molib.core.color import ColorMap
from molib.pdb.molscript.point import Point3D

from elmo.gl.buffers.molecule.secondary_structure.buffers import (
    SecondaryStructureBuffers,
)


def create_coil_ribbon_segment(
    p1: np.ndarray, p2: np.ndarray, is_first: bool = False, is_last: bool = False
):
    """Create a coil ribbon segment with proper geometry for connecting coil_regions."""
    try:
        # Calculate direction vector
        direction = np.array(p2) - np.array(p1)
        length = np.linalg.norm(direction)
        if length < 1e-6:  # Avoid division by zero
            return None

        direction = direction / length

        # Create perpendicular vectors for ribbon width
        if abs(direction[0]) < 0.9:
            perp1 = np.cross(direction, [1, 0, 0])
        else:
            perp1 = np.cross(direction, [0, 1, 0])
        perp1 = perp1 / np.linalg.norm(perp1)

        # Set coil ribbon dimensions (make more discreet)
        width = 0.6
        thickness = 0.15

        class SimpleSegment:
            def __init__(
                self,
                p1,
                p2,
                direction,
                perp1,
                width,
                thickness,
                is_first=False,
                is_last=False,
            ):
                half_width = width / 2

                # Create ribbon corners
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
                self.n = Point3D(n1[0], n1[1], n1[2])

                # Slightly lighter coil color handled in SS_COLORS; optionally dim further here
                self.c = ColorMap.get_ss_colors()[" "]
                self.p = Point3D(
                    (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, (p1[2] + p2[2]) / 2
                )

                # Add region markers for proper grouping
                self.region_start = is_first
                self.region_end = is_last

        return SimpleSegment(
            p1, p2, direction, perp1, width, thickness, is_first, is_last
        )
    except Exception as ex:
        log.message(f"Error creating coil ribbon segment: {ex}")
        return None


def create_coil_segments(
    residues: list["Residue"], secondary_struct: SecondaryStructureBuffers
):
    """Create coil segment geometry for connecting coil_regions between secondary structures."""
    log.message(f"Creating coil segment geometry for {len(residues)} residues")

    if len(residues) < 2:
        log.message(f"Not enough residues ({len(residues)}) for coil segment")
        return

    # Get CA coordinates for the coil segment
    ca_coords = []
    for i, residue in enumerate(residues):
        if hasattr(residue, "ca") and residue.ca is not None:
            ca_coords.append(residue.ca)
            log.message(f"  Residue {i}: found CA coords {residue.ca}")
        elif hasattr(residue, "coords"):
            ca_coords.append(residue.coords)
            log.message(f"  Residue {i}: found coords {residue.coords}")
        else:
            log.message(
                f"  Residue {i}: no CA or coords found, attributes: {dir(residue)}"
            )

    log.message(f"Found {len(ca_coords)} CA coordinates for coil")

    if len(ca_coords) < 2:
        log.message(f"Not enough CA coordinates ({len(ca_coords)}) for coil segment")
        return

    # Convert to numpy array
    coords = np.array(ca_coords)
    log.message(f"Created coords array with shape {coords.shape}")

    # Create coil segments between consecutive CA atoms
    for i in range(len(coords) - 1):
        p1 = coords[i]
        p2 = coords[i + 1]

        # Create a coil segment
        segment = create_coil_ribbon_segment(p1, p2)
        if segment:
            # Add region tracking attributes
            segment.region_start = i == 0  # First segment in region
            segment.region_end = i == len(coords) - 2  # Last segment in region

            secondary_struct.coil_segments.append(segment)
            secondary_struct.coil_segment_count += 1
            log.message(f"    Added coil segment {secondary_struct.coil_segment_count}")
        else:
            log.message(f"    Failed to create coil segment {i}")

    log.message(f"  Final coil count: {secondary_struct.coil_segment_count}")

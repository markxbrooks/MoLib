from dataclasses import dataclass

import numpy as np
from decologr import Decologr as log
from molib.core.color import ColorMap
from molib.pdb.molscript.point import Point3D


def normalize(vec):
    norm = np.linalg.norm(vec)
    return vec if norm < 1e-8 else vec / norm


class SimpleSegment:
    """
    Represents a simple 3D segment defined by four points.
    Used as the geometric basis for arrow parts or other structures.
    """

    def __init__(self, p1, p2, p3, p4):
        self.p1 = np.array(p1, dtype=np.float32)
        self.p2 = np.array(p2, dtype=np.float32)
        self.p3 = np.array(p3, dtype=np.float32)
        self.p4 = np.array(p4, dtype=np.float32)

    def as_array(self) -> np.ndarray:
        """Return vertices as a flat array (4x3)."""
        return np.vstack([self.p1, self.p2, self.p3, self.p4])


class ArrowSegment:
    """
    Base class for arrow segments, which wrap a SimpleSegment
    and provide geometry/VBO helpers.
    """

    def __init__(self, segment: SimpleSegment, color):
        self.segment = segment
        self.color = np.array(color, dtype=np.float32)

    def generate_colors(self) -> np.ndarray:
        """Repeat segment color for 4 vertices."""
        return np.tile(self.color, (4, 1)).astype(np.float32)


class ArrowBuilder:
    """
    Factory for building composite arrows from multiple segments.
    Handles setup of base, shaft, and head.
    """

    def __init__(self, base_pos, direction, normal, color):
        self.base_pos = base_pos
        self.direction = direction
        self.normal = normal
        self.color = color

        self.segments: list[ArrowSegment] = []

    def build(self):
        """Constructs the arrow parts and stores ArrowSegments."""
        # Example of constructing different parts
        seg1 = SimpleSegment([0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0])
        arrow_seg1 = ArrowSegment(seg1, self.color)
        self.segments.append(arrow_seg1)

        # Could add seg2, seg3, etc. for shaft/head variations


class ArrowPipeline:
    """
    High-level class: builds arrows, smooths them, and uploads VBOs.
    """

    def __init__(self, builder: ArrowBuilder):
        self.builder = builder
        self.vbos = []

    def setup(self):
        """Run pipeline: build segments, smooth, generate VBOs."""
        self.builder.build()
        for seg in self.builder.segments:
            colors = seg.generate_colors()
            verts = seg.segment.as_array()
            # → Replace with VBO setup
            self.vbos.append((verts, colors))


class SimpleSegmentNew:
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
        self.p = Point3D((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, (p1[2] + p2[2]) / 2)

        # Add region markers for proper grouping
        self.region_start = is_first
        self.region_end = is_last


@dataclass
class SimpleSegmentOld:
    p1: Point3D
    p2: Point3D
    p3: Point3D
    p4: Point3D
    n1: Point3D
    n2: Point3D
    n3: Point3D
    n4: Point3D
    n_side: Point3D
    n_side1: Point3D
    n_side2: Point3D
    n: Point3D
    c: tuple
    p: Point3D
    region_start: bool = False
    region_end: bool = False


def create_simple_segment(
    p1: np.ndarray,
    p2: np.ndarray,
    ss_type: str,
    is_first: bool = False,
    is_last: bool = False,
):
    """Factory for ribbon/arrow segments (helix, strand, coil)."""

    try:
        # Direction
        direction = np.array(p2) - np.array(p1)
        length = np.linalg.norm(direction)
        if length < 1e-6:
            return None
        direction /= length

        # Perpendiculars
        if abs(direction[0]) < 0.9:
            perp1 = np.cross(direction, [1, 0, 0])
        else:
            perp1 = np.cross(direction, [0, 1, 0])
        perp1 /= np.linalg.norm(perp1)
        perp2 = np.cross(direction, perp1)
        perp2 /= np.linalg.norm(perp2)

        # Dimensions by type
        if ss_type == "H":  # Helix
            width, thickness = 1.5, 0.8
        elif ss_type == "E":  # Strand
            width, thickness = 1.2, 0.3
        else:  # Coil
            width, thickness = 0.6, 0.15

        half_width = width / 2

        # --- Geometry
        if ss_type == "E" and is_last:
            # Arrowhead
            arrow_length = 0.8
            arrow_tip = p2 + direction * arrow_length
            pts = [
                Point3D(*(p1 - perp1 * half_width)),
                Point3D(*(p2 - perp1 * half_width)),
                Point3D(*arrow_tip),
                Point3D(*(p1 + perp1 * half_width)),
            ]
        else:
            # Standard quad
            pts = [
                Point3D(*(p1 - perp1 * half_width)),
                Point3D(*(p2 - perp1 * half_width)),
                Point3D(*(p2 + perp1 * half_width)),
                Point3D(*(p1 + perp1 * half_width)),
            ]

        # Normals
        n = np.cross(direction, perp1)
        n /= np.linalg.norm(n)
        if ss_type == "H":
            normal = Point3D(*direction)  # helical alignment
        else:
            normal = Point3D(*n)

        normals = [Point3D(*n)] * 4

        # Color
        color = ColorMap.get_ss_colors().get(ss_type, ColorMap.get_ss_colors()[" "])

        # Center
        center = Point3D(*((p1 + p2) / 2))

        # --- Build final object
        class SimpleSegment:
            def __init__(self):
                self.p1, self.p2, self.p3, self.p4 = pts
                self.n1, self.n2, self.n3, self.n4 = normals
                self.n = normal
                self.c = color
                self.p = center
                self.region_start = is_first
                self.region_end = is_last

        return SimpleSegment()

    except Exception as ex:
        log.message(f"Error creating segment: {ex}")
        return None

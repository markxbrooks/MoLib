"""
Vertex PNC
==========
Vertex with Positions, Normals, Colors
"""

from __future__ import annotations

from dataclasses import dataclass

from molib.core.vectorlike import Vector3Like
from molib.entities.ribbon.color import RibbonColor


@dataclass(slots=True)
class VertexPNC:
    """Vertex PNC"""
    position: tuple[float, float, float]
    normal: tuple[float, float, float]
    color: tuple[float, float, float]


def _extract_rgb(colour: RibbonColor) -> tuple[float, float, float]:
    """Extract RGB from color"""
    try:
        return float(colour[0]), float(colour[1]), float(colour[2])
    except Exception:
        return (
            float(getattr(colour, "x", 1.0)),
            float(getattr(colour, "y", 1.0)),
            float(getattr(colour, "z", 1.0)),
        )


def generate_vertex_pnc(positions_array: Vector3Like, normals_array: Vector3Like, c: RibbonColor) -> VertexPNC:
    """generate vertex pnc"""
    r, g, b = _extract_rgb(c)
    return VertexPNC(
        position=(float(positions_array.x), float(positions_array.y), float(positions_array.z)),
        normal=(float(normals_array.x), float(normals_array.y), float(normals_array.z)),
        color=(r, g, b),
    )

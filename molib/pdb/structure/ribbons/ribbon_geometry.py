from dataclasses import dataclass


@dataclass
class RibbonGeometryContext:
    """Ribbon Geometry"""
    plane_normal = None
    binormal = None
    left_edge = None
    right_edge = None

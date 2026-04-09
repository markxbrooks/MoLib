"""
Ribbon Aggregation
"""

from dataclasses import dataclass
from typing import Optional

from molib.core.vectorlike import Vector3Like
from molib.entities.ribbon.color import RibbonColor


@dataclass
class RibbonAggregation:
    """Ribbon Aggregation"""
    colors_by_chain: dict[str, list[RibbonColor]]
    coords_by_chain: dict[str, list[Vector3Like]]
    o_coords_by_chain: Optional[dict[str, list[Vector3Like]]]
    ss_types_by_chain: Optional[dict[str, list[int]]]
    coords_by_chain_res: dict[tuple[str, str], list[Vector3Like]]
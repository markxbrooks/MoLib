"""Ribbon build context"""

from dataclasses import dataclass

import numpy as np

from molib.core.color.provider import ColorProvider


@dataclass
class RibbonBuildContext:
    """Ribbon build context"""

    coords: np.ndarray
    chain_ids: list[str] | np.ndarray
    residue_ids: np.ndarray  # NEW: List of residue IDs for each coordinate
    color_provider: ColorProvider | None = None
    colors: dict | np.ndarray | None = None
    o_coords: np.ndarray | None = None
    ss_types: np.ndarray | None = None

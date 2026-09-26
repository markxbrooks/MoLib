"""
Info for electron density maps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from molib.xtal.map.density import CrystallographicInfo, MapType


@dataclass
class MapInfo:
    """Information about an electron density map"""

    map_id: str
    map_type: Union[str, "MapType"]  # e.g., "2Fo-Fc", "Fo-Fc", "FWT", "DELFWT"
    f_label: str  # F column label (e.g., "2FOFCWT", "FWT")
    phi_label: str  # PHI column label (e.g., "PH2FOFCWT", "PHWT")
    volume: np.ndarray
    crystallographic_info: Optional[CrystallographicInfo] = None
    sigma_level: float = 1.0
    is_visible: bool = True
    color: Tuple[float, float, float] = (0.0, 0.0, 0.8)  # Default blue
    is_difference_map: bool = False
    positive_visible: bool = True
    negative_visible: bool = True
    positive_color: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    negative_color: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    positive_sigma_level: Optional[float] = None
    negative_sigma_level: Optional[float] = None
    description: str = ""
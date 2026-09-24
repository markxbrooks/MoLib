"""
Info for electron density maps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class MapInfo:
    """Information about an electron density map"""

    map_id: str
    map_type: str  # e.g., "2Fo-Fc", "Fo-Fc", "FWT", "DELFWT"
    f_label: str  # F column label (e.g., "2FOFCWT", "FWT")
    phi_label: str  # PHI column label (e.g., "PH2FOFCWT", "PHWT")
    volume: np.ndarray
    crystallographic_info: Optional[Dict] = None
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
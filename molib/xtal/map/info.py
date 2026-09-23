from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from decologr import Decologr as log
from molib.xtal.map.density import CrystallographicInfo


@dataclass
class MapInfo:
    """Information about an electron density map"""

    map_id: str
    map_type: str  # e.g., "2Fo-Fc", "Fo-Fc", "FWT", "DELFWT"
    f_label: str  # F column label (e.g., "2FOFCWT", "FWT")
    phi_label: str  # PHI column label (e.g., "PH2FOFCWT", "PHWT")
    volume: Optional[np.ndarray] = None
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

    def __str__(self):
        """str"""
        return self.as_string

    def log_contents(self):
        """log contents"""
        log.debug(self.as_string, scope=self.__class__.__name__, silent=True)

    @property
    def as_string(self):
        """as string"""
        string = f"MapInfo: {self.map_id}, {self.map_type}, {self.f_label}, {self.phi_label}, {self.volume.shape}, {self.crystallographic_info}, {self.sigma_level}, {self.is_visible}, {self.color}, {self.is_difference_map}, {self.positive_visible}, {self.negative_visible}, {self.positive_color}, {self.negative_color}, {self.positive_sigma_level}, {self.negative_sigma_level}, {self.description}"
        return string
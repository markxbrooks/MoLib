"""
CoordinateData
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import gemmi
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


@dataclass
class CoordinateData:
    """Structured data extracted from a PDB file."""

    active: bool = False
    df: Optional[pd.DataFrame] = None
    cif_blocks: Optional[dict[str, gemmi.cif.Block]] = None
    coords: Optional[np.ndarray] = None
    atom_names: Optional[np.ndarray] = None
    num_atoms: int = 0
    num_hetatom_atoms: int = 0
    num_water_atoms: int = 0
    chain_ids: List[str] = field(default_factory=list)
    chain_colors: Dict[str, tuple] = field(default_factory=dict)
    element_symbols: Optional[np.ndarray] = None
    _kdtree = None
    _last_distance = None

    def build_kdtree(self, force: bool = False):
        """
        build_kdtree

        :param force: bool
        :return: None
        """
        if force or self._kdtree is None:
            if self.coords is not None:
                self._kdtree = cKDTree(self.coords)

    def find_closest_atom(self, pos: Tuple[float, float, float]) -> int:
        """
        find_closest_atom

        :param pos: Tuple
        :return: int
        """
        if self._kdtree is None:
            self.build_kdtree()
        if self._kdtree is None:
            # still no kDTree
            return -1
        distance, index = self._kdtree.query(pos)
        self._last_distance = distance  # optional: store for later inspection
        return int(index)

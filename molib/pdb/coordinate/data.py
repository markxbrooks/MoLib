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

    def find_closest_atom_to_ray(
            self,
            origin: np.ndarray,
            direction: np.ndarray,
            threshold: float = 2.0,
    ) -> int | None:
        """
        Find the closest atom to a ray.

        Parameters
        ----------
        origin : (3,) np.ndarray
            Ray origin (world space)
        direction : (3,) np.ndarray
            Normalized ray direction
        threshold : float
            Max perpendicular distance (Å)

        Returns
        -------
        int | None
            Index of closest atom, or None if no hit
        """

        positions = self.coords  # shape: (N, 3)
        if positions is None or len(positions) == 0:
            return None

        # Vector from origin to each atom
        v = positions - origin  # (N, 3)

        # Project onto ray direction
        t = np.dot(v, direction)  # (N,)

        # Only consider atoms in front of the camera
        forward_mask = t > 0
        if not np.any(forward_mask):
            return None

        v = v[forward_mask]
        t = t[forward_mask]
        indices = np.where(forward_mask)[0]

        # Closest point on ray for each atom
        proj = origin + np.outer(t, direction)  # (M, 3)

        # Perpendicular distances
        d = np.linalg.norm(positions[indices] - proj, axis=1)

        # Apply distance threshold
        hit_mask = d < threshold
        if not np.any(hit_mask):
            return None

        d = d[hit_mask]
        t = t[hit_mask]
        indices = indices[hit_mask]

        # Pick closest along ray (not just closest in space)
        best = np.argmin(t)

        return int(indices[best])

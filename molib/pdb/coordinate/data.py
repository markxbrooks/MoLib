"""
CoordinateData
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import gemmi
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from molib.pdb.calculate.atom_data import AtomData
from decologr import Decologr as log


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
    atom_ids = None

    def build_kdtree(self, force: bool = False):
        """
        build_kdtree

        :param force: bool
        :return: None
        """
        if force or self._kdtree is None:
            if self.coords is not None:
                self._kdtree = cKDTree(self.coords)

    def data_from_index(self, index: int) -> AtomData | None:
        """
        data_from_index

        :param self: CoordinateData object with atom DataFrames.
        :param index: Index into the ATOM dataframe (or other available type).
        :return: (residue_number, chain_id) tuple or None.

        Get residue number and chain ID for the atom at the given index in the coordinate data.
        """
        if not hasattr(self, "df"):
            return None
        try:
            atom_df = self.df

            if atom_df is None:
                log.message("⚠️ No atom dataframe found in self_main.df")
                return None

            if not (0 <= index < len(atom_df)):
                log.message(
                    f"⚠️ Index {index} out of range for dataframe of length {len(atom_df)}"
                )
                return None

            if 0 <= index < len(atom_df):
                atom_row = atom_df.iloc[index]
                atom_name = atom_row.get("atom_name")
                residue_name = atom_row.get("residue_name")
                residue_id = atom_row.get("residue_number")
                chain_id = atom_row.get("chain_id")
                record_type = atom_row.get("record_type", None)
                atom_data = AtomData(atom_row=atom_row,
                                     atom_name=atom_name,
                                     residue_name=residue_name,
                                     residue_id=residue_id,
                                     chain_id=chain_id,
                                     record_type=record_type)
                return atom_data

            return None
        except Exception as ex:
            log.error(f"Error reading atom metadata at index {index}: {ex}")

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

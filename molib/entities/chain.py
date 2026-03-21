"""
Chain class
"""

from typing import List, Optional, Union

import numpy as np
from decologr import Decologr as log
from molib.core.color.strategy import ColorScheme


class Chain3D:
    """Chain3D class - Pure Python class for performance."""

    def __init__(
        self,
        name: Optional[str] = None,
        chain_id: Optional[str] = None,
        residues: Optional[List["Res3D"]] = None,
        parent: Optional["Model3D"] = None,
        **kwargs,
    ):
        self.name = name
        self.chain_id = chain_id
        self.residues = residues or []
        self.parent = parent

    def add_residue(self, residue: "Res3D") -> None:
        """add_residue"""
        residue.parent = self
        self.residues.append(residue)

    def append_residue(self, res: "Res3D"):
        """append_residue"""
        self.residues.append(res)

    def get_ca_coords(self) -> np.ndarray:
        """get_ca_coords"""
        return np.array(
            [res.ca for res in self.residues if res.has_ca()], dtype=np.float32
        )

    def get_ca_colors(self) -> np.ndarray:
        """get_ca_colors"""
        return np.array(
            [res.color for res in self.residues if res.has_ca()], dtype=np.float32
        )

    def get_ca_data(self):
        """Get coordinates and colors for calphas in a chain"""
        data = [(res.ca, res.color) for res in self.residues if res.has_ca()]
        if not data:
            empty = np.zeros((0, 3), dtype=np.float32)
            return empty, empty.copy()
        coords, colors = zip(*data)
        return np.array(coords, dtype=np.float32), np.array(colors, dtype=np.float32)

    def get_coord_data(self):
        """Get coordinates and colors for atoms in a chain."""
        data = [(atom.coords, atom.color) for res in self.residues for atom in res]
        if not data:
            empty = np.zeros((0, 3), dtype=np.float32)
            return empty, empty.copy()
        coords, colors = zip(*data)
        return np.array(coords, dtype=np.float32), np.array(colors, dtype=np.float32)

    def get_backbone_trace(self) -> list[np.ndarray]:
        """get_backbone_trace"""
        return [res.ca for res in self.residues if res.has_ca()]

    def __iter__(self):
        return iter(self.residues)

    def __getitem__(self, idx):
        return self.residues[idx]

    def get_atoms(self, name: str) -> list["Atom3D"]:
        """Return all atoms with the given color_scheme in this chain."""
        return [res.atoms[name] for res in self.residues if name in res.atoms]

    def has_atom(self, name: str) -> bool:
        """True if any residue in the chain has an atom with the given color_scheme."""
        return any(name in res.atoms for res in self.residues)

    def apply_atom_color_scheme(
        self, color_scheme: Union[str, ColorScheme] = ColorScheme.ELEMENT
    ) -> None:
        """
        apply_atom_color_scheme - PERFORMANCE OPTIMIZED

        :param color_scheme: Union[str, ColorScheme]
        """
        try:
            # Collect all atoms for batch processing
            all_atoms = []
            for residue in self.residues:
                all_atoms.extend(residue.atoms.values())

            # Apply color scheme in batch
            from molib.entities.atom import Atom3D

            Atom3D.apply_color_scheme_batch(all_atoms, color_scheme)
        except Exception as ex:
            log.error(f"Error {ex} occurred applying atom coloring")

    def add_residue(self, residue: "Res3D") -> None:
        """
        add_residue

        :param residue: Res3D
        """
        residue.parent = self
        self.residues.append(residue)

    def set_color(self, r, g, b):
        """set color"""
        for residue in self.residues:
            residue.set_color(r, g, b)

"""
Res3D class
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from molib.core.constants import MoLibConstant
from molib.entities.secondary_structure_type import SecondaryStructureType
from molib.entities.structure import Structure3D


class Res3D(Structure3D):
    """3D Residue class - Pure Python class for performance."""

    def __init__(
        self,
        residue_number: int = 0,
        atoms: Optional[Dict[str, "Atom3D"]] = None,
        parent: Optional["Chain3D"] = None,
        residue_validated: Optional[bool] = None,
        residue_validation_error: Optional[str] = None,
        residue_selected: bool = False,
        # Structure3D parameters
        name: str = "",
        type: str = "",
        selected: bool = False,
        visible: bool = True,
        secstruc: Optional[SecondaryStructureType] = None,
        chain_id: str = "A",
        coords: tuple = (0.0, 0.0, 0.0),
        next: Optional["Structure3D"] = None,
        **kwargs,
    ):
        # Initialize Structure3D first
        super().__init__(
            name=name,
            type=type,
            selected=selected,
            visible=visible,
            secstruc=secstruc or SecondaryStructureType.COIL,
            chain_id=chain_id,
            coords=coords,
            next=next,
            **kwargs,
        )

        # Residue-specific attributes
        self.residue_number = residue_number
        self.atoms = atoms or {}
        self.parent = parent
        self.residue_validated = residue_validated
        self.residue_validation_error = residue_validation_error
        self.residue_selected = residue_selected

    def add_atom(self, atom: "Atom3D") -> None:
        """Add an atom to the residue, ensuring no shared mutable defaults."""
        atom.parent = self
        self.atoms[atom.name] = atom

    def has_atom(self, name: str) -> bool:
        """Check if the residue contains an atom with the given color_scheme."""
        return name in self.atoms

    def get_atom(self, name: str) -> Optional["Atom3D"]:
        """Get an atom by color_scheme, or None if not found."""
        return self.atoms.get(name)

    @property
    def ca(self) -> "np.ndarray":
        """Return CA atom position if present, otherwise fallback to residue coords."""
        if MoLibConstant.PEPTIDE_CHAIN_ATOMNAME in self.atoms:
            return self.atoms[MoLibConstant.PEPTIDE_CHAIN_ATOMNAME].pos
        return self.pos

    def has_ca(self) -> bool:
        """
        Return True if this residue contains an actual ``CA`` atom with
        non-zero coordinates.

        Notes
        -----
        ``Res3D.ca`` falls back to ``self.pos`` when ``MoLibConstant.PEPTIDE_CHAIN_ATOMNAME`` is missing.
        If ``has_ca`` were based on that fallback, residues without a CA atom
        would still be included in C-alpha traces, producing a "carbon-like"
        backbone.
        """
        if MoLibConstant.PEPTIDE_CHAIN_ATOMNAME not in self.atoms:
            return False
        return not np.allclose(self.atoms[MoLibConstant.PEPTIDE_CHAIN_ATOMNAME].pos, 0.0)

    def get_atom_positions(self) -> "np.ndarray":
        """Return all atom positions as a NumPy array."""
        return np.array([atom.pos for atom in self.atoms.values()], dtype=np.float32)

    def set_color(self, r: float, g: float, b: float) -> None:
        """
        Set colour for the residue and all its atoms.
        Updates the colour in-place to avoid breaking references.
        """
        # Update residue colour
        self.color[:] = (r, g, b)
        # Update all atom colors
        for atom in self.atoms.values():
            atom.color[:] = (r, g, b)

    def apply_atom_coloring_by_strategy(self, color_scheme: str) -> None:
        """
        apply_atom_coloring_by_strategy

        :param color_scheme: ColorScheme to apply to the Residue
        """
        for atom_name, atom in self.atoms.items():
            atom.apply_atom_color_scheme(color_scheme)

    # ------------------------------------------------------------------
    # Alt-loc helpers
    # ------------------------------------------------------------------

    def list_alt_locs(self) -> list[str]:
        """Return a sorted list of distinct alt_loc identifiers present in this residue."""
        alts = {getattr(atom, "alt_loc", None) for atom in self.atoms.values()}
        alts.discard(None)
        alts.discard("")
        return sorted(alts)

    def get_atoms_for_alt_loc(self, alt_loc: str) -> dict[str, "Atom3D"]:
        """
        Return a mapping of atom name -> Atom3D for a given alt_loc identifier.

        Atoms without alt_loc or with a different alt_loc are excluded.
        """
        return {
            name: atom
            for name, atom in self.atoms.items()
            if getattr(atom, "alt_loc", None) == alt_loc
        }

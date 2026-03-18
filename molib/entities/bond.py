"""
Bond3D class - structural bond representation between two Atom3D instances.

This is intentionally lightweight and free of OpenGL concerns so it can be
used as the core connectivity representation for:
- valence / coordination checks
- connectivity-based selections
- fragment / molecule splitting
- feeding rendering backends that need a bond graph
"""

from __future__ import annotations

from typing import Optional

from molib.entities.atom import Atom3D


class Bond3D:
    """
    Simple bond between two atoms, with optional validation metadata.

    This class is kept as a regular Python class (not a dataclass) to stay
    consistent with other performance-sensitive 3D classes in ElMo.
    """

    def __init__(
        self,
        atom1: Atom3D,
        atom2: Atom3D,
        order: Optional[int] = None,
        length: Optional[float] = None,
        ideal_length: Optional[float] = None,
        delta: Optional[float] = None,
        z_score: Optional[float] = None,
    ) -> None:
        self.atom1 = atom1
        self.atom2 = atom2
        self.order = order
        self.length = length
        self.ideal_length = ideal_length
        self.delta = delta
        self.z_score = z_score

    def other(self, atom: Atom3D) -> Atom3D:
        """Return the atom at the other end of this bond."""
        if atom is self.atom1:
            return self.atom2
        if atom is self.atom2:
            return self.atom1
        raise ValueError("Atom is not part of this bond.")


__all__ = ["Bond3D"]

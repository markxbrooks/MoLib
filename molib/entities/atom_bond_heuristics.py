"""
Optional UglyMol-style covalent bond *guesses* based on distance and element radii.

**Not** used by ElMo OpenGL bond buffers (those use ``BufferFactory`` / dedicated bond
perception). Use this module only when you want a quick, dependency-light heuristic
(e.g. prototyping or crystallography utilities) without implying parity with the
renderer's bond graph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from molib.entities.atom import Atom3D

# Upper bound used in legacy UglyMol ``Atom.is_bonded_to`` (squared angstroms).
_DEFAULT_MAX_DIST_SQ = 2.2 * 2.2


def uglymol_bond_radius(atom: Atom3D) -> float:
    """Crude covalent radius factor by element (Angstrom scale, UglyMol-compatible)."""
    el = (atom.element or "").strip().upper()
    if el == "H":
        return 1.3
    if el in ("S", "P"):
        return 2.43
    return 1.99


def uglymol_is_bonded_approximate(
    a: Atom3D,
    b: Atom3D,
    *,
    max_dist_sq: float = _DEFAULT_MAX_DIST_SQ,
) -> bool:
    """
    Return True if ``a`` and ``b`` pass a simple distance + conformer + element filter.

    This mirrors the historical UglyMol ``Atom.is_bonded_to`` logic; it is **not**
    a full chemistry perception algorithm.
    """
    if not a.is_same_conformer(b):
        return False
    dxyz2 = a.distance_sq(b)
    if dxyz2 > max_dist_sq:
        return False
    if a.is_hydrogen() and b.is_hydrogen():
        return False
    return dxyz2 <= uglymol_bond_radius(a) * uglymol_bond_radius(b)

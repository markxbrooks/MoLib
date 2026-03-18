"""
Lightweight selection helpers built on top of the existing 3D hierarchy.

These are intentionally implemented as simple Python utilities for now, with
an eye toward later replacement by a compiled selection core that operates
on array-backed metadata.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from molib.entities.atom import Atom3D
from elmo.gl.renderers.molecule import MoleculeRenderer
from molib.entities.residue import Res3D


def select_residue_range(
    mol: MoleculeRenderer,
    chain_id: str,
    start_resi: int,
    end_resi: int,
) -> List[Atom3D]:
    """
    Return all atoms in the given chain and residue range [start_resi, end_resi].
    """
    atoms: List[Atom3D] = []
    for model in mol.models:
        chain = model.chains.get(chain_id)
        if chain is None:
            continue
        for residue in chain.residues:
            if start_resi <= residue.residue_number <= end_resi:
                atoms.extend(residue.atoms.values())
    return atoms


def select_chain_atoms(mol: MoleculeRenderer, chain_id: str) -> List[Atom3D]:
    """
    Return all atoms in a given chain across all models.
    """
    atoms: List[Atom3D] = []
    for model in mol.models:
        chain = model.chains.get(chain_id)
        if chain is None:
            continue
        for residue in chain.residues:
            atoms.extend(residue.atoms.values())
    return atoms


def select_segment_atoms(mol: MoleculeRenderer, segment_id: str) -> List[Atom3D]:
    """
    Return all atoms whose segment_id matches the given identifier.
    """
    atoms: List[Atom3D] = []
    for atom in mol.get_all_atoms():
        if isinstance(atom, Atom3D) and atom.segment_id == segment_id:
            atoms.append(atom)
    return atoms


def select_residue_objects(
    mol: MoleculeRenderer,
    chain_id: Optional[str] = None,
    residue_numbers: Optional[Sequence[int]] = None,
) -> List[Res3D]:
    """
    Return residue objects matching an optional chain and set of residue numbers.
    """
    residues: List[Res3D] = []
    residue_set = set(residue_numbers) if residue_numbers is not None else None

    for model in mol.models:
        for cid, chain in model.chains.items():
            if chain_id is not None and cid != chain_id:
                continue
            for residue in chain.residues:
                if residue_set is None or residue.residue_number in residue_set:
                    residues.append(residue)
    return residues


__all__ = [
    "select_residue_range",
    "select_chain_atoms",
    "select_segment_atoms",
    "select_residue_objects",
]

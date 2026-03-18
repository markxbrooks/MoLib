"""
Segment3D abstraction and helpers.

Segments group residues (and therefore atoms) by a shared `segment_id`
identifier, similar to PyMOL's `segi` concept. This module provides:

- `Segment3D`: a lightweight container for residues sharing a segment ID.
- `build_segments_for_model()`: construct segments for a given Model3D.
- `build_segments_for_molecule()`: construct segments for all models in a Molecule3D.

This is implemented as an optional overlay on top of the existing hierarchy
and does not modify `Model3D` or `Molecule3D` state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from molib.entities.atom import Atom3D
from molib.entities.model import Model3D
from elmo.gl.renderers.molecule import MoleculeRenderer
from molib.entities.residue import Res3D


@dataclass
class Segment3D:
    """Logical grouping of residues by segment identifier (`segment_id`)."""

    segment_id: str
    residues: List[Res3D] = field(default_factory=list)

    @property
    def atoms(self) -> List[Atom3D]:
        """All atoms belonging to this segment across all residues."""
        atoms: List[Atom3D] = []
        for res in self.residues:
            atoms.extend(res.atoms.values())
        return atoms


def build_segments_for_model(model: Model3D) -> Dict[str, Segment3D]:
    """
    Build Segment3D objects for a Model3D based on atom.segment_id values.

    Returns a mapping: segment_id -> Segment3D.
    """
    segments: Dict[str, Segment3D] = {}

    for chain in model.chains.values():
        for residue in chain.residues:
            # Collect segment IDs from atoms in this residue
            seg_ids = {
                getattr(atom, "segment_id", None) for atom in residue.atoms.values()
            }
            seg_ids.discard(None)
            seg_ids.discard("")
            for seg_id in seg_ids:
                seg = segments.get(seg_id)
                if seg is None:
                    seg = segments[seg_id] = Segment3D(segment_id=seg_id)
                seg.residues.append(residue)

    return segments


def build_segments_for_molecule(
    mol: MoleculeRenderer,
) -> Dict[int, Dict[str, Segment3D]]:
    """
    Build Segment3D objects for all models in a Molecule3D.

    Returns a nested mapping:
        { model_id: { segment_id: Segment3D, ... }, ... }
    """
    result: Dict[int, Dict[str, Segment3D]] = {}
    for model in mol.models:
        result[model.model_id] = build_segments_for_model(model)
    return result


__all__ = [
    "Segment3D",
    "build_segments_for_model",
    "build_segments_for_molecule",
]

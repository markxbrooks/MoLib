"""
Pure structural model layer (no OpenGL).

This module defines lightweight data models that capture the molecular
hierarchy without any rendering concerns:

- AtomModel
- ResidueModel
- ChainModel
- ModelModel
- MoleculeModel

They can be constructed from the existing 3D classes (`Atom3D`, `Res3D`,
`Chain3D`, `Model3D`, `Molecule3D`) and used by non‑rendering code
(parsers, analysis, validation, serialization) without importing OpenGL
or buffer factories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from molib.entities.atom import Atom3D
from molib.entities.chain import Chain3D
from molib.entities.model import Model3D
from molib.entities.residue import Res3D

from elmo.gl.renderers.molecule import MoleculeRenderer


@dataclass
class AtomModel:
    name: str
    element: Optional[str]
    coords: Tuple[float, float, float]
    chain_id: str
    residue_number: int
    residue_name: str
    segment_id: Optional[str] = None
    alt_loc: Optional[str] = None
    b_factor: float = 0.0
    occupancy: float = 1.0


@dataclass
class ResidueModel:
    residue_number: int
    name: str
    chain_id: str
    atoms: List[AtomModel] = field(default_factory=list)


@dataclass
class ChainModel:
    chain_id: str
    residues: List[ResidueModel] = field(default_factory=list)


@dataclass
class ModelModel:
    model_id: int
    chains: Dict[str, ChainModel] = field(default_factory=dict)


@dataclass
class MoleculeModel:
    name: str
    models: List[ModelModel] = field(default_factory=list)


def atom3d_to_model(atom: Atom3D) -> AtomModel:
    """Convert an Atom3D instance into a pure AtomModel."""
    parent_res: Optional[Res3D] = atom.parent
    residue_number = getattr(parent_res, "residue_number", 0) if parent_res else 0
    residue_name = getattr(parent_res, "name", "") if parent_res else ""

    x, y, z = map(float, atom.coords)
    return AtomModel(
        name=atom.name,
        element=atom.element,
        coords=(x, y, z),
        chain_id=atom.chain_id,
        residue_number=residue_number,
        residue_name=residue_name,
        segment_id=atom.segment_id,
        alt_loc=atom.alt_loc,
        b_factor=float(getattr(atom, "b_factor", 0.0) or 0.0),
        occupancy=float(getattr(atom, "occupancy", 1.0) or 1.0),
    )


def residue3d_to_model(res: Res3D) -> ResidueModel:
    """Convert a Res3D into a ResidueModel (including atoms)."""
    atoms = [atom3d_to_model(atom) for atom in res.atoms.values()]
    return ResidueModel(
        residue_number=res.residue_number,
        name=res.name,
        chain_id=res.chain_id,
        atoms=atoms,
    )


def chain3d_to_model(chain: Chain3D) -> ChainModel:
    """Convert a Chain3D into a ChainModel."""
    residues = [residue3d_to_model(res) for res in chain.residues]
    return ChainModel(chain_id=chain.chain_id or "", residues=residues)


def model3d_to_model(model: Model3D) -> ModelModel:
    """Convert a Model3D into a ModelModel."""
    chains = {cid: chain3d_to_model(chain) for cid, chain in model.chains.items()}
    return ModelModel(model_id=model.model_id, chains=chains)


def molecule3d_to_model(mol: MoleculeRenderer) -> MoleculeModel:
    """
    Convert a Molecule3D (rendering-aware) into a pure MoleculeModel.

    This provides a GL‑free representation that can be used by parsing,
    analysis, and validation code.
    """
    models = [model3d_to_model(m) for m in mol.models]
    return MoleculeModel(name=mol.name, models=models)


__all__ = [
    "AtomModel",
    "ResidueModel",
    "ChainModel",
    "ModelModel",
    "MoleculeModel",
    "atom3d_to_model",
    "residue3d_to_model",
    "chain3d_to_model",
    "model3d_to_model",
    "molecule3d_to_model",
]

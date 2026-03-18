"""
Represents information about a ligand molecule including its RDKit-derived
properties and various identifiers.

This module defines the `LigandInfo` class, which encapsulates attributes
and methods relevant to describing a single ligand molecule. It supports
the calculation of various chemical properties using RDKit tools and
facilitates structured representation of ligand data including identifiers,
SMILES, and molecular properties.
"""

from dataclasses import dataclass
from typing import Optional

from rdkit import Chem

from molib.ligand.attrs import LigandAttrs
from molib.ligand.file.parser import Crippen, Descriptors, rdMolDescriptors


@dataclass
class LigandInfo:
    """Information about a single ligand molecule"""

    name: str
    smiles: str

    mol: Optional[Chem.Mol] = None
    ligand_id: int | None = None
    sdf_data: Optional[str] = None
    inchikey: str | None = None
    canonical_smiles: str | None = None
    murcko_scaffold_id: str | None = None
    cluster_id: int | None = None

    def __repr__(self):
        return f"<LigandInfo: {self.name} {self.smiles} self.mol={self.mol}>"

    # ---------- RDKit-derived properties ----------

    @property
    def molecular_weight(self) -> float:
        return Descriptors.MolWt(self.mol) if self.mol else float("nan")

    @property
    def logp(self) -> float:
        return Crippen.MolLogP(self.mol) if self.mol else float("nan")

    @property
    def tpsa(self) -> float:
        return rdMolDescriptors.CalcTPSA(self.mol) if self.mol else float("nan")

    @property
    def hbd(self) -> int:
        return rdMolDescriptors.CalcNumHBD(self.mol) if self.mol else 0

    @property
    def hba(self) -> int:
        return rdMolDescriptors.CalcNumHBA(self.mol) if self.mol else 0

    @property
    def rotatable_bonds(self) -> int:
        return rdMolDescriptors.CalcNumRotatableBonds(self.mol) if self.mol else 0

    @property
    def heavy_atoms(self) -> int:
        return self.mol.GetNumHeavyAtoms() if self.mol else 0

    @property
    def aromatic_rings(self) -> int:
        return rdMolDescriptors.CalcNumAromaticRings(self.mol) if self.mol else 0

    @property
    def formula(self) -> str:
        return rdMolDescriptors.CalcMolFormula(self.mol) if self.mol else ""

    @property
    def fraction_sp3(self) -> float:
        return rdMolDescriptors.CalcFractionCSP3(self.mol) if self.mol else float("nan")

    def to_dict(self):
        data = {
            LigandAttrs.LIGAND_ID: self.ligand_id,
            LigandAttrs.NAME: self.name,
            LigandAttrs.SMILES: self.smiles,
            LigandAttrs.CANONICAL_SMILES: self.canonical_smiles,
            LigandAttrs.INCHIKEY: self.inchikey,
            LigandAttrs.MURCKO_SCAFFOLD_ID: self.murcko_scaffold_id,
            LigandAttrs.CLUSTER_ID: self.cluster_id,
        }

        for attr in LigandAttrs.DESCRIPTOR_ATTRS:
            data[attr] = getattr(self, attr)

        return data

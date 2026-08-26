"""
Representation of information related to a PDB ligand molecule as a data structure.

This module defines a data class for storing detailed information about a ligand
molecule found in Protein Data Bank (PDB) files. It encompasses structural,
chemical, and molecular details essential for computational and structural analyses.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from rdkit.Chem import Mol, Descriptors, Lipinski, rdMolDescriptors, MolToSmiles, MolToInchi, InchiToInchiKey


@dataclass
class PDBLigandInfo:
    """Information about a PDB ligand molecule.

    Fully backward-compatible with older attribute-assignment pipelines,
    while utilizing RDKit as a dynamic backend when available.
    """
    # Required Identification Fields
    ligand_id: str
    ligand_name: str
    chain_id: str
    res_seq: int
    insertion_code: str

    # Core engine link (Optional for legacy compatibility)
    mol: Optional[Mol] = None

    # Clustering metadata fields
    murcko_scaffold_id: Optional[str] = None
    cluster_id: Optional[int] = None

    # Legacy private backing stores for backward-compatible setters
    _atom_count: Optional[int] = None
    _coordinates: Optional[List[Tuple[float, float, float]]] = None
    _atom_names: Optional[List[str]] = None
    _element_symbols: Optional[List[str]] = None
    _smiles: Optional[str] = None
    _molecular_weight: Optional[float] = None
    _formula: Optional[str] = None
    _logp: Optional[float] = None
    _hbd: Optional[int] = None
    _hba: Optional[int] = None
    _rotatable_bonds: Optional[int] = None
    _aromatic_rings: Optional[int] = None
    _heavy_atoms: Optional[int] = None
    _inchikey: Optional[str] = None
    _canonical_smiles: Optional[str] = None
    _fraction_sp3: Optional[float] = None

    # --- Structural Properties (RDKit Backend with Manual Fallbacks) ---

    @property
    def atom_count(self) -> int:
        return self.mol.GetNumAtoms() if self.mol else (self._atom_count or 0)

    @atom_count.setter
    def atom_count(self, value: int):
        self._atom_count = value

    @property
    def coordinates(self) -> List[Tuple[float, float, float]]:
        if self.mol and self.mol.GetNumConformers() > 0:
            return [tuple(self.mol.GetConformer().GetAtomPosition(i)) for i in range(self.atom_count)]
        return self._coordinates or []

    @coordinates.setter
    def coordinates(self, value: List[Tuple[float, float, float]]):
        self._coordinates = value

    @property
    def atom_names(self) -> List[str]:
        if self.mol:
            return [a.GetMonomerInfo().GetName().strip() if a.GetMonomerInfo() else a.GetSymbol() for a in
                    self.mol.GetAtoms()]
        return self._atom_names or []

    @atom_names.setter
    def atom_names(self, value: List[str]):
        self._atom_names = value

    @property
    def element_symbols(self) -> List[str]:
        return [a.GetSymbol() for a in self.mol.GetAtoms()] if self.mol else (self._element_symbols or [])

    @element_symbols.setter
    def element_symbols(self, value: List[str]):
        self._element_symbols = value

    # --- Chemical Properties & Descriptors ---

    @property
    def smiles(self) -> str:
        return MolToSmiles(self.mol) if self.mol else (self._smiles or "")

    @smiles.setter
    def smiles(self, value: str):
        self._smiles = value

    @property
    def canonical_smiles(self) -> str:
        return MolToSmiles(self.mol, canonical=True) if self.mol else (self._canonical_smiles or "")

    @canonical_smiles.setter
    def canonical_smiles(self, value: str):
        self._canonical_smiles = value

    @property
    def molecular_weight(self) -> float:
        return Descriptors.MolWt(self.mol) if self.mol else (self._molecular_weight or 0.0)

    @molecular_weight.setter
    def molecular_weight(self, value: float):
        self._molecular_weight = value

    @property
    def mw(self) -> float:
        """Alias matching MolecularProperties framework."""
        return self.molecular_weight

    @property
    def formula(self) -> str:
        return rdMolDescriptors.CalcMolFormula(self.mol) if self.mol else (self._formula or "")

    @formula.setter
    def formula(self, value: str):
        self._formula = value

    @property
    def logp(self) -> float:
        return Descriptors.MolLogP(self.mol) if self.mol else (self._logp or 0.0)

    @logp.setter
    def logp(self, value: float):
        self._logp = value

    @property
    def hbd(self) -> int:
        return Lipinski.NumHDonors(self.mol) if self.mol else (self._hbd or 0)

    @hbd.setter
    def hbd(self, value: int):
        self._hbd = value

    @property
    def hba(self) -> int:
        return Lipinski.NumHAcceptors(self.mol) if self.mol else (self._hba or 0)

    @hba.setter
    def hba(self, value: int):
        self._hba = value

    @property
    def tpsa(self) -> float:
        return Descriptors.TPSA(self.mol) if self.mol else (self._logp or 0.0)  # mapping safety

    @property
    def rotatable_bonds(self) -> int:
        return Lipinski.NumRotatableBonds(self.mol) if self.mol else (self._rotatable_bonds or 0)

    @rotatable_bonds.setter
    def rotatable_bonds(self, value: int):
        self._rotatable_bonds = value

    @property
    def aromatic_rings(self) -> int:
        return Lipinski.NumAromaticRings(self.mol) if self.mol else (self._aromatic_rings or 0)

    @aromatic_rings.setter
    def aromatic_rings(self, value: int):
        self._aromatic_rings = value

    @property
    def heavy_atoms(self) -> int:
        return self.mol.GetNumHeavyAtoms() if self.mol else (self._heavy_atoms or 0)

    @heavy_atoms.setter
    def heavy_atoms(self, value: int):
        self._heavy_atoms = value

    @property
    def inchikey(self) -> Optional[str]:
        if self.mol:
            try:
                return InchiToInchiKey(MolToInchi(self.mol))
            except:
                return None
        return self._inchikey

    @inchikey.setter
    def inchikey(self, value: Optional[str]):
        self._inchikey = value

    @property
    def fraction_sp3(self) -> float:
        return Lipinski.FractionCSP3(self.mol) if self.mol else (self._fraction_sp3 or 0.0)

    @fraction_sp3.setter
    def fraction_sp3(self, value: float):
        self._fraction_sp3 = value

"""
Representation of information related to a PDB ligand molecule as a data structure.

This module defines a data class for storing detailed information about a ligand
molecule found in Protein Data Bank (PDB) files. It encompasses structural,
chemical, and molecular details essential for computational and structural analyses.
"""

from typing import Any, Dict, List, Optional, Tuple

from rdkit.Chem import (
    Descriptors,
    InchiToInchiKey,
    Lipinski,
    Mol,
    MolToInchi,
    MolToSmiles,
    rdMolDescriptors,
)


class PDBLigandInfo:
    """Information about a PDB ligand molecule.

    Uses explicit Python descriptors to intercept legacy kwargs seamlessly.
    When ``mol`` is set, chemical properties are derived from RDKit on demand.
    """

    def __init__(
        self,
        ligand_id: str,
        ligand_name: str,
        chain_id: str,
        res_seq: int,
        insertion_code: str = "",
        mol: Optional[Mol] = None,
        murcko_scaffold_id: Optional[str] = None,
        cluster_id: Optional[int] = None,
        docking_score: Optional[float] = None,
        **kwargs,  # Gracefully absorbs any historical fields passed by tests
    ):
        self.ligand_id = ligand_id
        self.ligand_name = ligand_name
        self.chain_id = chain_id
        self.res_seq = res_seq
        self.insertion_code = insertion_code
        self.mol = mol
        self.murcko_scaffold_id = murcko_scaffold_id
        self.cluster_id = cluster_id
        self.docking_score = docking_score

        # Route legacy parameters directly into backing slots
        self._atom_count = kwargs.get("atom_count")
        self._coordinates = kwargs.get("coordinates")
        self._atom_names = kwargs.get("atom_names")
        self._element_symbols = kwargs.get("element_symbols")
        self._smiles = kwargs.get("smiles")
        self._molecular_weight = kwargs.get("molecular_weight") or kwargs.get("mw")
        self._formula = kwargs.get("formula")
        self._logp = kwargs.get("logp")
        self._tpsa = kwargs.get("tpsa")
        self._hbd = kwargs.get("hbd")
        self._hba = kwargs.get("hba")
        self._rotatable_bonds = kwargs.get("rotatable_bonds")
        self._aromatic_rings = kwargs.get("aromatic_rings")
        self._heavy_atoms = kwargs.get("heavy_atoms")
        self._inchikey = kwargs.get("inchikey")
        self._canonical_smiles = kwargs.get("canonical_smiles")
        self._fraction_sp3 = kwargs.get("fraction_sp3")

    # --- Structural Properties ---

    @property
    def atom_count(self) -> int:
        return self.mol.GetNumAtoms() if self.mol else (self._atom_count or 0)

    @atom_count.setter
    def atom_count(self, value: int):
        self._atom_count = value

    @property
    def coordinates(self) -> List[Tuple[float, float, float]]:
        if self.mol and self.mol.GetNumConformers() > 0:
            return [
                tuple(self.mol.GetConformer().GetAtomPosition(i))
                for i in range(self.atom_count)
            ]
        return self._coordinates or []

    @coordinates.setter
    def coordinates(self, value: List[Tuple[float, float, float]]):
        self._coordinates = value

    @property
    def atom_names(self) -> List[str]:
        if self.mol:
            return [
                a.GetMonomerInfo().GetName().strip()
                if a.GetMonomerInfo()
                else a.GetSymbol()
                for a in self.mol.GetAtoms()
            ]
        return self._atom_names or []

    @atom_names.setter
    def atom_names(self, value: List[str]):
        self._atom_names = value

    @property
    def element_symbols(self) -> List[str]:
        return (
            [a.GetSymbol() for a in self.mol.GetAtoms()]
            if self.mol
            else (self._element_symbols or [])
        )

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
        return (
            MolToSmiles(self.mol, canonical=True)
            if self.mol
            else (self._canonical_smiles or "")
        )

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

    @mw.setter
    def mw(self, value: float):
        self.molecular_weight = value

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
        return Descriptors.TPSA(self.mol) if self.mol else (self._tpsa or 0.0)

    @tpsa.setter
    def tpsa(self, value: float):
        self._tpsa = value

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
            except Exception:
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

    def to_dict(self) -> Dict[str, Any]:
        """Flatten identity and key descriptors for CSV / DataFrame export."""
        return {
            "ligand_id": self.ligand_id,
            "ligand_name": self.ligand_name,
            "chain_id": self.chain_id,
            "res_seq": self.res_seq,
            "smiles": self.smiles,
            "mw": self.molecular_weight,
            "logp": self.logp,
            "tpsa": self.tpsa,
            "rotatable_bonds": self.rotatable_bonds,
            "aromatic_rings": self.aromatic_rings,
            "heavy_atoms": self.heavy_atoms,
            "atom_count": self.atom_count,
            "docking_score": self.docking_score,
        }

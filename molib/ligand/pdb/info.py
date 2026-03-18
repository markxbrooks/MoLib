"""
Representation of information related to a PDB ligand molecule as a data structure.

This module defines a data class for storing detailed information about a ligand
molecule found in Protein Data Bank (PDB) files. It encompasses structural,
chemical, and molecular details essential for computational and structural analyses.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class PDBLigandInfo:
    """Information about a PDB ligand molecule"""

    # Ligand identification
    ligand_id: str  # 3-letter ligand code (e.g., 'ANP', 'MPD', 'MRD')
    ligand_name: str  # Full ligand name from PDB

    # Ligand structure
    chain_id: str  # Chain identifier
    res_seq: int  # Residue sequence number
    insertion_code: str  # Insertion code if present

    # Structural properties
    atom_count: int  # Number of atoms in the ligand
    coordinates: List[tuple]  # List of (x, y, z) coordinates
    atom_names: List[str]  # List of atom names
    element_symbols: List[str]  # List of element symbols

    # Chemical properties  from RDKit
    smiles: str  # SMILES representation (if RDKit available)
    molecular_weight: float  # Molecular weight
    formula: str  # Molecular formula
    logp: float  # LogP value
    hbd: int  # Hydrogen bond donors
    hba: int  # Hydrogen bond acceptors
    tpsa: float  # Topological polar surface area
    rotatable_bonds: int
    aromatic_rings: int
    heavy_atoms: int
    inchikey: str | None = None
    canonical_smiles: str | None = None  # <-didn't exist!
    fraction_sp3: float | None = None
    murcko_scaffold_id: str | None = None
    cluster_id: int | None = None

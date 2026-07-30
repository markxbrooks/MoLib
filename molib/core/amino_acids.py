"""Canonical PDB three-letter amino-acid residue names."""

from __future__ import annotations

STANDARD_AMINO_ACIDS: frozenset[str] = frozenset(
    {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    }
)

# Ribbon / polypeptide recognition (includes common modified residues).
STANDARD_POLYPEPTIDE_RESIDUES: frozenset[str] = STANDARD_AMINO_ACIDS | frozenset(
    {"SEC", "PYL", "MSE"}
)

# Uglymol ligand-exclusion base (standard + MSE + unknown).
AMINO_ACIDS: frozenset[str] = STANDARD_AMINO_ACIDS | frozenset({"MSE", "UNK"})

# Back-compat alias used by molib.core.structure and docs.
STANDARD_AA_RESIDUES: frozenset[str] = STANDARD_AMINO_ACIDS

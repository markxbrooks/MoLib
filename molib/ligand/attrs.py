"""
This module defines the `LigandAttrs` class which contains a collection of
constants representing various attributes and descriptors related to a ligand.
The class includes identifiers, structural information, clustering attributes,
and molecular descriptors, as well as predefined sets of export columns and
descriptors for easy reference and usage.
"""


class LigandAttrs:
    """Ligand attributes"""

    # identifiers
    LIGAND_ID = "ligand_id"
    NAME = "name"

    # structure
    SMILES = "smiles"
    CANONICAL_SMILES = "canonical_smiles"
    INCHIKEY = "inchikey"
    MOL = "mol"

    # clustering
    MURCKO_SCAFFOLD_ID = "murcko_scaffold_id"
    CLUSTER_ID = "cluster_id"

    # descriptors
    FRACTION_SP3 = "fraction_sp3"
    MOLECULAR_WEIGHT = "molecular_weight"
    LOGP = "logp"
    TPSA = "tpsa"
    HBD = "hbd"
    HBA = "hba"
    ROTATABLE_BONDS = "rotatable_bonds"
    HEAVY_ATOMS = "heavy_atoms"
    AROMATIC_RINGS = "aromatic_rings"
    FORMULA = "formula"

    EXPORT_COLUMNS = [
        LIGAND_ID,
        NAME,
        SMILES,
        CANONICAL_SMILES,
        INCHIKEY,
        MURCKO_SCAFFOLD_ID,
        CLUSTER_ID,
        MOLECULAR_WEIGHT,
        LOGP,
        TPSA,
        HBD,
        HBA,
        ROTATABLE_BONDS,
        HEAVY_ATOMS,
        AROMATIC_RINGS,
        FORMULA,
        FRACTION_SP3,
    ]

    DESCRIPTOR_ATTRS = [
        MOLECULAR_WEIGHT,
        LOGP,
        TPSA,
        HBD,
        HBA,
        ROTATABLE_BONDS,
        HEAVY_ATOMS,
        AROMATIC_RINGS,
        FORMULA,
        FRACTION_SP3,
    ]

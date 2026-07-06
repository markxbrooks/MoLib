"""Backward-compatible re-exports for PDB ligand RDKit helpers."""

from molib.ligand.bond import (
    add_bond,
    add_conformer,
    detect_bonds,
    determine_bond_order_based_on_distance,
)
from molib.ligand.element import (
    ELEMENTS,
    Element,
    calculate_distance,
    get_covalent_radii,
)
from molib.ligand.rdkit.smiles.component import (
    SmilesComponent,
    add_hydrogen_and_optimize_geometry,
    check_rdkit_availability,
    create_common_ligand_molecule,
    create_mol_with_conformer,
    create_molecule_alternative,
    create_molecule_from_coordinates,
    create_pdb_ligand_info,
    create_sulfate_from_coordinates,
    embed_and_optimize,
    generate_clean_smiles,
    is_connected_molecule,
    validate_molecule,
)
from molib.ligand.rdkit.smiles.symbol import SmilesSymbol, StrEnum

__all__ = [
    "ELEMENTS",
    "Element",
    "SmilesComponent",
    "SmilesSymbol",
    "StrEnum",
    "add_bond",
    "add_conformer",
    "add_hydrogen_and_optimize_geometry",
    "calculate_distance",
    "check_rdkit_availability",
    "create_common_ligand_molecule",
    "create_mol_with_conformer",
    "create_molecule_alternative",
    "create_molecule_from_coordinates",
    "create_pdb_ligand_info",
    "create_sulfate_from_coordinates",
    "detect_bonds",
    "determine_bond_order_based_on_distance",
    "embed_and_optimize",
    "generate_clean_smiles",
    "get_covalent_radii",
    "is_connected_molecule",
    "validate_molecule",
]

"""Distance-based bond detection for coordinate-built RDKit molecules."""

from dataclasses import dataclass
from typing import Dict, List

from decologr import Decologr as log
from molib.ligand.element import (
    ELEMENTS,
    Element,
    calculate_distance,
    get_covalent_radii,
)

COVALENT_RADII = {
    symbol: element.covalent_radius for symbol, element in ELEMENTS.items()
}

try:
    from rdkit import Chem
    from rdkit.Chem import Mol

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    Chem = None
    Mol = None  # type: ignore[misc, assignment]


@dataclass(frozen=True, slots=True)
class BondSpec:
    """Candidate bond between two atoms."""

    atom1: int
    atom2: int

    element1: Element
    element2: Element

    distance: float
    order: int


def add_bond_from_spec(
    mol: Mol, atom_indices: dict[int, int], bond: BondSpec, bonds_added: int
) -> int:
    bonds_added = add_bond(
        mol=mol,
        bonds_added=bonds_added,
        atom_indices=atom_indices,
        bond_order=bond.order,
        distance=bond.distance,
        elem1=bond.element1.symbol,
        elem2=bond.element2.symbol,
        i=bond.atom1,
        j=bond.atom2,
    )


def add_bond(
    atom_indices: dict[int, int],
    bond_order: int,
    bonds_added: int,
    distance: float,
    elem1: str,
    elem2: str,
    i: int,
    j: int,
    mol: Mol,
) -> int:
    """Add bond"""
    try:
        mol.AddBond(atom_indices[i], atom_indices[j], Chem.BondType(bond_order))
        bonds_added += 1
        log.debug(
            f"  🔗 Bond {bonds_added}: {elem1}-{elem2} (distance: {distance:.2f}Å, order: {bond_order})"
        )
    except Exception as e:
        log.warning(f"  ⚠️ Failed to add bond {elem1}-{elem2}: {e}")
    return bonds_added


def determine_bond_order_based_on_distance(
    distance: float, elem1: str, elem2: str, radius1: float, radius2: float
):
    """Determine bond order based on distance"""
    bond_order = 1
    if distance <= (radius1 + radius2) * 0.9:  # Very close = double/triple bond
        if elem1 == "C" and elem2 == "C":
            bond_order = 2  # Assume double bond for C-C
        elif elem1 in ["C", "N", "O"] and elem2 in ["C", "N", "O"]:
            bond_order = 2
    return bond_order


def add_conformer(atom_indices: dict[int, int], coordinates: list, mol: Mol):
    """Add conformer with 3D coordinates"""
    conf = Chem.Conformer(len(atom_indices))
    for i, coord in enumerate(coordinates):
        conf.SetAtomPosition(i, (coord[0], coord[1], coord[2]))
    mol.AddConformer(conf)


def detect_bonds(
    mol: "Chem.RWMol",
    coordinates: List[tuple],
    element_symbols: List[str],
    atom_indices: Dict[int, int],
) -> int:
    """Detect bonds between atoms based on distance and chemical rules"""
    bonds_added = 0

    # --- Check all pairs of atoms
    for i in range(len(coordinates)):
        for j in range(i + 1, len(coordinates)):
            distance = calculate_distance(coordinates, i, j)

            elem1, elem2, radius1, radius2 = get_covalent_radii(
                COVALENT_RADII, element_symbols, i, j
            )

            element1 = ELEMENTS[element_symbols[i]]
            element2 = ELEMENTS[element_symbols[j]]

            # Bond if distance is less than sum of covalent radii + tolerance
            # Use different tolerances for different element pairs
            if "H" in (element1.symbol, element2.symbol):
                tolerance = 1.2
            else:
                tolerance = 1.3

            max_bond_distance = (
                element1.covalent_radius + element2.covalent_radius
            ) * tolerance

            if distance <= max_bond_distance:
                bond_order = determine_bond_order_based_on_distance(
                    distance, elem1, elem2, radius1, radius2
                )

                bonds_added = add_bond(
                    atom_indices,
                    bond_order,
                    bonds_added,
                    distance,
                    elem1,
                    elem2,
                    i,
                    j,
                    mol,
                )

    return bonds_added

"""
Defines utilities for handling chemical elements, bond detection, and molecule
manipulations using RDKit. Includes classes and functions for element
properties, bond specifications, and various chemical computations.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence

from decologr import Decologr as log
from molib.ligand.pdb.info import PDBLigandInfo

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, Mol, rdMolDescriptors

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    Chem = None
    Descriptors = None
    rdMolDescriptors = None
    AllChem = None


@dataclass(frozen=True, slots=True)
class Element:
    """
    Represents a chemical element with its symbol and covalent radius.

    This class provides a model for storing basic information about a chemical
    element. It includes a symbol representing the element and its covalent
    radius, which is a key property used in various chemical computations.
    """

    symbol: str
    covalent_radius: float

    @property
    def is_hydrogen(self) -> bool:
        return self.symbol == "H"


ELEMENTS: dict[str, Element] = {
    "H": Element("H", 0.31),
    "C": Element("C", 0.76),
    "N": Element("N", 0.71),
    "O": Element("O", 0.66),
    "F": Element("F", 0.57),
    "P": Element("P", 1.07),
    "S": Element("S", 1.05),
    "Cl": Element("Cl", 0.99),
    "Br": Element("Br", 1.20),
    "I": Element("I", 1.39),
}


def get_covalent_radii(covalent_radii, element_symbols, i, j):
    """Get covalent radii"""
    elem1 = element_symbols[i]
    elem2 = element_symbols[j]
    radius1 = covalent_radii.get(elem1, 1.0)
    radius2 = covalent_radii.get(elem2, 1.0)
    return elem1, elem2, radius1, radius2


def calculate_distance(coordinates, i, j):
    """Calculate distance between atoms"""
    coord1 = coordinates[i]
    coord2 = coordinates[j]
    distance = (
        (coord1[0] - coord2[0]) ** 2
        + (coord1[1] - coord2[1]) ** 2
        + (coord1[2] - coord2[2]) ** 2
    ) ** 0.5
    return distance


def create_mol_with_conformer(coordinates, ligand_id, smiles):
    """Create molecule from SMILES"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        # --- Add conformer with actual coordinates if we have them
        if coordinates and len(coordinates) >= mol.GetNumAtoms():
            conf = Chem.Conformer(mol.GetNumAtoms())
            for i, coord in enumerate(coordinates[: mol.GetNumAtoms()]):
                conf.SetAtomPosition(i, (coord[0], coord[1], coord[2]))
            mol.AddConformer(conf)
            # log.info(f"✅ PDBLigandParser: Added conformer for {ligand_id}")

        return mol
    else:
        log.warning(
            f"⚠️ PDBLigandParser: Failed to create molecule from SMILES: {smiles}"
        )
        return None

def create_molecule_alternative(
    coordinates: List[tuple], element_symbols: List[str], atom_names: List[str]
) -> Optional["Chem.Mol"]:
    """Alternative molecule creation using RDKit's built-in methods"""
    try:
        print("🔄 PDBLigandParser: Trying alternative molecule creation...")
        log.info("🔄 PDBLigandParser: Trying alternative molecule creation...")

        # Try to create molecule from SMILES if we can guess the structure
        if len(element_symbols) == 1 and element_symbols[0] == "O":
            # Water molecule
            mol = Chem.MolFromSmiles("O")
            if mol:
                print("✅ PDBLigandParser: Created water molecule from SMILES")
                log.info("✅ PDBLigandParser: Created water molecule from SMILES")
                return mol
        elif (
            len(element_symbols) == 5
            and element_symbols.count("C") == 1
            and element_symbols.count("H") == 4
        ):
            # Methane molecule
            mol = Chem.MolFromSmiles("C")
            if mol:
                print("✅ PDBLigandParser: Created methane molecule from SMILES")
                log.info("✅ PDBLigandParser: Created methane molecule from SMILES")
                return mol

        # ---For other cases, try to create a simple structure
        # --- This is a fallback for when bond detection fails
        log.warning("⚠️ PDBLigandParser: Using fallback molecule creation")

        # --- Create a simple chain structure
        mol = Chem.RWMol()
        atom_indices = {}

        for i, element in enumerate(element_symbols):
            atomic_num = Chem.GetPeriodicTable().GetAtomicNumber(element)
            atom = Chem.Atom(atomic_num)
            atom_idx = mol.AddAtom(atom)
            atom_indices[i] = atom_idx

            # --- Connect to previous atom if possible
            if i > 0:
                try:
                    mol.AddBond(
                        atom_indices[i - 1], atom_indices[i], Chem.BondType.SINGLE
                    )
                except:
                    pass

        mol = mol.GetMol()

        # --- Add conformer
        if mol.GetNumAtoms() > 0:
            conf = Chem.Conformer(mol.GetNumAtoms())
            for i, coord in enumerate(coordinates[: mol.GetNumAtoms()]):
                conf.SetAtomPosition(i, (coord[0], coord[1], coord[2]))
            mol.AddConformer(conf)

        log.info(
            f"✅ PDBLigandParser: Alternative molecule created ({mol.GetNumAtoms()} atoms, {mol.GetNumBonds()} bonds)"
        )
        return mol

    except Exception as e:
        log.warning(f"❌ PDBLigandParser: Alternative molecule creation failed: {e}")
        return None


def is_connected_molecule(mol: "Chem.Mol") -> bool:
    """Check if all atoms in the molecule are connected"""
    try:
        # --- Get all atoms
        atoms = [mol.GetAtomWithIdx(i) for i in range(mol.GetNumAtoms())]
        if not atoms:
            return False

        # --- Start BFS from first atom
        visited = set()
        queue = [0]  # Start with first atom
        visited.add(0)

        while queue:
            atom_idx = queue.pop(0)
            atom = mol.GetAtomWithIdx(atom_idx)

            # --- Visit all neighbours
            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                if neighbor_idx not in visited:
                    visited.add(neighbor_idx)
                    queue.append(neighbor_idx)

        # --- Check if all atoms were visited
        return len(visited) == mol.GetNumAtoms()

    except Exception as e:
        print(f"❌ PDBLigandParser: Error checking connectivity: {e}")
        log.warning(f"❌ PDBLigandParser: Error checking connectivity: {e}")
        return False


def add_hydrogen_and_optimize_geometry(mol):
    """Try to add hydrogens and optimize meshdata"""
    try:
        log.info("🔄 PDBLigandParser: Adding hydrogens and optimizing meshdata...")

        # --- First, sanitize the molecule to fix valence issues
        try:
            Chem.SanitizeMol(mol)
            log.info("✅ PDBLigandParser: Molecule sanitization successful")
        except Exception as e:
            log.warning(f"⚠️ PDBLigandParser: Molecule sanitization failed: {e}")

        # --- Add hydrogens
        mol_with_h = Chem.AddHs(mol)

        mol = embed_and_optimize(mol_with_h)

    except Exception as e:
        log.warning(f"⚠️ PDBLigandParser: Hydrogen addition failed: {e}")
        # --- If hydrogen addition fails, use the basic molecule
        pass
    return mol


def validate_molecule(mol: "Chem.Mol") -> bool:
    """Validate that the molecule is chemically reasonable"""
    try:
        if mol is None:
            return False

        # --- Check if molecule has any bonds
        if mol.GetNumBonds() == 0:
            log.warning("❌ PDBLigandParser: Molecule has no bonds")
            return False

        # --- Check if molecule is connected (no disconnected fragments)
        if not is_connected_molecule(mol):
            log.warning("❌ PDBLigandParser: Molecule has disconnected fragments")
            return False

        # Check for reasonable atom counts
        num_atoms = mol.GetNumAtoms()
        if num_atoms > 1000:  # Unreasonably large
            log.warning(f"❌ PDBLigandParser: Molecule too large ({num_atoms} atoms)")
            return False

        log.info(
            f"✅ PDBLigandParser: Molecule validation passed ({num_atoms} atoms, {mol.GetNumBonds()} bonds)"
        )
        return True

    except Exception as e:
        print(f"❌ PDBLigandParser: Molecule validation error: {e}")
        log.warning(f"❌ PDBLigandParser: Molecule validation error: {e}")
        return False


def create_molecule_from_coordinates(
    coordinates: List[tuple], element_symbols: List[str], atom_names: List[str]
) -> Optional["Chem.Mol"]:
    """Create RDKit molecule from 3D coordinates with proper bond detection"""
    try:
        if not RDKIT_AVAILABLE:
            return None

        log.info(f"🔄 PDBLigandParser: Creating molecule from {len(coordinates)} atoms")

        # --- Create molecule
        mol = Chem.RWMol()

        # --- Add atoms
        atom_indices = {}
        for i, (coord, element) in enumerate(zip(coordinates, element_symbols)):
            # --- Get atomic number from element symbol
            atomic_num = Chem.GetPeriodicTable().GetAtomicNumber(element)
            atom = Chem.Atom(atomic_num)
            atom_idx = mol.AddAtom(atom)
            atom_indices[i] = atom_idx
            log.debug(f"  📍 Atom {i}: {element} at {coord}")

        # --- Detect bonds based on distance and chemical rules
        bonds_added = detect_bonds(mol, coordinates, element_symbols, atom_indices)
        log.info(f"🔗 PDBLigandParser: Detected {bonds_added} bonds")

        add_conformer(atom_indices, coordinates, mol)

        # --- Convert to molecule
        mol = mol.GetMol()

        # --- Validate the molecule
        if not validate_molecule(mol):
            log.warning(
                "⚠️ PDBLigandParser: Molecule validation failed, trying alternative approach"
            )
            return create_molecule_alternative(coordinates, element_symbols, atom_names)

        return add_hydrogen_and_optimize_geometry(mol)

    except Exception as e:
        log.warning(f"Error creating molecule from coordinates: {e}")
        return None

def check_rdkit_availability() -> bool:
    """Check if RDKit is available and working"""
    return RDKIT_AVAILABLE

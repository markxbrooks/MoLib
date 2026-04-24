from typing import Dict, List, Optional

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
):
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


def embed_and_optimize(mol_with_h: Mol):
    """Try to embed and optimize"""
    try:
        AllChem.EmbedMolecule(mol_with_h)
        AllChem.MMFFOptimizeMolecule(mol_with_h)
        mol = Chem.RemoveHs(mol_with_h)
        log.info("✅ PDBLigandParser: Geometry optimization successful")
    except Exception as e:
        log.warning(
            f"⚠️ PDBLigandParser: Geometry optimization failed, using basic molecule: {e}"
        )
        # Use the molecule without optimization
        mol = Chem.RemoveHs(mol_with_h)
    return mol


def detect_bonds(
    mol: "Chem.RWMol",
    coordinates: List[tuple],
    element_symbols: List[str],
    atom_indices: Dict[int, int],
) -> int:
    """Detect bonds between atoms based on distance and chemical rules"""
    bonds_added = 0

    # --- Define covalent radii for common elements (in Angstroms)
    covalent_radii = {
        "H": 0.31,
        "C": 0.76,
        "N": 0.71,
        "O": 0.66,
        "F": 0.57,
        "P": 1.07,
        "S": 1.05,
        "Cl": 0.99,
        "Br": 1.20,
        "I": 1.39,
    }

    # --- Check all pairs of atoms
    for i in range(len(coordinates)):
        for j in range(i + 1, len(coordinates)):
            distance = calculate_distance(coordinates, i, j)

            elem1, elem2, radius1, radius2 = get_covalent_radii(
                covalent_radii, element_symbols, i, j
            )

            # Bond if distance is less than sum of covalent radii + tolerance
            # Use different tolerances for different element pairs
            if elem1 == "H" or elem2 == "H":
                max_bond_distance = (
                    radius1 + radius2
                ) * 1.2  # 20% tolerance for H bonds
            else:
                max_bond_distance = (
                    radius1 + radius2
                ) * 1.3  # 30% tolerance for other bonds

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


def create_sulfate_from_coordinates(
    coordinates: List[tuple], element_symbols: List[str], atom_names: List[str]
) -> Optional["Chem.Mol"]:
    """Create sulfate ion from coordinates with proper tetrahedral meshdata"""
    try:
        log.info(f"🔄 PDBLigandParser: Creating sulfate from {len(coordinates)} atoms")

        # --- Create sulfate ion from SMILES
        mol = Chem.MolFromSmiles("[O-]S(=O)(=O)[O-]")
        if mol is None:
            log.error("❌ PDBLigandParser: Failed to create sulfate from SMILES")
            return None

        # --- Find the sulfur atom in coordinates
        sulfur_idx = None
        for i, element in enumerate(element_symbols):
            if element == "S":
                sulfur_idx = i
                break

        if sulfur_idx is None:
            print("❌ PDBLigandParser: No sulfur atom found in coordinates")
            log.error("❌ PDBLigandParser: No sulfur atom found in coordinates")
            return None

        # Create conformer with actual coordinates
        conf = Chem.Conformer(mol.GetNumAtoms())

        # --- Place sulfur at the sulfur coordinate
        sulfur_coord = coordinates[sulfur_idx]
        conf.SetAtomPosition(0, (sulfur_coord[0], sulfur_coord[1], sulfur_coord[2]))

        # --- Place oxygen atoms at the oxygen coordinates
        oxygen_coords = [
            coord for i, coord in enumerate(coordinates) if element_symbols[i] == "O"
        ]
        for i, oxy_coord in enumerate(oxygen_coords[:4]):  # Take up to 4 oxygen atoms
            conf.SetAtomPosition(i + 1, (oxy_coord[0], oxy_coord[1], oxy_coord[2]))

        mol.AddConformer(conf)

        log.info(f"✅ PDBLigandParser: Created sulfate with {mol.GetNumAtoms()} atoms")
        return mol

    except Exception as e:
        log.warning(f"❌ PDBLigandParser: Error creating sulfate from coordinates: {e}")
        return None


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
            log.info(f"✅ PDBLigandParser: Added conformer for {ligand_id}")

        return mol
    else:
        log.warning(
            f"⚠️ PDBLigandParser: Failed to create molecule from SMILES: {smiles}"
        )
        return None


def create_common_ligand_molecule(
    ligand_id: str,
    coordinates: List[tuple],
    element_symbols: List[str],
    atom_names: List[str],
) -> Optional["Chem.Mol"]:
    """Create molecules for common biological ligand using known structures"""
    try:
        log.info(f"🔄 PDBLigandParser: Checking for common ligand: {ligand_id}")

        # --- Define common biological ligand and their correct SMILES
        common_ligands = {
            "HOH": "O",  # Water
            "SO4": "[O-]S(=O)(=O)[O-]",  # Sulfate ion
            "PO4": "[O-]P(=O)([O-])[O-]",  # Phosphate ion
            "CL": "Cl",  # Chloride ion
            "NA": "Na",  # Sodium ion
            "MG": "Mg",  # Magnesium ion
            "CA": "Ca",  # Calcium ion
            "ZN": "Zn",  # Zinc ion
            "FE": "Fe",  # Iron ion
            "MN": "Mn",  # Manganese ion
            "CU": "Cu",  # Copper ion
            "NI": "Ni",  # Nickel ion
            "CO": "Co",  # Cobalt ion
            "EDO": "CCO",  # Ethylene glycol
            "GOL": "C(CO)O",  # Glycerol
            "ACT": "CC(=O)O",  # Acetate
            "ZN2": "Zn",  # Zinc ion (alternative name)
            "CA2": "Ca",  # Calcium ion (alternative name)
            "MG2": "Mg",  # Magnesium ion (alternative name)
        }

        # --- Check if this is a known common ligand
        if ligand_id in common_ligands:
            smiles = common_ligands[ligand_id]
            log.info(f"✅ PDBLigandParser: Found common ligand {ligand_id} -> {smiles}")

            return create_mol_with_conformer(coordinates, ligand_id, smiles)

        # --- Special handling for ligand that need coordinate-based creation
        if (
            ligand_id == "SO4"
            and len(element_symbols) == 5
            and element_symbols.count("S") == 1
            and element_symbols.count("O") == 4
        ):
            log.info(f"🔄 PDBLigandParser: Creating SO4 from coordinates")
            return create_sulfate_from_coordinates(
                coordinates, element_symbols, atom_names
            )

        elif (
            ligand_id == "HOH"
            and len(element_symbols) == 1
            and element_symbols[0] == "O"
        ):
            log.info(f"🔄 PDBLigandParser: Creating H2O from single O atom")
            return Chem.MolFromSmiles("O")

        log.info(
            f"ℹ️ PDBLigandParser: {ligand_id} not in common ligand, will use coordinate-based creation"
        )
        return None

    except Exception as e:
        log.warning(
            f"❌ PDBLigandParser: Error creating common ligand {ligand_id}: {e}"
        )
        return None


def generate_clean_smiles(mol: "Chem.Mol") -> str:
    """Generate a clean, chemically accurate SMILES string"""
    try:
        if mol is None:
            return ""

        # Generate canonical SMILES
        smiles = Chem.MolToSmiles(mol, canonical=True)

        # Check if SMILES contains disconnected components (dots)
        if "." in smiles:
            log.warning(
                f"⚠️ PDBLigandParser: SMILES contains disconnected components: {smiles}"
            )

            # Try to get the largest connected component
            try:
                # Split by dots and get the largest component
                components = smiles.split(".")
                largest_component = max(components, key=len)
                log.info(
                    f"🔄 PDBLigandParser: Using largest component: {largest_component}"
                )
                return largest_component
            except:
                # If that fails, return the original
                return smiles

        # Validate SMILES by parsing it back
        try:
            test_mol = Chem.MolFromSmiles(smiles)
            if test_mol is None:
                log.warning(
                    f"⚠️ PDBLigandParser: Generated SMILES is invalid: {smiles}"
                )
                return ""

            log.info(f"✅ PDBLigandParser: Valid SMILES generated: {smiles}")
            return smiles

        except Exception as e:
            log.warning(f"⚠️ PDBLigandParser: Error validating SMILES: {e}")
            return smiles

    except Exception as e:
        print(f"❌ PDBLigandParser: Error generating SMILES: {e}")
        log.warning(f"❌ PDBLigandParser: Error generating SMILES: {e}")
        return ""


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


def create_pdb_ligand_info(ligand_data: "PDBLigandData") -> Optional["PDBLigandInfo"]:
    """Create PDBLigandInfo from grouped ligand data"""
    try:
        res_name = ligand_data.res_name
        chain_id = ligand_data.chain_id
        res_seq = ligand_data.res_seq
        insertion_code = ligand_data.insertion_code
        atoms = ligand_data.atoms

        # --- Extract coordinates and atom informationMS first?
        coordinates = [atom["coordinates"] for atom in atoms]

        atom_names = [atom["atom_name"] for atom in atoms]
        element_symbols = [atom["element"] for atom in atoms]

        # --- Create ligand identifier
        ligand_id = res_name
        ligand_name = f"{res_name} (Chain {chain_id}, Res {res_seq})"

        # --- Calculate basic properties
        atom_count = len(atoms)
        heavy_atoms = len([elem for elem in element_symbols if elem != "H"])

        # --- Try to create 3D molecule and convert to SMILES
        smiles = ""
        molecular_weight = 0.0
        formula = ""
        logp = 0.0
        hbd = 0
        hba = 0
        tpsa = 0.0
        rotatable_bonds = 0
        aromatic_rings = 0

        # --- Normalize coordinates to floats (defensive)
        try:
            coordinates = [(float(x), float(y), float(z)) for (x, y, z) in coordinates]
        except Exception as e:
            log.error(
                f"Invalid coordinates for ligand {ligand_id}: {coordinates[:3]}... ({e})"
            )
            return None

        assert all(
            isinstance(c, float) for coord in coordinates for c in coord
        ), "Non-float coordinates detected"

        if RDKIT_AVAILABLE:
            try:
                # --- Check if this is a common biological ligand that needs special handling
                mol = create_common_ligand_molecule(
                    ligand_id, coordinates, element_symbols, atom_names
                )

                if mol is None:
                    # --- Fall back to coordinate-based molecule creation
                    mol = create_molecule_from_coordinates(
                        coordinates, element_symbols, atom_names
                    )

                if not mol:
                    return None

                # --- Calculate molecular properties
                smiles = generate_clean_smiles(mol)
                molecular_weight = Descriptors.MolWt(mol)
                formula = rdMolDescriptors.CalcMolFormula(mol)
                logp = Descriptors.MolLogP(mol)
                hbd = Descriptors.NumHDonors(mol)
                hba = Descriptors.NumHAcceptors(mol)
                tpsa = Descriptors.TPSA(mol)
                rotatable_bonds = Descriptors.NumRotatableBonds(mol)
                aromatic_rings = Descriptors.NumAromaticRings(mol)

                log.info(f"✅ PDBLigandParser: Generated SMILES: {smiles}")

            except Exception as e:
                log.warning(
                    f"Error calculating molecular properties for {ligand_id}: {e}"
                )

        return PDBLigandInfo(
            ligand_id=ligand_id,
            ligand_name=ligand_name,
            chain_id=chain_id,
            res_seq=res_seq,
            insertion_code=insertion_code,
            atom_count=atom_count,
            coordinates=coordinates,
            atom_names=atom_names,
            element_symbols=element_symbols,
            smiles=smiles,
            molecular_weight=molecular_weight,
            formula=formula,
            logp=logp,
            hbd=hbd,
            hba=hba,
            tpsa=tpsa,
            rotatable_bonds=rotatable_bonds,
            aromatic_rings=aromatic_rings,
            heavy_atoms=heavy_atoms,
        )

    except Exception as e:
        log.error(f"Error creating PDB ligand info: {e}")
        return None


def check_rdkit_availability() -> bool:
    """Check if RDKit is available and working"""
    return RDKIT_AVAILABLE

"""
Symmetry mates generation for crystallographic structures.

This module provides functionality to generate contacting symmetry mates
using gemmi's crystallographic operations.
"""

from typing import Any, Dict, List, Optional, Tuple

import gemmi
import numpy as np
from decologr import Decologr as log


class SymmetryMatesGenerator:
    """
    Generates contacting symmetry mates for crystallographic structures.

    This class uses gemmi to apply space group symmetry operations
    and generate symmetry-related molecules that contact the original
    asymmetric unit.
    """

    def __init__(self):
        """Initialize the symmetry mates generator."""
        self.contact_distance = (
            8.0  # Å, distance for considering molecules "contacting"
        )
        self.max_mates = 50  # Maximum number of symmetry mates to generate

    def generate_contacting_symmetry_mates(
        self,
        structure: gemmi.Structure,
        contact_distance: Optional[float] = None,
        max_mates: Optional[int] = None,
    ) -> List[gemmi.Model]:
        """
        Generate symmetry mates that contact the original asymmetric unit.

        Args:
            structure: gemmi.Structure object
            contact_distance: Distance in Å to consider molecules "contacting" (default: 8.0)
            max_mates: Maximum number of symmetry mates to generate (default: 50)

        Returns:
            List of gemmi.Model objects representing symmetry mates
        """
        if contact_distance is not None:
            self.contact_distance = contact_distance
        if max_mates is not None:
            self.max_mates = max_mates

        try:
            # Get the original model (asymmetric unit)
            original_model = structure[0]
            if not original_model:
                log.error("No model found in structure")
                return []

            # Get crystallographic information
            cell = structure.cell
            space_group = structure.spacegroup_hm

            log.info(f"Generating symmetry mates for space group: {space_group}")
            log.info(f"📐 Unit cell: a={cell.a:.2f}, b={cell.b:.2f}, c={cell.c:.2f} Å")
            log.info(f"📐 Contact distance: {self.contact_distance:.1f} Å")

            # Find the space group and get symmetry operations
            sg = gemmi.find_spacegroup_by_name(space_group)
            if not sg:
                log.error(f"Could not find space group: {space_group}")
                return []

            sym_ops = sg.operations()
            log.info(f"Found {len(sym_ops)} symmetry operations")

            # Generate symmetry mates
            contacting_mates = self._generate_contacting_mates(
                original_model, cell, sym_ops
            )

            log.info(f"✅ Generated {len(contacting_mates)} contacting symmetry mates")
            return contacting_mates

        except Exception as e:
            log.error(f"Error generating symmetry mates: {e}")
            return []

    def _generate_contacting_mates(
        self, original_model: gemmi.Model, cell: gemmi.UnitCell, sym_ops: gemmi.GroupOps
    ) -> List[gemmi.Model]:
        """
        Generate symmetry mates that contact the original model.

        Args:
            original_model: Original asymmetric unit model
            cell: Unit cell parameters
            sym_ops: Symmetry operations

        Returns:
            List of contacting symmetry mate models
        """
        contacting_mates = []

        # Get bounding box of original model
        original_bounds = self._get_model_bounds(original_model)
        log.info(f"📦 Original model bounds: {original_bounds}")

        # Generate symmetry mates for each operation
        # Note: GroupOps is not subscriptable, we need to iterate
        mate_id = 1
        for i, op in enumerate(sym_ops):
            if len(contacting_mates) >= self.max_mates:
                break

            # Skip identity operation (first operation is usually identity)
            if i == 0:
                continue

            # Apply symmetry operation to create mate
            mate_model = self._apply_symmetry_operation(
                original_model, cell, op, mate_id
            )

            # Check if this mate contacts the original
            if self._models_contact(original_model, mate_model, cell):
                contacting_mates.append(mate_model)
                log.info(f"✅ Mate {mate_id} contacts original model")
            else:
                log.info(
                    f"❌ Mate {mate_id} does not contact original model (distance > {self.contact_distance} Å)"
                )
                # Clean up non-contacting mate
                del mate_model

            mate_id += 1

        return contacting_mates

    def _apply_symmetry_operation(
        self, model: gemmi.Model, cell: gemmi.UnitCell, op: gemmi.Op, mate_id: int
    ) -> gemmi.Model:
        """
        Apply a symmetry operation to create a symmetry mate.

        Args:
            model: Original model
            cell: Unit cell parameters
            op: Symmetry operation
            mate_id: Unique identifier for this mate

        Returns:
            New model with applied symmetry operation
        """
        # Create new model for the symmetry mate
        mate_model = gemmi.Model(mate_id)
        # Note: Model doesn't have a name attribute in this gemmi version

        # Apply symmetry operation to each chain
        for chain in model:
            new_chain = gemmi.Chain(chain.name)

            for residue in chain:
                new_res = gemmi.Residue()
                new_res.name = residue.name
                new_res.seqid = residue.seqid
                new_res.subchain = residue.subchain
                new_res.flag = residue.flag

                # Apply symmetry operation to each atom
                for atom in residue:
                    new_atom = gemmi.Atom()
                    new_atom.name = atom.name
                    new_atom.element = atom.element
                    new_atom.altloc = atom.altloc
                    new_atom.charge = atom.charge
                    new_atom.occ = atom.occ
                    new_atom.b_iso = atom.b_iso
                    new_atom.aniso = atom.aniso

                    # Apply symmetry operation to atom position
                    # Convert to fractional coordinates, apply operation, convert back
                    frac_pos = cell.fractionalize(atom.pos)
                    frac_sym = op.apply_to_xyz([frac_pos.x, frac_pos.y, frac_pos.z])

                    # Ensure fractional coordinates are within 0-1 range (periodic boundary conditions)
                    frac_sym = [coord % 1.0 for coord in frac_sym]

                    # Debug: log the transformation
                    log.info(
                        f"Atom {atom.name} in {residue.name}: frac_pos={frac_pos.x:.3f},{frac_pos.y:.3f},{frac_pos.z:.3f} -> frac_sym={frac_sym[0]:.3f},{frac_sym[1]:.3f},{frac_sym[2]:.3f}"
                    )

                    new_atom.pos = cell.orthogonalize(gemmi.Fractional(*frac_sym))

                    new_res.add_atom(new_atom)

                new_chain.add_residue(new_res)
            mate_model.add_chain(new_chain)

        return mate_model

    def _get_model_bounds(self, model: gemmi.Model) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the bounding box of a model.

        Args:
            model: gemmi.Model object

        Returns:
            Tuple of (min_coords, max_coords) as numpy arrays
        """
        if not model:
            return np.array([0, 0, 0]), np.array([0, 0, 0])

        min_coords = np.array([float("inf")] * 3)
        max_coords = np.array([float("-inf")] * 3)

        for chain in model:
            for residue in chain:
                for atom in residue:
                    pos = atom.pos
                    coords = np.array([pos.x, pos.y, pos.z])
                    min_coords = np.minimum(min_coords, coords)
                    max_coords = np.maximum(max_coords, coords)

        return min_coords, max_coords

    def _models_contact(
        self, model1: gemmi.Model, model2: gemmi.Model, cell: gemmi.UnitCell
    ) -> bool:
        """
        Check if two models contact each other within the contact distance.

        Args:
            model1: First model
            model2: Second model
            cell: Unit cell parameters (for periodic boundary conditions)

        Returns:
            True if models contact, False otherwise
        """
        # Get bounds of both models
        bounds1 = self._get_model_bounds(model1)
        bounds2 = self._get_model_bounds(model2)

        # Check for overlap in bounding boxes
        min1, max1 = bounds1
        min2, max2 = bounds2

        log.info(f"📦 Model 1 bounds: {min1} to {max1}")
        log.info(f"📦 Model 2 bounds: {min2} to {max2}")

        # Apply periodic boundary conditions
        for i in range(3):
            # Check if models overlap in this dimension
            if (
                max1[i] < min2[i] - self.contact_distance
                or min1[i] > max2[i] + self.contact_distance
            ):
                log.info(
                    f"❌ Models don't overlap in dimension {i}: {max1[i]} < {min2[i] - self.contact_distance} or {min1[i]} > {max2[i] + self.contact_distance}"
                )
                return False

        log.info("✅ Bounding boxes overlap, checking atom contacts...")
        # If bounding boxes overlap, check individual atoms
        return self._check_atom_contacts(model1, model2, cell)

    def _check_atom_contacts(
        self, model1: gemmi.Model, model2: gemmi.Model, cell: gemmi.UnitCell
    ) -> bool:
        """
        Check if any atoms in two models are within contact distance.

        Args:
            model1: First model
            model2: Second model
            cell: Unit cell parameters

        Returns:
            True if any atoms contact, False otherwise
        """
        contact_distance_sq = self.contact_distance**2

        # Sample atoms from both models (don't check every single atom for performance)
        atoms1 = list(self._sample_atoms(model1, max_atoms=100))
        atoms2 = list(self._sample_atoms(model2, max_atoms=100))

        for atom1 in atoms1:
            pos1 = np.array([atom1.pos.x, atom1.pos.y, atom1.pos.z])

            for atom2 in atoms2:
                pos2 = np.array([atom2.pos.x, atom2.pos.y, atom2.pos.z])

                # Calculate distance (considering periodic boundary conditions)
                distance_sq = self._periodic_distance_sq(pos1, pos2, cell)

                if distance_sq <= contact_distance_sq:
                    return True

        return False

    def _sample_atoms(
        self, model: gemmi.Model, max_atoms: int = 100
    ) -> List[gemmi.Atom]:
        """
        Sample atoms from a model for contact checking.

        Args:
            model: gemmi.Model object
            max_atoms: Maximum number of atoms to sample

        Returns:
            List of sampled atoms
        """
        atoms = []
        total_atoms = sum(len(residue) for chain in model for residue in chain)

        if total_atoms <= max_atoms:
            # Return all atoms if we have fewer than max_atoms
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        atoms.append(atom)
        else:
            # Sample atoms evenly across the model
            step = total_atoms // max_atoms
            count = 0
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        if count % step == 0:
                            atoms.append(atom)
                        count += 1
                        if len(atoms) >= max_atoms:
                            break
                    if len(atoms) >= max_atoms:
                        break
                if len(atoms) >= max_atoms:
                    break

        return atoms

    def _periodic_distance_sq(
        self, pos1: np.ndarray, pos2: np.ndarray, cell: gemmi.UnitCell
    ) -> float:
        """
        Calculate squared distance between two positions considering periodic boundary conditions.

        Args:
            pos1: First position
            pos2: Second position
            cell: Unit cell parameters

        Returns:
            Squared distance in Å²
        """
        # Convert to fractional coordinates
        frac1 = cell.fractionalize(gemmi.Position(pos1[0], pos1[1], pos1[2]))
        frac2 = cell.fractionalize(gemmi.Position(pos2[0], pos2[1], pos2[2]))

        # Calculate fractional distance
        frac_diff = np.array([frac1.x - frac2.x, frac1.y - frac2.y, frac1.z - frac2.z])

        # Apply periodic boundary conditions (fractional coordinates are 0-1)
        frac_diff = frac_diff - np.round(frac_diff)

        # Convert back to orthogonal coordinates
        diff_orth = cell.orthogonalize(gemmi.Fractional(*frac_diff))
        diff_array = np.array([diff_orth.x, diff_orth.y, diff_orth.z])

        return np.sum(diff_array**2)

    def get_symmetry_info(self, structure: gemmi.Structure) -> Dict[str, Any]:
        """
        Get information about the symmetry of a structure.

        Args:
            structure: gemmi.Structure object

        Returns:
            Dictionary containing symmetry information
        """
        try:
            cell = structure.cell
            space_group = structure.spacegroup_hm

            sg = gemmi.find_spacegroup_by_name(space_group)
            if not sg:
                return {"error": f"Could not find space group: {space_group}"}

            sym_ops = sg.operations()

            info = {
                "space_group": space_group,
                "space_group_number": getattr(sg, "number", "Unknown"),
                "symmetry_operations": len(sym_ops),
                "unit_cell": {
                    "a": cell.a,
                    "b": cell.b,
                    "c": cell.c,
                    "alpha": cell.alpha,
                    "beta": cell.beta,
                    "gamma": cell.gamma,
                },
                "volume": cell.volume,
                "z_value": getattr(sg, "z", "Unknown"),
                "operations": [],
            }

            # Add information about each symmetry operation
            for i, op in enumerate(sym_ops):
                # Note: We can't access matrix/translation directly in this gemmi version
                op_info = {
                    "index": i,
                    "is_identity": i == 0,  # First operation is usually identity
                    "description": str(op),
                }
                info["operations"].append(op_info)

            return info

        except Exception as e:
            log.error(f"Error getting symmetry info: {e}")
            return {"error": str(e)}


def generate_symmetry_mates_from_pdb(
    pdb_path: str, contact_distance: float = 8.0, max_mates: int = 50
) -> Tuple[List[gemmi.Model], Dict[str, Any]]:
    """
    Generate symmetry mates from a PDB file.

    Args:
        pdb_path: Path to PDB file
        contact_distance: Distance in Å to consider molecules "contacting"
        max_mates: Maximum number of symmetry mates to generate

    Returns:
        Tuple of (symmetry_mates, symmetry_info)
    """
    try:
        log.info(f"Loading PDB structure: {pdb_path}")

        # Load structure with gemmi
        structure = gemmi.read_structure(pdb_path)

        # Create symmetry generator
        generator = SymmetryMatesGenerator()

        # Get symmetry information
        symmetry_info = generator.get_symmetry_info(structure)

        # Generate contacting symmetry mates
        symmetry_mates = generator.generate_contacting_symmetry_mates(
            structure, contact_distance, max_mates
        )

        log.info(f"✅ Generated {len(symmetry_mates)} symmetry mates")

        return symmetry_mates, symmetry_info

    except Exception as e:
        log.error(f"Error generating symmetry mates from PDB: {e}")
        return [], {"error": str(e)}


def generate_symmetry_mates_from_mtz(
    mtz_path: str, pdb_path: str, contact_distance: float = 8.0, max_mates: int = 50
) -> Tuple[List[gemmi.Model], Dict[str, Any]]:
    """
    Generate symmetry mates using crystallographic data from MTZ file.

    Args:
        mtz_path: Path to MTZ file
        pdb_path: Path to PDB file
        contact_distance: Distance in Å to consider molecules "contacting"
        max_mates: Maximum number of symmetry mates to generate

    Returns:
        Tuple of (symmetry_mates, symmetry_info)
    """
    try:
        log.info(f"Loading MTZ file: {mtz_path}")

        # Load MTZ file to get crystallographic information
        mtz = gemmi.read_mtz_file(mtz_path)

        # Load PDB structure
        structure = gemmi.read_structure(pdb_path)

        # Update structure with MTZ crystallographic data if available
        if hasattr(mtz, "cell") and mtz.cell:
            structure.cell = mtz.cell
            log.info("✅ Updated structure with MTZ unit cell parameters")

        if hasattr(mtz, "spacegroup") and mtz.spacegroup:
            structure.spacegroup_hm = str(mtz.spacegroup)
            log.info("✅ Updated structure with MTZ space group")

        # Create symmetry generator
        generator = SymmetryMatesGenerator()

        # Get symmetry information
        symmetry_info = generator.get_symmetry_info(structure)

        # Generate contacting symmetry mates
        symmetry_mates = generator.generate_contacting_symmetry_mates(
            structure, contact_distance, max_mates
        )

        log.info(f"✅ Generated {len(symmetry_mates)} symmetry mates using MTZ data")

        return symmetry_mates, symmetry_info

    except Exception as e:
        log.error(f"Error generating symmetry mates from MTZ: {e}")
        return [], {"error": str(e)}

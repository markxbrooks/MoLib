"""
Integration module for symmetry mates in ElMo.

This module provides functions to integrate symmetry mates generation
with the existing ElMo molecular visualization system.
"""

from typing import Any, Dict, List, Tuple

import gemmi
from decologr import Decologr as log

from .symmetry import generate_symmetry_mates_from_pdb


def load_structure_with_symmetry_mates(
    pdb_data,
    contact_distance: float = 8.0,
    max_mates: int = 50,
    include_original: bool = True,
) -> Tuple[List[gemmi.Model], Dict[str, Any]]:
    """
    Load a structure and generate its contacting symmetry mates.

    This function is designed to be integrated with ElMo's molecular
    visualization system.

    Args:
        pdb_data: Either a path to PDB file (str) or a pandas_pdb object
        contact_distance: Distance in Å to consider molecules "contacting"
        max_mates: Maximum number of symmetry mates to generate
        include_original: Whether to include the original asymmetric unit

    Returns:
        Tuple of (all_models, symmetry_info)
    """
    try:
        # Determine if we have a file path or pandas_pdb object
        if isinstance(pdb_data, str):
            # File path - use existing logic
            log.info(f"Loading structure with symmetry mates from file: {pdb_data}")

            # Generate symmetry mates
            symmetry_mates, symmetry_info = generate_symmetry_mates_from_pdb(
                pdb_data, contact_distance, max_mates
            )

            # Load original structure
            structure = gemmi.read_structure(pdb_data)
            original_model = structure[0]

        else:
            # pandas_pdb object - need to save to temporary file or work directly
            log.info("Loading structure with symmetry mates from pandas_pdb object")

            # For now, we'll need to save the pandas_pdb to a temporary file
            # This is a limitation that could be improved in the future
            import os
            import tempfile

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".pdb", delete=False
            ) as tmp_file:
                # Write pandas_pdb to temporary file
                pdb_data.to_pdb(tmp_file.name)
                tmp_path = tmp_file.name

            try:
                # Generate symmetry mates from temporary file
                symmetry_mates, symmetry_info = generate_symmetry_mates_from_pdb(
                    tmp_path, contact_distance, max_mates
                )

                # Load original structure
                structure = gemmi.read_structure(tmp_path)
                original_model = structure[0]

            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        # Prepare all models for visualization
        all_models = []

        if include_original:
            # Add original asymmetric unit
            all_models.append(("original", original_model))
            log.info("✅ Added original asymmetric unit")

        # Add symmetry mates
        for i, mate in enumerate(symmetry_mates):
            all_models.append((f"symmetry_mate_{i+1}", mate))

        log.info(f"✅ Prepared {len(all_models)} models for visualization")

        return all_models, symmetry_info

    except Exception as e:
        log.error(f"Error loading structure with symmetry mates: {e}")
        return [], {"error": str(e)}


def create_symmetry_mates_widget_info(symmetry_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create widget information for displaying symmetry data in the UI.

    Args:
        symmetry_info: Symmetry information dictionary

    Returns:
        Dictionary formatted for UI widgets
    """
    try:
        # Extract key information
        space_group = symmetry_info.get("space_group", "Unknown")
        unit_cell = symmetry_info.get("unit_cell", {})
        operations = symmetry_info.get("symmetry_operations", 0)

        # Create display strings
        display_info = {
            "space_group": space_group,
            "unit_cell_display": f"a={unit_cell.get('a', '--'):.2f}, "
            f"b={unit_cell.get('b', '--'):.2f}, "
            f"c={unit_cell.get('c', '--'):.2f} Å",
            "symmetry_operations": operations,
            "volume": f"{symmetry_info.get('volume', 0):.1f} Å³",
            "operations_list": [],
        }

        # Format symmetry operations for display
        for op in symmetry_info.get("operations", []):
            op_display = {
                "index": op.get("index", 0),
                "description": str(op.get("description", "Unknown")),
                "is_identity": op.get("is_identity", False),
            }
            display_info["operations_list"].append(op_display)

        return display_info

    except Exception as e:
        log.error(f"Error creating widget info: {e}")
        return {"error": str(e)}


def apply_symmetry_mates_coloring(
    models: List[Tuple[str, gemmi.Model]], color_scheme: str = "chain"
) -> List[Tuple[str, gemmi.Model, str]]:
    """
    Apply colour coding to symmetry mates for visualization.

    Args:
        models: List of (name, model) tuples
        color_scheme: Color scheme to apply ('chain', 'symmetry', 'random')

    Returns:
        List of (name, model, colour) tuples
    """
    try:
        colored_models = []

        if color_scheme == "symmetry":
            # Color by symmetry relationship
            colors = [
                "#FF6B6B",  # Red for original
                "#4ECDC4",  # Teal for mates
                "#45B7D1",  # Blue for mates
                "#96CEB4",  # Green for mates
                "#FFEAA7",  # Yellow for mates
                "#DDA0DD",  # Plum for mates
            ]

            for i, (name, model) in enumerate(models):
                if i == 0:  # Original
                    color = colors[0]
                else:  # Symmetry mates
                    color = colors[(i - 1) % (len(colors) - 1) + 1]

                colored_models.append((name, model, color))

        elif color_scheme == "chain":
            # Color by chain (default ElMo behavior)
            for name, model in models:
                colored_models.append((name, model, "default"))

        else:  # random or other schemes
            # Generate random colors for each model
            import random

            for name, model in models:
                color = f"#{random.randint(0, 0xFFFFFF):06x}"
                colored_models.append((name, model, color))

        log.info(
            f"✅ Applied {color_scheme} colour scheme to {len(colored_models)} models"
        )
        return colored_models

    except Exception as e:
        log.error(f"Error applying coloring: {e}")
        return [(name, model, "default") for name, model in models]


def export_symmetry_mates_to_pdb(
    models: List[Tuple[str, gemmi.Model]], output_path: str
) -> bool:
    """
    Export all models (original + symmetry mates) to a single PDB file.

    Args:
        models: List of (name, model) tuples
        output_path: Output PDB file path

    Returns:
        True if successful, False otherwise
    """
    try:
        log.info(f"💾 Exporting {len(models)} models to: {output_path}")

        # Create a new structure
        structure = gemmi.Structure()

        # Add all models
        for i, (name, model) in enumerate(models):
            # Create a copy of the model
            new_model = gemmi.Model(i)

            # Copy chains from the model
            for chain in model:
                new_chain = gemmi.Chain(chain.name)

                for residue in chain:
                    new_res = gemmi.Residue()
                    new_res.name = residue.name
                    new_res.seqid = residue.seqid
                    new_res.subchain = residue.subchain
                    new_res.flag = residue.flag

                    for atom in residue:
                        new_atom = gemmi.Atom()
                        new_atom.name = atom.name
                        new_atom.element = atom.element
                        new_atom.altloc = atom.altloc
                        new_atom.charge = atom.charge
                        new_atom.occ = atom.occ
                        new_atom.b_iso = atom.b_iso
                        new_atom.aniso = atom.aniso
                        new_atom.pos = atom.pos

                        new_res.add_atom(new_atom)

                    new_chain.add_residue(new_res)
                new_model.add_chain(new_chain)

            structure.add_model(new_model)

        # Write to PDB file
        structure.write_pdb(output_path)

        log.info(f"✅ Successfully exported to: {output_path}")
        return True

    except Exception as e:
        log.error(f"Error exporting symmetry mates: {e}")
        return False


def get_symmetry_mates_statistics(
    models: List[Tuple[str, gemmi.Model]],
) -> Dict[str, Any]:
    """
    Get statistics about the loaded models and symmetry mates.

    Args:
        models: List of (name, model) tuples

    Returns:
        Dictionary with statistics
    """
    try:
        stats = {
            "total_models": len(models),
            "original_model": None,
            "symmetry_mates": 0,
            "total_atoms": 0,
            "total_residues": 0,
            "total_chains": 0,
            "model_details": [],
        }

        for name, model in models:
            # Count atoms, residues, chains
            atom_count = sum(len(residue) for chain in model for residue in chain)
            residue_count = sum(len(chain) for chain in model)
            chain_count = len(model)

            # Update totals
            stats["total_atoms"] += atom_count
            stats["total_residues"] += residue_count
            stats["total_chains"] += chain_count

            # Model details
            model_detail = {
                "name": name,
                "atoms": atom_count,
                "residues": residue_count,
                "chains": chain_count,
            }
            stats["model_details"].append(model_detail)

            # Identify original vs symmetry mates
            if name == "original":
                stats["original_model"] = model_detail
            else:
                stats["symmetry_mates"] += 1

        log.info(
            f"📊 Statistics: {stats['total_models']} models, "
            f"{stats['total_atoms']} atoms, {stats['symmetry_mates']} symmetry mates"
        )

        return stats

    except Exception as e:
        log.error(f"Error calculating statistics: {e}")
        return {"error": str(e)}

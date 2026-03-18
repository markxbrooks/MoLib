"""
Load PDB/mmCIF via Gemmi and expose a PandasPdb-compatible interface.

This provides a fast path that bypasses biopandas parsing: Gemmi reads the file
and we build DataFrames with the columns expected by generate_coordinate_data
and parse_pdb_atoms_to_mol3d. Use pdb_file_load_gemmi() for PDB/mmCIF files
when Gemmi is available and you want to avoid PandasPdb.read_pdb().
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, Optional

import gemmi
import pandas as pd
from decologr import Decologr as log
from molib.core.entity import MolEntityType

# Column names expected by coordinate generator and mol3d parser
_ATOM_COLUMNS = [
    "record_name",
    "atom_number",
    "atom_name",
    "residue_name",
    "residue_number",
    "chain_id",
    "x_coord",
    "y_coord",
    "z_coord",
    "occupancy",
    "b_factor",
    "segment_id",
    "element_symbol",
    "charge",
    "alt_loc",
]


def _structure_to_dfs(structure: gemmi.Structure) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build ATOM and HETATM DataFrames from a Gemmi Structure."""
    atom_rows: list[dict[str, Any]] = []
    hetatm_rows: list[dict[str, Any]] = []

    serial = 0
    for model in structure:
        for chain in model:
            chain_id = chain.name
            for residue in chain:
                res_name = residue.name
                res_num = residue.seqid.num
                seg_id = residue.segment or ""
                for atom in residue:
                    serial += 1
                    pos = atom.pos
                    # Gemmi: het_flag 'H' (or other non-polymer) = HETATM; ' ' or 'A'/'R' = ATOM
                    is_het = (getattr(residue, "het_flag", None) or " ") not in (
                        " ",
                        "A",
                        "R",
                    )
                    row = {
                        "record_name": "HETATM" if is_het else "ATOM",
                        "atom_number": serial,
                        "atom_name": atom.name,
                        "residue_name": res_name,
                        "residue_number": res_num,
                        "chain_id": chain_id,
                        "x_coord": pos.x,
                        "y_coord": pos.y,
                        "z_coord": pos.z,
                        "occupancy": atom.occ,
                        "b_factor": atom.b_iso,
                        "segment_id": seg_id,
                        "element_symbol": (
                            atom.element.name if atom.element else atom.name[:1]
                        ),
                        "charge": "",
                        "alt_loc": atom.altloc or "",
                    }
                    if is_het:
                        hetatm_rows.append(row)
                    else:
                        atom_rows.append(row)

    atom_df = (
        pd.DataFrame(atom_rows, columns=_ATOM_COLUMNS)
        if atom_rows
        else pd.DataFrame(columns=_ATOM_COLUMNS)
    )
    hetatm_df = (
        pd.DataFrame(hetatm_rows, columns=_ATOM_COLUMNS)
        if hetatm_rows
        else pd.DataFrame(columns=_ATOM_COLUMNS)
    )
    return atom_df, hetatm_df


def pdb_file_load_gemmi(file_path: str) -> Optional[Any]:
    """
    Load a PDB or mmCIF file using Gemmi and return a PandasPdb-compatible object.

    The returned object has:
    - .df["ATOM"], .df["HETATM"]: pandas DataFrames with columns expected by
      generate_coordinate_data and parse_pdb_atoms_to_mol3d
    - .pdb_text: PDB-format string (from structure.make_pdb_string()) for
      secondary structure parsing etc.

    Args:
        file_path: Path to .pdb, .cif, or .cif.gz file.

    Returns:
        A namespace with .df and .pdb_text, or None if loading fails.
    """
    try:
        structure = gemmi.read_structure(file_path)
        atom_df, hetatm_df = _structure_to_dfs(structure)
        pdb_text = structure.make_pdb_string()

        result = SimpleNamespace()
        result.df = {
            MolEntityType.ATOM.value: atom_df,
            MolEntityType.HETATM.value: hetatm_df,
        }
        result.pdb_text = pdb_text
        result.pdb_path = file_path
        log.message(
            f"Loaded structure from {file_path} via Gemmi ({len(atom_df)} ATOM, {len(hetatm_df)} HETATM)",
            level=logging.DEBUG,
        )
        return result
    except Exception as e:
        log.warning(f"Gemmi load failed for {file_path}: {e}")
        return None

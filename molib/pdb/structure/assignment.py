"""
Assign Secondary Structure
"""

import pandas as pd
from Bio.PDB import DSSP, PDBParser

from molib.core.constants import MoLibConstant
from molib.entities.molecule import Molecule3D


def assign_secondary_structure(pdb_path: str) -> dict:
    """
    assign_secondary_structure

    :param pdb_path: str
    :return: dict
    Run DSSP and return a dict mapping (chain_id, res_id)
    to secondary structure.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("X", pdb_path)
    model = structure[0]  # First model
    dssp = DSSP(model, pdb_path)

    # DSSP keys are ((chain_id, res_id), ...)
    sec_dict = {}
    for key in dssp.keys():
        chain_id, res_id = key
        ss = dssp[key][2]  # Secondary structure: 'H', 'E', 'T', etc., or ' ' for coil
        sec_dict[(chain_id, res_id[1])] = ss

    return sec_dict


def convert_pdb_df_to_mol3d_with_ss_prediction(
    atom_df: pd.DataFrame, sec_dict: dict
) -> Molecule3D:
    """
    convert_pdb_df_to_mol3d_with_ss_prediction
    :param atom_df: pd.DataFrame
    :param sec_dict: dict
    :return: Mol3D
    """
    mol = Molecule3D()
    seen_residues = set()

    # Performance optimization: Use vectorized operations instead of iterrows
    # Filter for CA atoms first
    ca_atoms = atom_df[atom_df["atom_name"].str.strip() == MoLibConstant.PEPTIDE_CHAIN_ATOMNAME]

    for i, row in ca_atoms.iterrows():
        res_id = (row["chain_id"].strip(), int(row["residue_number"]))
        if res_id in seen_residues:
            continue
        seen_residues.add(res_id)

        coords = (row["x_coord"], row["y_coord"], row["z_coord"])
        secstruc = sec_dict.get(res_id, " ")  # Use DSSP result or default to ' '
        residue = Res3D(
            name=f"{res_id[0]}{res_id[1]}",
            residue_number=int(row["residue_number"]),
            chain=row["chain_id"].strip(),
            type=row["residue_name"].strip(),
            coords=coords,
            secstruc=secstruc,
        )
        mol.append_residue(residue)

    return mol

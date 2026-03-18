"""
CoordinateData Generator
========================

Parse pandas pdb file to return coordinate data
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
from decologr import Decologr as log
from molib.core.entity import MolEntityType
from molib.pdb.color import generate_chain_colors
from molib.pdb.coordinate.data import CoordinateData

from molib.xtal.cif import get_unique_hetatm_cif_dicts
from molib.xtal.validate_hetatm import validate_hetatm_atom_names


def generate_coordinate_data(
    pdb_pandas: PandasPdb,
    include_atom: bool = True,
    atom_filter: Optional[str] = None,
    include_hetatm: bool = True,
    chain_from: str = "atom",  # "atom", "ca", "water, or "hetatm"
) -> CoordinateData | None:
    """
    Generate structured coordinate data from a PandasPdb object.

    Args:
        pdb_pandas (PandasPdb): Parsed PDB object.
        include_atom (bool): Include ATOM records.
        atom_filter (Optional[str]): Filter ATOMs by atom color_scheme (e.g., 'CA').
        include_hetatm (bool): Include HETATM records.
        chain_from (str): Determines which source defines the chains (atom / ca / hetatm).

    Returns:
        CoordinateData | None: Coordinate structure object or None if input is empty.
    """
    if not pdb_pandas:
        return None

    atom_df = pd.DataFrame()
    hetatm_df = pd.DataFrame()
    hetatm_no_water_df = pd.DataFrame()
    cif_blocks = {}

    if include_atom and MolEntityType.ATOM.value in pdb_pandas.df:
        atom_df = pdb_pandas.df[MolEntityType.ATOM.value]
        if atom_filter:
            atom_df = atom_df[atom_df["atom_name"] == atom_filter]

    if include_hetatm and MolEntityType.HETATM.value in pdb_pandas.df:
        hetatm_df = pdb_pandas.df[MolEntityType.HETATM.value]
        hetatm_no_water_df = hetatm_df[~hetatm_df["residue_name"].isin(["HOH", "WAT"])]
    else:
        hetatm_df = pd.DataFrame()
        hetatm_no_water_df = pd.DataFrame()

    water_df = hetatm_df[
        (hetatm_df["residue_name"] == "HOH") & (hetatm_df["atom_name"] == "O")
    ]
    num_water_atoms = len(water_df)
    log.message(num_water_atoms)

    # Tag record type so downstream picking can distinguish ATOM vs HETATM reliably
    if not atom_df.empty:
        atom_df = atom_df.copy()
        atom_df["record_type"] = "ATOM"
    if not hetatm_df.empty:
        hetatm_df = hetatm_df.copy()
        hetatm_df["record_type"] = "HETATM"

    all_atom_df = pd.concat([atom_df, hetatm_df], ignore_index=True)

    num_atoms = len(atom_df)
    num_hetatom_atoms = len(hetatm_df)

    if chain_from == "all_atom":
        source_df = all_atom_df
    elif chain_from == "atom":
        source_df = atom_df
    elif chain_from == "ca":
        source_df = atom_df[atom_df["atom_name"] == "CA"]
    elif chain_from == "water":
        source_df = water_df
    elif chain_from == "hetatm_no_water":
        source_df = hetatm_no_water_df
    elif chain_from == "hetatm":
        source_df = hetatm_df
        cif_blocks = get_unique_hetatm_cif_dicts(hetatm_df)
        validated_df = validate_hetatm_atom_names(hetatm_df, cif_blocks)

        num_invalid = len(validated_df[~validated_df["atom_valid"]])
        if num_invalid > 0:
            log.message(
                f"HETATM validation: {num_invalid} atoms marked invalid (residues with missing CIF or atom-name mismatch)",
                level=logging.DEBUG,
            )

    else:
        raise ValueError(f"Invalid value for 'chain_from': {chain_from}")

    # 🔑 Use source_df for coords to ensure consistency
    coords = source_df[["x_coord", "y_coord", "z_coord"]].to_numpy(dtype=np.float32)
    atom_names = source_df["atom_name"].to_numpy()
    element_symbols = source_df["element_symbol"].to_numpy()

    chain_ids = source_df["chain_id"].tolist()
    chain_colors = generate_chain_colors(chain_ids)
    # log.parameter("source_df", source_df)
    # log.parameter("coords", coords)
    return CoordinateData(
        df=source_df,  # Use source_df for metadata consistency
        cif_blocks=cif_blocks,
        coords=coords,
        atom_names=atom_names,
        element_symbols=element_symbols,
        num_atoms=num_atoms,
        num_hetatom_atoms=num_hetatom_atoms,
        num_water_atoms=num_water_atoms,
        chain_ids=chain_ids,
        chain_colors=chain_colors,
    )

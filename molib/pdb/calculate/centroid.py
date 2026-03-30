"""
Calculate centroid of PDB file
"""

from typing import Optional

import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
from decologr import Decologr as log
from molib.core.entity import MolEntityType


def mol_calculate_atoms_only_centroid(
    pdb_pandas: PandasPdb, chain_id: str | None = None
) -> Optional[np.ndarray]:
    """
    Calculate the centroid of ATOM records in a PDB.

    :param pdb_pandas: PandasPdb instance
    :param chain_id: Optional chain ID filter
    :return: centroid (x, y, z) as np.ndarray, or None if no valid ATOM data
    """
    if not pdb_pandas:
        # log.error("❌ No PDB data provided for centroid calculation.")
        return None
    log.parameter("pdb_pandas", pdb_pandas)
    """if not isinstance(pdb_pandas, PandasPdb):
        log.error("❌ Input must be a PandasPdb instance.")
        return None"""

    df = pdb_pandas.df.get(MolEntityType.ATOM.value)
    if df is None or df.empty:
        log.debug("⚠️ ATOM dataframe is missing or empty.")
        return None

    if chain_id:
        df = df[df["chain_id"] == chain_id]

    if df.empty:
        log.debug(f"⚠️ No ATOM records for chain {chain_id}.")
        return None

    coords = df[["x_coord", "y_coord", "z_coord"]].to_numpy()
    if coords.shape[0] == 0:
        return None

    centroid = np.nanmean(coords, axis=0)
    log.info(f"✅ Calculated centroid: {centroid}")
    return centroid


def mol_calculate_atoms_only_centroid_old(
    pdb_pandas: PandasPdb,
) -> Optional[np.ndarray]:
    """
    Calculate centroid of all ATOM records in the PDB.

    :param pdb_pandas: PandasPdb instance
    :return: centroid (x, y, z) as np.ndarray or None
    """
    if not pdb_pandas:
        # log.error("❌ No PDB data provided for centroid calculation.")
        return None

    df = pdb_pandas.df.get(MolEntityType.ATOM.value)
    if df is None or df.empty:
        log.warning("⚠️ ATOM dataframe is missing or empty.")
        return None

    coords = df[["x_coord", "y_coord", "z_coord"]].to_numpy()
    if coords.shape[0] == 0:
        log.warning("⚠️ No atomic coordinate_data_main found.")
        return None

    centroid = coords.mean(axis=0)
    log.info(f"✅ Calculated centroid: {centroid}")
    return centroid


def mol_calculate_centroid(pdb_pandas: PandasPdb) -> Optional[np.ndarray]:
    """
    Calculate the centroid of all atomic coordinate_data_main (ATOM + HETATM) in a PDB file.

    :param pdb_pandas: PandasPdb instance
    :return: Centroid as np.ndarray [x, y, z] or None if no valid coordinate_data_main.
    """
    if not pdb_pandas:
        log.error("❌ No PDB data provided for centroid calculation.")
        return None

    atom_df = pdb_pandas.df.get(MolEntityType.ATOM.value)
    hetatm_df = pdb_pandas.df.get(MolEntityType.HETATM.value)

    # Filter out None or empty DataFrames
    frames = []
    if atom_df is not None and not atom_df.empty:
        frames.append(atom_df)
    if hetatm_df is not None and not hetatm_df.empty:
        frames.append(hetatm_df)

    if not frames:
        log.warning("⚠️ No valid ATOM or HETATM data found for centroid calculation.")
        return None

    combined_df = pd.concat(frames, ignore_index=True)
    coords = combined_df[["x_coord", "y_coord", "z_coord"]].to_numpy()

    if coords.size == 0:
        log.warning("⚠️ No coordinate_data_main found after merging ATOM and HETATM.")
        return None

    centroid = coords.mean(axis=0)
    log.info(f"✅ Calculated centroid from {len(coords)} atoms: {centroid}")
    return centroid

"""
contact

Example Usage:
==============
>>>df_A = df[df['chain_id']=='A']
...df_B = df[df['chain_id']=='B']
...contact_atoms, node_pairs = get_contact_atoms(df_A, df_B, 7.)
"""

import numpy as np
import pandas as pd


def get_contact_atoms(
    df1: pd.DataFrame,
    df2: pd.DataFrame = None,
    threshold: float = 7.0,
    coord_names=["x_coord", "y_coord", "z_coord"],
):
    # Extract coordinate_data_main from dataframes
    coords1 = df1[coord_names].to_numpy()
    coords2 = df2[coord_names].to_numpy()

    # Compute pairwise distances between atoms
    dist_matrix = np.sqrt(((coords1[:, None] - coords2) ** 2).sum(axis=2))

    # Create a new dataframe containing pairs of atoms whose distance is below the threshold
    pairs = np.argwhere(dist_matrix < threshold)
    atoms1, atoms2 = df1.iloc[pairs[:, 0]], df2.iloc[pairs[:, 1]]
    atoms1_id = (
        atoms1["chain_id"].map(str)
        + ":"
        + atoms1["residue_name"].map(str)
        + ":"
        + atoms1["residue_number"].map(str)
    )
    atoms2_id = (
        atoms2["chain_id"].map(str)
        + ":"
        + atoms2["residue_name"].map(str)
        + ":"
        + atoms2["residue_number"].map(str)
    )
    node_pairs = np.vstack((atoms1_id.values, atoms2_id.values)).T
    result = pd.concat(
        [df1.iloc[np.unique(pairs[:, 0])], df2.iloc[np.unique(pairs[:, 1])]]
    )

    return result, node_pairs

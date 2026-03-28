"""
CDR

Annotation and contact functions
"""

import pandas as pd
from decologr import Decologr as log
from molib.calc.geometry.distance import euclidean_distance
from molib.calc.inter_chain_contacts import get_contacting_atom_pairs

LOGGING_ENABLED = False

IMGT_CDR1 = set(range(27, 38 + 1))
IMGT_CDR2 = set(range(56, 65 + 1))
IMGT_CDR3 = set(range(105, 117 + 1))
CONTACT_DISTANCE = 5  # Angstroms (Å)


def log_if_enabled(message: str):
    """
    log_if_enabled

    :param message: str
    :return: none
    """
    if LOGGING_ENABLED:
        log.message(message)


def assign_cdr_number(seq_id: int) -> int:
    """
    Map imgt_id to CDR domains, return number associated with domain or return None if input is not in a CDR
    domain.
    """
    if seq_id in IMGT_CDR1:
        return 1

    if seq_id in IMGT_CDR2:
        return 2

    if seq_id in IMGT_CDR3:
        return 3

    return None


def get_tcr_contacting_residues(chain_c_df: pd.DataFrame, tcr_cdrs_df: pd.DataFrame):
    """
    get_contacting_residues

    :param chain_c_df: pd.DataFrame
    :param tcr_cdrs_df: pd.DataFrame
    :return: pd.DataFrame contacting_residues
    """
    interface = tcr_cdrs_df.merge(
        chain_c_df, how="cross", suffixes=("_tcr", "_peptide")
    )
    log_if_enabled(f"interface: {interface.head()}")
    interface["atom_distances"] = euclidean_distance(
        interface["x_coord_tcr"],
        interface["y_coord_tcr"],
        interface["z_coord_tcr"],
        interface["x_coord_peptide"],
        interface["y_coord_peptide"],
        interface["z_coord_peptide"],
    )
    contacting_atoms = interface[interface["atom_distances"] <= CONTACT_DISTANCE]
    contacting_residues = contacting_atoms[
        [
            "chain_id_tcr",
            "residue_number_tcr",
            "insertion_tcr",
            "cdr",
            "chain_type",
            "residue_number_peptide",
            "insertion_peptide",
        ]
    ].drop_duplicates()
    log_if_enabled(f"contacting_residues: {contacting_residues.head()}")
    return contacting_residues


def get_contacting_atoms(chain1: pd.DataFrame, chain2: pd.DataFrame) -> pd.DataFrame:
    """
    Find atom pairs between chain1 and chain2 that are within CONTACT_DISTANCE.

    Both dataframes must share biopandas-style columns ``x_coord``, ``y_coord``,
    ``z_coord``. Suffixes in the result are ``_a`` / ``_b`` (see
    :func:`molib.calc.inter_chain_contacts.get_contacting_atom_pairs`).

    :param chain1: pd.DataFrame with atoms from chain 1
    :param chain2: pd.DataFrame with atoms from chain 2
    :return: pd.DataFrame of contacting atom pairs
    """
    pairs = get_contacting_atom_pairs(chain1, chain2, CONTACT_DISTANCE)
    # Historically TCR code expected _1/_2 suffixes; normalize for callers in this module.
    rename = {
        c: c.replace("_a", "_1").replace("_b", "_2")
        for c in pairs.columns
        if "_a" in c or "_b" in c
    }
    return pairs.rename(columns=rename)


def annotate_alpha_beta_chains(tcr_df: pd.DataFrame):
    """
    annotate_alpha_beta_chains

    :param tcr_df: pd.DataFrame
    :return: pd.DataFrame
    """
    tcr_df["chain_type"] = tcr_df["chain_id"].map(
        lambda chain_id: "alpha" if chain_id == "D" else "beta"
    )


def annotate_contacts(df: pd.DataFrame):
    """
    annotate_contacts

    :param df: pd.DataFrame
    :return: pd.DataFrame contacting_residues
    """
    chain_a_df = df.query("chain_id == 'A'")  # MHC dataframe
    log_if_enabled(str(chain_a_df.head))
    chain_c_df = df.query("chain_id == 'C'")  # peptide dataframe
    chain_c_residues = chain_c_df.groupby(
        ["residue_number", "insertion"], dropna=False
    )  # peptide residues
    log_if_enabled(f"Chain A is a {len(chain_c_residues)}-mer!")
    chain_d_e = df.query("chain_id == 'D' or chain_id == 'E'")  # TCR df
    log_if_enabled(str(chain_d_e.head()))
    chain_a_df = df.query("chain_id == 'A'")
    chain_d_e = (
        chain_d_e.copy()
    )  # Doing this on a copy of the dataframe since it is originally a slice of df!
    chain_d_e["cdr"] = chain_d_e["residue_number"].map(assign_cdr_number)
    annotate_alpha_beta_chains(chain_d_e)
    tcr_cdrs_df = chain_d_e.query("cdr.notnull()")
    cdr_lengths = (
        tcr_cdrs_df[["chain_type", "cdr", "residue_number", "insertion"]]
        .drop_duplicates()
        .groupby(["chain_type", "cdr"], dropna=False)
        .size()
    )
    log_if_enabled(cdr_lengths)
    contacting_residues = get_tcr_contacting_residues(chain_c_df, tcr_cdrs_df)
    contacting_atoms = get_contacting_atoms(chain_c_df, tcr_cdrs_df)
    log_if_enabled(str(contacting_atoms.head()))
    return contacting_residues, None  # contacting_atoms

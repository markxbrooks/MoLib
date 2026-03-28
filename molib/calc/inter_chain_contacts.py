"""
Inter-chain contact detection from atom tables (e.g. biopandas ATOM DataFrame).

Provides configurable distance cutoffs and optional hydrogen exclusion for
residue-level interface summaries between two chain IDs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from molib.entities.molecule import Molecule3D


def _normalize_element_symbol(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def _prepare_chain_atoms(frame: pd.DataFrame, exclude_hydrogen: bool) -> pd.DataFrame:
    df = frame.copy()
    required = {"x_coord", "y_coord", "z_coord", "chain_id", "residue_number", "insertion"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"atom_df is missing required columns: {sorted(missing)}")

    if exclude_hydrogen and "element_symbol" in df.columns:
        es = _normalize_element_symbol(df["element_symbol"])
        df = df[~es.isin(("H", "D"))]

    return df


def get_contacting_atom_pairs(
    chain1: pd.DataFrame,
    chain2: pd.DataFrame,
    cutoff_angstrom: float,
) -> pd.DataFrame:
    """
    All atom pairs with one atom in ``chain1`` and one in ``chain2`` within
    ``cutoff_angstrom`` (Å).

    Both frames must share biopandas-style coordinate columns ``x_coord``,
    ``y_coord``, ``z_coord``. Overlapping column names receive suffixes ``_a``
    and ``_b`` (chain1 → _a, chain2 → _b).

    :param chain1: atoms belonging to the first chain (already filtered)
    :param chain2: atoms belonging to the second chain (already filtered)
    :param cutoff_angstrom: maximum Euclidean distance (Å)
    :return: merged rows with ``distance`` and suffixed atom/residue columns
    """
    if cutoff_angstrom <= 0:
        raise ValueError("cutoff_angstrom must be positive")

    if chain1.empty or chain2.empty:
        return pd.DataFrame()

    interface = chain1.merge(chain2, how="cross", suffixes=("_a", "_b"))
    dx = interface["x_coord_a"] - interface["x_coord_b"]
    dy = interface["y_coord_a"] - interface["y_coord_b"]
    dz = interface["z_coord_a"] - interface["z_coord_b"]
    interface["distance"] = np.sqrt(dx**2 + dy**2 + dz**2)
    out = interface[interface["distance"] <= cutoff_angstrom].copy()
    return out


def contacting_residue_pairs(
    atom_df: pd.DataFrame,
    chain_a: str,
    chain_b: str,
    cutoff_angstrom: float = 5.0,
    *,
    exclude_hydrogen: bool = True,
    return_atom_pairs: bool = False,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Residue-residue contacts between two chain IDs using a distance cutoff on
    inter-chain atom pairs.

    Typical input is ``pandas_pdb.df["ATOM"]`` from
    ``molib.pdb.biopandas.load.pdb_file_load_biopandas``.

    Each output row is a unique residue pair
    (chain A residue ↔ chain B residue) with
    ``min_distance`` = minimum inter-atomic distance between those two residues
    among pairs within the cutoff.

    :param atom_df: full ATOM (or ATOM-like) table
    :param chain_a: first chain ID (e.g. ``"A"``)
    :param chain_b: second chain ID (e.g. ``"B"``)
    :param cutoff_angstrom: maximum atom-atom distance (Å) to count as contact
    :param exclude_hydrogen: if True, drop rows whose ``element_symbol`` is H or D
    :param return_atom_pairs: if True, also return the underlying atom-pair table
    :return: ``(residue_pairs, atom_pairs_or_none)`` — atom_pairs is None unless
             ``return_atom_pairs`` is True; empty atom frame yields empty residue frame
    """
    if atom_df.empty:
        empty_r = pd.DataFrame(
            columns=[
                "chain_id_a",
                "residue_number_a",
                "insertion_a",
                "chain_id_b",
                "residue_number_b",
                "insertion_b",
                "min_distance",
            ]
        )
        return empty_r, (pd.DataFrame() if return_atom_pairs else None)

    work = _prepare_chain_atoms(atom_df, exclude_hydrogen=exclude_hydrogen)
    cid = work["chain_id"].astype(str).str.strip()
    ca = str(chain_a).strip()
    cb = str(chain_b).strip()

    present = set(cid.unique())
    if ca not in present:
        raise ValueError(f"chain_id {ca!r} not found in atom_df (have {sorted(present)})")
    if cb not in present:
        raise ValueError(f"chain_id {cb!r} not found in atom_df (have {sorted(present)})")

    dfa = work[cid == ca]
    dfb = work[cid == cb]

    atom_pairs = get_contacting_atom_pairs(dfa, dfb, cutoff_angstrom=cutoff_angstrom)
    if atom_pairs.empty:
        residue_cols = [
            "chain_id_a",
            "residue_number_a",
            "insertion_a",
            "chain_id_b",
            "residue_number_b",
            "insertion_b",
            "min_distance",
        ]
        empty_r = pd.DataFrame(columns=residue_cols)
        return empty_r, (atom_pairs if return_atom_pairs else None)

    residue_keys = [
        "chain_id_a",
        "residue_number_a",
        "insertion_a",
        "chain_id_b",
        "residue_number_b",
        "insertion_b",
    ]
    residue_pairs = (
        atom_pairs.groupby(residue_keys, dropna=False)["distance"]
        .min()
        .reset_index()
        .rename(columns={"distance": "min_distance"})
        .sort_values(residue_keys, kind="mergesort")
        .reset_index(drop=True)
    )

    atom_out = atom_pairs if return_atom_pairs else None
    return residue_pairs, atom_out


def _atom_is_hydrogen(atom) -> bool:
    el = getattr(atom, "element", None)
    if el is None:
        return False
    return str(el).strip().upper() in ("H", "D")


def apply_min_inter_chain_distances_to_molecule(
    molecule: "Molecule3D",
    chain_a: str,
    chain_b: str,
    *,
    exclude_hydrogen: bool = True,
) -> Tuple[int, int]:
    """
    For Phase C visualization: set ``atom_contact_distance`` on atoms in
    ``chain_a`` / ``chain_b`` to the minimum Euclidean distance to any atom on
    the **other** chain. All other atoms get ``atom_contact_distance = None``.

    Pair with :class:`~molib.core.color.strategy.ColorScheme.CONTACT_DISTANCE`.

    Hydrogens may be excluded from the distance sets (they still get
    ``atom_contact_distance = None`` when excluded).

    :param molecule: loaded :class:`~molib.entities.molecule.Molecule3D`
    :param chain_a: first chain ID
    :param chain_b: second chain ID (must differ from ``chain_a``)
    :param exclude_hydrogen: omit H/D from coordinate lists when computing minima
    :return: ``(n_atoms_used_chain_a, n_atoms_used_chain_b)``
    """
    ca = str(chain_a).strip()
    cb = str(chain_b).strip()
    if ca == cb:
        raise ValueError("chain_a and chain_b must differ")

    atoms_a: list = []
    coords_a: list = []
    atoms_b: list = []
    coords_b: list = []

    for model in molecule.models:
        for chain_key, chain in model.chains.items():
            scid = str(chain_key).strip()
            for residue in chain.residues:
                for atom in residue.atoms.values():
                    atom.atom_contact_distance = None
                    if exclude_hydrogen and _atom_is_hydrogen(atom):
                        continue
                    ac = str(
                        getattr(atom, "chain_id", None) or chain.chain_id or scid
                    ).strip()
                    c = np.asarray(atom.coords, dtype=np.float64).reshape(3)
                    if ac == ca:
                        atoms_a.append(atom)
                        coords_a.append(c)
                    elif ac == cb:
                        atoms_b.append(atom)
                        coords_b.append(c)

    if not atoms_a or not atoms_b:
        raise ValueError(
            f"No atoms found for chains {ca!r} and {cb!r} "
            f"(after hydrogen filter={exclude_hydrogen})"
        )

    PA = np.stack(coords_a, axis=0)
    PB = np.stack(coords_b, axis=0)

    try:
        from scipy.spatial.distance import cdist

        da = cdist(PA, PB).min(axis=1)
        db = cdist(PB, PA).min(axis=1)
    except Exception:
        da = np.sqrt(((PA[:, None, :] - PB[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
        db = np.sqrt(((PB[:, None, :] - PA[None, :, :]) ** 2).sum(axis=2)).min(axis=1)

    for atom, d in zip(atoms_a, da):
        atom.atom_contact_distance = float(d)
    for atom, d in zip(atoms_b, db):
        atom.atom_contact_distance = float(d)

    return len(atoms_a), len(atoms_b)

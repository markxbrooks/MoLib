"""
Validate hetatm atom names against residue CIF definitions.
"""

import gemmi
import pandas as pd
from decologr import Decologr as log


def validate_hetatm_atom_names(
    hetatm_df: pd.DataFrame, cif_blocks: dict[str, gemmi.cif.Block]
) -> pd.DataFrame:
    """
    Validate atom names in hetatm_df against the corresponding residue CIF definitions.

    Adds:
      - 'atom_valid': whether each atom_name is valid for its residue_name.
      - 'validated_by': which residue (block key) validated the atom (can be empty if not validated).

    :param hetatm_df: DataFrame with 'residue_name' and 'atom_name' columns.
    :param cif_blocks: Dictionary of residue_name -> gemmi.CifBlock.
    :return: Copy of DataFrame with additional validation columns.
    """
    # Clean up whitespace
    hetatm_df = hetatm_df.copy()
    hetatm_df["atom_name"] = hetatm_df["atom_name"].str.strip()
    hetatm_df["residue_name"] = hetatm_df["residue_name"].str.strip()

    # Prepare output columns
    hetatm_df["atom_valid"] = False
    hetatm_df["validated_by"] = ""

    # Group rows by residue_name for efficient lookup
    for resname, group in hetatm_df.groupby("residue_name"):
        block = cif_blocks.get(resname)
        if block is None:
            continue  # Skip unknown residues

        atom_ids = set(block.find_values("_chem_comp_atom.atom_id"))
        valid_mask = group["atom_name"].isin(atom_ids)

        hetatm_df.loc[valid_mask.index, "atom_valid"] = valid_mask
        hetatm_df.loc[valid_mask[valid_mask].index, "validated_by"] = resname

    return hetatm_df


def validate_hetatm_atoms(
    hetatm_df: pd.DataFrame, cif_block: gemmi.cif.Block
) -> pd.DataFrame | None:
    """
    validate_hetatm_atoms

    :param hetatm_df: pd.DataFrame
    :param cif_block: gemmi.cif.Block
    :return: pd.DataFrame or None
    """
    try:
        atom_id_loop = cif_block.find_loop("_chem_comp_atom.atom_id")
        if atom_id_loop is None:
            raise ValueError("CIF block does not contain _chem_comp_atom.atom_id")

        valid_cif_names = {
            atom_id_loop.get_value(i, 0).strip().upper()
            for i in range(atom_id_loop.length())
        }

        return hetatm_df["atom_name"].str.strip().str.upper().isin(valid_cif_names)
    except ValueError:
        raise ValueError("CIF block does not contain _chem_comp_atom.atom_name")
    except KeyError:
        raise ValueError("CIF block does not contain _chem_comp_atom.atom_name")
    except AttributeError:
        raise ValueError("CIF block does not contain _chem_comp_atom.atom_name")
    except IndexError:
        raise ValueError("CIF block does not contain _chem_comp_atom.atom_name")
    except Exception as ex:
        log.error(ex)


def validate_hetatm_atom_names_old(
    hetatm_df: pd.DataFrame, cif_blocks: dict[str, gemmi.cif.Block]
) -> pd.DataFrame:
    """
    Validate atom names in hetatm_df against the
    corresponding residue CIF definitions.

    Adds a new column `atom_valid` indicating whether
    each atom_name is valid for its residue.

    :param hetatm_df: DataFrame with 'residue_name' and 'atom_name' columns.
    :param cif_blocks: Dictionary of residue_name -> gemmi.CifBlock.
    :return: DataFrame with an additional 'atom_valid' column.
    """

    def is_valid(row):
        resname = row["residue_name"]
        atom_name = row["atom_name"].strip()

        block = cif_blocks.get(resname)
        if block is None:
            return False

        atom_ids = set(block.find_values("_chem_comp_atom.atom_id"))
        return atom_name in atom_ids

    hetatm_df = hetatm_df.copy()
    hetatm_df["atom_valid"] = hetatm_df.apply(is_valid, axis=1)

    return hetatm_df

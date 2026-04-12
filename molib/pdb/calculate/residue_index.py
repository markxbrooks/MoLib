from biopandas.pdb import PandasPdb
from decologr import Decologr as log
from molib.core.constants import MoLibConstant
from molib.core.entity import MolEntityType
from molib.pdb.coordinate.data import CoordinateData


def get_atom_data_from_index(coordinate_data: CoordinateData, index: int):
    """
    get_atom_data_from_index

    :param coordinate_data: CoordinateData object with atom DataFrames.
    :param index: Index into the ATOM dataframe (or other available type).
    :return: (residue_number, chain_id) tuple or None.

    Get residue number and chain ID for the atom at the given index in the coordinate data.
    """
    if coordinate_data is None or not hasattr(coordinate_data, "df"):
        return None
    try:
        atom_df = coordinate_data.df

        if atom_df is None:
            log.message("⚠️ No atom dataframe found in coordinate_data_main.df")
            return None

        if not (0 <= index < len(atom_df)):
            log.message(
                f"⚠️ Index {index} out of range for dataframe of length {len(atom_df)}"
            )
            return None

        if 0 <= index < len(atom_df):
            atom_row = atom_df.iloc[index]
            atom_name = atom_row.get("atom_name")
            residue_name = atom_row.get("residue_name")
            residue_id = atom_row.get("residue_number")
            chain_id = atom_row.get("chain_id")
            record_type = atom_row.get("record_type", None)
            return atom_name, residue_name, residue_id, chain_id, record_type

        return None
    except Exception as ex:
        log.error(f"Error reading atom metadata at index {index}: {ex}")


def get_residue_from_by_index_and_chain_id(
    pdb_pandas: PandasPdb, index: int, chain_id: str
):
    """
    get_residue_from_by_index_and_chain_id

    :param pdb_pandas: PandasPdb
    :param index: int (index into self.vertex_data)
    :param chain_id: str chain_id
    :return: (residue_id) or None if not found
    """
    if pdb_pandas is None or MolEntityType.ATOM.value not in pdb_pandas.df:
        return None

    atom_df = pdb_pandas.df[MolEntityType.ATOM.value]
    chain_atom_df = atom_df[atom_df["chain_id"] == chain_id].reset_index(drop=True)
    ca_df = chain_atom_df[chain_atom_df["atom_name"] == MoLibConstant.PEPTIDE_CHAIN_ATOMNAME].reset_index(drop=True)

    if 0 <= index < len(ca_df):
        residue_id = int(ca_df.iloc[index]["residue_number"])
        return residue_id
    return None

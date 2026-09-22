"""get residues by index"""

from biopandas.pdb import PandasPdb
from molib.core.constants import MoLibConstant
from molib.core.entity import MolEntityType


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
    ca_df = chain_atom_df[
        chain_atom_df["atom_name"] == MoLibConstant.PEPTIDE_CHAIN_ATOMNAME
    ].reset_index(drop=True)

    if 0 <= index < len(ca_df):
        residue_id = int(ca_df.iloc[index]["residue_number"])
        return residue_id
    return None

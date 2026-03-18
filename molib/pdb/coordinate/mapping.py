from biopandas.pdb import PandasPdb
from decologr import Decologr as log
from molib.pdb.calculate.residue_index import get_residue_from_by_index_and_chain_id


def get_pdb_residue_mapping(
    pandas_pdb: PandasPdb, chain_id: str, start_res: int, end_res: int
) -> tuple:
    """
    get_pdb_residue_mapping

    :param pandas_pdb: PandasPdb
    :param chain_id: str
    :param start_res: int
    :param end_res:int
    :return: tuple
    """
    try:
        start_pdb = get_residue_from_by_index_and_chain_id(
            pandas_pdb, start_res - 1, chain_id
        )
        end_pdb = get_residue_from_by_index_and_chain_id(
            pandas_pdb, end_res - 1, chain_id
        )
        return start_pdb, end_pdb
    except Exception as ex:
        log.error(f"Error mapping residues to PDB: {ex}")
        return None, None

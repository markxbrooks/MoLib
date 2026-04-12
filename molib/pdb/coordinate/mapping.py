from biopandas.pdb import PandasPdb
from decologr import Decologr as log
from molib.core.constants import MoLibConstant
from molib.core.entity import MolEntityType
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


def get_fasta_index_range_for_pdb_residue_span(
    pandas_pdb: PandasPdb, chain_id: str, pdb_start: int, pdb_end: int
) -> tuple:
    """
    Map an inclusive PDB residue-number span on ``chain_id`` to 1-based FASTA indices.

    Indices follow the same CA ordering as :func:`get_residue_from_by_index_and_chain_id`
    (and the ElMo sequence editor lines built from :meth:`PandasPdb.amino3to1`).
    """
    try:
        if pandas_pdb is None or MolEntityType.ATOM.value not in pandas_pdb.df:
            return None, None
        if pdb_start > pdb_end:
            pdb_start, pdb_end = pdb_end, pdb_start
        atom_df = pandas_pdb.df[MolEntityType.ATOM.value]
        chain_atom_df = atom_df[atom_df["chain_id"] == chain_id].reset_index(drop=True)
        ca_df = chain_atom_df[chain_atom_df["atom_name"] == MoLibConstant.PEPTIDE_CHAIN_ATOMNAME].reset_index(drop=True)
        hit_indices: list[int] = []
        for ord_idx in range(len(ca_df)):
            rid = int(ca_df.iloc[ord_idx]["residue_number"])
            if pdb_start <= rid <= pdb_end:
                hit_indices.append(ord_idx)
        if not hit_indices:
            return None, None
        # 1-based FASTA positions (matches get_pdb_residue_mapping convention)
        return hit_indices[0] + 1, hit_indices[-1] + 1
    except Exception as ex:
        log.error(f"Error mapping PDB span to FASTA indices: {ex}")
        return None, None

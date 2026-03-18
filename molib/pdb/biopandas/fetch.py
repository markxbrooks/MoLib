"""
ELMO
PDB Utils for loading PDB files
"""

from typing import Optional

import gemmi
from biopandas.pdb import PandasPdb
from decologr import Decologr as log

from elmo.xtal.cif import validate_structure


def fetch_pdb_biopandas(
    uniprot_id: str = None, pdb_code: str = None, validate: bool = True
) -> Optional[PandasPdb]:
    """
    Fetch a structure from AlphaFold (via UniProt ID) or RCSB PDB (via PDB code),
    validate a residue against monomer CIF, and return the PandasPdb object.

    :param validate: bool validate residue against monomer CIF
    :param pdb_code: str PDB code string
    :param uniprot_id: str UniProt ID string
    :return: PandasPdb object or None
    """
    try:
        if uniprot_id:
            log.message(f"pdb_fetch_as_biopandas uniprot_id {uniprot_id}", silent=True)
            ppdb = PandasPdb().fetch_pdb(uniprot_id=uniprot_id, source="alphafold2-v4")
        elif pdb_code:
            log.message(f"pdb_fetch_as_biopandas pdb_code {pdb_code}", silent=True)
            ppdb = PandasPdb().fetch_pdb(pdb_code=pdb_code)
        else:
            log.warning("No UniProt ID or PDB code provided.")
            return None
        if validate:
            validation_report = perform_cif_dict_validation(ppdb)
            # Store validation report in the PandasPdb object for later use
            ppdb.validation_report = validation_report

        return ppdb
    except Exception as ex:
        log.error(f"Error {ex} occurred in fetch_pdb_biopandas")
        return None


def perform_cif_dict_validation(pandas_pdb: PandasPdb) -> dict | None:
    """
    perform_cif_dict_validation

    :param pandas_pdb: PandasPdb object
    :return: validation_report dict or None
    """
    structure = gemmi.read_pdb_string(pandas_pdb.pdb_text)
    log.parameter("structure", structure)
    validation_report = validate_structure(structure)
    # Get unit cell
    cell = structure.cell
    sg_hm = structure.spacegroup_hm
    sg = gemmi.find_spacegroup_by_name(sg_hm)
    sym_ops = sg.operations()
    log.parameter("cell", cell)
    log.parameter("sg", sg)
    if validation_report:
        log.message(f"Validation result: {validation_report}")
    else:
        log.warning("No validation could be performed")
    # Original model
    model = structure[0]
    log.parameter("model", model)
    try:
        # Create symmetry mates
        mates = gemmi.Model(0)
        # mates.color_scheme = "symmetry_mates"
        # log.parameter("mates", mates)
        for op in sym_ops:
            for chain in model:
                # log.parameter("chain", chain)
                new_chain = gemmi.Chain(chain.name)
                # log.parameter("new_chain", new_chain)
                for residue in chain:
                    new_res = gemmi.Residue()
                    # log.parameter("new_res", new_res)
                    new_res.name = residue.name
                    new_res.seqid = residue.seqid
                    for atom in residue:
                        new_atom = gemmi.Atom()
                        # log.parameter("new_atom", new_atom)
                        new_atom.name = atom.name
                        new_atom.element = atom.element
                        frac = cell.fractionalize(atom.pos)  # gemmi.Fractional
                        frac_sym = op.apply_to_xyz(
                            [frac.x, frac.y, frac.z]
                        )  # list[float]
                        frac_sym = gemmi.Fractional(
                            *frac_sym
                        )  # convert list → Fractional
                        new_atom.pos = cell.orthogonalize(frac_sym)
                        new_res.add_atom(new_atom)
                    new_chain.add_residue(new_res)
                mates.add_chain(new_chain)

        structure.add_model(mates)
        structure.write_pdb("6eqb_with_symmetry.pdb")
    except Exception as ex:
        log.error(f"Error {ex} occurred in perform_cif_dict_validation")

    return validation_report


def fetch_pdb_by_source_and_id(identifier: str, source: str) -> PandasPdb:
    """
    fetch_pdb_by_source_and_id

    :param identifier: str
    :param source: str
    :return: pdb_data
    :rtype: PandasPdb
    """
    if source == "pdb":
        pdb_data = fetch_pdb_biopandas(pdb_code=identifier)
    elif source == "alphafold":
        pdb_data = fetch_pdb_biopandas(uniprot_id=identifier)
    else:
        raise ValueError(f"Unknown source: {source}")
    return pdb_data

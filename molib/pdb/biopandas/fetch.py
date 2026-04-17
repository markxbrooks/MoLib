"""
ELMO
PDB Utils for loading PDB files
"""

from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import gemmi
from biopandas.pdb import PandasPdb
from decologr import Decologr as log
from molib.xtal.cif import validate_structure


def _http_get_text(url: str, *, quiet: bool = False) -> Optional[str]:
    """GET url and return decoded UTF-8 text, or None on failure."""
    try:
        response = urlopen(url)
        raw = response.read()
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        return text if text and text.strip() else None
    except HTTPError as e:
        if not quiet:
            if e.code == 404:
                log.warning(f"Not found (404): {url}")
            else:
                log.warning(f"HTTP {e.code} fetching {url}")
        return None
    except URLError as e:
        if not quiet:
            log.warning(f"URL error fetching {url}: {e}")
        return None


def _pdb_from_text(pdb_text: str, source_url: str) -> PandasPdb:
    """Build PandasPdb from raw PDB text (avoids biopandas bug when fetch returns None)."""
    ppdb = PandasPdb().read_pdb_from_list(pdb_text.splitlines(True))
    ppdb.pdb_path = source_url
    return ppdb


def _fetch_alphafold_ebi(uniprot_id: str) -> Optional[PandasPdb]:
    """
    Download AlphaFold PDB from EBI. Tries model v6, then v4, then v3 (DB layout varies).
    Biopandas' fetch_pdb leaves pdb_text None on 404 then crashes in splitlines().
    """
    uid = uniprot_id.strip().upper()
    log.message(f"pdb_fetch_as_biopandas uniprot_id {uid}", silent=True)
    last_url = ""
    for version in (6, 4, 3):
        last_url = (
            f"https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_v{version}.pdb"
        )
        text = _http_get_text(last_url, quiet=True)
        if text:
            return _pdb_from_text(text, last_url)
    log.warning(
        f"No AlphaFold PDB at alphafold.ebi.ac.uk for UniProt {uid} "
        f"(tried model v6, v4, and v3; 404 or empty). Last URL: {last_url}"
    )
    return None


def _fetch_rcsb(pdb_code: str) -> Optional[PandasPdb]:
    code = pdb_code.strip().lower()
    log.message(f"pdb_fetch_as_biopandas pdb_code {code}", silent=True)
    url = f"https://files.rcsb.org/download/{code}.pdb"
    text = _http_get_text(url, quiet=False)
    if not text:
        return None
    return _pdb_from_text(text, url)


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
            ppdb = _fetch_alphafold_ebi(uniprot_id)
        elif pdb_code:
            ppdb = _fetch_rcsb(pdb_code)
        else:
            log.warning("No UniProt ID or PDB code provided.")
            return None
        if ppdb is None:
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


def fetch_pdb_by_source_and_id(
    identifier: str, source: str
) -> Optional[PandasPdb]:
    """
    fetch_pdb_by_source_and_id

    :param identifier: str
    :param source: str
    :return: pdb_data or None if the structure is not available
    """
    if source == "pdb":
        pdb_data = fetch_pdb_biopandas(pdb_code=identifier)
    elif source == "alphafold":
        pdb_data = fetch_pdb_biopandas(uniprot_id=identifier)
    else:
        raise ValueError(f"Unknown source: {source}")
    return pdb_data

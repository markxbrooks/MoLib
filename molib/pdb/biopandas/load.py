"""
Module for loading and converting PDB and mmCIF files to a PandasPdb object.

This module provides functionality for reading protein structure files in PDB
and mmCIF formats. It enables conversion of these formats to a PandasPdb format
for easier manipulation, analysis, and visualization of structural data.

It utilizes Biopandas for PDB file handling and Gemmi for mmCIF operations.
Logging is enabled for tracking success or failure of file operations.
"""

from pathlib import Path

import gemmi
from biopandas.pdb import PandasPdb
from decologr import Decologr as log


def pdb_file_load_biopandas(file_path: str):
    """
    pdb_file_read_as_biopandas

    :param file_path: str
    :return:
    """
    ppdb = PandasPdb().read_pdb(file_path)
    return ppdb


def mmcif_file_load_biopandas(file_path: str):
    """
    Load mmCIF file and convert to PandasPdb format using gemmi.

    :param file_path: str - Path to mmCIF file (.cif or .cif.gz)
    :return: PandasPdb object or None if loading fails
    """
    try:
        # Load structure using gemmi
        structure = gemmi.read_structure(file_path)

        # Convert gemmi structure to PDB format string
        pdb_string = structure.make_pdb_string()

        # Create PandasPdb object from the PDB string
        ppdb = PandasPdb()
        ppdb.read_pdb_from_list(pdb_string.split("\n"))

        log.info(f"✅ Successfully loaded mmCIF file: {file_path}")
        return ppdb

    except Exception as e:
        log.error(f"❌ Failed to load mmCIF file {file_path}: {e}")
        return None

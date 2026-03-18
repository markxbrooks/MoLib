"""
Ligand handling module for ElMo.

This module provides functionality for parsing and displaying ligand files
in SDF and CSV formats with molecular visualization using RDKit.
"""

from molib.ligand.file.parser import LigandFileParser, check_rdkit_availability
from molib.ligand.pdb.info import PDBLigandInfo

# Import GUI components only if PySide6 is available
try:
    from elmo_inspect.ui.ligands.pdb_ligands_widget import (
        PDBLigandsDisplayWidget,
        PDBLigandTableWidget,
    )
    from elmo_inspect.ui.ligands.widget import LigandDisplayWidget, LigandTableWidget

    _GUI_AVAILABLE = True
except ImportError:
    _GUI_AVAILABLE = False
    LigandDisplayWidget = None
    LigandTableWidget = None
    PDBLigandsDisplayWidget = None
    PDBLigandTableWidget = None

__all__ = [
    "LigandFileParser",
    "LigandInfo",
    "PDBLigandParser",
    "PDBLigandInfo",
    "check_rdkit_availability",
]

if _GUI_AVAILABLE:
    __all__.extend(
        [
            "LigandDisplayWidget",
            "LigandTableWidget",
            "PDBLigandsDisplayWidget",
            "PDBLigandTableWidget",
        ]
    )

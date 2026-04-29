"""
Molecular Entity Type
"""

from enum import Enum


class MolEntityType(str, Enum):
    """Mol Entity Type"""

    ATOM = "ATOM"
    #: Canonical façade key for the drawable primary structure (Phase C); resolves like ATOM.
    PRIMARY = "PRIMARY"
    ATOM_HETATM = "ATOM_HETATM"
    CALPHAS = "CALPHAS"
    HETATM = "HETATM"
    WATER = "WATER"
    # Not-strict
    TEXT = "TEXT"
    LIGAND = "LIGAND"

    ALL = [
        ATOM,
        HETATM,
        ATOM_HETATM,
        CALPHAS,
        WATER,
    ]
    NON_WATER = [
        ATOM_HETATM,
        ATOM,
        HETATM,
        CALPHAS
    ]
    NON_WATER = REGULAR

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value

    def get_name(self):
        return self.value

    def get_display_name(self):
        return self.value + "_model"


"""
Molecular Entity Type
"""

from enum import Enum


class MolEntityType(str, Enum):
    """Mol Entity Type"""

    ATOM = "ATOM"
    ATOM_HETATM = "ATOM_HETATM"
    CALPHAS = "CALPHAS"
    HETATM = "HETATM"
    LIGAND = "LIGAND"
    WATER = "WATER"
    TEXT = "TEXT"

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value

    def get_name(self):
        return self.value

    def get_display_name(self):
        return self.value + "_model"

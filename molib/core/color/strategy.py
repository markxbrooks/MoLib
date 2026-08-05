"""
An enumeration defining various color schemes for Molecule coloring.

This module provides a ColorScheme Enum to represent different methods of
coloring molecules. Each value in the Enum corresponds to a specific
coloring method, and it can be used for visualization or representation
purposes in molecular modeling.
"""

from enum import Enum, auto


class ColorScheme(Enum):
    """Color Scheme Enum for Molecule coloring"""

    UNIFORM = auto()
    SECONDARY_STRUCTURE = auto()
    ELEMENT = auto()
    CHAIN = auto()
    B_FACTOR = auto()
    VALIDATION = auto()
    CONTACT_DISTANCE = auto()
    #: UniProt domain colors (PDBe mapping); ElMo applies per-residue RGB on top of chain fallbacks.
    DOMAIN = auto()

    def __str__(self) -> str:
        """
        Return a human-readable string representation of the object,
        showing its class name and current colour color_scheme.
        """
        return f"<{self.__class__.__name__} color_scheme={self.name}>"

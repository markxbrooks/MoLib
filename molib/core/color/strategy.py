"""Color Scheme Enum for Molecule coloring"""

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

    def __str__(self) -> str:
        """
        Return a human-readable string representation of the object,
        showing its class name and current colour color_scheme.
        """
        return f"<{self.__class__.__name__} color_scheme={self.name}>"

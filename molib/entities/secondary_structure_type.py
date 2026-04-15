from __future__ import annotations

from enum import Enum

from picogl.utils.strenum import StrEnum


class SecondaryStructureType(StrEnum):
    """Secondary structure types with string mappings for backward compatibility."""

    ALPHA_HELIX = "H"  # Alpha helix
    ALPHA_HELIX2 = "A"  # Alpha helix
    BETA_STRAND = "E"  # Beta strand/sheet
    TURN = "T"  # Turn
    BEND = "S"  # Bend
    HELIX_3_10 = "G"  # 3-10 helix
    HELIX_3_10_2 = "3"  # 3-10 helix
    HELIX_3_10_3 = "L"  # 3-10 helix
    PI_HELIX = "I"  # Pi helix
    BETA_BRIDGE = "B"  # Beta bridge
    COIL = " "  # Coil/loop (space character for compatibility)
    COIL2 = "C"  # Coil/loop (space character for compatibility)

    @classmethod
    def from_string(cls, value: str) -> "SecondaryStructureType":
        """Convert string value to enum, with fallback to COIL."""
        # Handle common string representations
        string_map = {
            "H": cls.ALPHA_HELIX,
            "E": cls.BETA_STRAND,
            "T": cls.TURN,
            "S": cls.BEND,
            "G": cls.HELIX_3_10,
            "I": cls.PI_HELIX,
            "B": cls.BETA_BRIDGE,
            " ": cls.COIL,
            "-": cls.COIL,
            "C": cls.COIL,
        }
        return string_map.get(value, cls.COIL)

    def to_string(self) -> str:
        """Convert enum to string representation."""
        return self.value

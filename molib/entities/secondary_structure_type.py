from __future__ import annotations

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

HELIX_TYPES = {
    SecondaryStructureType.ALPHA_HELIX,
    SecondaryStructureType.ALPHA_HELIX2,
    SecondaryStructureType.HELIX_3_10,
    SecondaryStructureType.HELIX_3_10_2,
    SecondaryStructureType.HELIX_3_10_3,
    SecondaryStructureType.PI_HELIX,
}

SHEET_TYPES = {
    SecondaryStructureType.BETA_STRAND,
    SecondaryStructureType.BETA_BRIDGE,
}

COIL_TYPES = {
    SecondaryStructureType.COIL,
    SecondaryStructureType.COIL2,
    SecondaryStructureType.TURN,
    SecondaryStructureType.BEND,
}


def normalize_ss(ss: str | SecondaryStructureType) -> SecondaryStructureType:
    if isinstance(ss, SecondaryStructureType):
        return ss
    return SecondaryStructureType.from_string(ss)


class SecondaryStructureWidth:
    """Secondary Structure Width"""
    HELIX = 0.6
    SHEET = 0.8
    COIL = 0.5


def _width_for(ss: SecondaryStructureType) -> float:
    if ss in HELIX_TYPES:
        return SecondaryStructureWidth.HELIX
    elif ss in SHEET_TYPES:
        return SecondaryStructureWidth.SHEET
    else:
        return SecondaryStructureWidth.COIL





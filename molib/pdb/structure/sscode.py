"""
Representation and management of secondary structure codes and their associated labels.

This module provides functionality to handle secondary structure codes, such as
helix, strand, turn, coil, and others. It also facilitates the translation between
codes and their human-readable labels or UI display labels. Additionally, it
offers utilities for working with predefined sets of codes and label pairs.
"""


class SSCode:
    """Secondary Structure Code with Label"""

    HELIX = "H"
    STRAND = "E"
    TURN = "T"
    OTHER = " "
    COIL = "C"
    COIL2 = "-"

    ALL = [HELIX, STRAND, TURN, COIL, COIL2, OTHER]
    ALL_STRUCTURED = [HELIX, STRAND]
    ALL_COIL = [TURN, COIL, COIL2, OTHER]

    ALL_LABELS = ["Helix", "Strand", "Turn", "Coil", "Other"]

    @classmethod
    def codes(cls) -> list[str]:
        return [cls.HELIX, cls.STRAND, cls.TURN, cls.COIL, cls.OTHER]

    @classmethod
    def labels(cls) -> list[str]:
        return list(cls.ALL_LABELS)

    @classmethod
    def get_label(cls, code: str) -> str:
        codes = cls.codes()
        if code not in codes:
            return "Other"
        return cls.labels()[codes.index(code)]

    @classmethod
    def get_code(cls, label: str) -> str:
        labels = cls.labels()
        codes = cls.codes()
        if label not in labels:
            return cls.OTHER
        return codes[labels.index(label)]

    @classmethod
    def get_display_label(cls, code: str) -> str:
        """Human-readable label for *code*, or *code* unchanged if not in :meth:`codes`."""
        return cls.get_label(code) if code in cls.codes() else code

    @classmethod
    def code_label_pairs(cls) -> list[tuple[str, str]]:
        """Ordered (one-letter code, UI label) pairs for previews and combos."""
        return list(zip(cls.codes(), cls.labels(), strict=True))

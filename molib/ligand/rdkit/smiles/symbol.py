"""
Defines utilities for handling chemical elements, bond detection, and molecule
manipulations using RDKit. Includes classes and functions for element
properties, bond specifications, and various chemical computations.
"""

from enum import Enum


class StrEnum(str, Enum):
    """
    Python 3.10 compatible StrEnum.

    Behaves like Python 3.11 enum.StrEnum.
    """

    def __str__(self) -> str:
        return str(self.value)


class SmilesSymbol(StrEnum):
    """Smiles Symbols"""

    COMPONENT_SEPARATOR = "."
    BRANCH_START = "("
    BRANCH_END = ")"

    SINGLE_BOND = "-"
    DOUBLE_BOND = "="
    TRIPLE_BOND = "#"
    AROMATIC_BOND = ":"

    RING_0 = "0"
    RING_1 = "1"
    RING_2 = "2"
    RING_3 = "3"
    RING_4 = "4"
    RING_5 = "5"
    RING_6 = "6"
    RING_7 = "7"
    RING_8 = "8"
    RING_9 = "9"

    BRACKET_START = "["
    BRACKET_END = "]"

    CHARGE_POSITIVE = "+"
    CHARGE_NEGATIVE = "-"

    DISCONNECTED = "."

    @staticmethod
    def is_ring_digit(ch: str) -> bool:
        return ch.isdigit()

    @staticmethod
    def is_bond(ch: str) -> bool:
        return ch in {
            SmilesSymbol.SINGLE_BOND,
            SmilesSymbol.DOUBLE_BOND,
            SmilesSymbol.TRIPLE_BOND,
            SmilesSymbol.AROMATIC_BOND,
        }

    @staticmethod
    def is_branch(ch: str) -> bool:
        return ch in {
            SmilesSymbol.BRANCH_START,
            SmilesSymbol.BRANCH_END,
        }

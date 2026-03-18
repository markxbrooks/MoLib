"""
CoordinateSelection
"""

from dataclasses import dataclass


@dataclass
class CoordinateSelection:
    """
    CoordinateSelection class
    """

    start_residue: int = None
    end_residue: int = None
    chain_id: str = None

    def __str__(self):
        return f"Selection: {self.chain_id} {self.start_residue}-{self.end_residue}"

    def reset(self):
        """Reset the selection to its default state."""
        self.start_residue = None
        self.end_residue = None
        self.chain_id = None

    def update(
        self, start_residue: int = None, end_residue: int = None, chain_id: str = None
    ) -> None:
        """
        update

        :param self:
        :param start_residue: int
        :param end_residue: int
        :param chain_id: int
        :return: None
        """
        if start_residue is not None:
            self.start_residue = start_residue
        if end_residue is not None:
            self.end_residue = end_residue
        if chain_id is not None:
            self.chain_id = chain_id

"""
Atom Data Structure
"""

from dataclasses import dataclass

from pandas import Series

from decologr import Decologr as log
from elmo.chem.amino_acids import is_amino_acid


@dataclass
class AtomData:
    """Atom Data"""
    atom_row: Series = None
    atom_name: str = None
    residue_name: str = None
    residue_id: int = None
    chain_id: str = None
    record_type: str = None

    def log_contents(self):
        """log contents of the Atom Data"""
        log.message(f"Atom data: {self.atom_row} {self.atom_name} {self.residue_name} {self.residue_id} {self.chain_id} {self.record_type}")

    @property
    def is_hetatm(self) -> bool:
        return self.record_type == "HETATM" if self.record_type is not None else False

    @property
    def residue_name_available(self) -> str | bool:
        return (self.residue_name
                and (not is_amino_acid(self.residue_name))
                and (len(str(self.residue_name)) <= 3))

"""
This module provides a `DeduplicatedLigand` dataclass to manage ligand
information with deduplicated instances from PDB data.

The module defines a dataclass that includes information about
the primary ligand and its instances. It also provides a property
to determine the count of the ligand's instances within the dataset.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DeduplicatedLigand:
    """DeduplicatedLigand"""

    ligand: "PDBLigandInfo"
    instances: list["PDBLigandInfo"]

    @property
    def instance_count(self) -> int:
        return len(self.instances)

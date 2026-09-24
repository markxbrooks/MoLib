from __future__ import annotations

from dataclasses import dataclass

from molib.xtal.ccp4.mtz.column_pair import MtzColumnPair


@dataclass
class MtzFileSpec:
    """Contents of an MTZ file"""

    file_path: str
    map_coefficients: MtzColumnPair
    difference_coefficients: MtzColumnPair

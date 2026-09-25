from __future__ import annotations

from dataclasses import dataclass

from molib.xtal.ccp4.mtz.column_pair import MtzColumnPair
from molib.xtal.map.density import MapType


@dataclass(frozen=True, slots=True)
class MtzFileSpec:
    """Contents of an MTZ file"""

    file_path: str
    map_coefficients: MtzColumnPair
    difference_coefficients: MtzColumnPair


@dataclass(slots=True)
class MtzDensitySpec:
    """Which MTZ coefficients to grid and at what sampling rate."""

    map_type: MapType = MapType.TWO_FO_FC
    f_label: str | None = None
    phi_label: str | None = None
    sample_rate: float = 0.0

    def __post_init__(self):
        if (self.f_label is None) != (self.phi_label is None):
            raise ValueError(
                "f_label and phi_label must both be provided or both be None"
            )
        if self.sample_rate < 0:
            raise ValueError(
                f"sample_rate must be >= 0, got {self.sample_rate}"
            )
from __future__ import annotations

from dataclasses import dataclass

from molib.xtal.map.density import MapType


@dataclass(frozen=True, slots=True)
class MtzColumnPair:
    """F/PHI coefficients describing a density map."""

    f_label: str
    phi_label: str
    map_type: MapType

    @classmethod
    def map(
        cls,
        f_label: str,
        phi_label: str,
    ) -> "MtzColumnPair":
        """Build a 2Fo-Fc coefficient pair."""
        return cls(
            f_label=f_label,
            phi_label=phi_label,
            map_type=MapType.TWO_FO_FC,
        )

    @classmethod
    def difference(
        cls,
        f_label: str,
        phi_label: str,
    ) -> "MtzColumnPair":
        """Build an Fo-Fc difference coefficient pair."""
        return cls(
            f_label=f_label,
            phi_label=phi_label,
            map_type=MapType.FO_FC,
        )

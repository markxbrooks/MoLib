from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MapCoefficients:
    """MTZ coefficient labels used to generate a map."""

    f_label: str
    phi_label: str

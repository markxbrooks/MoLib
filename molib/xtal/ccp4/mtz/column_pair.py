from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MtzColumnPair:
    """F/PHI column pair used to generate a map."""

    f_label: str
    phi_label: str

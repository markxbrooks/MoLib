from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MtzColumnPair:
    f_label: str
    phi_label: str
    are_difference_coefficients: bool = False

    @classmethod
    def map(
        cls,
        f_label: str,
        phi_label: str,
    ) -> "MtzColumnPair":
        return cls(
            f_label=f_label,
            phi_label=phi_label,
            are_difference_coefficients=False,
        )

    @classmethod
    def difference(
        cls,
        f_label: str,
        phi_label: str,
    ) -> "MtzColumnPair":
        return cls(
            f_label=f_label,
            phi_label=phi_label,
            are_difference_coefficients=True,
        )

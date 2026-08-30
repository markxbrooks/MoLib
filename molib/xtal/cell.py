"""Canonical crystallographic unit-cell value object."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

CELL_PARAMETERS = ("a", "b", "c", "alpha", "beta", "gamma")


def unit_cell_volume(
    a: float,
    b: float,
    c: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> float:
    """Return crystallographic unit-cell volume in Å³."""
    return a * b * c * math.sqrt(_volume_radicand(alpha, beta, gamma))


def _volume_radicand(alpha: float, beta: float, gamma: float) -> float:
    """Return the expression under the volume square root."""
    cos_alpha = math.cos(math.radians(alpha))
    cos_beta = math.cos(math.radians(beta))
    cos_gamma = math.cos(math.radians(gamma))
    return (
        1.0
        - cos_alpha**2
        - cos_beta**2
        - cos_gamma**2
        + 2.0 * cos_alpha * cos_beta * cos_gamma
    )


def _optional_string(value: object) -> str | None:
    return None if value is None else str(value)


@dataclass(frozen=True, slots=True)
class UnitCell:
    """Crystallographic unit-cell parameters."""

    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    space_group: str | None = None
    crystal_system: str | None = None

    def __post_init__(self) -> None:
        for name in ("a", "b", "c"):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError("Unit-cell lengths must be positive.")
        for name in ("alpha", "beta", "gamma"):
            angle = float(getattr(self, name))
            if not 0.0 < angle < 180.0:
                raise ValueError(
                    "Unit-cell angles must be strictly between 0 and 180 degrees."
                )
        radicand = _volume_radicand(self.alpha, self.beta, self.gamma)
        if radicand <= 0.0:
            raise ValueError(
                "Unit-cell metric is not positive definite "
                "(volume radicand must be positive)."
            )

    @property
    def parameters(self) -> tuple[float, float, float, float, float, float]:
        """Return the six unit-cell parameters."""
        return (
            self.a,
            self.b,
            self.c,
            self.alpha,
            self.beta,
            self.gamma,
        )

    def value(self, name: str) -> float:
        """Return a named unit-cell parameter."""
        if name not in CELL_PARAMETERS:
            raise AttributeError(f"Unknown unit-cell parameter: {name!r}")
        return float(getattr(self, name))

    @property
    def volume(self) -> float:
        """Return the cell volume in Å³."""
        return unit_cell_volume(*self.parameters)

    @classmethod
    def from_mapping(cls, mapping: Mapping) -> UnitCell:
        """Build a unit cell from a Gemmi-style parameter mapping."""
        return cls(
            a=float(mapping["a"]),
            b=float(mapping["b"]),
            c=float(mapping["c"]),
            alpha=float(mapping["alpha"]),
            beta=float(mapping["beta"]),
            gamma=float(mapping["gamma"]),
            space_group=_optional_string(mapping.get("space_group")),
            crystal_system=_optional_string(mapping.get("crystal_system")),
        )

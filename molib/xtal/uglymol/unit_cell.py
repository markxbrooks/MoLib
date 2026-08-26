"""Coordinate transforms derived from a crystallographic unit cell."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from molib.xtal.cell import UnitCell
from molib.xtal.uglymol.math.helpers import multiply


@dataclass(frozen=True, slots=True)
class UnitCellGeometry:
    """Coordinate transforms derived from a unit cell.

    Matrices use the row-vector convention employed by ``multiply``.
    """

    cell: UnitCell
    fractional_to_orthogonal: tuple[float, ...] = field(init=False)
    orthogonal_to_fractional: tuple[float, ...] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fractional_to_orthogonal",
            self._calculate_orthogonalization_matrix(self.cell),
        )
        object.__setattr__(
            self,
            "orthogonal_to_fractional",
            self._calculate_fractionalization_matrix(self.cell),
        )

    @classmethod
    def from_parameters(
        cls,
        a: float,
        b: float,
        c: float,
        alpha: float,
        beta: float,
        gamma: float,
    ) -> UnitCellGeometry:
        """Build geometry from six crystallographic parameters."""
        return cls(UnitCell(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma))

    @property
    def orth(self) -> tuple[float, ...]:
        """Alias for :attr:`fractional_to_orthogonal`."""
        return self.fractional_to_orthogonal

    @property
    def frac(self) -> tuple[float, ...]:
        """Alias for :attr:`orthogonal_to_fractional`."""
        return self.orthogonal_to_fractional

    @property
    def parameters(self) -> tuple[float, float, float, float, float, float]:
        """Expose the underlying cell parameters."""
        return self.cell.parameters

    def fractionalize(self, xyz):
        """Convert orthogonal coordinates to fractional coordinates."""
        return multiply(xyz, self.orthogonal_to_fractional)

    def orthogonalize(self, xyz):
        """Convert fractional coordinates to orthogonal coordinates."""
        return multiply(xyz, self.fractional_to_orthogonal)

    @staticmethod
    def _calculate_orthogonalization_matrix(
        cell: UnitCell,
    ) -> tuple[float, ...]:
        a, b, c, alpha, beta, gamma = cell.parameters
        deg2rad = math.pi / 180.0
        cos_alpha = math.cos(deg2rad * alpha)
        cos_beta = math.cos(deg2rad * beta)
        cos_gamma = math.cos(deg2rad * gamma)
        sin_beta = math.sin(deg2rad * beta)
        sin_gamma = math.sin(deg2rad * gamma)
        cos_alpha_star_sin_beta = (cos_beta * cos_gamma - cos_alpha) / sin_gamma
        cos_alpha_star = cos_alpha_star_sin_beta / sin_beta
        s1rca2 = math.sqrt(1.0 - cos_alpha_star * cos_alpha_star)
        return (
            a,
            b * cos_gamma,
            c * cos_beta,
            0.0,
            b * sin_gamma,
            -c * cos_alpha_star_sin_beta,
            0.0,
            0.0,
            c * sin_beta * s1rca2,
        )

    @staticmethod
    def _calculate_fractionalization_matrix(
        cell: UnitCell,
    ) -> tuple[float, ...]:
        a, b, c, alpha, beta, gamma = cell.parameters
        deg2rad = math.pi / 180.0
        cos_alpha = math.cos(deg2rad * alpha)
        cos_beta = math.cos(deg2rad * beta)
        cos_gamma = math.cos(deg2rad * gamma)
        sin_beta = math.sin(deg2rad * beta)
        sin_gamma = math.sin(deg2rad * gamma)
        cos_alpha_star_sin_beta = (cos_beta * cos_gamma - cos_alpha) / sin_gamma
        cos_alpha_star = cos_alpha_star_sin_beta / sin_beta
        s1rca2 = math.sqrt(1.0 - cos_alpha_star * cos_alpha_star)
        return (
            1.0 / a,
            -cos_gamma / (sin_gamma * a),
            -(cos_gamma * cos_alpha_star_sin_beta + cos_beta * sin_gamma)
            / (sin_beta * s1rca2 * sin_gamma * a),
            0.0,
            1.0 / (sin_gamma * b),
            cos_alpha_star / (s1rca2 * sin_gamma * b),
            0.0,
            0.0,
            1.0 / (sin_beta * s1rca2 * c),
        )


# Compatibility alias for existing imports / call sites.
UnitCellTransform = UnitCellGeometry

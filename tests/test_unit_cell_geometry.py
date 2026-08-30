"""Tests for UnitCell value object and UnitCellGeometry transforms."""

from __future__ import annotations

import math

import pytest

from molib.xtal.cell import UnitCell
from molib.xtal.uglymol.unit_cell import UnitCellGeometry, UnitCellTransform


def test_unit_cell_transform_alias() -> None:
    assert UnitCellTransform is UnitCellGeometry


def test_orthogonal_cell_volume() -> None:
    cell = UnitCell(a=10.0, b=20.0, c=30.0, alpha=90.0, beta=90.0, gamma=90.0)
    assert cell.volume == pytest.approx(6000.0)
    assert cell.parameters == (10.0, 20.0, 30.0, 90.0, 90.0, 90.0)


def test_orthogonalize_fractionalize_round_trip() -> None:
    geometry = UnitCellGeometry.from_parameters(50.0, 60.0, 70.0, 90.0, 90.0, 90.0)
    frac = [0.25, 0.5, 0.75]
    orth = geometry.orthogonalize(frac)
    back = geometry.fractionalize(orth)
    assert back[0] == pytest.approx(frac[0], abs=1e-9)
    assert back[1] == pytest.approx(frac[1], abs=1e-9)
    assert back[2] == pytest.approx(frac[2], abs=1e-9)
    assert geometry.orth == geometry.fractional_to_orthogonal
    assert geometry.frac == geometry.orthogonal_to_fractional


def test_rejects_non_positive_length() -> None:
    with pytest.raises(ValueError, match="lengths must be positive"):
        UnitCell(a=0.0, b=10.0, c=10.0, alpha=90.0, beta=90.0, gamma=90.0)


def test_rejects_invalid_angle() -> None:
    with pytest.raises(ValueError, match="between 0 and 180"):
        UnitCell(a=10.0, b=10.0, c=10.0, alpha=0.0, beta=90.0, gamma=90.0)
    with pytest.raises(ValueError, match="between 0 and 180"):
        UnitCell(a=10.0, b=10.0, c=10.0, alpha=180.0, beta=90.0, gamma=90.0)


def test_rejects_non_positive_definite_metric() -> None:
    # Angles that make the volume radicand negative.
    with pytest.raises(ValueError, match="positive definite"):
        UnitCell(a=10.0, b=10.0, c=10.0, alpha=5.0, beta=5.0, gamma=170.0)


def test_from_mapping() -> None:
    cell = UnitCell.from_mapping(
        {
            "a": 63.1,
            "b": 50.17,
            "c": 111.07,
            "alpha": 90.0,
            "beta": 96.19,
            "gamma": 90.0,
            "space_group": "P 1 21 1",
        }
    )
    assert cell.a == pytest.approx(63.1)
    assert cell.space_group == "P 1 21 1"
    assert cell.volume > 0.0
    assert math.isfinite(cell.volume)

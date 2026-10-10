"""Unified density carving around MapGrid geometry."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from molib.xtal.map.density import (
    AxisOrder,
    GridOrigin,
    GridSpacing,
    MapGrid,
)
from molib.xtal.map.helper import (
    _CARVE_SLAB_SIZE,
    _carve_density_near_coordinates,
    carve_density_around_position,
    carve_density_around_protein,
)


def _grid(
    shape: tuple[int, int, int],
    *,
    origin=(0.0, 0.0, 0.0),
    spacing=(1.0, 1.0, 1.0),
    axis_order: AxisOrder = AxisOrder.XYZ,
) -> MapGrid:
    return MapGrid(
        dimensions=shape,
        origin=GridOrigin(*origin),
        spacing=GridSpacing(*spacing),
        axis_order=axis_order,
    )


def test_carve_near_coordinates_retains_nearby_density() -> None:
    shape = (5, 5, 5)
    density = np.ones(shape, dtype=np.float32)
    grid = _grid(shape)
    # Origin at (0,0,0), spacing 1 → voxel (2,2,2) is at Cartesian (2,2,2)
    carved = _carve_density_near_coordinates(
        density,
        np.array([[2.0, 2.0, 2.0]], dtype=np.float64),
        grid,
        cutoff_distance=1.01,
    )
    assert carved[2, 2, 2] == 1.0
    assert carved[0, 0, 0] == 0.0
    assert carved[4, 4, 4] == 0.0


def test_carve_around_position_wrapper() -> None:
    shape = (6, 6, 6)
    density = np.full(shape, 2.0, dtype=np.float32)
    grid = _grid(shape)
    carved = carve_density_around_position(
        density,
        (0.0, 0.0, 0.0),
        grid,
        cutoff_distance=0.5,
    )
    assert carved[0, 0, 0] == 2.0
    assert carved[3, 3, 3] == 0.0
    assert density[3, 3, 3] == 2.0  # non-mutating


def test_shape_mismatch_raises() -> None:
    density = np.ones((4, 4, 4), dtype=np.float32)
    grid = _grid((5, 5, 5))
    with pytest.raises(ValueError, match="does not match"):
        _carve_density_near_coordinates(
            density,
            np.array([[0.0, 0.0, 0.0]]),
            grid,
            1.0,
        )


def test_non_xyz_axis_order_raises() -> None:
    shape = (3, 3, 3)
    density = np.ones(shape, dtype=np.float32)
    grid = _grid(shape, axis_order=AxisOrder.ZYX)
    with pytest.raises(ValueError, match="XYZ"):
        carve_density_around_position(density, (0.0, 0.0, 0.0), grid, 1.0)


def test_negative_cutoff_raises() -> None:
    shape = (2, 2, 2)
    with pytest.raises(ValueError, match="cutoff_distance"):
        _carve_density_near_coordinates(
            np.ones(shape, dtype=np.float32),
            np.array([[0.0, 0.0, 0.0]]),
            _grid(shape),
            -1.0,
        )


def test_protein_empty_atoms_raises() -> None:
    shape = (2, 2, 2)
    density = np.ones(shape, dtype=np.float32)
    grid = _grid(shape)
    with patch(
        "molib.xtal.map.helper.gemmi.read_structure",
        return_value=MagicMock(),
    ), patch(
        "molib.xtal.map.helper._collect_atom_coordinates",
        return_value=[],
    ):
        with pytest.raises(ValueError, match="No atom coordinates"):
            carve_density_around_protein(
                density,
                "fake.pdb",
                grid,
                cutoff_distance=4.0,
            )


def test_slab_path_covers_tall_volume() -> None:
    """Volumes taller than the slab size still carve correctly."""
    assert _CARVE_SLAB_SIZE == 32
    shape = (8, 8, 40)
    density = np.ones(shape, dtype=np.float32)
    grid = _grid(shape)
    # Point at z=39 (last slab)
    carved = _carve_density_near_coordinates(
        density,
        np.array([[0.0, 0.0, 39.0]], dtype=np.float64),
        grid,
        cutoff_distance=0.5,
    )
    assert carved[0, 0, 39] == 1.0
    assert carved[0, 0, 0] == 0.0


def test_carve_with_nonunit_spacing() -> None:
    """Vectorized origin/spacing must place voxels at index * spacing."""
    shape = (5, 5, 5)
    density = np.ones(shape, dtype=np.float32)
    spacing = (2.0, 2.0, 2.0)
    grid = _grid(shape, spacing=spacing)
    # Voxel (1,1,1) sits at Cartesian (2,2,2)
    carved = _carve_density_near_coordinates(
        density,
        np.array([[2.0, 2.0, 2.0]], dtype=np.float64),
        grid,
        cutoff_distance=0.5,
    )
    assert carved[1, 1, 1] == 1.0
    assert carved[0, 0, 0] == 0.0
    assert carved[2, 2, 2] == 0.0

"""Unified density carving around DensityMapData geometry."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from molib.xtal.map.density import (
    AxisOrder,
    CoordinateTransforms,
    CrystallographicInfo,
    GridOrigin,
    GridSpacing,
    MapGrid,
    UnitCell,
)
from molib.xtal.map.helper import (
    DensityMapData,
    _CARVE_SLAB_SIZE,
    _carve_density_near_coordinates,
    carve_density_around_position,
    carve_density_around_protein,
)


def _map_data(
    density: np.ndarray,
    *,
    origin=(0.0, 0.0, 0.0),
    spacing=(1.0, 1.0, 1.0),
    axis_order: AxisOrder = AxisOrder.XYZ,
) -> DensityMapData:
    shape = tuple(int(x) for x in density.shape)
    info = CrystallographicInfo(
        unit_cell=UnitCell(
            a=1.0,
            b=1.0,
            c=1.0,
            alpha=90.0,
            beta=90.0,
            gamma=90.0,
        ),
        space_group="P1",
        grid=MapGrid(
            dimensions=shape,
            origin=GridOrigin(*origin),
            spacing=GridSpacing(*spacing),
            axis_order=axis_order,
        ),
        transforms=CoordinateTransforms(
            frac_to_orth=np.eye(3),
            orth_to_frac=np.eye(3),
        ),
    )
    return DensityMapData(volume=density, crystallographic_info=info)


def test_carve_near_coordinates_retains_nearby_density() -> None:
    density = np.ones((5, 5, 5), dtype=np.float32)
    map_data = _map_data(density)
    # Origin at (0,0,0), spacing 1 → voxel (2,2,2) is at Cartesian (2,2,2)
    carved = _carve_density_near_coordinates(
        map_data,
        np.array([[2.0, 2.0, 2.0]], dtype=np.float64),
        cutoff_distance=1.01,
    )
    assert carved[2, 2, 2] == 1.0
    assert carved[0, 0, 0] == 0.0
    assert carved[4, 4, 4] == 0.0


def test_carve_around_position_wrapper() -> None:
    density = np.full((6, 6, 6), 2.0, dtype=np.float32)
    map_data = _map_data(density)
    carved = carve_density_around_position(
        map_data,
        (0.0, 0.0, 0.0),
        cutoff_distance=0.5,
    )
    assert carved[0, 0, 0] == 2.0
    assert carved[3, 3, 3] == 0.0
    assert map_data.volume[3, 3, 3] == 2.0  # non-mutating


def test_shape_mismatch_raises() -> None:
    density = np.ones((4, 4, 4), dtype=np.float32)
    map_data = _map_data(density)
    map_data.crystallographic_info.grid.dimensions = (5, 5, 5)
    with pytest.raises(ValueError, match="does not match"):
        _carve_density_near_coordinates(
            map_data,
            np.array([[0.0, 0.0, 0.0]]),
            1.0,
        )


def test_non_xyz_axis_order_raises() -> None:
    density = np.ones((3, 3, 3), dtype=np.float32)
    map_data = _map_data(density, axis_order=AxisOrder.ZYX)
    with pytest.raises(ValueError, match="XYZ"):
        carve_density_around_position(map_data, (0.0, 0.0, 0.0), 1.0)


def test_negative_cutoff_raises() -> None:
    map_data = _map_data(np.ones((2, 2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="cutoff_distance"):
        _carve_density_near_coordinates(
            map_data,
            np.array([[0.0, 0.0, 0.0]]),
            -1.0,
        )


def test_protein_empty_atoms_raises() -> None:
    map_data = _map_data(np.ones((2, 2, 2), dtype=np.float32))
    with patch(
        "molib.xtal.map.helper.gemmi.read_structure",
        return_value=MagicMock(),
    ), patch(
        "molib.xtal.map.helper._collect_atom_coordinates",
        return_value=[],
    ):
        with pytest.raises(ValueError, match="No atom coordinates"):
            carve_density_around_protein(
                map_data,
                "fake.pdb",
                cutoff_distance=4.0,
            )


def test_slab_path_covers_tall_volume() -> None:
    """Volumes taller than the slab size still carve correctly."""
    assert _CARVE_SLAB_SIZE == 32
    density = np.ones((8, 8, 40), dtype=np.float32)
    map_data = _map_data(density)
    carved = _carve_density_near_coordinates(
        map_data,
        np.array([[0.0, 0.0, 39.0]], dtype=np.float64),
        cutoff_distance=0.5,
    )
    assert carved[0, 0, 39] == 1.0
    assert carved[0, 0, 0] == 0.0


def test_carve_with_nonunit_spacing() -> None:
    """Vectorized origin/spacing must place voxels at index * spacing."""
    density = np.ones((5, 5, 5), dtype=np.float32)
    map_data = _map_data(density, spacing=(2.0, 2.0, 2.0))
    # Voxel (1,1,1) sits at Cartesian (2,2,2)
    carved = _carve_density_near_coordinates(
        map_data,
        np.array([[2.0, 2.0, 2.0]], dtype=np.float64),
        cutoff_distance=0.5,
    )
    assert carved[1, 1, 1] == 1.0
    assert carved[0, 0, 0] == 0.0
    assert carved[2, 2, 2] == 0.0

"""DensityMapData ownership: volume canonical, cached stats, grid invariants."""

from __future__ import annotations

import numpy as np
import pytest

from molib.xtal.ccp4.map.volume import VolumeStatistics
from molib.xtal.map.axis import AxisOrder
from molib.xtal.map.crystal import CoordinateTransforms, CrystallographicInfo
from molib.xtal.map.grid import GridOrigin, GridSpacing, MapGrid
from molib.xtal.map.helper import DensityMapData
from molib.xtal.map.unit_cell import UnitCell


def _cryst(shape: tuple[int, int, int]) -> CrystallographicInfo:
    return CrystallographicInfo(
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
            origin=GridOrigin(0.0, 0.0, 0.0),
            spacing=GridSpacing(1.0, 1.0, 1.0),
            axis_order=AxisOrder.XYZ,
        ),
        transforms=CoordinateTransforms(
            frac_to_orth=np.eye(3),
            orth_to_frac=np.eye(3),
        ),
    )


def test_stats_derived_from_volume() -> None:
    volume = np.ones((2, 2, 2), dtype=np.float32)
    data = DensityMapData(volume=volume)
    assert data.volume_stats.mean == 1.0
    assert data.volume_data.statistics is data.volume_stats
    assert data.volume_data.volume is data.volume


def test_rejects_non_3d_volume() -> None:
    with pytest.raises(ValueError, match="3D"):
        DensityMapData(volume=np.ones((2, 2), dtype=np.float32))


def test_rejects_non_finite_volume() -> None:
    volume = np.ones((2, 2, 2), dtype=np.float32)
    volume[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        DensityMapData(volume=volume)


def test_grid_mismatch_raises_on_construction() -> None:
    volume = np.ones((2, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="incompatible"):
        DensityMapData(volume=volume, crystallographic_info=_cryst((3, 3, 3)))


def test_with_volume_same_shape_keeps_cryst() -> None:
    volume = np.ones((2, 2, 2), dtype=np.float32)
    data = DensityMapData(volume=volume, crystallographic_info=_cryst((2, 2, 2)))
    replacement = volume * 2.0
    updated = data.with_volume(replacement)
    assert updated.volume_stats.mean == 2.0
    assert updated.crystallographic_info is data.crystallographic_info


def test_with_volume_shape_change_requires_new_cryst() -> None:
    volume = np.ones((2, 2, 2), dtype=np.float32)
    data = DensityMapData(volume=volume, crystallographic_info=_cryst((2, 2, 2)))
    larger = np.ones((3, 3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="incompatible"):
        data.with_volume(larger)
    updated = data.with_volume(larger, crystallographic_info=_cryst((3, 3, 3)))
    assert updated.volume.shape == (3, 3, 3)
    assert tuple(updated.crystallographic_info.grid.dimensions) == (3, 3, 3)


def test_replace_volume_refreshes_stats() -> None:
    volume = np.ones((2, 2, 2), dtype=np.float32)
    data = DensityMapData(volume=volume, crystallographic_info=_cryst((2, 2, 2)))
    data.replace_volume(volume * 3.0)
    assert data.volume_stats.mean == 3.0
    assert isinstance(data.volume_stats, VolumeStatistics)
    assert data.volume_data.statistics is data.volume_stats


def test_volume_stats_not_an_init_field() -> None:
    with pytest.raises(TypeError):
        DensityMapData(
            volume=np.ones((2, 2, 2), dtype=np.float32),
            volume_stats=VolumeStatistics(0, 0, 0, 0),  # type: ignore[call-arg]
        )


def test_zero_variance_mean_near_zero() -> None:
    stats_zero = VolumeStatistics.from_array(np.zeros((2, 2, 2), dtype=np.float32))
    assert stats_zero.std == 0.0
    assert stats_zero.is_mean_near_zero is True
    stats_const = VolumeStatistics.from_array(np.ones((2, 2, 2), dtype=np.float32))
    assert stats_const.std == 0.0
    assert stats_const.is_mean_near_zero is False

"""VolumeGeometry and _apply_symmetry_mates parameter-object API."""

from __future__ import annotations

import numpy as np

from molib.xtal.ccp4.map.globals import CCP4_HEADER_SIZE
from molib.xtal.map.helper import (
    CANONICAL_AXIS_ORDER,
    CCP4Map,
    VolumeGeometry,
    _apply_symmetry_mates,
)
from molib.xtal.map.density import AxisOrder


def _symop_chunk(text: bytes) -> bytes:
    """Pad a symmetry operator string to an 80-byte CCP4 record."""
    assert len(text) <= 80
    return text + b" " * (80 - len(text))


def _ccp4_map_with_symops(symops: bytes, *, n_grid=(4, 4, 4)) -> CCP4Map:
    nsymbt = len(symops)
    buffer = b"\x00" * CCP4_HEADER_SIZE + symops
    return CCP4Map(
        buffer=buffer,
        nsymbt=nsymbt,
        n_grid=tuple(n_grid),
        start=(0, 0, 0),
        end=tuple(n_grid),
    )


def test_volume_geometry_from_ccp4_map() -> None:
    ccp4_map = CCP4Map(
        buffer=b"",
        nsymbt=0,
        n_grid=(10, 20, 30),
        start=(1, 2, 3),
        end=(11, 22, 33),
    )
    geometry = VolumeGeometry.from_ccp4_map(ccp4_map, [4, 5, 6])
    assert geometry.n_grid == (10, 20, 30)
    assert geometry.start == (1, 2, 3)
    assert geometry.end == (11, 22, 33)
    assert geometry.center_offset == (4, 5, 6)


def test_canonical_axis_order_is_xyz() -> None:
    assert CANONICAL_AXIS_ORDER is AxisOrder.XYZ
    assert (
        CANONICAL_AXIS_ORDER.x_axis,
        CANONICAL_AXIS_ORDER.y_axis,
        CANONICAL_AXIS_ORDER.z_axis,
    ) == (0, 1, 2)


def test_apply_symmetry_mates_applies_inversion() -> None:
    volume = np.ones((4, 4, 4), dtype=np.float32)
    expanded = np.zeros((8, 8, 8), dtype=np.float32)
    center = (2, 2, 2)
    expanded[
        center[0] : center[0] + 4,
        center[1] : center[1] + 4,
        center[2] : center[2] + 4,
    ] = volume

    symops = _symop_chunk(b"x,y,z") + _symop_chunk(b"-x,-y,-z")
    ccp4_map = _ccp4_map_with_symops(symops)
    geometry = VolumeGeometry.from_ccp4_map(ccp4_map, center)

    count = _apply_symmetry_mates(volume, expanded, ccp4_map, geometry)
    assert count == 1
    assert np.count_nonzero(expanded) > np.count_nonzero(volume)


def test_apply_symmetry_mates_skips_identity_only() -> None:
    volume = np.ones((4, 4, 4), dtype=np.float32)
    expanded = np.zeros((8, 8, 8), dtype=np.float32)
    symops = _symop_chunk(b"x,y,z") + _symop_chunk(b"x,y,z")
    ccp4_map = _ccp4_map_with_symops(symops)
    geometry = VolumeGeometry.from_ccp4_map(ccp4_map, (2, 2, 2))

    count = _apply_symmetry_mates(volume, expanded, ccp4_map, geometry)
    assert count == 0


def test_apply_symmetry_mates_skips_malformed_symop() -> None:
    volume = np.ones((4, 4, 4), dtype=np.float32)
    expanded = np.zeros((8, 8, 8), dtype=np.float32)
    # Not three comma-separated terms -> ValueError from parser
    symops = _symop_chunk(b"not-a-valid-symop")
    ccp4_map = _ccp4_map_with_symops(symops)
    geometry = VolumeGeometry.from_ccp4_map(ccp4_map, (2, 2, 2))

    count = _apply_symmetry_mates(volume, expanded, ccp4_map, geometry)
    assert count == 0

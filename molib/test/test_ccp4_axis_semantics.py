"""CCP4 CRS/XYZ axis semantics, header validation, and pure grid conversion."""

from __future__ import annotations

import struct
from types import SimpleNamespace
from unittest.mock import mock_open, patch

import numpy as np
import pytest

from molib.xtal.ccp4.map.globals import CCP4_HEADER_SIZE
from molib.xtal.map.density import AxisOrder, CrystallographicInfo, GridOrigin, GridSpacing
from molib.xtal.map.helper import (
    CCP4Map,
    CCP4MiniHeader,
    _apply_mask_and_finalize,
    _build_header_from_ccp4_map,
    _crs_to_xyz_start_size,
    _grid_to_xyz_array,
    _xyz_to_dict,
    expand_ccp4_symmetry,
    read_symmetry_ops,
)


def _pack_ccp4_header(
    *,
    nc: int = 10,
    nr: int = 20,
    ns: int = 30,
    ncstart: int = 1,
    nrstart: int = 2,
    nsstart: int = 3,
    nx: int = 10,
    ny: int = 20,
    nz: int = 30,
    mapc: int = 1,
    mapr: int = 2,
    maps: int = 3,
    nsymbt: int = 0,
) -> bytes:
    """Build a little-endian 1024-byte CCP4 header with selected int fields."""
    ints = [0] * 256
    ints[0], ints[1], ints[2] = nc, nr, ns
    ints[4], ints[5], ints[6] = ncstart, nrstart, nsstart
    ints[7], ints[8], ints[9] = nx, ny, nz
    ints[16], ints[17], ints[18] = mapc, mapr, maps
    ints[23] = nsymbt
    return struct.pack("<256i", *ints)


def test_crs_to_xyz_standard_mapc() -> None:
    header = CCP4MiniHeader(
        nsymbt=0,
        nc=10,
        nr=20,
        ns=30,
        nx=10,
        ny=20,
        nz=30,
        ncstart=1,
        nrstart=2,
        nsstart=3,
        mapc=1,
        mapr=2,
        maps=3,
    )
    start, size = _crs_to_xyz_start_size(header)
    assert start == (1, 2, 3)
    assert size == (10, 20, 30)


def test_crs_to_xyz_nonstandard_mapc() -> None:
    """MAPC,MAPR,MAPS = 3,2,1 remaps CRS starts/sizes into XYZ."""
    header = CCP4MiniHeader(
        nsymbt=0,
        nc=10,
        nr=20,
        ns=30,
        nx=30,
        ny=20,
        nz=10,
        ncstart=5,
        nrstart=6,
        nsstart=7,
        mapc=3,
        mapr=2,
        maps=1,
    )
    start, size = _crs_to_xyz_start_size(header)
    # mapc=3 -> X gets nsstart/ns; mapr=2 -> Y gets nrstart/nr; maps=1 -> Z gets ncstart/nc
    assert start == (7, 6, 5)
    assert size == (30, 20, 10)


def test_read_symmetry_ops_remaps_start_end() -> None:
    header = CCP4MiniHeader(
        nsymbt=80,
        nc=4,
        nr=5,
        ns=6,
        nx=6,
        ny=5,
        nz=4,
        ncstart=1,
        nrstart=2,
        nsstart=3,
        mapc=3,
        mapr=2,
        maps=1,
    )
    buffer = _pack_ccp4_header(
        nc=4,
        nr=5,
        ns=6,
        ncstart=1,
        nrstart=2,
        nsstart=3,
        nx=6,
        ny=5,
        nz=4,
        mapc=3,
        mapr=2,
        maps=1,
        nsymbt=80,
    ) + (b"x,y,z" + b" " * 75)
    with patch("builtins.open", mock_open(read_data=buffer)):
        ccp4_map = read_symmetry_ops("fake.map", header)
    assert ccp4_map.n_grid == (6, 5, 4)
    assert ccp4_map.start == (3, 2, 1)
    assert ccp4_map.end == (3 + 6, 2 + 5, 1 + 4)
    assert ccp4_map.nsymbt == 80


def test_read_symmetry_ops_rejects_negative_nsymbt() -> None:
    header = CCP4MiniHeader(nsymbt=-1, nx=1, ny=1, nz=1, nc=1, nr=1, ns=1)
    with patch("builtins.open", mock_open(read_data=b"\x00" * CCP4_HEADER_SIZE)):
        with pytest.raises(ValueError, match="NSYMBT"):
            read_symmetry_ops("fake.map", header)


def test_read_symmetry_ops_rejects_short_file() -> None:
    header = CCP4MiniHeader(nsymbt=80, nx=1, ny=1, nz=1, nc=1, nr=1, ns=1)
    with patch("builtins.open", mock_open(read_data=b"\x00" * 10)):
        with pytest.raises(ValueError, match="shorter"):
            read_symmetry_ops("fake.map", header)


def test_build_header_from_raw_ccp4_bytes() -> None:
    raw = _pack_ccp4_header(
        nc=8,
        nr=9,
        ns=10,
        ncstart=11,
        nrstart=12,
        nsstart=13,
        nx=8,
        ny=9,
        nz=10,
        mapc=1,
        mapr=2,
        maps=3,
        nsymbt=160,
    )
    ccp4_map = SimpleNamespace(ccp4_header=raw)
    header = _build_header_from_ccp4_map(ccp4_map)
    assert header.nc == 8
    assert header.nr == 9
    assert header.ns == 10
    assert header.ncstart == 11
    assert header.nrstart == 12
    assert header.nsstart == 13
    assert header.nx == 8
    assert header.nsymbt == 160
    assert (header.mapc, header.mapr, header.maps) == (1, 2, 3)


def test_build_header_rejects_invalid_mapc() -> None:
    raw = _pack_ccp4_header(mapc=1, mapr=1, maps=1)
    with pytest.raises(ValueError, match="axis mapping"):
        _build_header_from_ccp4_map(SimpleNamespace(ccp4_header=raw))


def test_expand_uses_ccp4_map_attributes() -> None:
    """Attribute access (not tuple unpack) must drive expansion."""
    volume = np.ones((4, 4, 4), dtype=np.float32)
    header = CCP4MiniHeader(
        nsymbt=160,
        nc=4,
        nr=4,
        ns=4,
        nx=4,
        ny=4,
        nz=4,
        mapc=1,
        mapr=2,
        maps=3,
    )
    symops = (b"x,y,z" + b" " * 75) + (b"-x,-y,-z" + b" " * 72)
    buffer = _pack_ccp4_header(nc=4, nr=4, ns=4, nx=4, ny=4, nz=4, nsymbt=160) + symops
    with patch("builtins.open", mock_open(read_data=buffer)):
        result = expand_ccp4_symmetry(volume, "fake.map", header)
    assert result is not volume
    assert result.shape == (8, 8, 8)


def test_ccp4_map_requires_fields() -> None:
    with pytest.raises(TypeError):
        CCP4Map()  # type: ignore[call-arg]


def test_sync_to_xyz_volume_updates_axis_and_dimensions() -> None:
    from molib.xtal.map.density import CoordinateTransforms, MapGrid, UnitCell

    info = CrystallographicInfo(
        unit_cell=UnitCell(a=1, b=1, c=1, alpha=90, beta=90, gamma=90),
        space_group="P1",
        grid=MapGrid(
            dimensions=(2, 3, 4),
            origin=GridOrigin(0.0, 0.0, 0.0),
            spacing=GridSpacing(1.0, 1.0, 1.0),
            axis_order=AxisOrder.ZYX,
        ),
        transforms=CoordinateTransforms(
            frac_to_orth=np.eye(4),
            orth_to_frac=np.eye(4),
        ),
    )
    volume = np.zeros((4, 3, 2), dtype=np.float32)
    assert info.grid.axis_order is AxisOrder.ZYX
    info.sync_to_xyz_volume(volume)
    assert info.grid.axis_order is AxisOrder.XYZ
    assert info.grid.dimensions == (4, 3, 2)


def test_grid_to_xyz_wrapper_is_pure() -> None:
    """``_grid_to_xyz_array`` accepts only the grid (no metadata mutation arg)."""
    import inspect

    sig = inspect.signature(_grid_to_xyz_array)
    assert list(sig.parameters) == ["grid"]


def test_xyz_to_dict_accepts_mapping() -> None:
    assert _xyz_to_dict({"x": 1, "y": 2, "z": 3}) == {"x": 1.0, "y": 2.0, "z": 3.0}
    assert _xyz_to_dict(GridOrigin(1.0, 2.0, 3.0)) == {"x": 1.0, "y": 2.0, "z": 3.0}


def test_apply_mask_and_finalize_empty_map_no_divzero() -> None:
    density = np.zeros((2, 2, 2), dtype=np.float32)
    mask = np.zeros((2, 2, 2), dtype=bool)
    out = _apply_mask_and_finalize(
        density,
        mask,
        progress_callback=None,
        cutoff_distance=1.0,
        label="empty",
        guard_division=False,
    )
    assert out.shape == density.shape
    assert np.count_nonzero(out) == 0

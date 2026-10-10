"""Canonical-XYZ apply_symmetry_to_volume API."""

from __future__ import annotations

import numpy as np
import pytest

from molib.xtal.map.helper import (
    VolumeGeometry,
    _as_4x4_transform,
    apply_symmetry_to_volume,
)


def _geometry(
    *,
    shape=(4, 4, 4),
    start=(0, 0, 0),
    center_offset=(0, 0, 0),
) -> VolumeGeometry:
    return VolumeGeometry(
        n_grid=shape,
        start=start,
        end=tuple(s + n for s, n in zip(start, shape)),
        center_offset=center_offset,
    )


def test_as_4x4_transform_from_3x4() -> None:
    mat3x4 = [[1, 0, 0, 2], [0, 1, 0, 3], [0, 0, 1, 4]]
    out = _as_4x4_transform(mat3x4)
    assert out.shape == (4, 4)
    assert np.allclose(out[:3, :], mat3x4)
    assert np.allclose(out[3], [0, 0, 0, 1])


def test_as_4x4_transform_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="3x4 or 4x4"):
        _as_4x4_transform([[1, 0], [0, 1]])


def test_identity_transform_copies_into_offset_region() -> None:
    source = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    target = np.zeros((6, 6, 6), dtype=np.float32)
    geometry = _geometry(shape=(3, 3, 3), center_offset=(1, 2, 0))
    apply_symmetry_to_volume(source, target, np.eye(4), geometry)
    assert np.array_equal(target[1:4, 2:5, 0:3], source)


def test_out_of_bounds_transform_writes_nothing() -> None:
    source = np.ones((3, 3, 3), dtype=np.float32)
    target = np.zeros((4, 4, 4), dtype=np.float32)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = (100.0, 100.0, 100.0)
    apply_symmetry_to_volume(
        source,
        target,
        transform,
        _geometry(shape=(3, 3, 3)),
    )
    assert np.count_nonzero(target) == 0


def test_bad_transform_shape_raises() -> None:
    source = np.ones((2, 2, 2), dtype=np.float32)
    target = np.zeros((2, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="4x4"):
        apply_symmetry_to_volume(
            source,
            target,
            np.eye(3),
            _geometry(shape=(2, 2, 2)),
        )

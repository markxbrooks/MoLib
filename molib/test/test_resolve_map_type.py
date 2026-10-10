"""Tests for metadata-first :func:`resolve_map_type`."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from molib.xtal.ccp4.map.volume import VolumeData, resolve_map_type
from molib.xtal.map.density import MAP_NEGATIVE_RATIO_THRESHOLD
from molib.xtal.map.map_type import MapType


def _symmetric_difference_like_volume() -> np.ndarray:
    """Volume that looks Fo-Fc-like by statistics alone (high negative ratio)."""
    # Values spanning ±1.0 so |min|/max == 1.0 > threshold, mean near 0.
    return np.array(
        [[[-1.0, -0.8], [0.8, 1.0]], [[-0.9, -0.7], [0.7, 0.9]]],
        dtype=np.float64,
    )


def test_resolve_map_type_none_is_unknown() -> None:
    assert resolve_map_type(None) is MapType.UNKNOWN


def test_resolve_map_type_hint_difference() -> None:
    assert (
        resolve_map_type(None, map_type_hint=MapType.DIFFERENCE)
        is MapType.DIFFERENCE
    )
    assert resolve_map_type(None, map_type_hint="Fo-Fc") is MapType.DIFFERENCE


def test_resolve_map_type_metadata_fofc() -> None:
    cryst = SimpleNamespace(map_type="DELFWT / PHDELWT")
    assert (
        resolve_map_type(None, crystallographic_info=cryst) is MapType.DIFFERENCE
    )


def test_resolve_map_type_metadata_2fofc() -> None:
    cryst = SimpleNamespace(map_type="2Fo-Fc from FWT")
    assert resolve_map_type(None, crystallographic_info=cryst) is MapType.NORMAL


def test_resolve_map_type_hint_beats_metadata() -> None:
    cryst = SimpleNamespace(map_type="Fo-Fc")
    assert (
        resolve_map_type(
            None,
            crystallographic_info=cryst,
            map_type_hint=MapType.NORMAL,
        )
        is MapType.NORMAL
    )


def test_resolve_map_type_conservative_stats_default() -> None:
    volume = VolumeData(_symmetric_difference_like_volume())
    assert volume.detect_type() is MapType.DIFFERENCE
    assert abs(volume.statistics.min_value) / volume.statistics.max_value > (
        MAP_NEGATIVE_RATIO_THRESHOLD
    )
    assert resolve_map_type(volume) is MapType.NORMAL
    assert (
        resolve_map_type(volume, allow_statistical_difference=False)
        is MapType.NORMAL
    )


def test_resolve_map_type_allow_statistical_difference() -> None:
    volume = VolumeData(_symmetric_difference_like_volume())
    assert (
        resolve_map_type(volume, allow_statistical_difference=True)
        is MapType.DIFFERENCE
    )


def test_resolve_map_type_unknown_stats_become_normal() -> None:
    # Strictly positive density → detect_type UNKNOWN → resolve NORMAL.
    volume = VolumeData(np.ones((2, 2, 2), dtype=np.float64))
    assert volume.detect_type() is MapType.UNKNOWN
    assert resolve_map_type(volume) is MapType.NORMAL


@pytest.mark.parametrize(
    "label,expected",
    [
        ("2fofcwt", MapType.NORMAL),
        ("fofc", MapType.DIFFERENCE),
        ("difference map", MapType.DIFFERENCE),
    ],
)
def test_resolve_map_type_label_keywords(label: str, expected: MapType) -> None:
    cryst = SimpleNamespace(map_type=label)
    assert resolve_map_type(None, crystallographic_info=cryst) is expected

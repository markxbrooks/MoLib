"""Tests for load_ccp4_map_from_processing as a thin translator."""

from __future__ import annotations

from unittest.mock import patch

from molib.xtal.map.builders.processing import build_map_processing_settings
from molib.xtal.map.helper import load_ccp4_map_from_processing


def test_from_processing_forwards_settings_and_centroid():
    processing = build_map_processing_settings(
        carve_density=False,
        carve_density_centroid=True,
        carve_cutoff=3.5,
        centroid_cutoff=12.0,
        convert_to_cartesian=True,
    )
    progress = object()
    centroid = (1.0, 2.0, 3.0)
    expected = object()

    with patch(
        "molib.xtal.map.helper.load_ccp4_map",
        return_value=expected,
    ) as mock_load:
        result = load_ccp4_map_from_processing(
            "map.ccp4",
            processing,
            progress,
            pdb_centroid_or_clicked_position=centroid,
        )

    assert result is expected
    mock_load.assert_called_once_with(
        "map.ccp4",
        convert_to_cartesian=True,
        expand_symmetry=False,
        carve_density=False,
        carve_cutoff=3.5,
        progress_callback=progress,
        carve_density_centroid=True,
        pdb_centroid_or_clicked_position=centroid,
        centroid_cutoff=12.0,
    )


def test_from_processing_allows_none_centroid():
    processing = build_map_processing_settings()
    with patch(
        "molib.xtal.map.helper.load_ccp4_map",
        return_value=None,
    ) as mock_load:
        load_ccp4_map_from_processing("map.ccp4", processing, None)

    kwargs = mock_load.call_args.kwargs
    assert kwargs["pdb_centroid_or_clicked_position"] is None
    assert kwargs["expand_symmetry"] is False

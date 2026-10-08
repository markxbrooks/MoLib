"""MapInfo visibility and render-state ownership."""

from __future__ import annotations

import numpy as np

from molib.xtal.map.builder import build_map_info
from molib.xtal.map.map_type import MapType
from molib.xtal.map.manager import MapManager


def test_fofc_is_visible_when_only_positive_lobe():
    """Fo-Fc maps report visible if either difference lobe is on."""
    map_info = build_map_info(
        crystallographic_info=None,
        map_id="diff",
        map_type=MapType.DIFFERENCE,
        f_label="DELFWT",
        phi_label="PHDELWT",
        volume=np.ones((3, 3, 3), dtype=np.float32),
        sigma_level=3.0,
    )
    map_info.render.settings.normal.is_visible = False
    map_info.render.settings.positive.is_visible = True
    map_info.render.settings.negative.is_visible = False
    assert map_info.is_visible is True


def test_fofc_is_visible_setter_toggles_both_lobes():
    map_info = build_map_info(
        crystallographic_info=None,
        map_id="diff",
        map_type=MapType.DIFFERENCE,
        f_label="DELFWT",
        phi_label="PHDELWT",
        volume=np.ones((3, 3, 3), dtype=np.float32),
        sigma_level=3.0,
    )
    map_info.is_visible = False
    assert map_info.render.settings.positive.is_visible is False
    assert map_info.render.settings.negative.is_visible is False
    map_info.is_visible = True
    assert map_info.render.settings.positive.is_visible is True
    assert map_info.render.settings.negative.is_visible is True


def test_twofofc_sigma_facade_reads_render_bundle():
    map_info = build_map_info(
        crystallographic_info=None,
        map_id="2fofc",
        map_type=MapType.NORMAL,
        f_label="FWT",
        phi_label="PHWT",
        volume=np.ones((3, 3, 3), dtype=np.float32),
        sigma_level=1.0,
    )
    map_info.render.sigma_level = 2.25
    assert map_info.sigma_level == 2.25
    assert map_info.render.settings.normal.sigma_level == 2.25


def test_difference_sigma_setter_syncs_lobes():
    map_info = build_map_info(
        map_id="diff",
        map_type=MapType.DIFFERENCE,
        volume=np.ones((3, 3, 3), dtype=np.float32),
        sigma_level=2.5,
    )
    map_info.sigma_level = 1.5
    assert map_info.positive_sigma_level == 1.5
    assert map_info.negative_sigma_level == -1.5
    manager = MapManager()
    manager.add_map_from_map_info(map_info)
    manager.update_map_sigma_level("diff", 2.0)
    assert map_info.positive_sigma_level == 2.0
    assert map_info.negative_sigma_level == -2.0

"""
build map info
"""

from __future__ import annotations

from typing import Any

from numpy import ndarray

from molib.xtal.map.density import CrystallographicInfo, MapType
from molib.xtal.map.info import MapInfo

DEFAULT_2FOFC_SIGMA_LEVEL = 0.2
DEFAULT_FOFC_SIGMA_LEVEL = 1.0


def default_sigma_level_for_map(
    map_type: str, *, is_difference_map: bool = False
) -> float:
    """Return the default contour sigma for a map type.

    :param map_type: map type label (e.g. ``2Fo-Fc``, ``Fo-Fc``)
    :param is_difference_map: whether the map is a difference map
    :return: default sigma level
    """
    if is_difference_map:
        return DEFAULT_FOFC_SIGMA_LEVEL
    normalized = str(map_type).strip().lower().replace("_", "-")
    if normalized.startswith("2fo") or normalized.startswith("2mfo"):
        return DEFAULT_2FOFC_SIGMA_LEVEL
    if normalized in {"2fofc", "fwt"}:
        return DEFAULT_2FOFC_SIGMA_LEVEL
    if normalized in {"fo-fc", "fofc", "delfwt", "difference"}:
        return DEFAULT_FOFC_SIGMA_LEVEL
    return DEFAULT_FOFC_SIGMA_LEVEL


def build_map_info(
    crystallographic_info: CrystallographicInfo,
    f_label: str = "2FOFCWT",
    is_difference_map: bool = False,
    map_id: str = "map",
    map_type: str = "2Fo-Fc",
    phi_label: str = "PH2FOFCWT",
    volume: ndarray = None,
    sigma_level: float | None = None,
) -> MapInfo:
    """build map info"""
    if sigma_level is None:
        sigma_level = default_sigma_level_for_map(
            map_type, is_difference_map=is_difference_map
        )
    return MapInfo(
        map_id=map_id,
        map_type=map_type,
        f_label=f_label,
        phi_label=phi_label,
        volume=volume,
        crystallographic_info=crystallographic_info,
        is_difference_map=is_difference_map,
        sigma_level=sigma_level,
        description=f"{map_type} map ({f_label}/{phi_label})",
    )


def build_map_information_specs(
    crystallographic_info, mtz_id: str, volume_2fofc, volume_fofc
) -> dict[str, Any]:
    """build map information specs"""
    map_information_specs: dict[str, MapInfo] = {
        "map_2fofc_info": build_map_info(
            map_id=f"{mtz_id}_2Fo-Fc",
            map_type=MapType.TWO_FO_FC,
            f_label="2FOFCWT",
            phi_label="PH2FOFCWT",
            volume=volume_2fofc,
            crystallographic_info=crystallographic_info,
            is_difference_map=False,
            sigma_level=DEFAULT_2FOFC_SIGMA_LEVEL,
        ),
        "map_fofc_info": build_map_info(
            map_id=f"{mtz_id}_Fo-Fc",
            map_type="Fo-Fc",
            f_label="DELFWT",
            phi_label="PHDELWT",
            volume=volume_fofc,
            crystallographic_info=crystallographic_info,
            is_difference_map=True,
            sigma_level=DEFAULT_FOFC_SIGMA_LEVEL,
        ),
    }
    return map_information_specs

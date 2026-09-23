"""
build map info
"""

from __future__ import annotations

from numpy import ndarray

from molib.xtal.map.density import CrystallographicInfo
from molib.xtal.map.info import MapInfo


def build_map_info(
    crystallographic_info: CrystallographicInfo,
    f_label: str = "2FOFCWT",
    is_difference_map: bool = False,
    map_id: str = "map",
    map_type: str = "2Fo-Fc",
    phi_label: str = "PH2FOFCWT",
    volume: ndarray = None,
) -> MapInfo:
    """build map info"""
    return MapInfo(
        map_id=map_id,
        map_type=map_type,
        f_label=f_label,
        phi_label=phi_label,
        volume=volume,
        crystallographic_info=crystallographic_info,
        is_difference_map=is_difference_map,
        description=f"{map_type} map ({f_label}/{phi_label})",
    )


def build_map_information_specs(
    crystallographic_info, mtz_id: str, volume_2fofc, volume_fofc
) -> dict[str, Any]:
    """build map information specs"""
    map_information_specs: dict[str, Mapinfo] = {
        "map_2fofc_info": build_map_info(
            map_id=f"{mtz_id}_2Fo-Fc",
            map_type="2Fo-Fc",
            f_label="2FOFCWT",
            phi_label="PH2FOFCWT",
            volume=volume_2fofc,
            crystallographic_info=crystallographic_info,
        ),
        "map_fofc_info": build_map_info(
            map_id=f"{mtz_id}_Fo-Fc",
            map_type="Fo-Fc",
            f_label="DELFWT",
            phi_label="PHDELWT",
            volume=volume_fofc,
            crystallographic_info=crystallographic_info,
        ),
    }
    return map_information_specs
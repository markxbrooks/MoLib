"""
build map info
"""

from __future__ import annotations

from typing import Any

from numpy import ndarray

from molib.xtal.map.builders.processing import build_map_processing_settings
from molib.xtal.map.density import CrystallographicInfo
from molib.xtal.map.map_type import MapType
from molib.xtal.map.spec import DensityMapSpec
from molib.xtal.map.info import MapInfo, MapRenderSettings, MapPolaritySettings, MapProcessingSettings
from molib.xtal.map.render.mode import IsosurfaceMapRenderMode

DEFAULT_2FOFC_SIGMA_LEVEL = 1.0
DEFAULT_FOFC_SIGMA_LEVEL = 2.5


def default_sigma_level_for_map(
    map_type: MapType | str, *, is_difference_map: bool | None = None
) -> float:
    """Return the default contour sigma for a map type.

    :param map_type: map type label or :class:`MapType`
    :param is_difference_map: optional override; when ``None``, derived from type
    :return: default sigma level
    """
    try:
        resolved = MapType.coerce(map_type)
    except ValueError:
        resolved = None
    if is_difference_map is None and resolved is not None:
        is_difference_map = resolved is MapType.DIFFERENCE
    if is_difference_map:
        return DEFAULT_FOFC_SIGMA_LEVEL
    if resolved is MapType.NORMAL:
        return DEFAULT_2FOFC_SIGMA_LEVEL
    normalized = str(getattr(map_type, "value", map_type)).strip().lower().replace(
        "_", "-"
    )
    if normalized.startswith("2fo") or normalized.startswith("2mfo"):
        return DEFAULT_2FOFC_SIGMA_LEVEL
    if normalized in {"2fofc", "fwt"}:
        return DEFAULT_2FOFC_SIGMA_LEVEL
    if normalized in {"fo-fc", "fofc", "delfwt", "difference"}:
        return DEFAULT_FOFC_SIGMA_LEVEL
    return DEFAULT_FOFC_SIGMA_LEVEL


def build_map_info(
    crystallographic_info: CrystallographicInfo | None = None,
    f_label: str = "2FOFCWT",
    is_difference_map: bool | None = None,
    map_id: str = "map",
    map_type: MapType | str = MapType.NORMAL,
    phi_label: str = "PH2FOFCWT",
    volume: ndarray = None,
    sigma_level: float | None = None,
    render_mode: IsosurfaceMapRenderMode | str | None = None,
) -> MapInfo:
    """Build a :class:`MapInfo` with coerced :class:`MapType`."""

    resolved_type = MapType.coerce(map_type)

    if is_difference_map is True and resolved_type is not MapType.DIFFERENCE:
        resolved_type = MapType.DIFFERENCE

    if sigma_level is None:
        sigma_level = default_sigma_level_for_map(resolved_type)

    mode = (
        IsosurfaceMapRenderMode.coerce(render_mode)
        if render_mode is not None
        else IsosurfaceMapRenderMode.UNIT_CELL
    )
    type_label = resolved_type.value
    settings = MapPolaritySettings.for_map(
        resolved_type,
        float(sigma_level),
    )

    return MapInfo(
        map_id=map_id,
        map_type=resolved_type,
        f_label=f_label,
        phi_label=phi_label,
        volume=volume,
        crystallographic_info=crystallographic_info,
        description=f"{type_label} map ({f_label}/{phi_label})",
        render=MapRenderSettings(
            mode=mode,
            settings=settings,
        ),
    )


def build_map_information_specs(
    crystallographic_info, mtz_id: str, volume_2fofc, volume_fofc
) -> dict[str, Any]:
    """build map information specs"""
    map_information_specs: dict[str, MapInfo] = {
        "map_2fofc_info": build_map_info(
            map_id=f"{mtz_id}_2Fo-Fc",
            map_type=MapType.NORMAL,
            f_label="2FOFCWT",
            phi_label="PH2FOFCWT",
            volume=volume_2fofc,
            crystallographic_info=crystallographic_info,
            sigma_level=DEFAULT_2FOFC_SIGMA_LEVEL,
        ),
        "map_fofc_info": build_map_info(
            map_id=f"{mtz_id}_Fo-Fc",
            map_type=MapType.DIFFERENCE,
            f_label="DELFWT",
            phi_label="PHDELWT",
            volume=volume_fofc,
            crystallographic_info=crystallographic_info,
            sigma_level=DEFAULT_FOFC_SIGMA_LEVEL,
        ),
    }
    return map_information_specs


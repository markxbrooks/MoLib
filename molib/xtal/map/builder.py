"""
build map info
"""

from __future__ import annotations

from typing import Any

from numpy import ndarray

from molib.xtal.map.density import CrystallographicInfo, MapType
from molib.xtal.map.info import MapInfo, MapRenderSettings
from molib.xtal.map.render.mode import MapRenderMode

DEFAULT_2FOFC_SIGMA_LEVEL = 0.2
DEFAULT_FOFC_SIGMA_LEVEL = 1.0


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
        is_difference_map = resolved is MapType.FO_FC
    if is_difference_map:
        return DEFAULT_FOFC_SIGMA_LEVEL
    if resolved is MapType.TWO_FO_FC:
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
    map_type: MapType | str = MapType.TWO_FO_FC,
    phi_label: str = "PH2FOFCWT",
    volume: ndarray = None,
    sigma_level: float | None = None,
    render_mode: MapRenderMode | str | None = None,
) -> MapInfo:
    """Build a :class:`MapInfo` with coerced :class:`MapType`.

    ``is_difference_map`` is kept for call-site compatibility. When ``True`` and
    ``map_type`` still resolves to 2Fo-Fc, the type is forced to Fo-Fc. Difference
    status on the result is always derived from ``map_type``.
    """
    resolved_type = MapType.coerce(map_type)
    if is_difference_map is True and resolved_type is not MapType.FO_FC:
        resolved_type = MapType.FO_FC
    if sigma_level is None:
        sigma_level = default_sigma_level_for_map(resolved_type)
    mode = (
        MapRenderMode.coerce(render_mode)
        if render_mode is not None
        else MapRenderMode.ISOSURFACE
    )
    type_label = resolved_type.value
    return MapInfo(
        map_id=map_id,
        map_type=resolved_type,
        f_label=f_label,
        phi_label=phi_label,
        volume=volume,
        crystallographic_info=crystallographic_info,
        description=f"{type_label} map ({f_label}/{phi_label})",
        render=MapRenderSettings(mode=mode, sigma_level=float(sigma_level)),
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
            sigma_level=DEFAULT_2FOFC_SIGMA_LEVEL,
        ),
        "map_fofc_info": build_map_info(
            map_id=f"{mtz_id}_Fo-Fc",
            map_type=MapType.FO_FC,
            f_label="DELFWT",
            phi_label="PHDELWT",
            volume=volume_fofc,
            crystallographic_info=crystallographic_info,
            sigma_level=DEFAULT_FOFC_SIGMA_LEVEL,
        ),
    }
    return map_information_specs

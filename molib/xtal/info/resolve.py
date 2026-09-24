"""
Resolve CrystallographicInfo to dict
"""
from typing import Any

from decologr import Decologr as log
from molib.xtal.map.density import CrystallographicInfo, GridOrigin, GridSpacing


def normalize_crystallographic_info_to_dict(crystallographic_info: CrystallographicInfo | dict) -> dict:
    """normalize crystallographic info to dict"""
    if isinstance(crystallographic_info, CrystallographicInfo):
        log.message("crystallographic_info is a CrystallographicInfo object")
        return crystallographic_info.to_dict()
    elif isinstance(crystallographic_info, dict):
        log.message("crystallographic_info is a dict")
        return crystallographic_info
    else:
        raise TypeError(f"crystallographic_info is of type {type(crystallographic_info)}")


def normalize_crystallographic_info_from_dict(crystallographic_info: CrystallographicInfo | dict) -> CrystallographicInfo:
    """normalize crystallographic info from dict"""
    if isinstance(crystallographic_info, CrystallographicInfo):
        log.message("crystallographic_info is a CrystallographicInfo object")
        return crystallographic_info
    elif isinstance(crystallographic_info, dict):
        log.message("crystallographic_info was converted from a dict")
        return CrystallographicInfo.from_dict(crystallographic_info)
    else:
        raise TypeError(f"crystallographic_info is of type {type(crystallographic_info)}")


def normalize_grid_objects(grid_origin: dict[Any, Any], grid_spacing: dict[Any, Any]) -> tuple[GridOrigin, GridSpacing]:
    """Return both grid spacing and origin for proper coordinate alignment"""
    from molib.xtal.map.density import GridOrigin, GridSpacing
    if not isinstance(grid_spacing, GridSpacing):
        grid_spacing = GridSpacing(grid_spacing.x, grid_spacing.y, grid_spacing.z)
    if not isinstance(grid_origin, GridOrigin):
        grid_origin = GridOrigin(grid_origin['x'], grid_origin['y'], grid_origin['z'])
    return grid_origin, grid_spacing

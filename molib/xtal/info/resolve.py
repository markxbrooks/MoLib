"""
Resolve CrystallographicInfo to dict
"""
from decologr import Decologr as log
from molib.xtal.map.density import CrystallographicInfo


def resolve_crystallographic_info_to_dict(crystallographic_info: CrystallographicInfo | dict) -> dict:
    """resolve crystallographic info to dict"""
    if isinstance(crystallographic_info, CrystallographicInfo):
        log.message("crystallographic_info is a CrystallographicInfo object")
        crystallographic_info = crystallographic_info.to_dict()
    elif isinstance(crystallographic_info, dict):
        log.message("crystallographic_info is a dict")
    else:
        raise TypeError(f"crystallographic_info is of type {type(crystallographic_info)}")
    return crystallographic_info
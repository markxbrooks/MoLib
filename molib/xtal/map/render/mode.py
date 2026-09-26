"""
Defines rendering modes for maps as string enumerations.

This module provides the `MapRenderMode` class, which specifies
predefined rendering modes for maps using string values. These rendering
modes dictate how a map is rendered in various contexts.
"""
from picogl.utils.strenum import StrEnum


class MapRenderMode(StrEnum):
    """
    Represents different rendering modes for a map.

    This class defines a set of rendering modes that can be used to
    control the appearance and behavior of maps in a rendering context.
    Each rendering mode is a predefined string value, allowing precise
    specification of the desired mode for rendering operations.
    """
    ISOSURFACE: str = "isosurface"
    CONTINUOUS_LOCAL: str = "continuous_local"

    ALL = [
        ISOSURFACE,
        CONTINUOUS_LOCAL
    ]

    
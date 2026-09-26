"""
Defines rendering modes for maps as string enumerations.

This module provides the `MapRenderMode` class, which specifies
predefined rendering modes for maps using string values. These rendering
modes dictate how a map is rendered in various contexts.
"""
from __future__ import annotations

from enum import Enum


class MapRenderMode(str, Enum):
    """Rendering strategy for electron-density maps."""

    ISOSURFACE = "isosurface"
    CONTINUOUS_LOCAL = "continuous_local"

    @property
    def requires_mesh(self) -> bool:
        """Whether this mode draws from full-volume isosurface meshes."""
        return self is MapRenderMode.ISOSURFACE

    @property
    def supports_density_coloring(self) -> bool:
        """Whether this mode samples local density continuously."""
        return self is MapRenderMode.CONTINUOUS_LOCAL

    @classmethod
    def coerce(cls, value: "MapRenderMode | str") -> "MapRenderMode":
        """Normalize a mode string or enum member."""
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip())
        except ValueError as exc:
            raise ValueError(
                f"Unsupported map render mode: {value!r}. "
                f"Expected one of: {[m.value for m in cls]}"
            ) from exc


MAP_RENDER_MODES: tuple[MapRenderMode, ...] = (
    MapRenderMode.ISOSURFACE,
    MapRenderMode.CONTINUOUS_LOCAL,
)

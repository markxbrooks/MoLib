"""
Info for electron density maps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from molib.xtal.map.density import CrystallographicInfo, MapType
from molib.xtal.map.render.mode import MapRenderMode
from picogl.core.rgbcolor import RGBTuple, RGB

DEFAULT_MAP_COLOR: RGB = RGBTuple.BLUE
DEFAULT_POSITIVE_COLOR: RGB = RGBTuple.GREEN
DEFAULT_NEGATIVE_COLOR: RGB = RGBTuple.RED


@dataclass
class MapRenderSettings:
    """How the user wants a map displayed (strategy + contour/colour state)."""

    mode: MapRenderMode = MapRenderMode.ISOSURFACE
    sigma_level: float = 1.0
    is_visible: bool = True
    color: RGBTuple = DEFAULT_MAP_COLOR
    positive_visible: bool = True
    negative_visible: bool = True
    positive_color: RGBTuple = DEFAULT_POSITIVE_COLOR
    negative_color: RGBTuple = DEFAULT_NEGATIVE_COLOR
    positive_sigma_level: Optional[float] = None
    negative_sigma_level: Optional[float] = None
    # Per-map load / carve options (not session-global)
    carve_density: bool = False
    carve_density_centroid: bool = False
    carve_cutoff: float = 4.0
    centroid_cutoff: float = 15.0
    convert_to_cartesian: bool = True


@dataclass
class MapInfo:
    """Scientific identity and density for an electron-density map.

    Display state lives on :attr:`render`. Compatibility properties
    (``sigma_level``, ``color``, …) read/write through ``render``.
    """

    map_id: str
    map_type: MapType
    f_label: str
    phi_label: str
    volume: np.ndarray
    crystallographic_info: Optional[CrystallographicInfo] = None
    description: str = ""
    render: MapRenderSettings = field(default_factory=MapRenderSettings)

    def __post_init__(self) -> None:
        if self.render is None:
            self.render = MapRenderSettings()

    @property
    def is_difference_map(self) -> bool:
        """Whether this map is a Fo-Fc (difference) map."""
        return self.map_type is MapType.FO_FC

    # --- display shims (delegate to render) ---

    @property
    def sigma_level(self) -> float:
        return self.render.sigma_level

    @sigma_level.setter
    def sigma_level(self, value: float) -> None:
        self.render.sigma_level = float(value)

    @property
    def is_visible(self) -> bool:
        return self.render.is_visible

    @is_visible.setter
    def is_visible(self, value: bool) -> None:
        self.render.is_visible = bool(value)

    @property
    def color(self) -> RGBTuple:
        return self.render.color

    @color.setter
    def color(self, value: RGBTuple) -> None:
        self.render.color = value

    @property
    def positive_visible(self) -> bool:
        return self.render.positive_visible

    @positive_visible.setter
    def positive_visible(self, value: bool) -> None:
        self.render.positive_visible = bool(value)

    @property
    def negative_visible(self) -> bool:
        return self.render.negative_visible

    @negative_visible.setter
    def negative_visible(self, value: bool) -> None:
        self.render.negative_visible = bool(value)

    @property
    def positive_color(self) -> RGBTuple:
        return self.render.positive_color

    @positive_color.setter
    def positive_color(self, value: RGBTuple) -> None:
        self.render.positive_color = value

    @property
    def negative_color(self) -> RGBTuple:
        return self.render.negative_color

    @negative_color.setter
    def negative_color(self, value: RGBTuple) -> None:
        self.render.negative_color = value

    @property
    def positive_sigma_level(self) -> Optional[float]:
        return self.render.positive_sigma_level

    @positive_sigma_level.setter
    def positive_sigma_level(self, value: Optional[float]) -> None:
        self.render.positive_sigma_level = value

    @property
    def negative_sigma_level(self) -> Optional[float]:
        return self.render.negative_sigma_level

    @negative_sigma_level.setter
    def negative_sigma_level(self, value: Optional[float]) -> None:
        self.render.negative_sigma_level = value

    @property
    def carve_density(self) -> bool:
        return self.render.carve_density

    @carve_density.setter
    def carve_density(self, value: bool) -> None:
        self.render.carve_density = bool(value)

    @property
    def carve_density_centroid(self) -> bool:
        return self.render.carve_density_centroid

    @carve_density_centroid.setter
    def carve_density_centroid(self, value: bool) -> None:
        self.render.carve_density_centroid = bool(value)

    @property
    def carve_cutoff(self) -> float:
        return self.render.carve_cutoff

    @carve_cutoff.setter
    def carve_cutoff(self, value: float) -> None:
        self.render.carve_cutoff = float(value)

    @property
    def centroid_cutoff(self) -> float:
        return self.render.centroid_cutoff

    @centroid_cutoff.setter
    def centroid_cutoff(self, value: float) -> None:
        self.render.centroid_cutoff = float(value)

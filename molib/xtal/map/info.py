"""
Info for electron density maps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from molib.xtal.map.crystal import CrystallographicInfo
from molib.xtal.map.map_type import MapType
from molib.xtal.map.render.mode import IsosurfaceMapRenderMode
from picogl.core.rgbcolor import RGBTuple, RGB

DEFAULT_MAP_COLOR: RGB = RGBTuple.BLUE
DEFAULT_POSITIVE_COLOR: RGB = RGBTuple.GREEN
DEFAULT_NEGATIVE_COLOR: RGB = RGBTuple.RED


@dataclass(slots=True)
class MapProcessingSettings:
    """Settings controlling density-map processing."""

    carve_density: bool | None = True
    carve_cutoff: float | None = 4.0
    carve_density_centroid: bool | None = False
    centroid_cutoff: float | None = 15.0
    convert_to_cartesian: bool | None = False

    def with_overrides(
        self,
        *,
        carve_density: bool | None = None,
        carve_density_centroid: bool | None = None,
        carve_cutoff: float | None = None,
        centroid_cutoff: float | None = None,
        convert_to_cartesian: bool | None = None,
    ) -> "MapProcessingSettings":
        """Return a copy with non-None values overridden."""

        return MapProcessingSettings(
            carve_density=(
                self.carve_density
                if carve_density is None
                else carve_density
            ),
            carve_density_centroid=(
                self.carve_density_centroid
                if carve_density_centroid is None
                else carve_density_centroid
            ),
            carve_cutoff=(
                self.carve_cutoff
                if carve_cutoff is None
                else float(carve_cutoff)
            ),
            centroid_cutoff=(
                self.centroid_cutoff
                if centroid_cutoff is None
                else float(centroid_cutoff)
            ),
            convert_to_cartesian=(
                self.convert_to_cartesian
                if convert_to_cartesian is None
                else convert_to_cartesian
            ),
        )


@dataclass
class MapContourSettings:
    """Map Contour Settings"""
    is_visible: bool = True
    sigma_level: float = 1.0
    color: RGB = DEFAULT_MAP_COLOR


@dataclass
class MapPolaritySettings:
    """Contour settings for the standard map polarities."""

    normal: MapContourSettings = field(
        default_factory=lambda: MapContourSettings(
            sigma_level=1.0,
            color=DEFAULT_MAP_COLOR,
        )
    )
    positive: MapContourSettings = field(
        default_factory=lambda: MapContourSettings(
            sigma_level=2.5,
            color=DEFAULT_POSITIVE_COLOR,
        )
    )
    negative: MapContourSettings = field(
        default_factory=lambda: MapContourSettings(
            sigma_level=-2.5,
            color=DEFAULT_NEGATIVE_COLOR,
        )
    )

    @property
    def twofofc(self) -> MapContourSettings:
        return self.normal

    @property
    def fofc_positive(self) -> MapContourSettings:
        return self.positive

    @property
    def fofc_negative(self) -> MapContourSettings:
        return self.negative

    @classmethod
    def for_map(
        cls,
        map_type: MapType,
        sigma_level: float,
    ) -> "MapPolaritySettings":
        settings = cls()

        if map_type is MapType.NORMAL:
            settings.normal.sigma_level = sigma_level
        elif map_type is MapType.DIFFERENCE:
            settings.normal.sigma_level = abs(sigma_level)
            settings.positive.sigma_level = abs(sigma_level)
            settings.negative.sigma_level = -abs(sigma_level)

        return settings


@dataclass
class MapRenderSettings:
    """How the user wants a map displayed (strategy + contour/colour state)."""

    mode: IsosurfaceMapRenderMode = IsosurfaceMapRenderMode.UNIT_CELL
    settings: MapPolaritySettings = field(
        default_factory=MapPolaritySettings
    )

    @property
    def bundle(self) -> MapPolaritySettings:
        return self.settings

    ####  ========== Migration shims ==================== ############

    # ========= Visibility - now deprecated - please use map contour info ===== #

    @property
    def is_visible(self):
        return self.settings.normal.is_visible

    @is_visible.setter
    def is_visible(self, value):
        self.settings.normal.is_visible = value

    # ========= Map Colors - now deprecated - please use map contour info ===== #

    @property
    def positive_color(self):
        return self.settings.positive.color

    @positive_color.setter
    def positive_color(self, value):
        self.settings.positive.color = value

    @property
    def negative_color(self):
        return self.settings.negative.color

    @negative_color.setter
    def negative_color(self, value):
        self.settings.negative.color = value

    # ========= Sigma Levels - now deprecated - please use map contour info ===== #

    @property
    def sigma_level(self):
        return self.settings.normal.sigma_level

    @sigma_level.setter
    def sigma_level(self, value):
        self.settings.normal.sigma_level = value

    @property
    def positive_sigma_level(self):
        return self.settings.positive.sigma_level

    @positive_sigma_level.setter
    def positive_sigma_level(self, value):
        self.settings.positive.sigma_level = value

    @property
    def negative_sigma_level(self):
        return self.settings.negative.sigma_level

    @negative_sigma_level.setter
    def negative_sigma_level(self, value):
        self.settings.negative.sigma_level = value

    # =========    Carving settings - now deprecated - please use map processing info ===== #

    @property
    def carve_density(self):
        return self.processing.carve_density

    @carve_density.setter
    def carve_density(self, value):
        self.processing.carve_density = value

    @property
    def carve_density_centroid(self):
        return self.processing.carve_density_centroid

    @carve_density_centroid.setter
    def carve_density_centroid(self, value):
        self.processing.carve_density_centroid = value

    @property
    def carve_cutoff(self):
        return self.processing.carve_cutoff

    @carve_cutoff.setter
    def carve_cutoff(self, value):
        self.processing.carve_cutoff = value

    @property
    def centroid_cutoff(self):
        return self.processing.centroid_cutoff

    @centroid_cutoff.setter
    def centroid_cutoff(self, value):
        self.processing.centroid_cutoff = value

    @property
    def convert_to_cartesian(self):
        return self.processing.convert_to_cartesian

    @convert_to_cartesian.setter
    def convert_to_cartesian(self, value):
        self.processing.convert_to_cartesian = value


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
    volume: np.ndarray # AKA the map density data
    crystallographic_info: Optional[CrystallographicInfo] = None
    description: str = ""
    render: MapRenderSettings = field(default_factory=MapRenderSettings)
    processing: MapProcessingSettings = field(
        default_factory=MapProcessingSettings
    )

    @property
    def is_visible(self) -> bool:
        """Return whether any contour of this map is currently visible."""

        if self.map_type is MapType.DIFFERENCE:
            settings = self.render.settings
            return (
                    settings.positive.is_visible
                    or settings.negative.is_visible
            )

        return self.render.settings.normal.is_visible

    def __post_init__(self) -> None:
        if self.render is None:
            self.render = MapRenderSettings()

    @property
    def is_difference_map(self) -> bool:
        """Whether this map is a Fo-Fc (difference) map."""
        return self.map_type is MapType.DIFFERENCE

    # --- display shims (delegate to render) ---

    @property
    def sigma_level(self) -> float:
        """Primary contour σ (normal map, or magnitude for difference maps)."""
        return self.render.settings.normal.sigma_level

    @sigma_level.setter
    def sigma_level(self, value: float) -> None:
        """Set contour σ, keeping Fo-Fc ± lobes in sync for difference maps.

        For :attr:`MapType.DIFFERENCE`, writes ``normal``, ``positive = +|σ|``,
        and ``negative = -|σ|`` so extraction and UI spinboxes stay aligned.
        """
        level = float(value)
        settings = self.render.settings
        settings.normal.sigma_level = level
        if self.map_type is MapType.DIFFERENCE:
            magnitude = abs(level)
            settings.positive.sigma_level = magnitude
            settings.negative.sigma_level = -magnitude
            settings.normal.sigma_level = magnitude

    @is_visible.setter
    def is_visible(self, value: bool) -> None:
        b = bool(value)
        if self.map_type is MapType.DIFFERENCE:
            self.render.settings.positive.is_visible = b
            self.render.settings.negative.is_visible = b
        else:
            self.render.settings.normal.is_visible = b

    @property
    def color(self) -> RGBTuple:
        return self.render.settings.normal.color

    @color.setter
    def color(self, value: RGBTuple) -> None:
        self.render.settings.normal.color = value

    @property
    def positive_visible(self) -> bool:
        return self.render.settings.positive.is_visible

    @positive_visible.setter
    def positive_visible(self, value: bool) -> None:
        self.render.settings.positive.is_visible = bool(value)

    @property
    def negative_visible(self) -> bool:
        return self.render.settings.negative.is_visible

    @negative_visible.setter
    def negative_visible(self, value: bool) -> None:
        self.render.settings.negative.is_visible = bool(value)

    @property
    def positive_color(self) -> RGBTuple:
        return self.render.positive_color

    @positive_color.setter
    def positive_color(self, value: RGBTuple) -> None:
        self.render.positive_color = value

    @property
    def negative_color(self) -> RGBTuple:
        return self.render.settings.negative.color

    @negative_color.setter
    def negative_color(self, value: RGBTuple) -> None:
        self.render.settings.negative.color = value

    @property
    def positive_sigma_level(self) -> Optional[float]:
        return self.render.settings.positive.sigma_level

    @positive_sigma_level.setter
    def positive_sigma_level(self, value: Optional[float]) -> None:
        self.render.settings.positive.sigma_level = value

    @property
    def negative_sigma_level(self) -> Optional[float]:
        return self.render.settings.negative.sigma_level

    @negative_sigma_level.setter
    def negative_sigma_level(self, value: Optional[float]) -> None:
        if not value:
            return
        self.render.settings.negative.sigma_level = value

    @property
    def carve_density(self) -> bool:
        return self.render.processing.carve_density

    @carve_density.setter
    def carve_density(self, value: bool) -> None:
        self.render.processing.carve_density = bool(value)

    @property
    def carve_density_centroid(self) -> bool:
        return self.render.processing.carve_density_centroid

    @carve_density_centroid.setter
    def carve_density_centroid(self, value: bool) -> None:
        self.render.processing.carve_density_centroid = bool(value)

    @property
    def carve_cutoff(self) -> float:
        return self.render.processing.carve_cutoff

    @carve_cutoff.setter
    def carve_cutoff(self, value: float) -> None:
        self.render.processing.carve_cutoff = float(value)

    @property
    def centroid_cutoff(self) -> float:
        return self.render.processing.centroid_cutoff

    @centroid_cutoff.setter
    def centroid_cutoff(self, value: float) -> None:
        self.render.processing.centroid_cutoff = float(value)

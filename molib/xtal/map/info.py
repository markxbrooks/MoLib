"""
Info for electron density maps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from molib.xtal.map.density import CrystallographicInfo
from molib.xtal.map.map_type import MapType
from molib.xtal.map.render.mode import MapRenderMode
from picogl.core.rgbcolor import RGBTuple, RGB

DEFAULT_MAP_COLOR: RGB = RGBTuple.BLUE
DEFAULT_POSITIVE_COLOR: RGB = RGBTuple.GREEN
DEFAULT_NEGATIVE_COLOR: RGB = RGBTuple.RED


@dataclass(slots=True)
class MapProcessingSettings:
    """Settings controlling density-map processing."""

    carve_density: bool = True
    carve_cutoff: float = 4.0
    carve_density_centroid: bool = False
    centroid_cutoff: float = 15.0
    convert_to_cartesian: bool = False

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
    color: RGBTuple = DEFAULT_MAP_COLOR


@dataclass
class MapBundleContourSettings:
    """Contour settings for the standard map bundle."""

    twofofc: MapContourSettings = field(
        default_factory=lambda: MapContourSettings(
            sigma_level=1.0,
            color=DEFAULT_MAP_COLOR,
        )
    )
    fofc_positive: MapContourSettings = field(
        default_factory=lambda: MapContourSettings(
            sigma_level=3.0,
            color=DEFAULT_POSITIVE_COLOR,
        )
    )
    fofc_negative: MapContourSettings = field(
        default_factory=lambda: MapContourSettings(
            sigma_level=-3.0,
            color=DEFAULT_NEGATIVE_COLOR,
        )
    )

    @classmethod
    def for_map(
        cls,
        map_type: MapType,
        sigma_level: float,
    ) -> "MapBundleContourSettings":
        settings = cls()

        if map_type is MapType.TWO_FO_FC:
            settings.twofofc.sigma_level = sigma_level
        elif map_type is MapType.FO_FC:
            settings.fofc_positive.sigma_level = sigma_level

        return settings


@dataclass
class MapRenderSettings:
    """How the user wants a map displayed (strategy + contour/colour state)."""

    mode: MapRenderMode = MapRenderMode.ISOSURFACE
    bundle: MapBundleContourSettings = field(
        default_factory=MapBundleContourSettings
    )

    ####  ========== Migration shims ==================== ############

    # ========= Visibility - now deprecated - please use map contour info ===== #

    @property
    def is_visible(self):
        return self.bundle.twofofc.is_visible

    @is_visible.setter
    def is_visible(self, value):
        self.bundle.twofofc.is_visible = value

    # ========= Map Colors - now deprecated - please use map contour info ===== #

    @property
    def positive_color(self):
        return self.bundle.fofc_positive.color

    @positive_color.setter
    def positive_color(self, value):
        self.bundle.fofc_positive.color = value

    @property
    def negative_color(self):
        return self.bundle.fofc_negative.color

    @negative_color.setter
    def negative_color(self, value):
        self.bundle.fofc_negative.color = value

    # ========= Sigma Levels - now deprecated - please use map contour info ===== #

    @property
    def sigma_level(self):
        return self.bundle.twofofc.sigma_level

    @sigma_level.setter
    def sigma_level(self, value):
        self.bundle.twofofc.sigma_level = value

    @property
    def positive_sigma_level(self):
        return self.bundle.fofc_positive.sigma_level

    @positive_sigma_level.setter
    def positive_sigma_level(self, value):
        self.bundle.fofc_positive.sigma_level = value

    @property
    def negative_sigma_level(self):
        return self.bundle.fofc_negative.sigma_level

    @negative_sigma_level.setter
    def negative_sigma_level(self, value):
        self.bundle.fofc_negative.sigma_level = value

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

        if self.map_type is MapType.FO_FC:
            bundle = self.render.bundle
            return (
                    bundle.fofc_positive.is_visible
                    or bundle.fofc_negative.is_visible
            )

        return self.render.bundle.twofofc.is_visible

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
        return self.render.bundle.twofofc.sigma_level

    @sigma_level.setter
    def sigma_level(self, value: float) -> None:
        self.render.bundle.twofofc.sigma_level = float(value)

    @property
    def is_visible(self) -> bool:
        return self.render.bundle.twofofc.is_visible

    @is_visible.setter
    def is_visible(self, value: bool) -> None:
        self.render.bundle.twofofc.is_visible = bool(value)

    @property
    def color(self) -> RGBTuple:
        return self.render.bundle.twofofc.color

    @color.setter
    def color(self, value: RGBTuple) -> None:
        self.render.bundle.twofofc.color = value

    @property
    def positive_visible(self) -> bool:
        return self.render.bundle.fofc_positive.is_visible

    @positive_visible.setter
    def positive_visible(self, value: bool) -> None:
        self.render.bundle.fofc_positive.is_visible = bool(value)

    @property
    def negative_visible(self) -> bool:
        return self.render.bundle.fofc_negative.is_visible

    @negative_visible.setter
    def negative_visible(self, value: bool) -> None:
        self.render.bundle.fofc_negative.is_visible = bool(value)

    @property
    def positive_color(self) -> RGBTuple:
        return self.render.positive_color

    @positive_color.setter
    def positive_color(self, value: RGBTuple) -> None:
        self.render.positive_color = value

    @property
    def negative_color(self) -> RGBTuple:
        return self.render.bundle.fofc_negative.color

    @negative_color.setter
    def negative_color(self, value: RGBTuple) -> None:
        self.render.bundle.fofc_negative.color = value

    @property
    def positive_sigma_level(self) -> Optional[float]:
        return self.render.bundle.fofc_positive.sigma_level

    @positive_sigma_level.setter
    def positive_sigma_level(self, value: Optional[float]) -> None:
        self.render.bundle.fofc_positive.sigma_level = value

    @property
    def negative_sigma_level(self) -> Optional[float]:
        return self.render.bundle.fofc_negative.sigma_level

    @negative_sigma_level.setter
    def negative_sigma_level(self, value: Optional[float]) -> None:
        if not value:
            return
        self.render.bundle.fofc_negative.sigma_level = value

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

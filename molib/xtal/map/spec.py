from __future__ import annotations

from typing import Callable

import pathlib

from dataclasses import dataclass, field

from molib.xtal.ccp4.mtz import MtzDensitySpec
from molib.xtal.map.info import MapProcessingSettings


@dataclass(slots=True)
class DensityMapSpec:
    """Specification for loading and processing a density map."""

    map_path: str | pathlib.Path
    pdb_path: str | pathlib.Path | None = None

    mtz: MtzDensitySpec | None = None
    expand_symmetry: bool = True

    processing: MapProcessingSettings = field(
        default_factory=MapProcessingSettings
    )

    centroid: tuple[float, float, float] | None = None

    progress_callback: Callable | None = None

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

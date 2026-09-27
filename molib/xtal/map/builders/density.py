from __future__ import annotations

from typing import Callable

import pathlib

from molib.xtal.map.spec import DensityMapSpec
from molib.xtal.map.info import MapProcessingSettings


def build_density_map_spec(
    *,
    map_path: str | pathlib.Path,
    pdb_path: str | pathlib.Path | None = None,
    processing: MapProcessingSettings | None = None,
    expand_symmetry: bool = True,
    centroid: tuple[float, float, float] | None = None,
    progress_callback: Callable | None = None,
) -> DensityMapSpec:
    """Build a density-map loading specification."""

    return DensityMapSpec(
        map_path=map_path,
        pdb_path=pdb_path,
        expand_symmetry=expand_symmetry,
        processing=processing or MapProcessingSettings(),
        centroid=centroid,
        progress_callback=progress_callback,
    )

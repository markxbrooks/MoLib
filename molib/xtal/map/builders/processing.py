from __future__ import annotations

from molib.xtal.map.info import MapProcessingSettings


def build_map_processing_settings(
    *,
    carve_density: bool = True,
    carve_cutoff: float = 4.0,
    carve_density_centroid: bool = False,
    centroid_cutoff: float = 15.0,
    convert_to_cartesian: bool = False,
) -> MapProcessingSettings:
    """Build map-processing settings."""
    return MapProcessingSettings(
        carve_density=carve_density,
        carve_cutoff=float(carve_cutoff),
        carve_density_centroid=carve_density_centroid,
        centroid_cutoff=float(centroid_cutoff),
        convert_to_cartesian=convert_to_cartesian,
    )

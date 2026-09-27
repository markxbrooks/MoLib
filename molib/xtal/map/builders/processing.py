from __future__ import annotations

from molib.xtal.map.info import MapProcessingSettings


def build_map_processing_settings(carve_cutoff: float | int, carve_density: bool, carve_density_centroid: bool,
                                  centroid_cutoff: float | int, convert_to_cartesian: bool) -> MapProcessingSettings:
    processing = MapProcessingSettings(
        carve_density=carve_density,
        carve_cutoff=carve_cutoff,
        carve_density_centroid=carve_density_centroid,
        centroid_cutoff=centroid_cutoff,
        convert_to_cartesian=convert_to_cartesian,
    )
    return processing

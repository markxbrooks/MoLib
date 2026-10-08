"""
Map Manager for electron density maps

Manages multiple electron density maps (2Fo-Fc, Fo-Fc, etc.) and their associated data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Iterator, List, Tuple

import numpy as np

from decologr import LogMixin, Decologr as log
from molib.core.constants import MoLibConstant

from molib.xtal.ccp4.mtz.filespec import MtzFileSpec
from molib.xtal.map.builder import build_map_info, build_map_information_specs
from molib.xtal.map.helper import load_density_map
from molib.xtal.map.builders.density import build_density_map_spec
from molib.xtal.map.map_type import MapType
from molib.xtal.map.info import MapInfo, MapProcessingSettings


def mtz_id_from_file_name(mtz_file_path: str) -> str:
    """get mtz id from file name"""
    return Path(mtz_file_path).stem


def _extract_volume_and_info(result):
    """Unpack a load result that may be a (volume, info) tuple or a DensityMapData-like object."""
    if result is None:
        return None, None
    if hasattr(result, "volume"):
        return result.volume, result.crystallographic_info
    return result


def map_from_coefficients(
    spec: MtzFileSpec,
):
    """Load the 2Fo-Fc (map) and Fo-Fc (difference) from an MTZ file spec."""
    from molib.xtal.map.helper import load_maps_from_mtz_file_spec

    return load_maps_from_mtz_file_spec(spec)


def resolve_map_type(map_file_path: str, map_id: str) -> tuple[str, MapType]:
    stem_lower = Path(map_file_path).stem.lower()
    if stem_lower.endswith("_diff") or stem_lower.endswith("-diff"):
        map_type = MapType.DIFFERENCE
        full_id = map_id
    else:
        map_type = MapType.NORMAL
        full_id = f"{map_id}_2Fo-Fc"
    return full_id, map_type


def resolve_map_processing_setting(settings: MapProcessingSettings | None, existing: str) -> MapProcessingSettings:
    """Prefer explicit render, then existing map, then defaults"""
    if settings is None:
        if existing is not None and hasattr(existing, "processing"):
            settings = existing.processing
        else:
            settings = MapProcessingSettings()
    return settings


class MapManager(LogMixin):
    """
    Manager for multiple electron density maps.
    Supports different map types (2Fo-Fc, Fo-Fc, etc.) with individual settings.
    """

    default_map: Optional[str] = None

    # maps: dict[uuid, MapInfo]

    """
    def get_map_by_uuid(self, uuid: UUID):
        ""get map by uuid""
        return self.maps.get(uuid, None)
    """

    def __init__(self):
        self.maps: Dict[str, MapInfo] = {}

    def __len__(self) -> int:
        return len(self.maps)

    def __iter__(self) -> Iterator[tuple[str, MapInfo]]:
        return iter(self.maps.items())

    def __contains__(self, map_id: str) -> bool:
        return map_id in self.maps

    def __getitem__(self, map_id: str) -> MapInfo:
        return self.maps[map_id]

    def get_map_by_id(self, map_id: str) -> MapInfo | None:
        """get map by id"""
        return self.maps.get(map_id, None)

    def log_maps(self):
        """Debug: Check what's in the map manager"""
        self.log_message("Map Manager debug:")
        self.log_message(f"   Maps: {list(self.maps.keys())}")
        self.log_message(f"   Default map: {self.default_map}")

    def add_map_if_exists(self, map_file_path: str) -> str:
        """Add maps from an MTZ or CCP4/MAP path if not already present.

        :param map_file_path: Path to an MTZ or CCP4/MAP density file
        :return: Stem-based map id used for registration
        """
        map_id = mtz_id_from_file_name(map_file_path)
        suffix = Path(map_file_path).suffix.lower()

        # Check if this map already exists (map ids are "{map_id}_2Fo-Fc" etc.)
        if map_id not in self.maps and f"{map_id}_2Fo-Fc" not in self.maps:
            try:
                if suffix in (".map", ".ccp4", ".omap"):
                    self.load_ccp4_file(map_file_path, map_id)
                else:
                    self.load_2fofc_fofc(map_file_path, map_id)
            except Exception as ex:
                self.log_error("Error loading density map", ex)
                self.create_placeholder_map(map_id)
        else:
            self.log_message(f"Map {map_id} already exists in Map Manager")
        return map_id

    def load_ccp4_file(
        self,
        map_file_path: str,
        map_id: str,
    ) -> Optional[MapInfo]:
        """Load a single CCP4/MAP volume into the map manager.

        Carve / cartesian options are **per-map**: taken from ``render`` (or an
        existing managed map's settings), with explicit kwargs overriding.

        :param map_file_path: Path to a ``.map`` / ``.ccp4`` / ``.omap`` file
        :param map_id: Base id (usually the file stem)
        :param render: Optional settings to store on the resulting :class:`MapInfo`
        :return: Registered :class:`MapInfo`, or ``None`` on failure
        """
        full_id, map_type = resolve_map_type(map_file_path, map_id)

        processing = MapProcessingSettings()
        spec = build_density_map_spec(
            map_path=map_file_path,
            pdb_path=None,
            expand_symmetry=False,
            processing=processing,
            progress_callback=None,
        )
        result = load_density_map(spec)

        volume, crystallographic_info = _extract_volume_and_info(result)
        if volume is None:
            self.log_warning(f"Could not load CCP4 map from {map_file_path}")
            return None

        map_info = build_map_info(
            map_id=full_id,
            map_type=map_type,
            f_label="CCP4_DATA",
            phi_label="CCP4_DATA",
            volume=volume,
            crystallographic_info=crystallographic_info,
        )
        map_info.processing = spec.processing
        self.add_map_from_map_info(map_info, overwrite=True)
        self.log_message(f"Added CCP4 map to Map Manager: {full_id}")
        return map_info

    def resolve_map_id(self, full_id: str, map_id: str) -> MapInfo | None:
        """resolve the map id"""
        return self.get_map(full_id) or self.get_map(map_id)

    def create_placeholder_map(self, mtz_id: str):
        """Create placeholder maps anyway"""
        map_info = build_map_info(
            map_id=f"{mtz_id}_2Fo-Fc",
            map_type=MapType.NORMAL,
            f_label="2FOFCWT",
            phi_label="PH2FOFCWT",
            volume=np.zeros((10, 10, 10)),  # Placeholder
            crystallographic_info=None,
        )
        self.add_map_from_map_info(map_info, overwrite=True)

    def clear(self):
        """Clear the map manager when reinitializing"""
        try:
            # Clear the map manager
            self.maps.clear()
            self.default_map = None
            # Update the widget to reflect the cleared state
            # self.map_manager_widget.set_map_manager(self.map_manager)
            log.message(
                "Cleared map manager and updated widget",
            )
        except Exception as ex:
            log.warning(f"Could not clear map manager: {ex}")

    def add_map_from_map_info(
        self,
        map_info: MapInfo,
        overwrite: bool = False,
        *,
        render_mode: "MapRenderMode | str | None" = None,
    ) -> None:
        """Add a map from a MapInfo object.

        :param map_info: map to register
        :param overwrite: replace an existing map with the same id
        :param render_mode: optional session mode applied to ``map_info.render.mode``
        """
        from molib.xtal.map.render.mode import IsosurfaceMapRenderMode

        if map_info.map_id in self.maps and not overwrite:
            raise ValueError(f"Map ID '{map_info.map_id}' already exists.")

        # If overwrite, remove old mapping
        if overwrite and map_info.map_id in self.maps:
            del self.maps[map_info.map_id]

        # Ensure map_type is a MapType enum
        if not isinstance(map_info.map_type, MapType):
            map_info.map_type = MapType.coerce(map_info.map_type)

        if map_info.render is None:
            from molib.xtal.map.info import MapRenderSettings

            map_info.render = MapRenderSettings()

        if render_mode is not None:
            map_info.render.mode = IsosurfaceMapRenderMode.coerce(render_mode)

        self.maps[map_info.map_id] = map_info

        # Set as default if this is the first map
        if self.default_map is None:
            self.default_map = map_info.map_id

        log.message(
            f"Added map: {map_info.map_id} ({map_info.map_type}) with {map_info.volume.shape} volume"
        )

    def set_all_render_modes(self, mode: "MapRenderMode | str") -> None:
        """Set :attr:`MapRenderSettings.mode` on every managed map."""
        from molib.xtal.map.render.mode import IsosurfaceMapRenderMode

        resolved = IsosurfaceMapRenderMode.coerce(mode)
        for map_info in self.maps.values():
            map_info.render.mode = resolved
        log.message(f"Set all map render modes to: {resolved.value}")

    def get_map(self, map_id: str) -> Optional[MapInfo]:
        """Get a map by ID."""
        return self.maps.get(map_id)

    def get_default_map(self) -> Optional[MapInfo]:
        """Get the default map."""
        if self.default_map and self.default_map in self.maps:
            return self.maps[self.default_map]
        elif self.maps:
            # Auto-pick first available map
            self.default_map = next(iter(self.maps))
            return self.maps[self.default_map]
        return None

    def get_all_maps(self) -> List[str]:
        """Get list of all map IDs."""
        return list(self.maps.keys())

    def get_maps_by_type(self, map_type: str) -> List[MapInfo]:
        """Get all maps of a specific type."""
        return [
            map_info for map_info in self.maps.values() if map_info.map_type == map_type
        ]

    def remove_map(self, map_id: str) -> None:
        """Remove a map by ID."""
        if map_id not in self.maps:
            raise ValueError(f"Map ID '{map_id}' does not exist.")

        del self.maps[map_id]

        # Update default map if needed
        if self.default_map == map_id:
            self.default_map = next(iter(self.maps)) if self.maps else None

        log.message(f"Removed map: {map_id}")

    def set_default_map(self, map_id: str) -> None:
        """Set the default map."""
        if map_id not in self.maps:
            raise ValueError(f"Map ID '{map_id}' does not exist.")
        self.default_map = map_id
        log.message(f"Set default map to: {map_id}")

    def clear_all_maps(self) -> None:
        """Clear all maps."""
        self.maps.clear()
        self.default_map = None
        log.message("Cleared all maps")

    def update_map_visibility(self, map_id: str, is_visible: bool) -> None:
        """Update map visibility."""
        if map_id in self.maps:
            self.maps[map_id].render.is_visible = is_visible
            log.message(f"Map {map_id} visibility: {is_visible}")

    def update_map_sigma_level(self, map_id: str, sigma_level: float) -> None:
        """Update map contour σ via the type-aware :attr:`MapInfo.sigma_level` facade.

        Difference maps also update positive/negative lobe levels.
        """
        if map_id in self.maps:
            self.maps[map_id].sigma_level = float(sigma_level)
            log.message(f"Map {map_id} sigma level: {sigma_level}")

    def update_map_color(self, map_id: str, color: Tuple[float, float, float]) -> None:
        """Update map colour."""
        if map_id in self.maps:
            self.maps[map_id].render.color = color
            log.message(f"Map {map_id} colour: {color}")

    def update_difference_visibility(
        self,
        map_id: str,
        positive_visible: Optional[bool] = None,
        negative_visible: Optional[bool] = None,
    ) -> None:
        """Update Fo-Fc positive/negative contour visibility."""
        if map_id in self.maps:
            map_info = self.maps[map_id]
            if positive_visible is not None:
                map_info.render.settings.positive.is_visible = bool(positive_visible)
            if negative_visible is not None:
                map_info.render.settings.negative.is_visible = bool(negative_visible)

    def get_visible_maps(self) -> list[MapInfo]:
        """Get all currently visible maps."""
        return [
            map_info
            for map_info in self.maps.values()
            if map_info.is_visible
        ]

    def get_map_summary(self) -> Dict[str, Dict]:
        """Get summary information for all maps."""
        summary = {}
        for map_id, map_info in self.maps.items():
            summary[map_id] = {
                "type": map_info.map_type,
                "f_label": map_info.f_label,
                "phi_label": map_info.phi_label,
                "volume_shape": map_info.volume.shape,
                "sigma_level": map_info.sigma_level,
                "is_visible": map_info.is_visible,
                "description": map_info.description,
            }
        return summary

    def load_2fofc_fofc(self, mtz_file_path: str, mtz_id: str):
        """Load 2Fo-Fc and Fo-Fc using map-type-aware coefficient selection.

        Coefficients are chosen deterministically per requested map type
        (FWT/PH2FOFCWT for 2Fo-Fc, DELFWT/PHDELWT for Fo-Fc). The loader never
        silently substitutes unrelated coefficients.

        CCP4/MAP paths are rejected here; use the CCP4 loader instead.
        """
        suffix = Path(mtz_file_path).suffix.lower()
        if suffix in (".map", ".ccp4", ".omap"):
            self.log_warning(
                f"Refusing MTZ coefficient load for CCP4/MAP file: {mtz_file_path}"
            )
            return

        from molib.xtal.map.helper import (
            MapType,
            load_density_map_auto_mtz,
        )

        result_2fofc = load_density_map_auto_mtz(
            mtz_file_path, map_type=MapType.NORMAL
        )
        result_fofc = load_density_map_auto_mtz(
            mtz_file_path, map_type=MapType.DIFFERENCE
        )

        if result_2fofc:
            volume_2fofc, crystallographic_info = _extract_volume_and_info(result_2fofc)
            if result_fofc:
                volume_fofc, _ = _extract_volume_and_info(result_fofc)
            else:
                volume_fofc = volume_2fofc
                self.log_warning(
                    "Fo-Fc coefficients not found in MTZ; using fallback volume"
                )
            map_information = build_map_information_specs(
                crystallographic_info, mtz_id, volume_2fofc, volume_fofc
            )
            for map_name, map_info in map_information.items():
                self.add_map_from_map_info(map_info)
                self.log_message(f"Added maps to Map Manager: {map_name}")
        else:
            self.log_warning(f"Could not load density map from {mtz_file_path}")

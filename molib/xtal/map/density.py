"""
Utilities for loading and processing electron density maps from MTZ and CCP4 files.
"""
from __future__ import annotations

from typing import Any
import gemmi
import numpy as np
from gemmi import Mtz
from numpy import ndarray, dtype

from molib.xtal.map.crystal import CrystallographicInfo
from molib.xtal.map.map_type import MapSource
from decologr import Decologr as log

MAP_NEGATIVE_RATIO_THRESHOLD = 0.7


def load_density_map_with_columns(
    mtz_path: str, f_column: str, phi_column: str, sample_rate=0.0
) -> tuple[np.ndarray, CrystallographicInfo] | None:
    """
    Load density map from MTZ file with specific F and PHI column selections.

    Args:
        mtz_path: Path to MTZ file
        f_column: F column label
        phi_column: PHI column label
        sample_rate: Sampling rate for map generation (0.0 = full resolution)

    Returns:
        tuple of (numpy array, crystallographic_info) or None if loading fails
    """
    try:
        log.info(f"Loading MTZ file with specific columns: {mtz_path}")
        log.info(f"📊 F column: {f_column}, PHI column: {phi_column}")

        # Load the density map with specified columns
        result = load_density_map(mtz_path, f_column, phi_column, sample_rate)

        if result is not None:
            log.info(f"✅ Successfully loaded MTZ with {f_column}/{phi_column}")
            return result
        else:
            log.error(
                f"❌ Failed to load MTZ file with columns {f_column}/{phi_column}"
            )
            return None

    except Exception as e:
        log.error(
            f"❌ Error loading MTZ file {mtz_path} with columns {f_column}/{phi_column}: {e}"
        )
        return None


def load_density_map(
    mtz_path: str,
    f_label: str = "2FOFCWT",
    phi_label: str = "PH2FOFCWT",
    sample_rate: float = 0.0,
    map_type: MapSource = MapSource.CCP4_MAP
) -> tuple[ndarray[Any, dtype[Any]], CrystallographicInfo] | None | Any:
    try:
        mtz = gemmi.read_mtz_file(mtz_path)

        # Get available column labels
        f_labels = get_labels_for_col_type(col_type="F", mtz=mtz)
        phi_labels = get_labels_for_col_type(col_type="P", mtz=mtz)

        log.info(f"ℹ️ Available F labels: {f_labels}")
        log.info(f"ℹ️ Available PHI labels: {phi_labels}")

        # Check if requested labels exist
        if f_label not in f_labels:
            return log_available_f_labels(f_label, f_labels)

        if phi_label not in phi_labels:
            return log_available_phi_labels(phi_label, phi_labels)

        grid = mtz.transform_f_phi_to_map(
            f_label,
            phi_label,
            sample_rate=sample_rate,
        )

        # The returned array is guaranteed to be in X, Y, Z axis order
        # (numpy axis 0 -> X), matching the origin/spacing convention.
        crystallographic_info = CrystallographicInfo.from_grid(grid, map_type=map_type)
        np_array = CrystallographicInfo.grid_to_xyz_array(grid)
        crystallographic_info.sync_to_xyz_volume(np_array)

        crystallographic_info.log_summary()

        return np_array, crystallographic_info
    except Exception as e:
        log.error(f"❌ Could not load map from {mtz_path}: {e}")
        return None


def get_labels_for_col_type(col_type: str, mtz: Mtz) -> list[str]:
    """get labels for a given column type"""
    return [col.label for col in mtz.columns if col.type == col_type]


def log_available_phi_labels(phi_label: str, phi_labels: list[str]) -> Any:
    """Log available PHI labels"""
    log.error(f"❌ Requested PHI label '{phi_label}' not found in MTZ file")
    log.error(f"Available PHI labels: {phi_labels}")
    if phi_labels:
        log.info("💡 Try using one of these PHI labels instead")
        # Suggest common alternatives
        common_phi_labels = ["PHIC", "PHWT", "PHI", "PHIC_ALL"]
        for common in common_phi_labels:
            if common in phi_labels:
                log.info(f"💡 Suggested PHI label: {common}")
                break
    return None


def log_available_f_labels(f_label: str, f_labels: list[str]) -> Any:
    """Log available F labels"""
    log.error(f"❌ Requested F label '{f_label}' not found in MTZ file")
    log.error(f"Available F labels: {f_labels}")
    if f_labels:
        log.info("💡 Try using one of these F labels instead")
        # Suggest common alternatives
        common_f_labels = ["FP", "FWT", "F", "FC"]
        for common in common_f_labels:
            if common in f_labels:
                log.info(f"💡 Suggested F label: {common}")
                break
    return None

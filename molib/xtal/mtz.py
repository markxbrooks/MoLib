from typing import Any

import gemmi
import numpy as np
from gemmi import UnitCell
from numpy import dtype, ndarray


def load_mtz_density_array(
    mtz_path: str, column_labels=("FWT", "PHWT"), resolution=1.0
) -> tuple[ndarray[Any, dtype[Any]], UnitCell]:
    """
    load_mtz_density_array

    :param mtz_path: str
    :param column_labels: tuple
    :param resolution: float
    :return: tuple
    """
    mtz = gemmi.read_mtz_file(mtz_path)
    grid = mtz.transform_f_phi_to_map(
        column_labels[0], column_labels[1], sample_rate=resolution
    )
    array = np.array(grid, copy=False)  # shape: (z, y, x)
    return array, grid.unit_cell


def mtz_to_density_map(
    mtz_path: str, f_col: str = "F", phi_col: str = "PHI"
) -> tuple[np.ndarray, gemmi.Grid]:
    """
    mtz_to_density_map

    :param mtz_path: str
    :param f_col: str
    :param phi_col: str
    :return: tuple
    """
    # Load MTZ file
    mtz = gemmi.read_mtz_file(mtz_path)

    # Prepare structure factors
    sf = gemmi.MtzSF(mtz, f_col, phi_col)

    # Create reciprocal lattice
    f_transform = gemmi.FourierTransform()
    f_transform.prepare_viewport(sf)

    # Set grid resolution and compute the map
    grid = f_transform.fft_grid(clip=1.0)  # Full unit cell

    # Convert to NumPy
    data = np.array(grid, copy=False)  # shape: (z, y, x)

    return data, grid


def reorder_for_opengl(density_map: np.ndarray) -> np.ndarray:
    """
    reorder_for_opengl

    :param density_map: np.ndarray
    :return: np.ndarray
    """
    return np.transpose(density_map, (2, 1, 0))  # zyx → xyz

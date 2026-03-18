from typing import Tuple

import numpy as np
from molib.pdb.coordinate.data import CoordinateData


def find_closest_atom(
    coordinate_data: CoordinateData, pos: Tuple[float, float, float]
) -> int:
    """
    find_closest_atom

    :param coordinate_data: CoordinateData
    :param pos: tuple[float, float, float] A 3D position_array as a tuple (x, y, z)
    :return: Index of the closest atom (int)

    Returns the index of the atom in coordinates that is closest to the given position_array.
    """
    if coordinate_data is None or len(coordinate_data.coords) == 0:
        return -1  # or raise an error if appropriate

    pos_array = np.array(pos, dtype=np.float32)
    distances_squared = np.sum((coordinate_data.coords - pos_array) ** 2, axis=1)
    return int(np.argmin(distances_squared))

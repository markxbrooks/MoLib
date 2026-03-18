import numpy as np


def center_coordinates(coordinates: np.array) -> np.array:
    """
    center_coordinates
    :param coordinates: np.data
    :return: coordinate_data_main
    :rtype: np.data
    """
    center = np.mean(coordinates, axis=0)
    return coordinates - center

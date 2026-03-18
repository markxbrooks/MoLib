"""
Parser
"""

from elmo.ui.state.entity import MolEntityType


def parse_pdb_coordinates_from_file(file_path: str):
    """
    parse_pdb_coordinates_from_file

    :param file_path: str
    :return:
    """
    coordinates = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.startswith(MolEntityType.ATOM.value):
                x_coord = float(line[30:38])
                y_coord = float(line[38:46])
                z_coord = float(line[46:54])
                coordinates.append((x_coord, y_coord, z_coord))
    return coordinates

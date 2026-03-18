import numpy as np
from decologr import Decologr as log

from elmo.ui.state.scene.data import MolecularSceneState


def validate_coordinate_data(mol_scene_state: MolecularSceneState) -> bool:
    """
    validate_coordinate_data

    :param mol_scene_state: MolecularSceneState
    :return: bool
    """
    coords = mol_scene_state.get_coordinate_data("all_atom")
    if coords is None or not hasattr(coords, "shape"):
        log.message("No coordinate_data_main found, nothing to draw")
        return False
    if coords.shape[1] != 3 or coords.dtype != np.float32:
        raise ValueError("vertex_data must be Nx3 with dtype=np.float32")
    return True

from PySide6.QtCore import Signal

from elmo.ui.parameter.camera import CameraParameter
from elmo.ui.state.ui import UiState


def set_rotation_from_internal_value(
    parameter: CameraParameter,
    ui_state: UiState,
    camera_rotate_values: Signal,
    val: float,
    axis: int,
) -> None:
    """
    set rotation from internal uniform_value

    :param camera_rotate_values: Signal
    :param ui_state: UiState
    :param parameter: CameraParameter
    :param val: float uniform_value
    :param axis: int
    :return: None
    """
    clamped = max(parameter.min_value, min(parameter.max_value, val))
    parameter.value = clamped
    ui_state.camera.rotation[axis] = clamped

    # Emit percent (0–100) back to UI
    percent = (
        (clamped - parameter.min_value)
        / (parameter.max_value - parameter.min_value)
        * 100
    )
    camera_rotate_values.emit(
        int(ui_state.camera.rot_x.value),
        int(ui_state.camera.rot_y.value),
        int(ui_state.camera.rot_z.value),
    )

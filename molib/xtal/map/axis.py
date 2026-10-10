from __future__ import annotations

from enum import Enum

import gemmi
import numpy as np

from decologr import Decologr as log


class AxisOrder(str, Enum):
    """Mapping of NumPy array axes to crystallographic X, Y, Z axes.

    The string describes the crystallographic axis corresponding to
    NumPy axes 0, 1, and 2 respectively.

    Examples:
        XYZ:
            array axis 0 -> X
            array axis 1 -> Y
            array axis 2 -> Z

        ZYX:
            array axis 0 -> Z
            array axis 1 -> Y
            array axis 2 -> X
    """

    XYZ = "XYZ"
    XZY = "XZY"
    YXZ = "YXZ"
    YZX = "YZX"
    ZXY = "ZXY"
    ZYX = "ZYX"

    @property
    def x_axis(self) -> int:
        """NumPy axis corresponding to crystallographic X."""
        return self.value.index("X")

    @property
    def y_axis(self) -> int:
        """NumPy axis corresponding to crystallographic Y."""
        return self.value.index("Y")

    @property
    def z_axis(self) -> int:
        """NumPy axis corresponding to crystallographic Z."""
        return self.value.index("Z")

    @property
    def permutation(self) -> tuple[int, int, int]:
        """NumPy-axis permutation corresponding to X, Y, Z."""
        return (
            self.x_axis,
            self.y_axis,
            self.z_axis,
        )

    def transpose_to_xyz(
        self,
        array: np.ndarray,
    ) -> np.ndarray:
        """Return array with axes ordered X, Y, Z."""

        return np.transpose(array, self.permutation)

    @classmethod
    def from_gemmi(
            cls,
            axis_order: gemmi.AxisOrder,
    ) -> "AxisOrder":
        match axis_order:
            case gemmi.AxisOrder.XYZ:
                return cls.XYZ
            case gemmi.AxisOrder.ZYX:
                return cls.ZYX
            case gemmi.AxisOrder.Unknown:
                raise ValueError("Unknown Gemmi axis order")
            case _:
                raise ValueError(
                    f"Unsupported Gemmi axis order: {axis_order!r}"
                )


def _axis_order_from_gemmi(axis_order: gemmi.AxisOrder) -> AxisOrder:
    """Convert a gemmi axis order, defaulting to XYZ for unknown grids.

    CCP4 maps read without setup() report AxisOrder.Unknown; spacing/origin
    math in this module assumes the canonical XYZ ordering, which is the
    ordering gemmi itself guarantees for transform_f_phi_to_map() grids.
    """
    try:
        return AxisOrder.from_gemmi(axis_order)
    except ValueError:
        log.warning(
            "⚠️ Unsupported grid axis order %r; assuming XYZ",
            axis_order,
        )
        return AxisOrder.XYZ

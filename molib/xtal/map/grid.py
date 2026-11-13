from __future__ import annotations

from dataclasses import dataclass

import gemmi
import numpy as np

from decologr import Decologr as log
from molib.core.constants import MoLibConstant
from molib.xtal.map.axis import AxisOrder, _axis_order_from_gemmi
from picogl.core.mixin.vec3 import Vec3Mixin


@dataclass(frozen=True, slots=True)
class GridSpacing(Vec3Mixin):
    """Grid spacing along the Cartesian axes, in Å/grid."""

    x: float
    y: float
    z: float

    def to_tuple(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z

    @classmethod
    def from_grid(
        cls,
        grid: gemmi.FloatGrid,
        frac_to_orth: np.ndarray,
    ) -> "GridSpacing":
        """Calculate Cartesian grid spacing from a crystallographic grid.

        The columns of the fractional-to-orthogonal matrix are the lattice
        vectors, so the spacing along a crystallographic axis is the length of
        that lattice vector divided by the number of voxels along the axis.
        Maps from transform_f_phi_to_map() are always stored in XYZ order, so
        grid.shape[0] is the number of voxels along the X (a) axis. Using the
        row norms instead would mix components of different lattice vectors
        and give wrong results for monoclinic/triclinic cells.
        """
        return cls(
            x=np.linalg.norm(frac_to_orth[:, 0]) / grid.shape[0],
            y=np.linalg.norm(frac_to_orth[:, 1]) / grid.shape[1],
            z=np.linalg.norm(frac_to_orth[:, 2]) / grid.shape[2],
        )

    def log_summary(self):
        """log summary of grid spacing"""
        log.info(
            "🔧 Calculated grid spacing using "
            "crystallographic transformations:"
        )
        log.info(
            "   X spacing: %.4f Å/grid",
            self.x,
        )
        log.info(
            "   Y spacing: %.4f Å/grid",
            self.y,
        )
        log.info(
            "   Z spacing: %.4f Å/grid",
            self.z,
        )


@dataclass(frozen=True, slots=True)
class GridOrigin(Vec3Mixin):
    """Grid origin in Cartesian coordinates, in Å."""

    x: float
    y: float
    z: float

    def to_tuple(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z

    @classmethod
    def from_grid_center(
        cls,
        unit_cell_center: tuple[float, float, float],
        grid_center: tuple[float, float, float],
    ) -> "GridOrigin":
        """Calculate the grid origin from the unit-cell and grid centers."""
        return cls(
            x=unit_cell_center[0] - grid_center[0],
            y=unit_cell_center[1] - grid_center[1],
            z=unit_cell_center[2] - grid_center[2],
        )

    def log_summary(self):
        """log summary of grid origin"""
        log.info(
            "   Grid origin: (%.3f, %.3f, %.3f) Å",
            self.x,
            self.y,
            self.z,
        )


@dataclass(slots=True)
class MapGrid:
    """Geometry associated with a canonical XYZ density array.

    ``origin`` is the Cartesian position of index ``(0, 0, 0)``; ``spacing``
    is the Cartesian step along each array axis for the axis-aligned
    ``origin + index * spacing`` model used by carving and similar tools.
    """

    dimensions: tuple[int, int, int]
    origin: GridOrigin
    spacing: GridSpacing
    axis_order: AxisOrder

    @classmethod
    def from_gemmi(
        cls,
        grid: gemmi.FloatGrid,
        frac_to_orth: np.ndarray,
    ) -> "MapGrid":
        spacing, origin = _calculate_proper_grid_spacing(grid)
        return cls(
            dimensions=tuple(grid.shape),
            origin=origin,
            spacing=spacing,
            axis_order=_axis_order_from_gemmi(grid.axis_order),
        )


def _calculate_proper_grid_spacing(
    grid: gemmi.FloatGrid,
) -> tuple[GridSpacing, GridOrigin]:
    """Calculate physical spacing and origin for a crystallographic grid.

    The origin is the Cartesian position of grid point (0, 0, 0), matching
    gemmi's get_position()/point_to_position(). Maps produced by
    transform_f_phi_to_map() are XYZ-ordered and cover the full unit cell
    starting at the fractional origin, so the origin is the Cartesian origin
    of the unit cell.
    """
    frac_to_orth = get_grid_fractional_to_orthogonal_matrix(grid)

    if frac_to_orth is None:
        raise ValueError(
            "Failed to calculate fractional-to-orthogonal transformation"
        )

    spacing = GridSpacing.from_grid(
        grid,
        np.asarray(frac_to_orth),
    )

    if grid.axis_order != gemmi.AxisOrder.XYZ:
        log.warning(
            "⚠️ Grid axis order is %s; spacing/origin assume XYZ order",
            grid.axis_order,
        )

    position = grid.get_position(0, 0, 0)
    origin = GridOrigin(
        x=position.x,
        y=position.y,
        z=position.z,
    )

    origin.log_summary()
    spacing.log_summary()

    return spacing, origin


def _owned_matrix_3x3(source) -> np.ndarray:
    """Copy a 3×3 transform into a newly allocated float64 array.

    Gemmi's ``orth.mat`` buffer must not be passed to LAPACK. ``np.linalg.inv``
    on that buffer SIGBUS when called from a Qt worker thread on macOS.

    :param source: Gemmi matrix or any 3×3 sequence
    :return: Owned C-contiguous ``(3, 3)`` array
    """
    try:
        rows = [
            (float(source[i][0]), float(source[i][1]), float(source[i][2]))
            for i in range(3)
        ]
    except Exception:
        raw = np.asarray(source, dtype=np.float64).reshape(3, 3)
        rows = [tuple(float(value) for value in row) for row in raw]
    return np.array(rows, dtype=np.float64)


def _invert_3x3(matrix: np.ndarray) -> np.ndarray:
    """Invert a 3×3 matrix with the adjugate, without calling LAPACK.

    :param matrix: Square transform
    :return: Inverse
    :raises ValueError: If *matrix* is not 3×3 or is singular
    """
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected 3x3 matrix, got {matrix.shape}")
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    det = (
        a * (e * i - f * h)
        - b * (d * i - f * g)
        + c * (d * h - e * g)
    )
    if not np.isfinite(det) or abs(det) < MoLibConstant.EPSILON:
        raise ValueError(f"Matrix is singular (determinant: {det})")
    inv_det = 1.0 / det
    return np.array(
        [
            [
                (e * i - f * h) * inv_det,
                (c * h - b * i) * inv_det,
                (b * f - c * e) * inv_det,
            ],
            [
                (f * g - d * i) * inv_det,
                (a * i - c * g) * inv_det,
                (c * d - a * f) * inv_det,
            ],
            [
                (d * h - e * g) * inv_det,
                (b * g - a * h) * inv_det,
                (a * e - b * d) * inv_det,
            ],
        ],
        dtype=np.float64,
    )


def get_grid_fractional_to_orthogonal_matrix(grid: gemmi.FloatGrid) -> np.ndarray:
    """
    Get the transformation matrix from fractional to orthogonal coordinates.

    :param grid: Gemmi FloatGrid object

    :returns: 3x3 numpy array representing the transformation matrix
    """
    try:
        # The grid stores fractional coordinates in the conventional unit
        # cell, so we use the cell's standard orthogonalization matrix -- the
        # same frame as the PDB coordinates. primitive_orth_matrix() maps into
        # the primitive-cell frame, which differs for centred space groups.
        return _owned_matrix_3x3(grid.unit_cell.orth.mat)

    except Exception as e:
        log.error(f"❌ Error getting fractional to orthogonal matrix: {e}")
        return None


def get_grid_orthogonal_to_fractional_matrix(grid: gemmi.FloatGrid) -> np.ndarray:
    """
    Get the transformation matrix from orthogonal to fractional coordinates.

    Args:
        grid: Gemmi FloatGrid object

    Returns:
        3x3 numpy array representing the inverse transformation matrix
    """
    try:
        # Invert the fractional->orthogonal matrix (same frame as above).
        # Do not call np.linalg.inv: Apple Accelerate SIGBUS when that runs
        # on the density-carving QThread.
        return _invert_3x3(_owned_matrix_3x3(grid.unit_cell.orth.mat))

    except Exception as ex:
        log.error(f"❌ Error getting orthogonal to fractional matrix: {ex}")
        return None

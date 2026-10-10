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
        matrix = np.array(grid.unit_cell.orth.mat, dtype=np.float64)

        return matrix

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
        frac_to_orth = np.array(grid.unit_cell.orth.mat, dtype=np.float64)

        # Validate the matrix before inversion to prevent numerical issues.
        if frac_to_orth.shape != (3, 3):
            raise ValueError(f"Expected 3x3 matrix, got {frac_to_orth.shape}")

        det = np.linalg.det(frac_to_orth)
        if abs(det) < MoLibConstant.EPSILON:
            raise ValueError(f"Matrix is singular (determinant: {det})")

        orth_to_frac = np.linalg.inv(frac_to_orth)

        return np.ascontiguousarray(orth_to_frac, dtype=np.float64)

    except Exception as ex:
        log.error(f"❌ Error getting orthogonal to fractional matrix: {ex}")
        return None

"""
Utilities for loading and processing electron density maps from MTZ and CCP4 files.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Any

import gemmi
import numpy as np
from gemmi import FloatGrid, UnitCell, Mtz
from numpy import ndarray, dtype

from elmo.ui.widgets.gl.mol.base import CoordinateTuple
from picogl.core.mixin.vec3 import Vec3Mixin
from decologr import Decologr as log
from molib.pdb.coordinate.coordinate import Coordinates

MAP_NEGATIVE_RATIO_THRESHOLD = 0.7


def load_density_map_with_columns(
    mtz_path: str, f_column: str, phi_column: str, sample_rate=0.0
) -> tuple[np.ndarray, dict] | None:
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
        """Calculate Cartesian grid spacing from a crystallographic grid."""
        return cls(
            x=np.linalg.norm(frac_to_orth[0, :]) / grid.shape[0],
            y=np.linalg.norm(frac_to_orth[1, :]) / grid.shape[1],
            z=np.linalg.norm(frac_to_orth[2, :]) / grid.shape[2],
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


@dataclass(slots=True)
class UnitCell:
    """UnitCell"""
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    source: str = ""
    space_group: str = ""

    @property
    def center(self) -> tuple[float, float, float]:
        return (
            self.a / 2,
            self.b / 2,
            self.c / 2,
        )

    @classmethod
    def from_dict(cls, data: dict) -> "UnitCell":
        """Create a UnitCell from a dictionary."""
        return cls(
            a=float(data["a"]),
            b=float(data["b"]),
            c=float(data["c"]),
            alpha=float(data["alpha"]),
            beta=float(data["beta"]),
            gamma=float(data["gamma"]),
            source=data.get("source", ""),
            space_group=data.get("space_group", ""),
        )

    def to_dict(self) -> dict:
        """Convert the UnitCell to a dictionary."""
        return {
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "space_group": self.space_group,
            "source": self.source,
        }


    @property
    def fractional_center(self) -> tuple[float, float, float]:
        """Return the geometric center in fractional coordinates."""
        return 0.5, 0.5, 0.5

    @property
    def is_orthogonal(self) -> bool:
        """is_orthogonal"""
        return (
            abs(self.alpha - 90.0) < 0.1
            and abs(self.beta - 90.0) < 0.1
            and abs(self.gamma - 90.0) < 0.1
        )

    @property
    def is_monoclinic(self) -> bool:
        """is monoclinic"""
        return (abs(self.beta - 90.0) > 0.1
                or abs(self.alpha - 90.0) > 0.1
                or abs(self.gamma - 90.0) > 0.1)


@dataclass(slots=True)
class MapGrid:
    """MapGrid"""
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
        spacing = GridSpacing.from_grid(
            grid,
            frac_to_orth,
        )

        grid_center = (
            (grid.shape[0] - 1) * spacing.x / 2,
            (grid.shape[1] - 1) * spacing.y / 2,
            (grid.shape[2] - 1) * spacing.z / 2,
        )

        origin = GridOrigin.from_grid_center(
            grid.unit_cell.centroid,
            grid_center,
        )

        return cls(
            dimensions=tuple(grid.shape),
            origin=origin,
            spacing=spacing,
            axis_order=AxisOrder.from_gemmi(grid.axis_order),
        )


@dataclass(slots=True)
class CoordinateTransforms:
    """CoordinateTransforms"""
    frac_to_orth: np.ndarray
    orth_to_frac: np.ndarray


@dataclass(slots=True)
class CrystallographicInfo:
    """CrystallographicInfo"""
    unit_cell: UnitCell
    space_group: str
    grid: MapGrid
    transforms: CoordinateTransforms
    map_type: str = ""

    def log_summary(self) -> None:
        """Log the contents of this crystallographic information."""

        cell = self.unit_cell
        grid = self.grid

        log.info(
            "📐  Unit cell: a=%.2f, b=%.2f, c=%.2f Å, "
            "α=%.2f°, β=%.2f°, γ=%.2f°",
            cell.a,
            cell.b,
            cell.c,
            cell.alpha,
            cell.beta,
            cell.gamma,
        )

        log.info("📐 Space group: %s", self.space_group)
        log.info("📐 Grid dimensions: %s", grid.dimensions)
        log.info("📐 Grid origin: %s", grid.origin)
        log.info("📐 Grid spacing: %s", grid.spacing)
        log.info("📐Axis order: %s", grid.axis_order)

    @classmethod
    def from_dict(cls, data: dict) -> "CrystallographicInfo":
        """Create crystallographic information from a serialized dictionary."""

        unit_cell_data = data["unit_cell"]

        unit_cell = UnitCell(
            a=unit_cell_data["a"],
            b=unit_cell_data["b"],
            c=unit_cell_data["c"],
            alpha=unit_cell_data["alpha"],
            beta=unit_cell_data["beta"],
            gamma=unit_cell_data["gamma"],
        )
        empty_transform = np.empty((0, 0), dtype=float)

        transforms = CoordinateTransforms(
            frac_to_orth=data.get("frac_to_orth", empty_transform),
            orth_to_frac=data.get("orth_to_frac", empty_transform),
        )

        grid = MapGrid(
            dimensions=tuple(data["grid_dimensions"]),
            origin=tuple(data["grid_origin"]),
            axis_order=tuple(data["axis_order"]),
        )

        return cls(
            unit_cell=unit_cell,
            space_group=data["space_group"],
            grid=grid,
            transforms=transforms
        )

    def to_dict(self):
        """for use during refactoring"""
        crystallographic_info = {
            "unit_cell": {
                "a": self.unit_cell.a,
                "b": self.unit_cell.b,
                "c": self.unit_cell.c,
                "alpha": self.unit_cell.alpha,
                "beta": self.unit_cell.beta,
                "gamma": self.unit_cell.gamma,
            },
            "space_group": str(self.space_group),
            "grid_dimensions": self.grid.dimensions,
            "grid_origin": CoordinateTuple.ORIGIN,
            "axis_order": self.grid.axis_order,
        }
        return crystallographic_info


def load_density_map(
    mtz_path: str,
    f_label: str = "2FOFCWT",
    phi_label: str = "PH2FOFCWT",
    sample_rate: float = 0.0,
    map_type: str = "CCP4_MAP"
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

        crystallographic_info = crystallographic_info_from_grid(grid, map_type=map_type)

        crystallographic_info.log_summary()

        np_array = np.array(grid, copy=True)

        return np_array, crystallographic_info

    except Exception as e:
        log.error(f"❌ Could not load map from {mtz_path}: {e}")
        return None


def crystallographic_info_from_grid(
    grid: FloatGrid, map_type: str = "CCP4_MAP"
) -> CrystallographicInfo:
    """Create crystallographic information from a Gemmi map grid."""

    grid_spacing, grid_origin = _calculate_proper_grid_spacing(grid)

    grid_origin = _convert_grid_origin_to_cartesian(
        grid,
        grid_origin,
    )

    return CrystallographicInfo(
        unit_cell=UnitCell(
            a=grid.unit_cell.a,
            b=grid.unit_cell.b,
            c=grid.unit_cell.c,
            alpha=grid.unit_cell.alpha,
            beta=grid.unit_cell.beta,
            gamma=grid.unit_cell.gamma,
        ),
        space_group=str(grid.spacegroup),
        grid=MapGrid(
            dimensions=tuple(grid.shape),
            origin=grid_origin,
            spacing=grid_spacing,
            axis_order=grid.axis_order,
        ),
        transforms=CoordinateTransforms(
            frac_to_orth=get_grid_fractional_to_orthogonal_matrix(grid),
            orth_to_frac=get_grid_orthogonal_to_fractional_matrix(grid),
        ),
    )


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


def _convert_grid_origin_to_cartesian(
    grid: gemmi.FloatGrid, grid_origin: GridOrigin
) -> GridOrigin:
    """
    Convert grid origin from fractional coordinates to cartesian coordinates
    using the same approach as the orthoganalize function.

    Args:
        grid: Gemmi FloatGrid object
        grid_origin: Current grid origin dictionary

    Returns:
        Updated grid_origin dictionary with cartesian coordinates
    """
    try:
        # Get the transformation matrix from fractional to cartesian coordinates
        frac_to_cart_matrix = grid.unit_cell.orth.mat

        # Get the grid start offset from the header
        # For most CCP4 maps, the grid starts at (0,0,0)
        start_u, start_v, start_w = 0, 0, 0

        # Calculate the fractional coordinates of the grid origin
        # The grid origin represents the position of grid point (0,0,0)
        grid_coords = gemmi.Position(start_u, start_v, start_w)

        # Convert to fractional coordinates
        frac_coords = grid.unit_cell.fractionalize(grid_coords)

        # Convert fractional coordinates to cartesian coordinates
        # Convert gemmi objects to numpy arrays for matrix operations
        frac_array = np.array([frac_coords.x, frac_coords.y, frac_coords.z])
        matrix_array = np.array(frac_to_cart_matrix)
        cartesian_coords = frac_array @ matrix_array.T

        # Update the grid origin with cartesian coordinates
        cartesian_origin = GridOrigin(cartesian_coords[0], cartesian_coords[1], cartesian_coords[2])

        log.info("🔧 Converted grid origin to cartesian coordinates:")
        log.info(
            f"   Fractional origin: ({frac_coords.x:.3f}, {frac_coords.y:.3f}, {frac_coords.z:.3f})"
        )
        log.info(
            f"   Cartesian origin: ({cartesian_origin.x:.3f}, {cartesian_origin.y:.3f}, {cartesian_origin.z:.3f}) Å"
        )

        return cartesian_origin

    except Exception as e:
        log.error(f"❌ Error converting grid origin to cartesian: {e}")
        log.warning("⚠️ Returning original grid origin")
        return grid_origin


def _calculate_proper_grid_spacing(
    grid: gemmi.FloatGrid,
) -> tuple[GridSpacing, GridOrigin]:
    """Calculate physical spacing and origin for a crystallographic grid. @@@"""

    centring_type = grid.spacegroup.centring_type()
    log.message("centring_type: %s", centring_type)

    frac_to_orth = get_grid_fractional_to_orthogonal_matrix(grid)

    if frac_to_orth is None:
        raise ValueError(
            "Failed to calculate fractional-to-orthogonal transformation"
        )

    spacing = GridSpacing.from_grid(
        grid,
        np.asarray(frac_to_orth),
    )

    grid_center = (
        (grid.shape[0] - 1) * spacing.x / 2,
        (grid.shape[1] - 1) * spacing.y / 2,
        (grid.shape[2] - 1) * spacing.z / 2,
    )
    frac_to_orth = get_grid_fractional_to_orthogonal_matrix(grid)

    if frac_to_orth is None:
        raise ValueError("Could not determine fractional-to-orthogonal transform")

    matrix = np.asarray(frac_to_orth)
    fractional_center = np.array([0.5, 0.5, 0.5])

    unit_cell_center = matrix @ fractional_center
    origin = GridOrigin(
        x=unit_cell_center[0] - grid_center[0],
        y=unit_cell_center[1] - grid_center[1],
        z=unit_cell_center[2] - grid_center[2],
    )

    origin.log_summary()
    spacing.log_summary()

    return spacing, origin


def get_grid_fractional_to_orthogonal_matrix(grid: gemmi.FloatGrid) -> np.ndarray:
    """
    Get the transformation matrix from fractional to orthogonal coordinates.

    :param grid: Gemmi FloatGrid object
    :param centring_type: string type of crystallographic system. Default is "P"

    :returns: 3x3 numpy array representing the transformation matrix
    """
    try:
        # For monoclinic systems, gemmi's primitive_orth_matrix gives incorrect β angles
        # We need to construct the matrix manually using the correct convention

        centring_type = grid.spacegroup.centring_type()

        frac_to_orth = grid.unit_cell.primitive_orth_matrix(
            centring_type=centring_type
        )  # for Orthogonal P system

        matrix = np.array(frac_to_orth, dtype=np.float64)

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
        # Get the transformation matrix from Gemmi using the correct API
        # For orthogonal to fractional, we need to invert the primitive_orth_matrix

        centring_type = grid.spacegroup.centring_type()

        frac_to_orth = grid.unit_cell.primitive_orth_matrix(centring_type)

        # Convert Mat33 to numpy array with explicit memory layout for macOS compatibility
        matrix = np.array(frac_to_orth, dtype=np.float64)

        # Ensure the matrix is contiguous and properly aligned for macOS
        matrix = np.ascontiguousarray(matrix, dtype=np.float64)

        # Validate matrix before inversion to prevent SIGBUS on macOS
        if matrix.shape != (3, 3):
            raise ValueError(f"Expected 3x3 matrix, got {matrix.shape}")

        # Check for singular matrix
        det = np.linalg.det(matrix)
        if abs(det) < 1e-12:
            raise ValueError(f"Matrix is singular (determinant: {det})")

        log.info(
            f"DEBUG: Matrix inversion - shape: {matrix.shape}, dtype: {matrix.dtype}, det: {det}"
        )
        log.info(f"DEBUG: Matrix is contiguous: {matrix.flags.c_contiguous}")

        # Invert the matrix to get orthogonal to fractional transformation
        # Use manual LU decomposition to avoid SIGBUS issues with np.linalg.inv on macOS
        try:
            # Use scipy.linalg.solve instead of np.linalg.inv to avoid SIGBUS
            from scipy.linalg import solve

            identity = np.eye(3, dtype=np.float64)
            orth_to_frac = solve(matrix, identity)
            log.info("DEBUG: Used scipy.linalg.solve for matrix inversion")
        except ImportError:
            # Fallback to manual LU decomposition if scipy not available
            try:
                from scipy.linalg import lu_factor, lu_solve

                lu, piv = lu_factor(matrix)
                identity = np.eye(3, dtype=np.float64)
                orth_to_frac = lu_solve((lu, piv), identity)
                log.info("DEBUG: Used scipy LU decomposition for matrix inversion")
            except ImportError:
                # Final fallback to pseudo-inverse (less accurate but safer)
                orth_to_frac = np.linalg.pinv(matrix)
                log.warning(
                    "⚠️ Used pseudo-inverse as final fallback (scipy not available)"
                )
        except Exception as e:
            log.error(f"❌ Error in matrix inversion: {e}")
            # Fallback to pseudo-inverse for numerical stability
            orth_to_frac = np.linalg.pinv(matrix)
            log.warning("⚠️ Used pseudo-inverse as fallback due to error")

        # Ensure result is also contiguous
        orth_to_frac = np.ascontiguousarray(orth_to_frac, dtype=np.float64)

        log.info(
            f"DEBUG: Inversion successful - result shape: {orth_to_frac.shape}, dtype: {orth_to_frac.dtype}"
        )

        return orth_to_frac

    except Exception as ex:
        log.error(f"❌ Error getting orthogonal to fractional matrix: {ex}")
        return None
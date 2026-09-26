"""
Utilities for loading and processing electron density maps from MTZ and CCP4 files.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any
import gemmi
import numpy as np
from gemmi import FloatGrid, Mtz
from numpy import ndarray, dtype


from picogl.core.mixin.vec3 import Vec3Mixin
from decologr import Decologr as log

MAP_NEGATIVE_RATIO_THRESHOLD = 0.7


class MapType(str, Enum):
    """Requested electron-density map type used to select MTZ coefficients.

    Values match the canonical ElMo ``MapType`` (``"2Fo-Fc"`` / ``"Fo-Fc"``)
    so either enum (or the plain string) can be passed to the loaders.
    """

    TWO_FO_FC = "2Fo-Fc"
    FO_FC = "Fo-Fc"
    UNKNOWN = "unknown"

    @classmethod
    def coerce(cls, map_type: "MapType | str") -> "MapType":
        """Normalize a map type (str or StrEnum) to this enum; raise on unknown.

        Accepts canonical values (``"2Fo-Fc"``, ``"Fo-Fc"``) and common aliases
        such as ``"2fofc"``, ``"fofc"``, ``"delfwt"``, ``"fwt"``.
        """
        if isinstance(map_type, cls):
            return map_type
        if not isinstance(map_type, str):
            raise ValueError(f"Unsupported map type: {map_type!r}")
        raw = map_type.strip()
        try:
            return cls(raw)
        except ValueError:
            pass
        normalized = raw.lower().replace("_", "-")
        if (
            normalized.startswith("2fo")
            or normalized.startswith("2mfo")
            or normalized in {"2fofc", "fwt"}
        ):
            return cls.TWO_FO_FC
        if normalized in {"fo-fc", "fofc", "delfwt", "difference"}:
            return cls.FO_FC
        if normalized in {"unknown", ""}:
            return cls.UNKNOWN
        raise ValueError(
            f"Unsupported map type: {map_type!r}. "
            f"Expected one of: {[m.value for m in cls]}"
        )


def load_density_map_with_columns(
    mtz_path: str, f_column: str, phi_column: str, sample_rate=0.0
) -> tuple[np.ndarray, CrystallographicInfo] | None:
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
        spacing, origin = _calculate_proper_grid_spacing(grid)
        return cls(
            dimensions=tuple(grid.shape),
            origin=origin,
            spacing=spacing,
            axis_order=_axis_order_from_gemmi(grid.axis_order),
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

    def log_grid_metadata(self) -> None:
        """Log the unit-cell and grid metadata of a crystallographic info object."""
        log.info(
            f"📐 Unit cell: a={self.unit_cell.a:.2f}, "
            f"b={self.unit_cell.b:.2f}, "
            f"c={self.unit_cell.c:.2f} Å"
        )
        log.info(f"📐 Grid dimensions: {self.grid.dimensions}")
        log.info(f"📐 Grid origin: {self.grid.origin}")
        log.info(f"📐 Axis order: {self.grid.axis_order}")

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
            a=float(unit_cell_data["a"]),
            b=float(unit_cell_data["b"]),
            c=float(unit_cell_data["c"]),
            alpha=float(unit_cell_data["alpha"]),
            beta=float(unit_cell_data["beta"]),
            gamma=float(unit_cell_data["gamma"]),
            source=unit_cell_data.get("source", ""),
            space_group=unit_cell_data.get("space_group", ""),
        )

        grid_origin = data["grid_origin"]
        origin = GridOrigin(
            x=float(grid_origin["x"]),
            y=float(grid_origin["y"]),
            z=float(grid_origin["z"]),
        )

        grid_spacing = data["grid_spacing"]
        spacing = GridSpacing(
            x=float(grid_spacing["x"]),
            y=float(grid_spacing["y"]),
            z=float(grid_spacing["z"]),
        )

        transforms = CoordinateTransforms(
            frac_to_orth=np.array(data.get("frac_to_orth") or [], dtype=float),
            orth_to_frac=np.array(data.get("orth_to_frac") or [], dtype=float),
        )

        return cls(
            unit_cell=unit_cell,
            space_group=data["space_group"],
            grid=MapGrid(
                dimensions=tuple(data["grid_dimensions"]),
                origin=origin,
                spacing=spacing,
                axis_order=AxisOrder(str(data["axis_order"])),
            ),
            transforms=transforms,
            map_type=data.get("map_type", ""),
        )

    def to_dict(self) -> dict:
        """Serialize to a plain dict that round-trips with from_dict()."""
        return {
            "unit_cell": {
                "a": self.unit_cell.a,
                "b": self.unit_cell.b,
                "c": self.unit_cell.c,
                "alpha": self.unit_cell.alpha,
                "beta": self.unit_cell.beta,
                "gamma": self.unit_cell.gamma,
                "space_group": self.unit_cell.space_group,
                "source": self.unit_cell.source,
            },
            "space_group": str(self.space_group),
            "map_type": self.map_type,
            "grid_dimensions": list(self.grid.dimensions),
            "grid_origin": {
                "x": self.grid.origin.x,
                "y": self.grid.origin.y,
                "z": self.grid.origin.z,
            },
            "grid_spacing": {
                "x": self.grid.spacing.x,
                "y": self.grid.spacing.y,
                "z": self.grid.spacing.z,
            },
            "axis_order": str(self.grid.axis_order.value),
            "frac_to_orth": np.asarray(self.transforms.frac_to_orth).tolist(),
            "orth_to_frac": np.asarray(self.transforms.orth_to_frac).tolist(),
        }


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

        axis_order = _axis_order_from_gemmi(grid.axis_order)
        np_array = axis_order.transpose_to_xyz(
            np.array(grid, copy=True),
        )

        # The returned array is guaranteed to be in X, Y, Z axis order
        # (numpy axis 0 -> X), matching the origin/spacing convention.
        crystallographic_info = crystallographic_info_from_grid(grid, map_type=map_type)
        crystallographic_info.grid.dimensions = tuple(np_array.shape)
        crystallographic_info.grid.axis_order = axis_order

        crystallographic_info.log_summary()

        return np_array, crystallographic_info

    except Exception as e:
        log.error(f"❌ Could not load map from {mtz_path}: {e}")
        return None


def crystallographic_info_from_grid(
    grid: FloatGrid, map_type: str = "CCP4_MAP"
) -> CrystallographicInfo:
    """Create crystallographic information from a Gemmi map grid."""

    grid_spacing, grid_origin = _calculate_proper_grid_spacing(grid)

    return CrystallographicInfo(
        unit_cell=UnitCell(
            a=grid.unit_cell.a,
            b=grid.unit_cell.b,
            c=grid.unit_cell.c,
            alpha=grid.unit_cell.alpha,
            beta=grid.unit_cell.beta,
            gamma=grid.unit_cell.gamma,
            space_group=str(grid.spacegroup),
        ),
        space_group=str(grid.spacegroup),
        grid=MapGrid(
            dimensions=tuple(grid.shape),
            origin=grid_origin,
            spacing=grid_spacing,
            axis_order=_axis_order_from_gemmi(grid.axis_order),
        ),
        transforms=CoordinateTransforms(
            frac_to_orth=get_grid_fractional_to_orthogonal_matrix(grid),
            orth_to_frac=get_grid_orthogonal_to_fractional_matrix(grid),
        ),
        map_type=map_type,
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
        if abs(det) < 1e-12:
            raise ValueError(f"Matrix is singular (determinant: {det})")

        orth_to_frac = np.linalg.inv(frac_to_orth)

        return np.ascontiguousarray(orth_to_frac, dtype=np.float64)

    except Exception as ex:
        log.error(f"❌ Error getting orthogonal to fractional matrix: {ex}")
        return None


def transform_grid_vertices_to_cartesian(
    vertices: np.ndarray,
    dimensions: tuple[int, int, int],
    frac_to_orth: np.ndarray,
    origin: GridOrigin | tuple[float, float, float] | None = None,
) -> np.ndarray:
    """Convert marching-cubes grid vertices to Cartesian Å coordinates.

    The volume is assumed to be XYZ-ordered: vertex (i, j, k) refers to the
    voxel at grid index i along the crystallographic X axis, etc. For a grid
    covering the full unit cell from its fractional origin, vertex (i, j, k)
    has fractional coordinates (i/nx, j/ny, k/nz), and its Cartesian position
    is ``frac @ frac_to_orth.T + origin`` (the transpose matters for
    non-orthogonal cells; with orthorhombic cells the matrix is symmetric
    and both forms coincide).

    Args:
        vertices: (n, 3) float array of grid vertices from marching_cubes()
        dimensions: XYZ grid shape: (nx, ny, nz)
        frac_to_orth: 3x3 fractional-to-orthogonal matrix (columns are the
            lattice vectors), i.e. the conventional cell orthogonalization
        origin: Cartesian position of grid voxel (0, 0, 0) in Å; default None
            means (0, 0, 0)

    Returns:
        (n, 3) float array of Cartesian Å coordinates
    """
    try:
        vertices = np.asarray(vertices, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(
                f"Expected (n, 3) vertex array, got {vertices.shape}"
            )

        dims = np.asarray(tuple(dimensions), dtype=np.float64)
        if dims.shape != (3,) or np.any(dims <= 0):
            raise ValueError(f"Expected positive XYZ dimensions, got {dims}")

        frac_to_orth = np.asarray(frac_to_orth, dtype=np.float64)
        if frac_to_orth.shape != (3, 3):
            raise ValueError(
                f"Expected 3x3 frac_to_orth matrix, got {frac_to_orth.shape}"
            )

        fractional = vertices / dims
        # frac_to_orth is gemmi's conventional orthogonalization matrix whose
        # COLUMNS are the lattice vectors (a, b, c), so Cartesian coordinates
        # follow gemmi's orthogonalize(): cart = frac @ M^T. The transpose is
        # essential for non-orthogonal cells where M is not symmetric.
        cartesian = fractional @ frac_to_orth.T

        if origin is not None:
            if isinstance(origin, GridOrigin):
                origin_vec = np.array(
                    [origin.x, origin.y, origin.z], dtype=np.float64
                )
            else:
                origin_vec = np.asarray(tuple(origin), dtype=np.float64)
            if origin_vec.shape != (3,):
                raise ValueError(
                    f"Expected 3-element origin, got {origin_vec.shape}"
                )
            cartesian = cartesian + origin_vec

        return np.ascontiguousarray(cartesian, dtype=np.float64)

    except Exception as ex:
        log.error(f"❌ Error transforming grid vertices: {ex}")
        raise
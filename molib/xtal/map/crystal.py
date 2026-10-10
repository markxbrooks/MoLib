from __future__ import annotations

from dataclasses import dataclass

import gemmi
import numpy as np

from decologr import Decologr as log
from molib.xtal.map.axis import _axis_order_from_gemmi, AxisOrder
from molib.xtal.map.grid import MapGrid, _calculate_proper_grid_spacing, get_grid_fractional_to_orthogonal_matrix, \
    get_grid_orthogonal_to_fractional_matrix, GridOrigin, GridSpacing
from molib.xtal.map.map_type import MapSource
from molib.xtal.map.unit_cell import UnitCell


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
    map_source: MapSource = MapSource.UNKNOWN

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

    @classmethod
    def from_grid(
            cls,
            grid: gemmi.FloatGrid,
            map_type: MapSource = MapSource.CCP4_MAP,
    ) -> "CrystallographicInfo":
        """Build crystallographic metadata from a Gemmi density grid."""
        grid_spacing, grid_origin = _calculate_proper_grid_spacing(grid)
        space_group = str(grid.spacegroup)

        return cls(
            unit_cell=UnitCell(
                a=grid.unit_cell.a,
                b=grid.unit_cell.b,
                c=grid.unit_cell.c,
                alpha=grid.unit_cell.alpha,
                beta=grid.unit_cell.beta,
                gamma=grid.unit_cell.gamma,
                space_group=space_group,
            ),
            space_group=space_group,
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
            map_source=map_type,
        )

    @staticmethod
    def grid_to_xyz_array(
            grid: gemmi.FloatGrid,
    ) -> np.ndarray:
        """Return a copy of the grid data in canonical XYZ axis order.

        Pure conversion: does not mutate this object's grid metadata.
        Call :meth:`sync_to_xyz_volume` after conversion when metadata must
        describe the returned array.
        """
        array = np.array(grid, copy=True)

        try:
            axis_order = AxisOrder.from_gemmi(grid.axis_order)
        except ValueError:
            log.warning(
                "Unknown grid axis order %r; assuming XYZ",
                grid.axis_order,
            )
            return array

        if axis_order is not AxisOrder.XYZ:
            array = axis_order.transpose_to_xyz(array)

        return array

    def sync_to_xyz_volume(self, volume: np.ndarray) -> None:
        """Align ``grid.axis_order`` and ``grid.dimensions`` with an XYZ volume.

        Call after :meth:`grid_to_xyz_array` (or equivalent) so metadata matches
        the NumPy volume used for carving/rendering. Does not recompute
        spacing or origin.
        """
        self.grid.axis_order = AxisOrder.XYZ
        self.grid.dimensions = tuple(int(x) for x in volume.shape)

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
            map_source=MapSource.coerce(data.get("map_type", "")),
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
            "map_type": self.map_source,
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


def crystallographic_info_from_grid(
    grid: gemmi.FloatGrid, map_source: MapSource = MapSource.CCP4_MAP
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
        map_source=map_source,
    )

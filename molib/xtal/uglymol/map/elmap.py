"""Electron density map domain object (uglymol).

Format-specific loading lives in :mod:`molib.xtal.uglymol.map.ccp4_loader`
and :mod:`molib.xtal.uglymol.map.dsn6_loader`. Viewer MTZ/CCP4 paths use
gemmi :class:`~molib.xtal.map.helper.DensityMapData` → :class:`~molib.xtal.map.info.MapInfo`.
"""

from __future__ import annotations

from molib.xtal.ccp4.map.volume import VolumeStatistics
from molib.xtal.uglymol.block import Block
from molib.xtal.uglymol.map.grid_array import GridArray
from molib.xtal.uglymol.unit_cell import UnitCellGeometry


class ElMap:
    """Electron density map: unit cell, grid, statistics, and local block ops."""

    unit = "e/Å³"

    def __init__(
        self,
        unit_cell: UnitCellGeometry,
        grid: GridArray,
        statistics: VolumeStatistics,
        block: Block | None = None,
    ) -> None:
        self.unit_cell = unit_cell
        self.grid = grid
        self.statistics = statistics
        self.block = block if block is not None else Block()

    def abs_level(self, sigma: float) -> float:
        """Absolute density for *sigma* contour levels.

        :param sigma: Contour level in standard deviations
        :return: Absolute density value
        """
        return self.statistics.mean + float(sigma) * self.statistics.std

    def extract_block(self, radius, center) -> None:
        """Fill :attr:`block` with density within *radius* of *center*."""
        grid = self.grid
        unit_cell = self.unit_cell
        if grid is None or unit_cell is None:
            return

        fc = unit_cell.fractionalize(center)
        r = [
            radius / unit_cell.parameters[0],
            radius / unit_cell.parameters[1],
            radius / unit_cell.parameters[2],
        ]
        grid_min = grid.frac2grid([fc[0] - r[0], fc[1] - r[1], fc[2] - r[2]])
        grid_max = grid.frac2grid([fc[0] + r[0], fc[1] + r[1], fc[2] + r[2]])
        size = [
            grid_max[0] - grid_min[0] + 1,
            grid_max[1] - grid_min[1] + 1,
            grid_max[2] - grid_min[2] + 1,
        ]
        points = []
        values = []

        for i in range(grid_min[0], grid_max[0] + 1):
            for j in range(grid_min[1], grid_max[1] + 1):
                for k in range(grid_min[2], grid_max[2] + 1):
                    frac = grid.grid2frac(i, j, k)
                    orth = unit_cell.orthogonalize(frac)
                    points.append(orth)
                    map_value = grid.get_grid_value(i, j, k)
                    values.append(map_value)

        self.block.set(points, values, size)

    def isomesh_in_block(self, sigma: float, method):
        """Isosurface mesh for the current :attr:`block` at *sigma*."""
        abs_level = self.abs_level(sigma)
        return self.block.isosurface(abs_level, method)

    def __repr__(self) -> str:
        return (
            f"ElMap("
            f"unit_cell={self.unit_cell!r}, "
            f"grid={self.grid!r}, "
            f"statistics={self.statistics!r}"
            f")"
        )

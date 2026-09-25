"""Load DSN6 maps into :class:`~molib.xtal.uglymol.map.elmap.ElMap`."""

from __future__ import annotations

from typing import Iterable, SupportsBytes, SupportsIndex

from typing_extensions import Buffer

from molib.xtal.ccp4.map.header import Ccp4MapHeaderLocation
from molib.xtal.ccp4.map.volume import VolumeStatistics
from molib.xtal.uglymol.map.elmap import ElMap
from molib.xtal.uglymol.map.grid_array import GridArray
from molib.xtal.uglymol.math.helpers import calculate_stddev
from molib.xtal.uglymol.unit_cell import UnitCellGeometry


class Dsn6MapLoader:
    """DSN6 binary map loader."""

    def load(
        self, buffer: Iterable[SupportsIndex] | Buffer | SupportsBytes
    ) -> ElMap:
        """Decode a DSN6 buffer into an :class:`~molib.xtal.uglymol.map.elmap.ElMap`.

        :param buffer: Raw DSN6 file bytes
        :return: Populated ElMap
        """
        u8data = bytearray(buffer)
        header_ints = [
            int.from_bytes(u8data[i : i + 2], "little", signed=True)
            for i in range(0, len(u8data), 2)
        ]

        if header_ints[Ccp4MapHeaderLocation.MAPS] != 100:
            len_iview = len(header_ints)
            for n in range(len_iview):
                val = header_ints[n]
                header_ints[n] = ((val & 0xFF) << 8) | ((val >> 8) & 0xFF)

        if header_ints[Ccp4MapHeaderLocation.MAPS] != 100:
            raise ValueError("Endian swap failed")

        origin = [header_ints[0], header_ints[1], header_ints[2]]
        n_real = [header_ints[3], header_ints[4], header_ints[5]]
        n_grid = [header_ints[6], header_ints[7], header_ints[8]]
        cell_mult = 1.0 / header_ints[17]
        unit_cell = UnitCellGeometry.from_parameters(
            cell_mult * header_ints[9],
            cell_mult * header_ints[10],
            cell_mult * header_ints[11],
            cell_mult * header_ints[12],
            cell_mult * header_ints[13],
            cell_mult * header_ints[14],
        )
        grid = GridArray(n_grid)
        prod = header_ints[15] / 100
        plus = header_ints[16]
        offset = 512
        n_blocks = [-(n_real[0] // -8), -(n_real[1] // -8), -(n_real[2] // -8)]

        for zz in range(n_blocks[2]):
            for yy in range(n_blocks[1]):
                for xx in range(n_blocks[0]):
                    for k in range(8):
                        z = 8 * zz + k
                        for j in range(8):
                            y = 8 * yy + j
                            for i in range(8):
                                x = 8 * xx + i
                                if x < n_real[0] and y < n_real[1] and z < n_real[2]:
                                    density = (u8data[offset] - plus) / prod
                                    offset += 1
                                    grid.set_grid_value(
                                        origin[0] + x,
                                        origin[1] + y,
                                        origin[2] + z,
                                        density,
                                    )
                                else:
                                    offset += 8 - i
                                    break

        mean, rms = calculate_stddev(grid.values, 0)
        values = grid.values
        statistics = VolumeStatistics(
            mean=mean,
            std=rms,
            min_value=float(min(values)) if values else 0.0,
            max_value=float(max(values)) if values else 0.0,
        )
        return ElMap(unit_cell=unit_cell, grid=grid, statistics=statistics)

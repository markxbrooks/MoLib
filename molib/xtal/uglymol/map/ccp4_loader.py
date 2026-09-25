"""Load CCP4/MRC maps into :class:`~molib.xtal.uglymol.map.elmap.ElMap`."""

from __future__ import annotations

import numpy as np
from molib.xtal.ccp4.map.globals import CCP4_HEADER_SIZE
from molib.xtal.ccp4.map.header import Ccp4MapHeaderLocation
from molib.xtal.ccp4.map.parameters import Ccp4MapParameters
from molib.xtal.ccp4.map.volume import VolumeStatistics
from molib.xtal.uglymol.map.elmap import ElMap
from molib.xtal.uglymol.map.grid_array import GridArray
from molib.xtal.uglymol.map.grid_symmetry import expand_grid_symmetry
from molib.xtal.uglymol.math.helpers import calculate_stddev


class Ccp4MapLoader:
    """CCP4/MRC binary map loader."""

    def load(self, map_buffer: bytes, *, expand_symmetry: bool = False) -> ElMap:
        """Decode *map_buffer* into an :class:`~molib.xtal.uglymol.map.elmap.ElMap`.

        :param map_buffer: Raw CCP4/MRC file bytes
        :param expand_symmetry: Whether to expand symmetry into the full cell
        :return: Populated ElMap
        """
        parameters = Ccp4MapParameters.from_header(map_buffer)
        assert parameters.unit_cell is not None

        if parameters.nsymbt % 4 != 0:
            raise ValueError(
                "CCP4 map with NSYMBT not divisible by 4 is not supported."
            )

        data_view = (
            parameters.floats
            if parameters.map_mode == 2
            else np.frombuffer(map_buffer, dtype=np.int8)
        )
        idx = (CCP4_HEADER_SIZE + parameters.nsymbt) // parameters.bytes_per_voxel
        statistics = self._build_statistics(parameters, data_view, idx)

        grid = GridArray(list(parameters.n_grid))
        self._populate_grid(
            parameters=parameters,
            data_view=data_view,
            idx=idx,
            grid=grid,
            expand_symmetry=expand_symmetry,
            map_buffer=map_buffer,
        )

        return ElMap(
            unit_cell=parameters.unit_cell,
            grid=grid,
            statistics=statistics,
        )

    @staticmethod
    def _build_statistics(
        parameters: Ccp4MapParameters,
        data_view,
        idx: int,
    ) -> VolumeStatistics:
        mean = parameters.mean
        rms = parameters.rms
        min_value = parameters.min_value
        max_value = parameters.max_value

        if mean < min_value or mean > max_value or rms <= 0:
            mean, rms = calculate_stddev(data_view, idx)

        return VolumeStatistics(
            mean=mean,
            std=rms,
            min_value=min_value,
            max_value=max_value,
        )

    @staticmethod
    def _populate_grid(
        *,
        parameters: Ccp4MapParameters,
        data_view,
        idx: int,
        grid: GridArray,
        expand_symmetry: bool,
        map_buffer: bytes,
    ) -> None:
        b1 = 1.0
        b0 = 0.0
        if (
            parameters.map_mode == 0
            and parameters.ints[Ccp4MapHeaderLocation.LSKFLG] == 0
            and parameters.ints[Ccp4MapHeaderLocation.SKWTRN] == 127
        ):
            b1 = (parameters.max_value - parameters.min_value) / 255.0
            b0 = 0.5 * (parameters.min_value + parameters.max_value + b1)

        end = [
            parameters.start[Ccp4MapHeaderLocation.NC] + parameters.n_crs[0],
            parameters.start[Ccp4MapHeaderLocation.NR] + parameters.n_crs[1],
            parameters.start[Ccp4MapHeaderLocation.NS] + parameters.n_crs[2],
        ]
        ax, ay, az = parameters.ax, parameters.ay, parameters.az
        it = [0, 0, 0]
        for it[Ccp4MapHeaderLocation.NS] in range(
            parameters.start[Ccp4MapHeaderLocation.NS],
            end[Ccp4MapHeaderLocation.NS],
        ):
            for it[Ccp4MapHeaderLocation.NR] in range(
                parameters.start[Ccp4MapHeaderLocation.NR],
                end[Ccp4MapHeaderLocation.NR],
            ):
                for it[Ccp4MapHeaderLocation.NC] in range(
                    parameters.start[Ccp4MapHeaderLocation.NC],
                    end[Ccp4MapHeaderLocation.NC],
                ):
                    grid.set_grid_value(
                        it[ax], it[ay], it[az], b1 * data_view[idx] + b0
                    )
                    idx += 1

        if expand_symmetry and parameters.nsymbt > 0:
            expand_grid_symmetry(
                ax=ax,
                ay=ay,
                az=az,
                b0=b0,
                b1=b1,
                bytes_per_voxel=parameters.bytes_per_voxel,
                data_view=data_view,
                end=end,
                grid=grid,
                parameters=parameters,
                it=it,
                map_buffer=map_buffer,
            )

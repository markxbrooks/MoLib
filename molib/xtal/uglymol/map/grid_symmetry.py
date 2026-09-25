"""Expand a CCP4 ASU grid using symmetry operators from the map buffer."""

from __future__ import annotations

from molib.xtal.ccp4.map.globals import CCP4_HEADER_SIZE, CCP4_SYMOP_CHUNK_SIZE
from molib.xtal.ccp4.map.header import Ccp4MapHeaderLocation
from molib.xtal.ccp4.map.parameters import Ccp4MapParameters
from molib.xtal.uglymol.map.grid_array import GridArray
from molib.xtal.uglymol.map.helpers import (
    extract_symop_text,
    match_symop_text,
    parse_symmetry_operator_to_matrix,
)


def expand_grid_symmetry(
    *,
    ax: int,
    ay: int,
    az: int,
    b0: float,
    b1: float,
    bytes_per_voxel: int,
    data_view,
    end,
    grid: GridArray,
    parameters: Ccp4MapParameters,
    it,
    map_buffer: bytes,
) -> None:
    """Apply CCP4 symmetry operators to populate *grid* beyond the ASU.

    :param ax: Index of X in CRS order
    :param ay: Index of Y in CRS order
    :param az: Index of Z in CRS order
    :param b0: Linear density bias
    :param b1: Linear density scale
    :param bytes_per_voxel: 1 (mode 0) or 4 (mode 2)
    :param data_view: Flat density buffer
    :param end: Exclusive CRS end indices
    :param grid: Destination grid
    :param parameters: Parsed CCP4 parameters
    :param it: Mutable CRS index scratch ``[0, 0, 0]``
    :param map_buffer: Full map file bytes (for symop text)
    """
    for i in range(0, parameters.nsymbt, CCP4_SYMOP_CHUNK_SIZE):
        symop = extract_symop_text(map_buffer, i)
        if match_symop_text(symop):
            continue
        symop_matrix = parse_symmetry_operator_to_matrix(symop)
        for j in range(3):
            symop_matrix[j][3] = round(symop_matrix[j][3] * parameters.n_grid[j])
        idx = (CCP4_HEADER_SIZE + parameters.nsymbt) // bytes_per_voxel
        xyz = [0, 0, 0]
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
                    for j in range(3):
                        xyz[j] = (
                            it[ax] * symop_matrix[j][0]
                            + it[ay] * symop_matrix[j][1]
                            + it[az] * symop_matrix[j][2]
                            + symop_matrix[j][3]
                        )
                    grid.set_grid_value(
                        xyz[0], xyz[1], xyz[2], b1 * data_view[idx] + b0
                    )
                    idx += 1

"""CCP4 map header parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np
from molib.xtal.ccp4.map.globals import CCP4_HEADER_SIZE, CCP4_MAP_SIGNATURE
from molib.xtal.ccp4.map.header import Ccp4MapHeaderLocation
from molib.xtal.uglymol.map.helpers import (
    extract_symop_text,
    parse_symmetry_operator_to_matrix,
)
from molib.xtal.uglymol.unit_cell import UnitCellGeometry


@dataclass
class Ccp4MapParameters:
    """Interpreted CCP4/MRC map header (and buffers needed for voxel decode)."""

    map_mode: int = 2
    n_crs: Tuple[int, int, int] = (0, 0, 0)
    start: Tuple[int, int, int] = (0, 0, 0)
    n_grid: Tuple[int, int, int] = (0, 0, 0)
    map_crs: Tuple[int, int, int] = (1, 2, 3)
    nsymbt: int = 0
    min_value: float = 0.0
    max_value: float = 0.0
    mean: float = 0.0
    rms: float = 1.0
    unit_cell: Optional[UnitCellGeometry] = None
    ints: Any = None
    floats: Any = None
    map_buffer: Optional[bytes] = None
    data_view: Any = None
    ax: int = 0
    ay: int = 1
    az: int = 2
    b0: float = 0.0
    b1: float = 1.0
    end: Optional[List[int]] = None
    bytes_per_voxel: int = 4

    def extract_symop(self, i: int) -> str:
        """Extract one symmetry-operator string from the map buffer."""
        if self.map_buffer is None:
            raise ValueError("map_buffer is not set")
        return extract_symop_text(self.map_buffer, i)

    def parse_symop(self, symop: str):
        """Parse a symmetry operator string to a 3x4 matrix."""
        return parse_symmetry_operator_to_matrix(symop)

    @classmethod
    def from_header(cls, map_buffer: bytes) -> "Ccp4MapParameters":
        """Parse and validate a CCP4 header from *map_buffer*.

        :param map_buffer: Full CCP4/MRC file bytes
        :return: Populated parameters (unit cell set; voxel data not decoded)
        :raises ValueError: On short/invalid/unsupported maps
        """
        if len(map_buffer) < CCP4_HEADER_SIZE:
            raise ValueError("File shorter than 1024 bytes.")

        ints = np.frombuffer(map_buffer[:CCP4_HEADER_SIZE], dtype=np.int32)
        if ints[Ccp4MapHeaderLocation.MAP] != CCP4_MAP_SIGNATURE:
            raise ValueError("not a CCP4 map")

        floats = np.frombuffer(map_buffer, dtype=np.float32)
        map_mode = int(ints[Ccp4MapHeaderLocation.MODE])
        bytes_per_voxel = {2: 4, 0: 1}.get(map_mode)
        if bytes_per_voxel is None:
            raise ValueError("Only Mode 2 and Mode 0 are supported")

        n_crs = tuple(int(x) for x in ints[: Ccp4MapHeaderLocation.NS + 1])
        start = tuple(
            int(x)
            for x in ints[
                Ccp4MapHeaderLocation.NCSTART : Ccp4MapHeaderLocation.NSSTART + 1
            ]
        )
        n_grid = tuple(
            int(x)
            for x in ints[Ccp4MapHeaderLocation.NX : Ccp4MapHeaderLocation.NZ + 1]
        )
        nsymbt = int(ints[Ccp4MapHeaderLocation.NSYMBT])

        cls._validate_file_size(bytes_per_voxel, map_buffer, n_crs, nsymbt)
        cls._validate_header(ints, floats)

        map_crs = tuple(
            int(x)
            for x in ints[Ccp4MapHeaderLocation.MAPC : Ccp4MapHeaderLocation.MAPS + 1]
        )
        if sorted(map_crs) != [1, 2, 3]:
            raise ValueError(
                "Invalid axis mapping: MAPC, MAPR, MAPS must be unique and in [1, 2, 3]"
            )

        ax = list(map_crs).index(1)
        ay = list(map_crs).index(2)
        az = list(map_crs).index(3)

        unit_cell = UnitCellGeometry.from_parameters(
            floats[Ccp4MapHeaderLocation.X_LENGTH],
            floats[Ccp4MapHeaderLocation.Y_LENGTH],
            floats[Ccp4MapHeaderLocation.Z_LENGTH],
            floats[Ccp4MapHeaderLocation.ALPHA],
            floats[Ccp4MapHeaderLocation.BETA],
            floats[Ccp4MapHeaderLocation.GAMMA],
        )

        min_value = float(floats[Ccp4MapHeaderLocation.AMIN])
        max_value = float(floats[Ccp4MapHeaderLocation.AMAX])
        mean = float(floats[Ccp4MapHeaderLocation.AMEAN])
        rms = float(floats[Ccp4MapHeaderLocation.ARMS])

        return cls(
            map_mode=map_mode,
            n_crs=n_crs,
            start=start,
            n_grid=n_grid,
            map_crs=map_crs,
            nsymbt=nsymbt,
            min_value=min_value,
            max_value=max_value,
            mean=mean,
            rms=rms,
            unit_cell=unit_cell,
            ints=ints,
            floats=floats,
            map_buffer=map_buffer,
            ax=ax,
            ay=ay,
            az=az,
            bytes_per_voxel=bytes_per_voxel,
        )

    @staticmethod
    def _validate_header(
        header_ints: Sequence[int], header_floats: Sequence[float]
    ) -> None:
        if header_ints[Ccp4MapHeaderLocation.MAP] != CCP4_MAP_SIGNATURE:
            raise ValueError("Invalid CCP4 map: Missing 'MAP ' signature")
        if header_ints[Ccp4MapHeaderLocation.MODE] not in [0, 2]:
            raise ValueError(
                f"Unsupported CCP4 mode: {header_ints[Ccp4MapHeaderLocation.MODE]}"
            )
        if header_ints[Ccp4MapHeaderLocation.NSYMBT] % 4 != 0:
            raise ValueError("Invalid CCP4 map: NSYMBT not divisible by 4")
        if (
            header_floats[Ccp4MapHeaderLocation.AMIN]
            > header_floats[Ccp4MapHeaderLocation.AMAX]
        ):
            raise ValueError("Invalid CCP4 map: AMIN > AMAX")

    @staticmethod
    def _validate_file_size(
        bytes_per_voxel: int,
        map_buffer: bytes,
        n_crs: Sequence[int],
        nsymbt: int,
    ) -> None:
        expected_size = (
            CCP4_HEADER_SIZE
            + nsymbt
            + bytes_per_voxel
            * n_crs[Ccp4MapHeaderLocation.NC]
            * n_crs[Ccp4MapHeaderLocation.NR]
            * n_crs[Ccp4MapHeaderLocation.NS]
        )
        if expected_size != len(map_buffer):
            raise ValueError("CCP4 file size mismatch (too short or too long).")

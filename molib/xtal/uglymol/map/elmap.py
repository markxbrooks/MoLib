"""

For reference CCP4 Map header information

 1      NC              # of Columns    (fastest changing in map)
 2      NR              # of Rows
 3      NS              # of Sections   (slowest changing in map)
 4      MODE            Data type
                          0 = envelope stored as signed bytes (from
                              -128 lowest to 127 highest)
                          1 = Image     stored as Integer*2
                          2 = Image     stored as Reals
                          3 = Transform stored as Complex Integer*2
                          4 = Transform stored as Complex Reals
                          5 == 0

                          Note: Mode 2 is the normal mode used in
                                the CCP4 programs. Other modes than 2 and 0
                                may NOT WORK

 5      NCSTART         Number of first COLUMN  in map
 6      NRSTART         Number of first ROW     in map
 7      NSSTART         Number of first SECTION in map
 8      NX              Number of intervals along X
 9      NY              Number of intervals along Y
10      NZ              Number of intervals along Z
11      X length        Cell Dimensions (Angstroms)
12      Y length                     "
13      Z length                     "
14      Alpha           Cell Angles     (Degrees)
15      Beta                         "
16      Gamma                        "
17      MAPC            Which axis corresponds to Cols.  (1,2,3 for X,Y,Z)
18      MAPR            Which axis corresponds to Rows   (1,2,3 for X,Y,Z)
19      MAPS            Which axis corresponds to Sects. (1,2,3 for X,Y,Z)
20      AMIN            Minimum density value
21      AMAX            Maximum density value
22      AMEAN           Mean    density value    (Average)
23      ISPG            Space group number
24      NSYMBT          Number of bytes used for storing symmetry operators
25      LSKFLG          Flag for skew transformation, =0 none, =1 if foll
26-34   SKWMAT          Skew matrix S (in order S11, S12, S13, S21 etc) if
                        LSKFLG .ne. 0.
35-37   SKWTRN          Skew translation t if LSKFLG .ne. 0.
                        Skew transformation is from standard orthogonal
                        coordinate frame (as used for atoms) to orthogonal
                        map frame, as

                                Xo(map) = S * (Xo(atoms) - t)

38      future use       (some of these are used by the MSUBSX routines
 .          "              in MAPBRICK, MAPCONT and FRODO)
 .          "   (all set to zero by default)
 .          "
52          "

53  MAP         Character string 'MAP ' to identify file type
54  MACHST      Machine stamp indicating the machine type
            which wrote file
55      ARMS            Rms deviation of map from mean density
56      NLABL           Number of labels being used
57-256  LABEL(20,10)    10  80 character text labels (ie. A4 format)

"""

import re
from typing import Iterable, SupportsBytes, SupportsIndex

import numpy as np
from molib.xtal.ccp4.map.globals import (
    CCP4_HEADER_SIZE,
    CCP4_LABEL_SIZE,
    CCP4_MAP_SIGNATURE,
    CCP4_SYMOP_CHUNK_SIZE,
)
from molib.xtal.ccp4.map.header import Ccp4MapHeaderLocation
from molib.xtal.ccp4.map.parameters import Ccp4MapParameters
from molib.xtal.uglymol.block import Block
from molib.xtal.uglymol.map.grid_array import GridArray
from molib.xtal.uglymol.map.helpers import (
    extract_symop_text,
    match_symop_text,
    parse_symmetry_operator_to_matrix,
)
from molib.xtal.uglymol.math.helpers import calculate_stddev
from molib.xtal.uglymol.unit_cell import UnitCell
from typing_extensions import Buffer


class ElMap:
    """ElMap"""

    def __init__(self):
        self.parameters = None
        self._unit_cell = None
        self._grid = None
        self._stats = {"mean": 0.0, "rms": 1.0}
        self.block = Block()

    @property
    def unit_cell(self):
        return self._unit_cell

    @unit_cell.setter
    def unit_cell(self, unit_cell) -> None:
        self._unit_cell = unit_cell

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, grid) -> None:
        self._grid = grid

    @property
    def stats(self):
        return self._stats

    @stats.setter
    def stats(self, stats) -> None:
        self._stats = stats

    def _extract_labels(self, map_buffer: bytes) -> list[str]:
        """_extract_labels"""
        labels = []
        for i in range(56, 256, CCP4_LABEL_SIZE):
            label = (
                map_buffer[CCP4_HEADER_SIZE + i : CCP4_HEADER_SIZE + i + 20]
                .decode("ascii", errors="ignore")
                .strip()
            )
            if label:
                labels.append(label)
        return labels

    def _validate_ccp4_map_header(
        self, header_ints: list[int], header_floats: list[float]
    ) -> None:
        """_validate_ccp4_map_header"""
        if (
            header_ints[Ccp4MapHeaderLocation.MAP] != CCP4_MAP_SIGNATURE
        ):  # Check 'MAP ' signature
            raise ValueError("Invalid CCP4 map: Missing 'MAP ' signature")
        if header_ints[Ccp4MapHeaderLocation.MODE] not in [0, 2]:  # Supported modes
            raise ValueError(
                f"Unsupported CCP4 mode: {header_ints[Ccp4MapHeaderLocation.MODE]}"
            )
        if (
            header_ints[Ccp4MapHeaderLocation.NSYMBT] % 4 != 0
        ):  # NSYMBT must be divisible by 4
            raise ValueError("Invalid CCP4 map: NSYMBT not divisible by 4")
        if (
            header_floats[Ccp4MapHeaderLocation.AMIN]
            > header_floats[Ccp4MapHeaderLocation.AMAX]
        ):  # AMIN <= AMAX
            raise ValueError("Invalid CCP4 map: AMIN > AMAX")

    def abs_level(self, sigma: int) -> float:
        """abs_level"""
        return sigma * self.stats["rms"] + self.stats["mean"]

    def from_ccp4(self, map_buffer: bytes, expand_symmetry: bool = False) -> None:
        """
        Load a CCP4 density map from a binary buffer.

        Parameters
        ----------
        map_buffer : bytes
            The raw binary contents of a CCP4/MRC map file.
        expand_symmetry : bool, default=True
            Whether to expand symmetry operators from the file into the grid.

        Raises
        ------
        ValueError
            If the file is too short, incorrectly formatted, or has unsupported modes.
        """
        self.parameters = parameters = Ccp4MapParameters()
        # --- Basic header checks ---
        if len(map_buffer) < CCP4_HEADER_SIZE:
            raise ValueError("File shorter than 1024 bytes.")
        parameters.ints = np.frombuffer(map_buffer[:CCP4_HEADER_SIZE], dtype=np.int32)
        if parameters.ints[Ccp4MapHeaderLocation.MAP] != CCP4_MAP_SIGNATURE:
            raise ValueError("not a CCP4 map")
        parameters.floats = np.frombuffer(map_buffer, dtype=np.float32)

        # --- Check map mode ---
        self.parameters.map_mode = parameters.ints[Ccp4MapHeaderLocation.MODE]
        bytes_per_voxel = {2: 4, 0: 1}.get(self.parameters.map_mode)
        if bytes_per_voxel is None:
            raise ValueError("Only Mode 2 and Mode 0 are supported")

        # --- Parse map dimensions ---
        self.parameters.n_crs = parameters.ints[: Ccp4MapHeaderLocation.NS + 1]
        self.parameters.start = parameters.ints[
            Ccp4MapHeaderLocation.NCSTART : Ccp4MapHeaderLocation.NSSTART + 1
        ]
        parameters.n_grid = parameters.ints[
            Ccp4MapHeaderLocation.NX : Ccp4MapHeaderLocation.NZ + 1
        ]
        parameters.nsymbt = parameters.ints[Ccp4MapHeaderLocation.NSYMBT]

        self._validate_ccp4_map_file_size(
            bytes_per_voxel, map_buffer, self.parameters.n_crs, parameters.nsymbt
        )
        self._validate_ccp4_map_header(
            header_ints=parameters.ints, header_floats=parameters.floats
        )

        self._initialize_unit_cell(parameters)
        parameters.map_crs = parameters.ints[
            Ccp4MapHeaderLocation.MAPC : Ccp4MapHeaderLocation.MAPS + 1
        ]
        ax = list(parameters.map_crs).index(1)
        ay = list(parameters.map_crs).index(2)
        az = list(parameters.map_crs).index(3)

        if sorted(parameters.map_crs) != [1, 2, 3]:
            raise ValueError(
                "Invalid axis mapping: MAPC, MAPR, MAPS must be unique and in [1, 2, 3]"
            )

        parameters.min_val = parameters.floats[Ccp4MapHeaderLocation.AMIN]
        parameters.max_val = parameters.floats[Ccp4MapHeaderLocation.AMAX]
        grid = GridArray(parameters.n_grid)
        if parameters.nsymbt % 4 != 0:
            raise ValueError(
                "CCP4 map with NSYMBT not divisible by 4 is not supported."
            )
        data_view = (
            parameters.floats
            if self.parameters.map_mode == 2
            else np.frombuffer(map_buffer, dtype=np.int8)
        )
        idx = (CCP4_HEADER_SIZE + parameters.nsymbt) // bytes_per_voxel

        self._process_stats(
            data_view, parameters.floats, idx, parameters.max_val, parameters.min_val
        )
        self._populate_grid(
            ax,
            ay,
            az,
            bytes_per_voxel,
            data_view,
            expand_symmetry,
            grid,
            parameters,
            idx,
            map_buffer,
        )
        self.grid = grid

    def _initialize_unit_cell(self, parameters: Ccp4MapParameters):
        """
        _initialize_unit_cell

        :param parameters: CCP4MapParameters
        """
        self.unit_cell = UnitCell(
            parameters.floats[Ccp4MapHeaderLocation.X_LENGTH],
            parameters.floats[Ccp4MapHeaderLocation.Y_LENGTH],
            parameters.floats[Ccp4MapHeaderLocation.Z_LENGTH],
            parameters.floats[Ccp4MapHeaderLocation.ALPHA],
            parameters.floats[Ccp4MapHeaderLocation.BETA],
            parameters.floats[Ccp4MapHeaderLocation.GAMMA],
        )

    def _populate_grid(
        self,
        ax,
        ay,
        az,
        bytes_per_voxel,
        data_view,
        expand_symmetry,
        grid,
        parameters: Ccp4MapParameters,
        idx: int,
        map_buffer: bytes,
    ):
        b1 = 1
        b0 = 0
        if (
            self.parameters.map_mode == 0
            and parameters.ints[Ccp4MapHeaderLocation.LSKFLG] == 0
            and parameters.ints[Ccp4MapHeaderLocation.SKWTRN] == 127
        ):
            b1 = (parameters.max_val - parameters.min_val) / 255.0
            b0 = 0.5 * (parameters.min_val + parameters.max_val + b1)
        end = [
            parameters.start[Ccp4MapHeaderLocation.NC] + parameters.n_crs[0],
            parameters.start[Ccp4MapHeaderLocation.NR] + parameters.n_crs[1],
            parameters.start[Ccp4MapHeaderLocation.NS] + self.parameters.n_crs[2],
        ]
        it = [0, 0, 0]
        for it[Ccp4MapHeaderLocation.NS] in range(
            self.parameters.start[Ccp4MapHeaderLocation.NS],
            end[Ccp4MapHeaderLocation.NS],
        ):  # sections
            for it[Ccp4MapHeaderLocation.NR] in range(
                self.parameters.start[Ccp4MapHeaderLocation.NR],
                end[Ccp4MapHeaderLocation.NR],
            ):  # rows
                for it[Ccp4MapHeaderLocation.NC] in range(
                    self.parameters.start[Ccp4MapHeaderLocation.NC],
                    end[Ccp4MapHeaderLocation.NC],
                ):  # cols
                    grid.set_grid_value(
                        it[ax], it[ay], it[az], b1 * data_view[idx] + b0
                    )
                    idx += 1
        if expand_symmetry and parameters.nsymbt > 0:
            self.expand_grid_symmetry(
                ax,
                ay,
                az,
                b0,
                b1,
                bytes_per_voxel,
                data_view,
                end,
                grid,
                parameters,
                it,
                map_buffer,
            )

    def expand_grid_symmetry(
        self,
        ax,
        ay,
        az,
        b0,
        b1,
        bytes_per_voxel: int,
        data_view,
        end,
        grid,
        parameters: Ccp4MapParameters,
        it,
        map_buffer: bytes,
    ):
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
                self.parameters.start[Ccp4MapHeaderLocation.NS],
                end[Ccp4MapHeaderLocation.NS],
            ):  # sections
                for it[Ccp4MapHeaderLocation.NR] in range(
                    self.parameters.start[Ccp4MapHeaderLocation.NR],
                    end[Ccp4MapHeaderLocation.NR],
                ):  # rows
                    for it[Ccp4MapHeaderLocation.NC] in range(
                        self.parameters.start[Ccp4MapHeaderLocation.NC],
                        end[Ccp4MapHeaderLocation.NC],
                    ):  # cols
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

    def _process_stats(self, data_view, header_floats, idx, max_val, min_val):
        self.stats["mean"] = header_floats[Ccp4MapHeaderLocation.AMEAN]
        self.stats["rms"] = header_floats[Ccp4MapHeaderLocation.ARMS]
        if (
            self.stats["mean"] < min_val
            or self.stats["mean"] > max_val
            or self.stats["rms"] <= 0
        ):
            self.stats = calculate_stddev(data_view, idx)

    def _validate_ccp4_map_file_size(self, bytes_per_voxel, map_buffer, n_crs, nsymbt):
        # Validate file size
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

    def from_dsn6(self, buffer: Iterable[SupportsIndex] | Buffer | SupportsBytes):
        """from_dsn6"""
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
        self.unit_cell = UnitCell(
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

        self.stats = calculate_stddev(grid.values, 0)
        self.grid = grid

    def show_debug_info(self):
        """show_debug_info"""
        print("unit cell:", self.unit_cell.parameters if self.unit_cell else None)
        print("grid:", self.grid.dim if self.grid else None)

    def extract_block(self, radius, center):
        """extract block"""
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

    def isomesh_in_block(self, sigma: int, method):
        """isomesh_in_block(sigma, method)"""
        abs_level = self.abs_level(sigma)
        return self.block.isosurface(abs_level, method)

    unit = "e/Å³"

    def parse_symop(self, symop):
        """parse symop"""
        ops = symop.lower().replace(" ", "").split(",")
        if len(ops) != 3:
            raise ValueError("Unexpected symop: " + symop)

        mat = []
        for i in range(3):
            terms = re.split(r"(?=[+-])", ops[i])
            row = [0, 0, 0, 0]
            for term in terms:
                sign = -1 if term[0] == "-" else 1
                m = re.match(r"^[+-]?([xyz])$", term)
                if m:
                    pos = {"x": 0, "y": 1, "z": 2}[m[1]]
                    row[pos] = sign
                else:
                    m = re.match(r"^[+-]?(\d)/(\d)$", term)
                    if not m:
                        raise ValueError("What is " + term + " in " + symop)
                    row[3] = sign * int(m[1]) / int(m[2])
            mat.append(row)
        return mat

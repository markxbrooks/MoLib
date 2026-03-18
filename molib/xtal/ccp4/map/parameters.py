from dataclasses import dataclass

import numpy as np
from molib.xtal.uglymol.map.grid_array import GridArray
from molib.xtal.uglymol.map.helpers import (
    extract_symop_text,
    parse_symmetry_operator_to_matrix,
)


@dataclass
class Ccp4MapParameters:
    map_crs = None
    map_buffer: bytes = None
    ints: int | bytes | list[int] | np.ndarray | None = None
    floats: int | bytes | list[int] | np.ndarray = None
    nsymbt: int = 0
    nb: int = 0
    min_val = None
    max_val = None
    data_view: bytes = None
    start: list[int] = None
    end: list[int] = None
    ax: int = 0
    ay: int = 0
    az: int = 0
    b0: int = 0
    b1: int = 1
    n_grid: list[int] = None
    grid: GridArray = None

    def extract_symop(self, i):
        return extract_symop_text(self.map_buffer, i)

    def parse_symop(self, symop):
        return parse_symmetry_operator_to_matrix(symop)

    def extract_data(self):
        return self.data_view

    def extract_start(self):
        return self.start

    def extract_end(self):
        return self.end

    def extract_ax(self):
        return self.ax

    def extract_ay(self):
        return self.ay

    def extract_az(self):
        return self.az

    def extract_b0(self):
        return self.b0

    def extract_b1(self):
        return self.b1

    def extract_nsymbt(self):
        return self.nsymbt

    def extract_nb(self):
        return self.nb

    def extract_data_view(self):
        return self.data_view

    def extract_n_grid(self):
        return self.n_grid

    def extract_grid(self):
        return self.grid

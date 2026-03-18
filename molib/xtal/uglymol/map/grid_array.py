"""Grid array"""

from molib.xtal.uglymol.math.helpers import modulo


class GridArray:
    """Grid Array"""

    def __init__(self, dim):
        self.dim = dim  # dimensions of the grid for the entire unit cell
        self.values = [0.0] * (dim[0] * dim[1] * dim[2])

    def grid2index(self, i, j, k):
        """grid2index"""
        i = modulo(i, self.dim[0])
        j = modulo(j, self.dim[1])
        k = modulo(k, self.dim[2])
        return self.dim[2] * (self.dim[1] * i + j) + k

    def grid2index_unchecked(self, i, j, k):
        """grid2index_unchecked"""
        return self.dim[2] * (self.dim[1] * i + j) + k

    def grid2frac(self, i, j, k):
        """grid2frac"""
        return [i / self.dim[0], j / self.dim[1], k / self.dim[2]]

    def frac2grid(self, xyz):
        """frac2grid"""
        return [
            int(xyz[0] * self.dim[0]),
            int(xyz[1] * self.dim[1]),
            int(xyz[2] * self.dim[2]),
        ]

    def set_grid_value(self, i, j, k, value):
        """set_grid_value"""
        idx = self.grid2index(i, j, k)
        self.values[idx] = value

    def get_grid_value(self, i, j, k):
        """get_grid_value"""
        idx = self.grid2index(i, j, k)
        return self.values[idx]

    def set_grid_values(self, ax_indices, ay_indices, az_indices, data_values):
        pass

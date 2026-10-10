from __future__ import annotations

import numpy as np

from decologr import Decologr as log
from molib.xtal.map.grid import GridOrigin


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

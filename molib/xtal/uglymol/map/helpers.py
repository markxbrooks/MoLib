import re

import numpy as np

from molib.xtal.ccp4.map.globals import (
    CCP4_HEADER_SIZE,
    CCP4_SYMOP_CHUNK_SIZE,
    CCP4_SYMOP_REGEX_MATCH,
)


def vectorized_apply_symmetry(
    expand_symmetry,
    nsymbt,
    map_buffer,
    nb,
    data_view,
    start,
    end,
    ax,
    ay,
    az,
    b0,
    b1,
    n_grid,
    grid,
):
    if not (expand_symmetry and nsymbt > 0):
        return

    # Gather all non-identity symmetry operations first
    mats = []
    for i in range(0, nsymbt, CCP4_SYMOP_CHUNK_SIZE):
        symop = extract_symop_text(map_buffer, i)
        # Skip identity
        if match_symop_text(symop):
            continue
        symop_matrix = parse_symmetry_operator_to_matrix(symop)  # Expect 3x4
        # Scale translation part by n_grid
        for j in range(3):
            symop_matrix[j][3] = round(symop_matrix[j][3] * n_grid[j])
        mats.append(np.asarray(symop_matrix, dtype=np.float64))  # shape (3,4)

    if not mats:
        return

    # Prepare the 3D index grid for the target block
    # Create arrays of indices along each axis
    x_idx = np.arange(start[0], end[0], dtype=np.float64)
    y_idx = np.arange(start[1], end[1], dtype=np.float64)
    z_idx = np.arange(start[2], end[2], dtype=np.float64)

    # Create a 3D grid of coordinates. We'll use broadcasting to apply the affine
    # We treat each voxel coordinate as [i, j, k] and apply mat to it.
    X, Y, Z = np.meshgrid(x_idx, y_idx, z_idx, indexing="ij")  # shape (nx, ny, nz)

    # Flatten to N voxels for vectorized transform
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=0)  # shape (3, N)

    # idx mapping
    idx_start = (CCP4_HEADER_SIZE + nsymbt) // nb

    # Process each symmetry operation in a vectorized batch
    # We'll accumulate results for all mats, but to avoid huge memory, process per mat
    idx = idx_start

    for symop_matrix in mats:
        A = symop_matrix[:, :3]  # 3x3
        t = symop_matrix[:, 3]  # 3

        # Apply affine: xyz = A @ [i, j, k] + t
        xyz = (A @ pts) + t[:, None]  # shape (3, N)
        x = xyz[0, :]
        y = xyz[1, :]
        z = xyz[2, :]

        # If grid stores in a NumPy array, we can vectorize indexing
        # But ensure coordinates are within valid grid bounds
        # If grid has a set_grid_values method, adapt accordingly.
        # Here we assume grid.data is a 3D NumPy array with shape (X, Y, Z)

        # Build mask for valid integer indices (if grid uses integer indices)
        valid = (x >= 0) & (y >= 0) & (z >= 0)
        # If grid dimensions are known, add bounds check:
        # (x < grid.shape[0]) & (y < grid.shape[1]) & (z < grid.shape[2])

        # For valid indices, cast to int
        xi = x[valid].astype(np.int64)
        yi = y[valid].astype(np.int64)
        zi = z[valid].astype(np.int64)

        values = b1 * data_view[idx : idx + valid.sum()] + b0  # align values
        # Assign into grid with vectorized indexing
        # Example if grid.data is a NumPy array:
        grid.data[xi, yi, zi] = values

        idx += valid.sum()  # advance idx by number of voxels processed for this mat

    # end function


def parse_symmetry_operator_to_matrix(symmetry_operator: str) -> list:
    """parse_symmetry_operator_to_matrix"""
    ops = symmetry_operator.lower().replace(" ", "").split(",")
    if len(ops) != 3:
        raise ValueError("Unexpected symop: " + symmetry_operator)
    mat = []
    for i in range(3):
        terms = re.split(r"(?=[+-])", ops[i])
        row = [0, 0, 0, 0]
        for term in terms:
            if not term:  # Skip empty terms
                continue
            sign = -1 if term[0] == "-" else 1
            m = re.match(r"^[+-]?([xyz])$", term)
            if m:
                pos = {"x": 0, "y": 1, "z": 2}[m[1]]
                row[pos] = sign
            else:
                m = re.match(r"^[+-]?(\d)/(\d)$", term)
                if not m:
                    raise ValueError("What is " + term + " in " + symmetry_operator)
                row[3] = sign * int(m[1]) / int(m[2])
        mat.append(row)
    return mat


def extract_symop_text(map_buffer: bytes, offset: int) -> str:
    """Extract symmetry operation text from buffer at given offset

    Args:
        map_buffer: The CCP4 map buffer
        offset: Offset from the start of symmetry operations (in bytes)

    Returns:
        String containing the symmetry operation text
    """
    start_idx = CCP4_HEADER_SIZE + offset
    end_idx = start_idx + CCP4_SYMOP_CHUNK_SIZE
    return map_buffer[start_idx:end_idx].decode("ascii", errors="ignore")


def match_symop_text(symop: str):
    """match_symop_text"""
    return re.match(CCP4_SYMOP_REGEX_MATCH, symop, re.I)

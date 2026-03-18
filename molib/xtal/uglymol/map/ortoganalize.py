"""
orthoganalize
"""

import gemmi
import numpy as np


def ccp4_to_cartesian(map_file_path):
    """
    Reads a CCP4 map file and converts its grid points to Cartesian coordinates.

    Args:
        map_file_path (str): The path to the input CCP4 map file.

    Returns:
        tuple: A tuple containing:
               - cartesian_coords (np.ndarray): An array of shape (N, 3) with Cartesian coordinates.
               - density_values (np.ndarray): The electron density values corresponding to the coordinates.
    """
    # 1. Read the CCP4 map file
    try:
        ccp4_map = gemmi.read_ccp4_map(map_file_path)
    except FileNotFoundError:
        print(f"Error: Map file not found at {map_file_path}")
        return None, None

    # Get grid data as a numpy array. The data is accessed without copying.
    density_values = np.array(ccp4_map.grid, copy=False)

    # 2. Setup the map to expand it to the full unit cell if needed
    ccp4_map.setup()

    # 3. Get transformation matrices from the header
    grid_to_frac_matrix = ccp4_map.grid.get_unit_cell().frac_from_orthogonal_mat()
    frac_to_cart_matrix = ccp4_map.grid.get_unit_cell().orthogonal_mat()

    # Get grid dimensions
    size_x, size_y, size_z = ccp4_map.grid.nu, ccp4_map.grid.nv, ccp4_map.grid.nw

    # Create 3D arrays for grid indices (0-based)
    u, v, w = np.meshgrid(range(size_x), range(size_y), range(size_z), indexing="ij")

    # Flatten the grid indices
    u_flat = u.flatten()
    v_flat = v.flatten()
    w_flat = w.flatten()

    # Create an array of (u, v, w) grid indices
    grid_indices = np.stack([u_flat, v_flat, w_flat], axis=1)

    # 4. Convert grid indices to fractional coordinates
    # The header includes information about the grid start offset (nxstart, nystart, nzstart)
    start_u, start_v, start_w = ccp4_map.grid.unit_cell.first_grid_point

    # Shift indices based on the starting grid point
    grid_coords_shifted = grid_indices + np.array([start_u, start_v, start_w])

    # Apply the grid-to-fractional conversion
    frac_coords = ccp4_map.grid.unit_cell.fractionalize(grid_coords_shifted)

    # 5. Convert fractional coordinates to Cartesian coordinates
    cartesian_coords = frac_coords @ frac_to_cart_matrix.T

    return cartesian_coords, density_values.flatten()


# Example usage:
if __name__ == "__main__":
    map_file = "my_map.ccp4"  # Replace with your CCP4 file path

    # Example: Create a dummy map file for demonstration
    # In a real scenario, you would already have a .ccp4 file
    dummy_map = gemmi.Ccp4Map()
    dummy_map.grid = gemmi.CGrid(30, 30, 30)
    dummy_map.grid.set_unit_cell(gemmi.UnitCell(40, 50, 60, 90, 90, 90))
    dummy_map.grid.set_data(np.random.rand(30, 30, 30))
    dummy_map.write_ccp4_map(dummy_map.grid, dummy_map.unit_cell, map_file)

    cartesian_coords, density_values = ccp4_to_cartesian(map_file)

    if cartesian_coords is not None:
        # Displaying the first 5 coordinates and their density values
        print(f"Shape of coordinates array: {cartesian_coords.shape}")
        print("First 5 Cartesian coordinates and density values:")
        for i in range(5):
            print(f"Coord: {cartesian_coords[i]}, Density: {density_values[i]:.4f}")

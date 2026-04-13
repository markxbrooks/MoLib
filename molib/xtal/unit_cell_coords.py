"""
Unit cell coordinate generation using Gemmi.

This module provides crystallographically accurate unit cell coordinate generation
using Gemmi's crystallographic mathematics.
"""

from typing import Dict, List, Optional, Tuple

import gemmi
import numpy as np
from decologr import setup_logging

# Set up logging
log = setup_logging()


class UnitCellCoordinateGenerator:
    """
    Generates crystallographically accurate unit cell coordinates using Gemmi.

    This class handles the proper mathematical transformations needed to convert
    unit cell parameters (a, b, c, alpha, beta, gamma) into 3D coordinates
    that accurately represent the crystallographic unit cell.
    """

    def __init__(self):
        """Initialize the unit cell coordinate generator."""
        self.unit_cell = None
        self.space_group = None
        self.orthogonal_matrix = None
        self.fractional_matrix = None

    def set_unit_cell(self, unit_cell_info: Dict[str, float]) -> bool:
        """
        Set the unit cell parameters and calculate transformation matrices.

        Args:
            unit_cell_info: Dictionary containing unit cell parameters:
                - 'a', 'b', 'c': Unit cell lengths in Angstroms
                - 'alpha', 'beta', 'gamma': Unit cell angles in degrees
                - 'space_group': Space group information (optional)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract unit cell parameters
            a = unit_cell_info["a"]
            b = unit_cell_info["b"]
            c = unit_cell_info["c"]
            alpha = unit_cell_info["alpha"]
            beta = unit_cell_info["beta"]
            gamma = unit_cell_info["gamma"]
            space_group = unit_cell_info.get("space_group", "P 1")

            # Create Gemmi UnitCell object
            self.unit_cell = gemmi.UnitCell(a, b, c, alpha, beta, gamma)
            self.space_group = space_group

            # Calculate transformation matrices
            self._calculate_transformation_matrices()

            log.info(f"✅ Unit cell set: a={a:.2f}, b={b:.2f}, c={c:.2f} Å")
            log.info(f"✅ Angles: α={alpha:.2f}°, β={beta:.2f}°, γ={gamma:.2f}°")
            log.info(f"✅ Space group: {space_group}")

            return True

        except Exception as e:
            log.error(f"Error setting unit cell: {e}")
            return False

    def _calculate_transformation_matrices(self) -> None:
        """Calculate the transformation matrices between fractional and orthogonal coordinates."""
        if not self.unit_cell:
            return

        try:
            # Note: We don't actually need these matrices since we're using
            # the orthogonalize() method directly. The orthogonalize() method
            # handles the transformation internally.
            #
            # If you need the matrices for other purposes, you can calculate them
            # from the unit cell parameters, but for our current use case,
            # the orthogonalize() method is sufficient and more reliable.

            # Set to None since we're not using them
            self.orthogonal_matrix = None
            self.fractional_matrix = None

            log.debug(
                "✅ Transformation matrices calculation skipped (using orthogonalize() method)"
            )

        except Exception as e:
            log.error(f"Error in transformation matrices method: {e}")
            # Set to None on error
            self.orthogonal_matrix = None
            self.fractional_matrix = None

    def generate_unit_cell_corners(self) -> Optional[np.ndarray]:
        """
        Generate the 8 corners of the unit cell in orthogonal coordinates.

        Returns:
            numpy array of shape (8, 3) containing the corner coordinates in Angstroms
        """
        if not self.unit_cell:
            log.error("Unit cell not set")
            return None

        try:
            # Define the 8 corners in fractional coordinates (0 to 1)
            fractional_corners = np.array(
                [
                    [0.0, 0.0, 0.0],  # Origin
                    [1.0, 0.0, 0.0],  # +a
                    [0.0, 1.0, 0.0],  # +b
                    [0.0, 0.0, 1.0],  # +c
                    [1.0, 1.0, 0.0],  # +a+b
                    [1.0, 0.0, 1.0],  # +a+c
                    [0.0, 1.0, 1.0],  # +b+c
                    [1.0, 1.0, 1.0],  # +a+b+c
                ],
                dtype=np.float32,
            )

            # Convert to orthogonal coordinates using Gemmi
            orthogonal_corners = np.zeros_like(fractional_corners)

            for i, corner in enumerate(fractional_corners):
                # Convert fractional to orthogonal coordinates
                # Convert numpy array to gemmi.Fractional
                frac_corner = gemmi.Fractional(corner[0], corner[1], corner[2])
                ortho_pos = self.unit_cell.orthogonalize(frac_corner)
                orthogonal_corners[i] = [ortho_pos.x, ortho_pos.y, ortho_pos.z]

            log.debug(
                f"✅ Generated {len(orthogonal_corners)} unit cell corner coordinates"
            )
            return orthogonal_corners

        except Exception as e:
            log.error(f"Error generating unit cell corners: {e}")
            return None

    def generate_unit_cell_edges(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate the 12 edges of the unit cell.

        Returns:
            Tuple of (vertices, edges) where:
            - vertices: numpy array of shape (8, 3) containing corner coordinates
            - edges: list of tuples defining which corners are connected
        """
        corners = self.generate_unit_cell_corners()
        if corners is None:
            return None

        try:
            # Define the 12 edges of a unit cell
            # Each edge connects two corners
            edges = [
                # Edges along a-axis (4 edges)
                (0, 1),  # Origin to +a
                (2, 3),  # +b to +b+c
                (4, 5),  # +a+b to +a+b+c
                (6, 7),  # +b+c to +a+b+c
                # Edges along b-axis (4 edges)
                (0, 2),  # Origin to +b
                (1, 3),  # +a to +a+c
                (4, 6),  # +a+b to +b+c
                (5, 7),  # +a+c to +a+b+c
                # Edges along c-axis (4 edges)
                (0, 3),  # Origin to +c
                (1, 2),  # +a to +a+b
                (4, 7),  # +a+b to +a+b+c
                (5, 6),  # +a+c to +b+c
            ]

            log.debug(f"✅ Generated {len(edges)} unit cell edges")
            return corners, edges

        except Exception as e:
            log.error(f"Error generating unit cell edges: {e}")
            return None

    def generate_unit_cell_faces(self) -> Optional[Tuple[np.ndarray, List[List[int]]]]:
        """
        Generate the 6 faces of the unit cell.

        Returns:
            Tuple of (vertices, faces) where:
            - vertices: numpy array of shape (8, 3) containing corner coordinates
            - faces: list of lists defining which corners form each face
        """
        corners = self.generate_unit_cell_corners()
        if corners is None:
            return None

        try:
            # Define the 6 faces of a unit cell
            # Each face is defined by 4 corners in counter-clockwise order
            faces = [
                # Face 1: z=0 (bottom face)
                [0, 1, 4, 2],
                # Face 2: z=1 (top face)
                [3, 6, 7, 5],
                # Face 3: x=0 (left face)
                [0, 2, 6, 3],
                # Face 4: x=1 (right face)
                [1, 5, 7, 4],
                # Face 5: y=0 (back face)
                [0, 3, 5, 1],
                # Face 6: y=1 (front face)
                [2, 4, 7, 6],
            ]

            log.debug(f"✅ Generated {len(faces)} unit cell faces")
            return corners, faces

        except Exception as e:
            log.error(f"Error generating unit cell faces: {e}")
            return None

    def generate_axes_coordinates(
        self, axis_length: float = 50.0
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate coordinate axes that align with the crystallographic axes.

        Args:
            axis_length: Length of the axes in Angstroms

        Returns:
            Tuple of (vertices, colors) for rendering the axes
        """
        if not self.unit_cell:
            log.error("Unit cell not set")
            return None

        try:
            # Generate unit vectors along the crystallographic axes
            # These will be in the direction of the unit cell vectors
            # Convert to gemmi.Fractional objects first and extract numeric values
            a_pos = self.unit_cell.orthogonalize(gemmi.Fractional(1.0, 0.0, 0.0))
            origin_pos = self.unit_cell.orthogonalize(gemmi.Fractional(0.0, 0.0, 0.0))

            b_pos = self.unit_cell.orthogonalize(gemmi.Fractional(0.0, 1.0, 0.0))
            c_pos = self.unit_cell.orthogonalize(gemmi.Fractional(0.0, 0.0, 1.0))

            # Extract numeric values from gemmi.Position objects
            a_vector = np.array(
                [a_pos.x - origin_pos.x, a_pos.y - origin_pos.y, a_pos.z - origin_pos.z]
            )
            b_vector = np.array(
                [b_pos.x - origin_pos.x, b_pos.y - origin_pos.y, b_pos.z - origin_pos.z]
            )
            c_vector = np.array(
                [c_pos.x - origin_pos.x, c_pos.y - origin_pos.y, c_pos.z - origin_pos.z]
            )

            # Normalize and scale to desired length
            a_vector = a_vector / np.linalg.norm(a_vector) * axis_length
            b_vector = b_vector / np.linalg.norm(b_vector) * axis_length
            c_vector = c_vector / np.linalg.norm(c_vector) * axis_length

            # Create vertices for the three axes
            origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)

            vertices = np.array(
                [
                    # X-axis (a-axis, red)
                    origin,
                    a_vector,
                    # Y-axis (b-axis, green)
                    origin,
                    b_vector,
                    # Z-axis (c-axis, blue)
                    origin,
                    c_vector,
                ],
                dtype=np.float32,
            )

            # Create colors for the axes
            colors = np.array(
                [
                    # X-axis (a-axis, red)
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    # Y-axis (b-axis, green)
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    # Z-axis (c-axis, blue)
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )

            log.debug(
                f"✅ Generated crystallographic axes with length {axis_length:.1f} Å"
            )
            return vertices, colors

        except Exception as e:
            log.error(f"Error generating axes coordinates: {e}")
            return None

    def get_unit_cell_volume(self) -> Optional[float]:
        """Get the volume of the unit cell in cubic Angstroms."""
        if not self.unit_cell:
            return None
        return self.unit_cell.volume

    def get_unit_cell_info(self) -> Optional[Dict[str, float]]:
        """Get comprehensive unit cell information."""
        if not self.unit_cell:
            return None

        try:
            info = {
                "a": self.unit_cell.a,
                "b": self.unit_cell.b,
                "c": self.unit_cell.segment_color,
                "alpha": self.unit_cell.alpha,
                "beta": self.unit_cell.beta,
                "gamma": self.unit_cell.gamma,
                "volume": self.unit_cell.volume,
                "space_group": self.space_group,
            }
            return info

        except Exception as e:
            log.error(f"Error getting unit cell info: {e}")
            return None

"""
Custom symmetry conversion functionality for CCP4 maps.
Provides tools to apply specific symmetry operations to density maps using ELMAP.
"""

import struct
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from decologr import Decologr as log
from molib.xtal.uglymol.map.elmap import ElMap
from molib.xtal.uglymol.map.helpers import parse_symmetry_operator_to_matrix


class SymmetryConverter:
    """
    Handles custom symmetry operations on CCP4 maps using ELMAP.
    """

    def __init__(self):
        self.elmap = ElMap()

    def apply_symmetry_operation(
        self,
        input_map_path: str,
        symmetry_operation: str,
        output_map_path: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Apply a specific symmetry operation to a CCP4 map.

        Args:
            input_map_path: Path to input CCP4 map
            symmetry_operation: Symmetry operation string (e.g., "-x+1/2,-y,z+1/2")
            output_map_path: Optional output path. If None, generates from input path.

        Returns:
            Tuple of (success: bool, output_path: str)
        """
        try:
            log.info(f"🔄 Applying symmetry operation: {symmetry_operation}")
            log.info(f"📁 Input map: {input_map_path}")

            # Generate output path if not provided
            if output_map_path is None:
                input_path = Path(input_map_path)
                output_map_path = str(
                    input_path.parent / f"{input_path.stem}_symmetry{input_path.suffix}"
                )

            # Load the original map
            with open(input_map_path, "rb") as f:
                map_buffer = f.read()

            self.elmap.from_ccp4(map_buffer, expand_symmetry=False)

            log.info("✅ Original map loaded:")
            log.info(f"   - Grid dimensions: {self.elmap.grid.dim}")
            log.info(f"   - Unit cell: {self.elmap.unit_cell}")
            log.info(
                f"   - Data range: {np.min(self.elmap.grid.values):.6f} to {np.max(self.elmap.grid.values):.6f}"
            )

            # Parse symmetry operation
            symop_matrix = parse_symmetry_operator_to_matrix(symmetry_operation)
            log.info("   - Transformation matrix:")
            for i, row in enumerate(symop_matrix):
                log.info(
                    f"     [{row[0]:6.3f} {row[1]:6.3f} {row[2]:6.3f} {row[3]:6.3f}]"
                )

            # Apply symmetry transformation
            success = self._apply_transformation(symop_matrix, output_map_path)

            if success:
                log.info("✅ Symmetry conversion completed successfully!")
                log.info(f"📁 Output file: {output_map_path}")
                return True, output_map_path
            else:
                return False, ""

        except Exception as e:
            log.error(f"❌ Error during symmetry conversion: {e}")
            import traceback

            traceback.print_exc()
            return False, ""

    def _apply_transformation(self, symop_matrix: list, output_path: str) -> bool:
        """
        Apply the symmetry transformation to the map data.

        Args:
            symop_matrix: 3x4 transformation matrix
            output_path: Path for output file

        Returns:
            True if successful, False otherwise
        """
        try:
            original_shape = self.elmap.grid.dim
            original_data = np.array(self.elmap.grid.values).copy()
            new_data = np.zeros_like(original_data)

            total_points = original_shape[0] * original_shape[1] * original_shape[2]
            processed = 0

            log.info("🔄 Applying symmetry transformation...")

            for i in range(original_shape[0]):
                for j in range(original_shape[1]):
                    for k in range(original_shape[2]):
                        # Original grid coordinates
                        orig_coords = np.array([i, j, k], dtype=float)

                        # Apply symmetry transformation
                        new_coords = np.zeros(3)
                        for row in range(3):
                            new_coords[row] = (
                                symop_matrix[row][0] * orig_coords[0]
                                + symop_matrix[row][1] * orig_coords[1]
                                + symop_matrix[row][2] * orig_coords[2]
                                + symop_matrix[row][3]
                            )

                        # Convert to grid indices with proper wrapping
                        new_i = int(round(new_coords[0])) % original_shape[0]
                        new_j = int(round(new_coords[1])) % original_shape[1]
                        new_k = int(round(new_coords[2])) % original_shape[2]

                        # Ensure positive indices
                        if new_i < 0:
                            new_i += original_shape[0]
                        if new_j < 0:
                            new_j += original_shape[1]
                        if new_k < 0:
                            new_k += original_shape[2]

                        # Convert 3D indices to 1D index for GridArray
                        orig_idx = self.elmap.grid.grid2index(i, j, k)
                        new_idx = self.elmap.grid.grid2index(new_i, new_j, new_k)

                        # Copy density value
                        new_data[new_idx] = original_data[orig_idx]

                        processed += 1
                        if processed % (total_points // 10) == 0:
                            progress = (processed / total_points) * 100
                            log.info(f"   Progress: {progress:.1f}%")

            # Calculate statistics
            amin = float(np.min(new_data))
            amax = float(np.max(new_data))
            amean = float(np.mean(new_data))
            arms = float(np.std(new_data))

            log.info("📊 Statistics:")
            log.info(f"   - Min: {amin:.6f}")
            log.info(f"   - Max: {amax:.6f}")
            log.info(f"   - Mean: {amean:.6f}")
            log.info(f"   - RMS: {arms:.6f}")

            # Write CCP4 file
            return self._write_ccp4_file(output_path, original_shape, new_data)

        except Exception as e:
            log.error(f"❌ Error applying transformation: {e}")
            return False

    def _write_ccp4_file(
        self, output_path: str, grid_shape: tuple, data: np.ndarray
    ) -> bool:
        """
        Write the transformed data as a CCP4 file.

        Args:
            output_path: Path for output file
            grid_shape: Grid dimensions
            data: Transformed data array

        Returns:
            True if successful, False otherwise
        """
        try:
            log.info(f"💾 Writing CCP4 file: {output_path}")

            with open(output_path, "wb") as f:
                # Write header
                unit_cell_params = (
                    self.elmap.unit_cell.parameters[0],
                    self.elmap.unit_cell.parameters[1],
                    self.elmap.unit_cell.parameters[2],
                    self.elmap.unit_cell.parameters[3],
                    self.elmap.unit_cell.parameters[4],
                    self.elmap.unit_cell.parameters[5],
                )
                self._write_ccp4_header(f, grid_shape, unit_cell_params)

                # Write data (float32, little-endian)
                data_bytes = data.astype(np.float32).tobytes()
                f.write(data_bytes)

            log.info("✅ CCP4 file written successfully!")
            return True

        except Exception as e:
            log.error(f"❌ Error writing CCP4 file: {e}")
            return False

    def _write_ccp4_header(
        self,
        file_handle,
        grid_shape: tuple,
        unit_cell: tuple,
        space_group: int = 1,
        mode: int = 2,
    ):
        """
        Write a basic CCP4 header to file.

        Args:
            file_handle: Open file handle
            grid_shape: Tuple of (nx, ny, nz)
            unit_cell: Unit cell parameters (a, b, c, alpha, beta, gamma)
            space_group: Space group number (default: 1 for P1)
            mode: Data mode (2 for float32)
        """
        # Initialize header with zeros
        header = [0] * 256  # 256 integers = 1024 bytes

        # Basic map information
        header[0] = grid_shape[0]  # NC - number of columns
        header[1] = grid_shape[1]  # NR - number of rows
        header[2] = grid_shape[2]  # NS - number of sections
        header[3] = mode  # MODE - data type (2 = float32)

        # Grid start positions (usually 0)
        header[4] = 0  # NCSTART
        header[5] = 0  # NRSTART
        header[6] = 0  # NSSTART

        # Grid intervals
        header[7] = grid_shape[0]  # NX
        header[8] = grid_shape[1]  # NY
        header[9] = grid_shape[2]  # NZ

        # Unit cell parameters (in Angstroms)
        header[10] = int(unit_cell[0] * 1000)  # X length (scaled by 1000)
        header[11] = int(unit_cell[1] * 1000)  # Y length
        header[12] = int(unit_cell[2] * 1000)  # Z length

        # Unit cell angles (in degrees)
        header[13] = int(unit_cell[3] * 1000)  # Alpha
        header[14] = int(unit_cell[4] * 1000)  # Beta
        header[15] = int(unit_cell[5] * 1000)  # Gamma

        # Axis mapping (standard: X=1, Y=2, Z=3)
        header[16] = 1  # MAPC
        header[17] = 2  # MAPR
        header[18] = 3  # MAPS

        # Density statistics (will be calculated)
        header[19] = 0  # AMIN (will be updated)
        header[20] = 0  # AMAX (will be updated)
        header[21] = 0  # AMEAN (will be updated)

        # Space group
        header[22] = space_group  # ISPG

        # No symmetry operations
        header[23] = 0  # NSYMBT

        # Machine stamp (little-endian)
        header[52] = 0x00004144  # MAP signature
        header[53] = 0x11110000  # Machine stamp

        # Labels
        header[56] = 1  # NLABL - number of labels
        # Label 1: "ELMAP symmetry converted map"
        label_text = "ELMAP symmetry converted map".ljust(80)
        for i, char in enumerate(label_text):
            if i < 80:
                header[57 + i // 4] |= ord(char) << (8 * (i % 4))

        # Write header as 32-bit integers
        for value in header:
            file_handle.write(struct.pack("<i", value))


def apply_symmetry_to_map(
    input_map_path: str, symmetry_operation: str, output_map_path: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Convenience function to apply symmetry operation to a CCP4 map.

    Args:
        input_map_path: Path to input CCP4 map
        symmetry_operation: Symmetry operation string (e.g., "-x+1/2,-y,z+1/2")
        output_map_path: Optional output path. If None, generates from input path.

    Returns:
        Tuple of (success: bool, output_path: str)
    """
    converter = SymmetryConverter()
    return converter.apply_symmetry_operation(
        input_map_path, symmetry_operation, output_map_path
    )


# Common symmetry operations
COMMON_SYMMETRY_OPERATIONS = {
    "Identity": "x,y,z",
    "Inversion": "-x,-y,-z",
    "2-fold rotation": "y,x,-z",
    "Translation": "x+1/2,y+1/2,z+1/2",
    "Custom 2VUG": "-x+1/2,-y,z+1/2",
}

"""extract isosurface"""

import numpy as np
from decologr import Decologr as log

from molib.calc.math.numpy_util import generate_colors_from_positions
from skimage import measure


def extract_isosurface(volume: np.ndarray, level: float = 1.0):
    """
    extract_isosurface

    :param volume: np.ndarray
    :param level: level
    :return: tuple of (vertices, faces, normals) or None if extraction fails
    """
    log.message("extracting isosurface")
    try:
        vertices, faces, normals, _ = measure.marching_cubes(volume, level=level)

        # Compute better normals using PicoGL's method for improved lighting
        from picogl.buffers.vertex.normals.compute import compute_vertex_normals

        normals = compute_vertex_normals(vertices, faces)

        log.parameter("vertices", vertices)
        log.parameter("faces", faces)
        return vertices, faces, normals
    except Exception as ex:
        log.error(f"Error {ex} extracting isosurface")
        return None


def extract_isosurface_with_density(volume: np.ndarray, level: float = 1.0):
    """
    Extract isosurface with density values preserved at vertices.

    This function is specifically designed for fo-fc maps where we need
    to colour positive values in green and negative values in red.

    Args:
        volume: np.ndarray - 3D volume data
        level: float - isosurface level

    Returns:
        tuple: (vertices, faces, vertex_densities, normals) where vertex_densities
               contains the density values at each vertex for coloring and normals
               contains the surface normals for proper lighting
    """
    log.message("extracting isosurface with density values for fo-fc coloring")
    try:
        # Check volume data range and adjust level if necessary
        volume_min, volume_max = volume.min(), volume.max()
        volume_range = volume_max - volume_min

        log.info(f"Volume data range: {volume_min:.6f} to {volume_max:.6f}")
        log.info(f"Requested isosurface level: {level:.6f}")

        # If volume has no variation, return empty result
        if volume_range == 0:
            log.warning("⚠️ Volume has no variation - cannot extract isosurface")
            return None, None, None

        # If level is outside the data range, adjust it
        if level < volume_min:
            log.warning(
                f"⚠️ Isosurface level {level:.6f} below data minimum {volume_min:.6f}"
            )
            level = volume_min + 0.1 * volume_range
            log.info(f"🔧 Adjusted isosurface level to: {level:.6f}")
        elif level > volume_max:
            log.warning(
                f"⚠️ Isosurface level {level:.6f} above data maximum {volume_max:.6f}"
            )
            level = volume_max - 0.1 * volume_range
            log.info(f"🔧 Adjusted isosurface level to: {level:.6f}")

        vertices, faces, normals, _ = measure.marching_cubes(volume, level=level)

        # Compute better normals using PicoGL's method for improved lighting
        from picogl.buffers.vertex.normals.compute import compute_vertex_normals

        normals = compute_vertex_normals(vertices, faces)

        # Interpolate density values at the vertices
        vertex_densities = np.zeros(len(vertices))

        for i, vertex in enumerate(vertices):
            # Convert vertex coordinates to volume indices
            x, y, z = vertex.astype(int)

            # Ensure indices are within bounds
            x = max(0, min(x, volume.shape[0] - 1))
            y = max(0, min(y, volume.shape[1] - 1))
            z = max(0, min(z, volume.shape[2] - 1))

            # Get density value at this vertex
            vertex_densities[i] = volume[x, y, z]

        log.parameter("vertices", len(vertices))
        log.parameter("faces", len(faces))
        log.parameter(
            "density_range",
            f"{vertex_densities.min():.3f} to {vertex_densities.max():.3f}",
        )

        return vertices, faces, vertex_densities, normals

    except Exception as ex:
        log.error(f"Error {ex} extracting isosurface with density values")
        return None, None, None, None


def create_fofc_color_map(
    vertex_densities: np.ndarray,
    positive_color: tuple = (0.0, 1.0, 0.0),  # Green
    negative_color: tuple = (1.0, 0.0, 0.0),  # Red
    zero_color: tuple = (0.5, 0.5, 0.5),
):  # Gray
    """
    Create a colour map for fo-fc difference maps.

    Positive values (excess density) are colored green.
    Negative values (missing density) are colored red.
    Values near zero are colored gray.

    Args:
        vertex_densities: np.ndarray - density values at each vertex
        positive_color: tuple - RGB colour for positive values (default: green)
        negative_color: tuple - RGB colour for negative values (default: red)
        zero_color: tuple - RGB colour for values near zero (default: gray)

    Returns:
        np.ndarray - RGB colors for each vertex, shape (n_vertices, 3)
    """
    try:
        n_vertices = len(vertex_densities)
        colors = np.zeros((n_vertices, 3))

        # Calculate the range for smooth colour transitions
        density_range = np.max(np.abs(vertex_densities))
        if density_range == 0:
            # All values are zero, use zero colour
            return generate_colors_from_positions(
                vertex_densities,
                zero_color[0],
                zero_color[1],
                zero_color[2],
            )

        # Normalize densities to [-1, 1] range
        normalized_densities = vertex_densities / density_range

        for i, density in enumerate(normalized_densities):
            if abs(density) < 0.1:  # Values near zero
                colors[i] = zero_color
            elif density > 0:  # Positive values
                # Interpolate from zero_color to positive_color
                factor = min(density, 1.0)
                colors[i] = tuple(
                    z * (1 - factor) + p * factor
                    for z, p in zip(zero_color, positive_color)
                )
            else:  # Negative values
                # Interpolate from zero_color to negative_color
                factor = min(abs(density), 1.0)
                colors[i] = tuple(
                    z * (1 - factor) + n * factor
                    for z, n in zip(zero_color, negative_color)
                )

        log.info(
            f"✅ Created fo-fc colour map: {n_vertices} vertices, "
            f"density range: {vertex_densities.min():.3f} to {vertex_densities.max():.3f}"
        )

        return colors

    except Exception as ex:
        log.error(f"Error creating fo-fc colour map: {ex}")
        # Return default colors on error
        return generate_colors_from_positions(
            vertex_densities,
            positive_color[0],
            positive_color[1],
            positive_color[2],
        )


def create_2fofc_color_map(
    vertex_densities: np.ndarray,
    base_color: tuple = None,  # Will use default blue if None,
):
    """
    Create a colour map for 2Fo-Fc electron density maps.

    2Fo-Fc maps show the electron density and are typically colored in blue
    with intensity varying based on density values.

    Args:
        vertex_densities: np.ndarray - density values at each vertex
        base_color: tuple - base blue colour for the map
        intensity_factor: float - factor to control colour intensity

    Returns:
        np.ndarray - RGB colors for each vertex, shape (n_vertices, 3)
    """
    try:
        bc = base_color if base_color is not None else (0.0, 0.0, 1.0)
        return generate_colors_from_positions(
            vertex_densities, bc[0], bc[1], bc[2]
        )

    except Exception as ex:
        log.error(f"Error creating 2Fo-Fc colour map: {ex}")
        # Return default colors on error
        default_color = (0.0, 0.5, 1.0) if base_color is None else base_color
        return generate_colors_from_positions(
            vertex_densities,
            default_color[0],
            default_color[1],
            default_color[2],
        )


def extract_isosurface_elmo(volume: np.ndarray, level: float = 1.0):
    """
    Extract isosurface using ElMo's custom implementation.

    This is a simplified version that provides basic isosurface extraction.
    For production use, consider using the skimage implementation.

    :param volume: np.ndarray - 3D volume data
    :param level: float - isosurface level
    :return: tuple of (vertices, faces, normals) or None if extraction fails
    """
    log.message("extracting isosurface using ElMo implementation")
    try:
        # Simple threshold-based approach
        above_level = volume > level

        # Find indices where the condition is met
        indices = np.where(above_level)

        if len(indices[0]) == 0:
            log.warning("No points above isosurface level")
            return [], [], []

        # Convert to vertices (simplified - just using grid points)
        x, y, z = np.meshgrid(
            np.arange(volume.shape[0]),
            np.arange(volume.shape[1]),
            np.arange(volume.shape[2]),
            indexing="ij",
        )

        vertices = np.column_stack([x[indices], y[indices], z[indices]])

        # Generate simple faces (triangles) - this is a very basic approach
        # In a real implementation, you'd want proper triangulation
        faces = []
        for i in range(0, len(vertices) - 2, 3):
            if i + 2 < len(vertices):
                faces.append([i, i + 1, i + 2])

        # Convert faces to numpy array
        faces = np.array(faces, dtype=np.int32)

        # Compute normals using PicoGL's method for improved lighting
        from picogl.buffers.vertex.normals.compute import compute_vertex_normals

        normals = compute_vertex_normals(vertices, faces)

        log.parameter("vertices", len(vertices))
        log.parameter("faces", len(faces))
        return vertices, faces, normals

    except Exception as ex:
        log.error(f"Error {ex} extracting isosurface using ElMo implementation")
        return None

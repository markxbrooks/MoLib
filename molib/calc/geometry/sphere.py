import numpy as np


def make_sphere_geometry(lat_segments=12, lon_segments=12, radius=1.0):
    """
    Generate a UV sphere mesh.

    Returns:
        vertices: np.ndarray of shape (N, 3)
        indices: np.ndarray of shape (M,) for glDrawElements
    """
    vertices = []
    indices = []

    for i in range(lat_segments + 1):
        theta = np.pi * i / lat_segments
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        for j in range(lon_segments + 1):
            phi = 2 * np.pi * j / lon_segments
            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)

            x = radius * sin_theta * cos_phi
            y = radius * cos_theta
            z = radius * sin_theta * sin_phi
            vertices.append((x, y, z))

    for i in range(lat_segments):
        for j in range(lon_segments):
            first = i * (lon_segments + 1) + j
            second = first + lon_segments + 1

            indices.extend([first, second, first + 1, second, second + 1, first + 1])

    vertices_np = np.array(vertices, dtype=np.float32)
    indices_np = np.array(indices, dtype=np.uint32)

    return vertices_np, indices_np

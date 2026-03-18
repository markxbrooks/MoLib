import numpy as np


def average_normal(n1, n2):
    """
    Return a normalized vector that is the average of two normals.
    Safe to use even if one of them is None.
    """
    if n1 is None:
        return n2
    if n2 is None:
        return n1

    avg_x = (n1.x + n2.x) * 0.5
    avg_y = (n1.y + n2.y) * 0.5
    avg_z = (n1.z + n2.z) * 0.5

    # Normalize
    length = (avg_x**2 + avg_y**2 + avg_z**2) ** 0.5
    if length == 0:
        return n1  # fallback: just return first normal
    return type(n1)(avg_x / length, avg_y / length, avg_z / length)


def compute_side_normal(p1, p4):
    """Compute a normalized vector for the strand's side surface."""
    # Make a vector between points and a vertical direction for cross
    v1 = np.array([p4.x - p1.x, p4.y - p1.y, p4.z - p1.z])
    up = np.array([0, 0, 1])  # assume Z-up
    n = np.cross(v1, up)
    norm = np.linalg.norm(n)
    if norm == 0:
        return type(p1)(0, 0, 1)
    return type(p1)(*(n / norm))

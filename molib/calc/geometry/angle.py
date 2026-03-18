import numpy as np


def angle_between(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """
    angle_between(vector1, vector2)
    :param vector1:
    :param vector2:
    :return: float
    Returns angle in degrees between vectors v1 and v2
    """
    cos_angle = np.dot(vector1, vector2) / (
        np.linalg.norm(vector1) * np.linalg.norm(vector2)
    )
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def dihedral_angle(p0, p1, p2, p3):
    """
    Calculate the dihedral (torsion) angle between four points.
    Returns angle in radians.
    """
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2

    # Normalize b1
    b1 /= np.linalg.norm(b1)

    # v and w are normals to the planes
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    # Normalize, handling zero-length vectors
    v_norm = np.linalg.norm(v)
    w_norm = np.linalg.norm(w)

    if v_norm < 1e-8 or w_norm < 1e-8:
        # If either vector is too small, return 0 (no torsion)
        return 0.0

    v /= v_norm
    w /= w_norm

    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)

    angle = np.arctan2(y, x)
    return angle


def smooth_torsion_angles(angles, window=3):
    """
    Smooth torsion angles using moving average (window size).
    Optionally, use cubic spline for more advanced smoothing.
    """
    if len(angles) == 0:
        return angles

    if window < 1:
        window = 1

    if len(angles) < window:
        # If we have fewer angles than window size, just return the original
        return angles

    # Ensure window is odd for symmetric padding
    if window % 2 == 0:
        window += 1

    # Calculate padding - ensure non-negative
    pad_left = max(0, window // 2)
    pad_right = max(0, window - 1 - window // 2)

    padded = np.pad(angles, (pad_left, pad_right), mode="edge")
    smoothed = np.convolve(padded, np.ones(window) / window, mode="valid")
    return smoothed


def rotate_vector_around_axis(v, axis, angle):
    """
    Rotate vector v around axis by angle (radians).
    Uses Rodrigues' rotation formula.
    """
    axis = axis / np.linalg.norm(axis)
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    return (
        v * cos_theta
        + np.cross(axis, v) * sin_theta
        + axis * np.dot(axis, v) * (1 - cos_theta)
    )

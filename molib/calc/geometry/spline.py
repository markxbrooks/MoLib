"""
Catmull rom spline

Example Usage:
==============
>>>spline = catmull_rom_chain(ca_coords)
"""

import numpy as np
from numba import njit


def catmull_rom_chain(points: np.ndarray, samples_per_segment: int = 8) -> np.ndarray:
    """
    Generate interpolated points using Catmull-Rom spline (pure NumPy).

    :param points: (N, 3) input control points
    :param samples_per_segment: number of samples between each pair of points
    :return: (M, 3) interpolated points
    """
    n_points = len(points)
    if n_points < 4:
        raise ValueError(
            "At least 4 points are required for Catmull-Rom interpolation."
        )

    # Precompute interpolation samples
    t_vals = np.linspace(0, 1, samples_per_segment, endpoint=False)
    t = t_vals[:, None]  # (samples_per_segment, 1)

    result = []

    # For each segment
    for i in range(1, n_points - 2):
        p0, p1, p2, p3 = points[i - 1 : i + 3]

        # Compute spline for all t in vectorized form
        a = 2 * p1
        b = -p0 + p2
        c = 2 * p0 - 5 * p1 + 4 * p2 - p3
        d = -p0 + 3 * p1 - 3 * p2 + p3

        segment = 0.5 * (a + b * t + c * t**2 + d * t**3)
        result.append(segment)

    result.append(points[-2][None, :])  # add last real point
    return np.vstack(result)


@njit
def catmull_rom_chain_optimized(
    points: np.ndarray, samples_per_segment: int = 8
) -> np.ndarray:
    """
    Generate interpolated points using Catmull-Rom spline (Numba-accelerated).

    :param points: (N, 3) data of 3D points.
    :param samples_per_segment: Number of interpolated points per segment.
    :return: (M, 3) data of interpolated points.
    """
    n_points = len(points)
    n_segments = n_points - 3
    total_samples = n_segments * samples_per_segment + 1  # +1 for the last anchor

    result = np.zeros((total_samples, 3), dtype=np.float32)
    idx = 0

    for i in range(1, n_points - 2):
        p0 = points[i - 1]
        p1 = points[i]
        p2 = points[i + 1]
        p3 = points[i + 2]

        for j in range(samples_per_segment):
            t = j / samples_per_segment
            t2 = t * t
            t3 = t2 * t

            result[idx, :] = 0.5 * (
                2 * p1
                + (-p0 + p2) * t
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
            )
            idx += 1

    # Add final point for continuity
    result[idx, :] = points[-2]
    return result


def catmull_rom_chain_old(
    points: np.ndarray, samples_per_segment: int = 8
) -> np.ndarray:
    """
    Generate interpolated points using Catmull-Rom spline.

    :param samples_per_segment: int
    :param points: np.ndarray shape (N, 3)
    :return: np.ndarray of interpolated points, shape (M, 3)
    """

    def interpolate(p0, p1, p2, p3, t):
        return 0.5 * (
            2 * p1
            + (-p0 + p2) * t
            + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2
            + (-p0 + 3 * p1 - 3 * p2 + p3) * t**3
        )

    result = []
    n = len(points)
    for i in range(1, n - 2):
        for j in range(samples_per_segment):
            t = j / samples_per_segment
            pt = interpolate(points[i - 1], points[i], points[i + 1], points[i + 2], t)
            result.append(pt)
    result.append(points[-2])  # Add last known point
    return np.array(result)

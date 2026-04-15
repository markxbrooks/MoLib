"""
Ribbons-style B-spline ribbon meshdata

This module implements Ribbons' B-spline approach for generating accurate
3D ribbon meshdata, as opposed to the simpler Catmull-Rom approach used
in the original Elmo implementation.

Key differences from Catmull-Rom:
- Uses cubic B-splines (smoother, doesn't interpolate through all points)
- Calculates guide points from peptide plane meshdata (CA, O, CB atoms)
- Uses Frenet frame (tangent, normal, binormal) for proper 3D orientation
- Generates 3D tube meshdata with proper normals for lighting
"""

from typing import Optional, Tuple, Any, Protocol

import numpy as np
from numpy import ndarray
from elmo.core.calc.utils import compute_tangents

from molib.core.constants import MoLibConstant
from picogl.buffers.geometry import GeometryData
class _ResgeomContext(Protocol):
    num_threads: int
    num_samples: int
    arrow_base_width: float | np.floating[Any]
    arrow_head_width: float
    has_arrow: bool
    force_thru_ca: bool



class RibbonStyle:
    """Ribbon Style"""
    FLAT = "flat"
    CIRCLE = "circle"
    SQUARE = "square"
    ELLIPSE = "ellipse"


# B-spline basis matrices (from Ribbons makeguide.C)
# Standard cubic B-spline matrix
BS_MAT = np.array(
    [
        [-1.0 / 6.0, 3.0 / 6.0, -3.0 / 6.0, 1.0 / 6.0],
        [3.0 / 6.0, -6.0 / 6.0, 3.0 / 6.0, 0.0],
        [-3.0 / 6.0, 0.0, 3.0 / 6.0, 0.0],
        [1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0, 0.0],
    ],
    dtype=np.float32,
)

# First segment (beginning)
BS_MAT_A = np.array(
    [
        [-1.0, 21.0 / 12.0, -11.0 / 12.0, 1.0 / 6.0],
        [3.0, -9.0 / 2.0, 3.0 / 2.0, 0.0],
        [-3.0, 3.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)

# Second segment
BS_MAT_B = np.array(
    [
        [-1.0 / 4.0, 7.0 / 12.0, -1.0 / 2.0, 1.0 / 6.0],
        [3.0 / 4.0, -5.0 / 4.0, 1.0 / 2.0, 0.0],
        [-3.0 / 4.0, 1.0 / 4.0, 1.0 / 2.0, 0.0],
        [1.0 / 4.0, 7.0 / 12.0, 1.0 / 6.0, 0.0],
    ],
    dtype=np.float32,
)

# Next-to-last segment
BS_MAT_Y = np.array(
    [
        [-1.0 / 6.0, 1.0 / 2.0, -7.0 / 12.0, 1.0 / 4.0],
        [3.0 / 6.0, -1.0, 6.0 / 12.0, 0.0],
        [-3.0 / 6.0, 0.0, 6.0 / 12.0, 0.0],
        [1.0 / 6.0, 2.0 / 3.0, 2.0 / 12.0, 0.0],
    ],
    dtype=np.float32,
)

# Last segment
BS_MAT_Z = np.array(
    [
        [-1.0 / 6.0, 11.0 / 12.0, -7.0 / 4.0, 1.0],
        [3.0 / 6.0, -15.0 / 12.0, 3.0 / 4.0, 0.0],
        [-3.0 / 6.0, -3.0 / 12.0, 3.0 / 4.0, 0.0],
        [1.0 / 6.0, 7.0 / 12.0, 1.0 / 4.0, 0.0],
    ],
    dtype=np.float32,
)


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector."""
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return v
    return v / norm


def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cross product of two 3D vectors."""
    return np.cross(a, b)


def dot(a: np.ndarray, b: np.ndarray) -> float:
    """Dot product of two vectors."""
    return np.dot(a, b)


def get_width(ss1: str, ss2: str) -> float:
    """
    Get ribbon width based on secondary structure types (from Ribbons).

    Ribbons averages the width of current and next residue.

    Args:
        ss1: Current secondary structure type
        ss2: Next secondary structure type

    Returns:
        Width factor (average of wa and wb)
    """
    # Get width for current residue
    if not ss1 or ss1 == " " or ss1 == "T" or ss1 == "C" or ss1 == "c":
        wa = 0.5  # Coil/turn
    elif ss1 == "H" or ss1 == "L" or ss1 == "3" or ss1 == "h":
        wa = 0.6  # Helix
    elif ss1 == "S" or ss1 == "A" or ss1 == "E" or ss1 == "B":
        wa = 0.8  # Sheet
    else:
        wa = 0.5

    # Get width for next residue
    if not ss2 or ss2 == " " or ss2 == "T" or ss2 == "C" or ss2 == "c":
        wb = 0.5  # Coil/turn
    elif ss2 == "H" or ss2 == "L" or ss2 == "3" or ss2 == "h":
        wb = 0.6  # Helix
    elif ss2 == "S" or ss2 == "A" or ss2 == "E" or ss2 == "B":
        wb = 0.8  # Sheet
    else:
        wb = 0.5

    # Return average (like Ribbons)
    return 0.5 * (wa + wb)


def get_shift(ss: str) -> float:
    """
    Get helix shift amount (from Ribbons).

    Args:
        ss: Secondary structure type

    Returns:
        Shift amount in Angstroms
    """
    if ss == "H" or ss == "G" or ss == "I":  # Helix types
        return 0.3  # Shift towards helix center
    return 0.0


def is_helix(ss1: str, ss2: str) -> bool:
    """Check if secondary structure is a helix."""
    return (ss1 == "H" or ss1 == "G" or ss1 == "I") and (
        ss2 == "H" or ss2 == "G" or ss2 == "I"
    )


def is_sheet(ss1: str, ss2: str) -> bool:
    """Check if secondary structure is a sheet."""
    return (ss1 == "S" or ss1 == "E" or ss1 == "B") and (
        ss2 == "S" or ss2 == "E" or ss2 == "B"
    )


def calculate_guide_points(
    ca_coords: np.ndarray,
    o_coords: Optional[np.ndarray] = None,
    cb_coords: Optional[np.ndarray] = None,
    ss_types: Optional[np.ndarray] = None,
    width: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate B-spline guide points from CA coordinates (Ribbons-style).

    This implements MakeProBSguides from Ribbons, which calculates guide points
    based on peptide plane meshdata.

    Args:
        ca_coords: (N, 3) array of C-alpha coordinates
        o_coords: (N, 3) array of O (oxygen) coordinates, optional
        cb_coords: (N, 3) array of CB (C-beta) coordinates, optional
        ss_types: (N,) array of secondary structure types ('H', 'S', 'T', etc.), optional
        width: Ribbon width factor

    Returns:
        Tuple of (p_points, q_points) where each is (M, 3) array of guide points.
        p_points are centerline points, q_points are perpendicular points.
    """
    n_res = len(ca_coords)
    if n_res < 2:
        raise ValueError("Need at least 2 residues for guide point calculation")

    # If O/CB not provided, estimate from CA positions
    if o_coords is None:
        # Estimate O position: typically ~2.4A from CA, roughly perpendicular to CA-CA vector
        o_coords = np.zeros_like(ca_coords)
        for i in range(n_res):
            if i < n_res - 1:
                ca_ca = ca_coords[i + 1] - ca_coords[i]
                ca_ca = normalize(ca_ca)
                # O is roughly perpendicular to CA-CA, offset by ~2.4A
                perp = np.array([-ca_ca[1], ca_ca[0], 0.0])
                if np.linalg.norm(perp) < 1e-6:
                    perp = np.array([0.0, -ca_ca[2], ca_ca[1]])
                perp = normalize(perp)
                o_coords[i] = ca_coords[i] + perp * 2.4
            else:
                o_coords[i] = o_coords[i - 1] + (ca_coords[i] - ca_coords[i - 1])

    # Initialize guide points (add 2 extra at each end for B-spline)
    n_guides = n_res + 4  # n_res + 2 head + 2 tail
    p_points = np.zeros((n_guides, 3), dtype=np.float32)
    q_points = np.zeros((n_guides, 3), dtype=np.float32)

    # Head points (dummy points for B-spline)
    p_points[0] = ca_coords[0] - 0.01
    q_points[0] = ca_coords[0] + 0.01
    p_points[1] = ca_coords[0] - 0.02
    q_points[1] = ca_coords[0] + 0.02

    # Calculate guide points for each peptide plane
    up = True  # Track which side of the plane we're on
    dprev = None

    for k in range(1, n_res - 1):
        # Form peptide plane vectors (similar to MakeProBSguides)
        ca_curr = ca_coords[k]
        ca_next = ca_coords[k + 1]
        o_curr = o_coords[k]

        a = ca_next - ca_curr  # CA-CA vector
        b = o_curr - ca_curr  # CA-O vector
        p = ca_curr + 0.5 * a  # Midpoint

        # Calculate plane normal and parallel vector
        c = cross(a, b)  # Normal to peptide plane
        c = normalize(c)
        d = cross(c, a)  # Parallel to plane, perpendicular to CA-CA
        d = normalize(d)
        axis = normalize(ca_coords[k + 1] - ca_coords[k - 1])
        radial = normalize(cross(axis, a))
        if dprev is not None:
            if dot(radial, dprev) < 0.0:
                radial = -radial
                dprev = radial.copy()

        # Get width based on secondary structure (Ribbons style)
        if ss_types is not None and k < len(ss_types) and k + 1 < len(ss_types):
            ribbon_width = get_width(ss_types[k], ss_types[k + 1]) * width
            if ss_types is not None and is_helix(
                    ss_types[k],
                    ss_types[k] if k + 1 >= len(ss_types) else ss_types[k + 1],
            ):
                helix_scale = 5.2  # try 1.3–2.0
                ribbon_width *= helix_scale
        else:
            ribbon_width = width

        # Apply helix shift (Ribbons enhancement)
        if ss_types is not None and k < len(ss_types):
            if is_helix(
                ss_types[k], ss_types[k] if k + 1 >= len(ss_types) else ss_types[k + 1]
            ):
                shift = get_shift(ss_types[k])
                p = p + c * shift  # Shift along plane normal

        # Check if we need to flip sides (when planes cross)
        # Ribbons uses: fact = (k==1) ? 1 : dot(d, dprev)
        if k == 1:
            fact = 1.0
            """elif dprev is not None:
                fact = dot(d, dprev)
                if fact < 0.0:
                    up = not up"""
        else:
            fact = 1.0

        # Generate control line (p and q points)
        fact = 0.5
        if up:
            fact = -fact

        guide_idx = k + 1  # +1 for head point
        """p_points[guide_idx] = p + fact * ribbon_width * d
        q_points[guide_idx] = p - fact * ribbon_width * d"""

        p_points[guide_idx] = p + fact * ribbon_width * radial
        q_points[guide_idx] = p - fact * ribbon_width * radial

        dprev = d.copy()

    # Tail points (dummy points for B-spline). We have n_guides = n_res+4, so indices
    # 0..n_res+3. The loop above fills 2..n_res-1. We must set n_res and n_res+1 as well
    # as n_res+2 and n_res+3; otherwise n_res and n_res+1 stay (0,0,0) and the B-spline
    # draws a segment to the origin before the C-terminal.
    p_points[n_guides - 2] = ca_coords[-1] - 0.01
    q_points[n_guides - 2] = ca_coords[-1] + 0.01
    p_points[n_guides - 1] = ca_coords[-1] - 0.02
    q_points[n_guides - 1] = ca_coords[-1] + 0.02
    # Fill the two tail slots that were left at (0,0,0): extrapolate from last real guides
    if n_res >= 2:
        # dp = p_points[n_res - 1] - p_points[n_res - 2]
        dp = 0 # Extrapolation isn't working
        # dq = q_points[n_res - 1] - q_points[n_res - 2]
        dq = 0 # Extrapolation isn't working
        p_points[n_res] = p_points[n_res - 1] + dp
        q_points[n_res] = q_points[n_res - 1] + dq
        p_points[n_res + 1] = p_points[n_res] + dp
        q_points[n_res + 1] = q_points[n_res] + dq

    # Extend end points in a straight line (similar to Ribbons)
    # Use no extra extension at the N-terminus to avoid visible overshoot,
    # but keep the original extension behavior at the C-terminus.
    fact = 0.0
    head_fact = 0.0  # disable additional extrapolation before the first residue

    # First 2 fake planes (N-terminus)
    if n_guides >= 4:
        c = 0.5 * (p_points[0] + q_points[0])
        a = 0.5 * (p_points[2] + q_points[2])
        b = 0.5 * (p_points[3] + q_points[3])
        dprev = normalize(c - a) * head_fact
        p_points[0] += 1.5 * dprev
        q_points[0] += 1.5 * dprev
        p_points[1] += 0.5 * dprev
        q_points[1] += 0.5 * dprev

    # Last 2 fake planes (C-terminus)
    # Keep C-tail extrapolation softer than the original Ribbons-style values
    # to avoid a visibly wide/fat terminal cap when extension is enabled.
    """if n_guides >= 4:
        c = 0.5 * (p_points[-1] + q_points[-1])
        a = 0.5 * (p_points[-2] + q_points[-2])
        b = 0.5 * (p_points[-3] + q_points[-3])
        tail_fact = 0.5 * fact
        dprev = normalize(c - a) * tail_fact
        p_points[-1] += 0.75 * dprev
        q_points[-1] += 0.75 * dprev
        p_points[-2] += 0.25 * dprev
        q_points[-2] += 0.25 * dprev"""

    return p_points, q_points


def evaluate_bspline_segment(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    t: float,
    matrix: Optional[np.ndarray]= None,
) -> np.ndarray:
    """
    Evaluate a cubic B-spline segment at parameter t.

    Args:
        p0, p1, p2, p3: Four control points (3D)
        t: Parameter in [0, 1]
        matrix: B-spline basis matrix (defaults to BS_MAT)

    Returns:
        Interpolated point (3D)
    """
    if matrix is None:
        matrix = BS_MAT

    # Form meshdata matrix
    geom = np.array([p0, p1, p2, p3], dtype=np.float32)  # (4, 3)

    # B-spline evaluation: P(t) = [t^3 t^2 t 1] * matrix * geom
    t_vec = np.array([t**3, t**2, t, 1.0], dtype=np.float32)
    result = t_vec @ matrix @ geom

    return result


def evaluate_bspline_chain(
    guide_points: np.ndarray, samples_per_segment: int = 8
) -> np.ndarray:
    """
    Evaluate a B-spline chain through guide points.

    This implements Ribbons' bs_line function, using special matrices
    for beginning and ending segments.

    Args:
        guide_points: (N, 3) array of guide points
        samples_per_segment: Number of samples per B-spline segment

    Returns:
        (M, 3) array of interpolated points
    """
    n_guides = len(guide_points)
    if n_guides < 4:
        raise ValueError("Need at least 4 guide points for B-spline")

    result = []
    t_vals = np.linspace(0, 1, samples_per_segment, endpoint=False)

    # First segment (uses BS_MAT_A)
    if n_guides >= 4:
        for t in t_vals:
            pt = evaluate_bspline_segment(
                guide_points[0],
                guide_points[1],
                guide_points[2],
                guide_points[3],
                t,
                BS_MAT_A,
            )
            result.append(pt)

    # Second segment (uses BS_MAT_B)
    if n_guides >= 5:
        for t in t_vals:
            pt = evaluate_bspline_segment(
                guide_points[1],
                guide_points[2],
                guide_points[3],
                guide_points[4],
                t,
                BS_MAT_B,
            )
            result.append(pt)

    # Middle segments (use standard BS_MAT)
    for i in range(2, n_guides - 3):
        for t in t_vals:
            pt = evaluate_bspline_segment(
                guide_points[i],
                guide_points[i + 1],
                guide_points[i + 2],
                guide_points[i + 3],
                t,
                BS_MAT,
            )
            result.append(pt)

    # Next-to-last segment (uses BS_MAT_Y)
    if n_guides >= 5:
        for t in t_vals:
            pt = evaluate_bspline_segment(
                guide_points[-4],
                guide_points[-3],
                guide_points[-2],
                guide_points[-1],
                t,
                BS_MAT_Y,
            )
            result.append(pt)

    # Last segment (uses BS_MAT_Z)
    if n_guides >= 4:
        # Add the last guide point
        result.append(guide_points[-2])

    return np.array(result, dtype=np.float32)

def smooth(vectors, alpha=0.2):
    """Smooth vectors"""
    out = vectors.copy()
    for i in range(1, len(vectors)):
        out[i] = normalize((1 - alpha) * out[i] + alpha * out[i - 1])
    return out

def calculate_frenet_frame_from_edges(
    left_edge: np.ndarray, centerline: np.ndarray, right_edge: np.ndarray
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    Calculate Frenet frame from ribbon edges (Ribbons' SetSpaceCurve approach).

    This is more accurate than calculating from centerline alone, as it uses
    the actual ribbon meshdata.

    Args:
        left_edge: (N, 3) array of left edge points
        centerline: (N, 3) array of centerline points
        right_edge: (N, 3) array of right edge points

    Returns:
        Tuple of (tangents, normals, binormals, widths), each (N, 3) except widths (N,)
    """
    n_points = len(centerline)
    tangents = np.zeros((n_points, 3), dtype=np.float32)
    normals = np.zeros((n_points, 3), dtype=np.float32)
    binormals = np.zeros((n_points, 3), dtype=np.float32)
    widths = np.zeros(n_points, dtype=np.float32)

    for i in range(n_points):
        # --- Tangent ---
        if i == 0:
            t = centerline[1] - centerline[0]
        elif i == n_points - 1:
            t = centerline[-1] - centerline[-2]
        else:
            t = centerline[i + 1] - centerline[i - 1]
        t = normalize(t)
        tangents[i] = t

        # --- Edge / binormal ---
        edge_vec = right_edge[i] - left_edge[i]
        width = np.linalg.norm(edge_vec)
        widths[i] = width

        if width < 1e-6:
            edge_vec = binormals[i - 1] if i > 0 else np.array([0.0, 1.0, 0.0])
        else:
            edge_vec /= width

        if i > 0:
            edge_vec = normalize(0.7 * edge_vec + 0.3 * binormals[i - 1])
        b = edge_vec

        # --- Normal ---
        n = cross(t, b)
        if np.linalg.norm(n) < 1e-6:
            n = normals[i - 1] if i > 0 else np.array([0.0, 0.0, 1.0])
        else:
            n = normalize(n)

        # --- Recompute binormal ---
        b = normalize(cross(t, n))

        # --- Parallel transport / minimal rotation correction ---
        if i > 0:
            v = np.cross(tangents[i - 1], t)
            s = np.linalg.norm(v)
            if s > 1e-6:
                c = np.dot(tangents[i - 1], t)
                vx = np.array([[0, -v[2], v[1]],
                               [v[2], 0, -v[0]],
                               [-v[1], v[0], 0]], dtype=np.float32)
                R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
                n = normalize(R @ normals[i - 1])
                b = normalize(np.cross(t, n))

        # --- Flip fix ---
        if i > 0 and np.dot(n, normals[i - 1]) < 0:
            n *= -1
            b *= -1

        normals[i] = n
        binormals[i] = b

    # --- Smoothing pass (single pass, OUTSIDE loop) ---
    for i in range(1, n_points):
        normals[i] = normalize(0.8 * normals[i] + 0.2 * normals[i - 1])
        binormals[i] = normalize(0.8 * binormals[i] + 0.2 * binormals[i - 1])

    return tangents, normals, binormals, widths


def calculate_frenet_frame(centerline: np.ndarray) -> Tuple[ndarray, ndarray, ndarray]:
    """
    Calculate Frenet frame (tangent, normal, binormal) for a space curve.

    Fallback method when edge information is not available.

    Args:
        centerline: (N, 3) array of points along the curve

    Returns:
        Tuple of (tangents, normals, binormals), each (N, 3)
    """
    n_points = len(centerline)
    tangents = np.zeros((n_points, 3), dtype=np.float32)
    normals = np.zeros((n_points, 3), dtype=np.float32)
    binormals = np.zeros((n_points, 3), dtype=np.float32)

    # Up hint for initial normal calculation
    up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    for i in range(n_points):
        # Calculate tangent
        if i == 0:
            t = (
                centerline[1] - centerline[0]
                if n_points > 1
                else np.array([1.0, 0.0, 0.0])
            )
        elif i == n_points - 1:
            t = centerline[-1] - centerline[-2]
        else:
            t = centerline[i + 1] - centerline[i - 1]

        t = normalize(t)
        tangents[i] = t

        # Calculate normal (perpendicular to tangent)
        n = cross(t, up_hint)
        if np.linalg.norm(n) < 1e-3:
            up_hint = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            n = cross(t, up_hint)
        n = normalize(n)
        normals[i] = n

        # Calculate binormal (perpendicular to both)
        b = cross(t, n)
        b = normalize(b)
        binormals[i] = b

        # Update up_hint for next iteration (use current normal)
        up_hint = n

    return tangents, normals, binormals


def calculate_parallel_transport_frames(centerline: np.ndarray):
    """calculate transport frame"""
    n = len(centerline)

    tangents = np.gradient(centerline, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)

    normals = np.zeros_like(tangents)
    binormals = np.zeros_like(tangents)

    # Initial normal (robust choice)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(np.dot(up, tangents[0])) > 0.9:
        up = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    normals[0] = np.cross(tangents[0], up)
    normals[0] /= np.linalg.norm(normals[0])

    binormals[0] = np.cross(tangents[0], normals[0])

    for i in range(1, n):
        v = np.cross(tangents[i - 1], tangents[i])
        norm_v = np.linalg.norm(v)

        if norm_v < 1e-6:
            # No rotation needed (straight segment)
            normals[i] = normals[i - 1]
            binormals[i] = binormals[i - 1]
            continue

        v /= norm_v
        angle = np.arccos(np.clip(np.dot(tangents[i - 1], tangents[i]), -1.0, 1.0))

        # Rodrigues' rotation formula
        def rotate(vec):
            return (
                vec * np.cos(angle)
                + np.cross(v, vec) * np.sin(angle)
                + v * np.dot(v, vec) * (1 - np.cos(angle))
            )

        normals[i] = rotate(normals[i - 1])
        binormals[i] = rotate(binormals[i - 1])

    return tangents, normals, binormals


def generate_ribbon_geometry_ribbons_style(
    ca_coords: np.ndarray,
    o_coords: Optional[np.ndarray]= None,
    cb_coords: Optional[np.ndarray]= None,
    ss_types: Optional[np.ndarray]= None,
    width: float = 0.5,
    samples_per_segment: int = 8,
    style: str = RibbonStyle.SQUARE,  # "flat", "circle", "square", "ellipse" - default to square for 3D blocks
    num_threads: int = 8,
    helix_radius_scale: float = 1.0
) -> tuple[
         ndarray, ndarray, ndarray, ndarray, tuple[ndarray, ndarray] | None, tuple[ndarray, ndarray, ndarray] | None] | \
     tuple[GeometryData, tuple[float | Any, float | Any] | None | tuple[Any, Any], tuple[Any, Any, Any] | None]:
    """
    Generate ribbon meshdata using Ribbons' B-spline approach. RIBBON_PATH

    This creates 3D ribbon meshdata with proper Frenet frame orientation,
    similar to Ribbons' ResGeomCircle/ResGeomFlat functions.

    Args:
        ca_coords: (N, 3) array of C-alpha coordinates
        o_coords: (N, 3) array of O coordinates, optional
        cb_coords: (N, 3) array of CB coordinates, optional (not used yet)
        ss_types: (N,) array of secondary structure types, optional
        width: Ribbon width (half-width for flat, radius for circle)
        samples_per_segment: Samples per B-spline segment
        style: "flat", "circle", "square", or "ellipse" - ribbon cross-section style
        num_threads: Number of points around tube cross-section (for "circle" and "ellipse" styles)

    Returns:
        Tuple of (vertices, normals, indices, colors) arrays
    """
    # Calculate guide points
    p_points, q_points = calculate_guide_points(
        ca_coords, o_coords, cb_coords, ss_types, width
    )

    # Evaluate B-splines for centerline and edges
    centerline = evaluate_bspline_chain(
        0.5 * (p_points + q_points),  # Average of p and q gives centerline
        samples_per_segment,
    )

    # Also evaluate edge splines for flat ribbons
    p_spline = evaluate_bspline_chain(p_points, samples_per_segment)
    q_spline = evaluate_bspline_chain(q_points, samples_per_segment)

    # Calculate Frenet frame from edges (Ribbons' SetSpaceCurve approach)
    # This is more accurate as it uses actual ribbon meshdata
    try:
        tangents, normals, binormals, widths = calculate_frenet_frame_from_edges(
            p_spline, centerline, q_spline
        )
    except:
        # Fallback to centerline-only if edges don't match
        tangents, normals, binormals = calculate_frenet_frame(centerline)
        widths = np.ones(len(centerline), dtype=np.float32) * width

    n_points = len(centerline)
    vertices = []
    vertex_normals = []
    indices = []

    if style == RibbonStyle.FLAT:
        # Flat ribbon (like Ribbons' RIB_FLAT) - use actual edge splines
        # This gives better accuracy than offsetting from centerline
        for i in range(n_points):
            # Use the p and q splines directly (these are the actual ribbon edges)
            if i < len(p_spline) and i < len(q_spline):
                left = p_spline[i]
                right = q_spline[i]
            elif i < len(p_spline):
                left = p_spline[i]
                right = q_spline[min(i, len(q_spline) - 1)]
            else:
                # Fallback to centerline + offset using actual width
                center = centerline[i]
                actual_width = widths[i] if i < len(widths) else width
                n = normals[i]
                left = center - n * actual_width * 0.5
                right = center + n * actual_width * 0.5

            # Use the calculated normal from Frenet frame
            # This is more accurate than calculating from edge vectors
            perp = (
                normals[i]
                if i < len(normals)
                else np.array([0.0, 0.0, 1.0], dtype=np.float32)
            )

            vertices.append(left)
            vertices.append(right)
            vertex_normals.append(perp)
            vertex_normals.append(perp)

        # Generate triangle indices (quads as two triangles)
        for i in range(n_points - 1):
            base = i * 2
            # Triangle 1: left1, right1, left2
            indices.extend([base, base + 1, base + 2])
            # Triangle 2: right1, right2, left2
            indices.extend([base + 1, base + 3, base + 2])

    elif style == RibbonStyle.CIRCLE:
        # Circular tube (like Ribbons' RIB_CIRCLE)
        tube_radius = width
        for i in range(n_points):
            center = centerline[i]
            t = tangents[i]
            n = normals[i]
            b = binormals[i]

            # Generate circle of points around the centerline
            for j in range(num_threads):
                angle = 2.0 * np.pi * j / num_threads
                # Rotate normal and binormal around tangent
                offset = n * np.cos(angle) + b * np.sin(angle)
                vertex = center + offset * tube_radius
                vertex_normal = normalize(offset)

                vertices.append(vertex)
                vertex_normals.append(vertex_normal)

        # Generate triangle indices (connect adjacent circles)
        for i in range(n_points - 1):
            base1 = i * num_threads
            base2 = (i + 1) * num_threads

            for j in range(num_threads):
                j_next = (j + 1) % num_threads

                # Two triangles per quad
                indices.extend([base1 + j, base2 + j, base1 + j_next])
                indices.extend([base1 + j_next, base2 + j, base2 + j_next])

    elif style == RibbonStyle.SQUARE:
        # --- VECTORIZE SQUARE STYLE ---

        depth = width * 0.4
        n = len(centerline)

        # Corner coefficients (same as before)
        ca4 = np.array([-0.70710678, 0.70710678, 0.70710678, -0.70710678], dtype=np.float32)
        sa4 = np.array([0.70710678, 0.70710678, -0.70710678, -0.70710678], dtype=np.float32)

        # Ensure widths shape is correct
        if len(widths) != n:
            widths = np.full(n, width, dtype=np.float32)

        # --- Compute scaled basis vectors ---
        rmaj = 0.5 * widths[:, None]  # (n, 1)
        rmin = 0.5 * depth  # scalar

        a = binormals * rmaj  # (n, 3)
        b_scaled = normals * rmin  # (n, 3)

        # --- Compute all 4 corners for all points ---
        # Result: (n, 4, 3)
        corners = (
                centerline[:, None, :]
                + sa4[None, :, None] * a[:, None, :]
                + ca4[None, :, None] * b_scaled[:, None, :]
        )

        # Flatten vertices
        vertices = corners.reshape(-1, 3)

        # --- Compute edge_along (central differences) ---
        edge_along = compute_tangents(centerline)

        # --- Compute edge_across (per corner) ---
        # (n, 4, 3)
        edge_across = np.roll(corners, -1, axis=1) - corners

        # --- Compute normals (vectorized cross product) ---
        normals_all = np.cross(edge_across, edge_along[:, None, :])

        # Normalize safely
        norms = np.linalg.norm(normals_all, axis=2, keepdims=True)
        norms[norms < 1e-8] = 1.0
        normals_all /= norms

        # Flatten normals
        vertex_normals = normals_all.reshape(-1, 3)

        # --- Build indices (vectorized) ---
        # Each segment contributes 8 triangles (24 indices)
        n_segments = n - 1

        base1 = np.arange(n_segments) * 4
        base2 = (np.arange(n_segments) + 1) * 4

        # Build all faces in one shot
        indices = np.stack([
            # Face 0
            base1 + 0, base2 + 0, base1 + 1,
            base1 + 1, base2 + 0, base2 + 1,

            # Face 1
            base1 + 1, base2 + 1, base1 + 2,
            base1 + 2, base2 + 1, base2 + 2,

            # Face 2
            base1 + 2, base2 + 2, base1 + 3,
            base1 + 3, base2 + 2, base2 + 3,

            # Face 3
            base1 + 3, base2 + 3, base1 + 0,
            base1 + 0, base2 + 3, base2 + 0,
        ], axis=1)

        indices = indices.reshape(-1).astype(np.uint32)
        """# Rectangular block (like Ribbons' RIB_SQUARE) - 4 corners
        # Uses depth for thickness (perpendicular to width). Use 0.4 so ribbons
        # are clearly visible (legacy-style width scale makes width large; depth follows).
        depth = width * 0.4

        # Square uses 4 corners: ca4 and sa4 arrays from Ribbons
        # ca4 = [-0.707, 0.707, 0.707, -0.707]
        # sa4 = [0.707, 0.707, -0.707, -0.707]
        # Formula from calc_ellipse: x = xc + sa[i]*a + ca[i]*b
        # where a = vmaj * rmaj (binormal * 0.5*width), b = vmin * rmin (normal * 0.5*depth)
        ca4 = np.array(
            [-0.70710678, 0.70710678, 0.70710678, -0.70710678], dtype=np.float32
        )
        sa4 = np.array(
            [0.70710678, 0.70710678, -0.70710678, -0.70710678], dtype=np.float32
        )

        for i in range(n_points):
            center = centerline[i]
            t = tangents[i]
            n = normals[i]  # vmin (depth direction)
            b = binormals[i]  # vmaj (width direction)

            # Get actual width and depth for this point
            actual_width = widths[i] if i < len(widths) else width
            # Depth could vary by secondary structure, but for now use constant
            actual_depth = depth

            # Scale basis vectors (like calc_ellipse: a = vmaj * rmaj, b = vmin * rmin)
            rmaj = 0.5 * actual_width
            rmin = 0.5 * actual_depth
            a = b * rmaj  # Major axis (width direction)
            b_scaled = n * rmin  # Minor axis (depth direction)

            # Generate 4 corners using Ribbons' formula
            corners = []
            for corner_idx in range(4):
                # x = xc + sa[i]*a + ca[i]*b
                offset = sa4[corner_idx] * a + ca4[corner_idx] * b_scaled
                vertex = center + offset
                corners.append(vertex)
                vertices.append(vertex)

            # Calculate normals for each face (like Ribbons' SetLineNormals for RIB_SQUARE)
            # Ribbons stores 2 normals per face (2*nt normals), but we'll use one per vertex
            # For each corner, calculate normal from adjacent edges
            # We need to get next/prev points for proper normal calculation
            if i < n_points - 1:
                # Get next centerline point for edge calculation
                next_center = centerline[i + 1]
                next_t = tangents[i + 1] if i + 1 < len(tangents) else t
            else:
                next_center = centerline[i] + t * 0.1  # Extrapolate
                next_t = t

            if i > 0:
                prev_center = centerline[i - 1]
            else:
                prev_center = centerline[i] - t * 0.1  # Extrapolate

            # Calculate normals for each corner using adjacent edges
            # Corner 0: between faces 0 and 3
            edge_along = next_center - prev_center  # Edge along curve
            edge_across_0 = corners[1] - corners[0]  # Edge across face 0
            normal_0 = cross(edge_across_0, edge_along)
            normal_0 = normalize(normal_0)

            # Corner 1: between faces 0 and 1
            edge_across_1 = corners[2] - corners[1]  # Edge across face 1
            normal_1 = cross(edge_across_1, edge_along)
            normal_1 = normalize(normal_1)

            # Corner 2: between faces 1 and 2
            edge_across_2 = corners[3] - corners[2]  # Edge across face 2
            normal_2 = cross(edge_across_2, edge_along)
            normal_2 = normalize(normal_2)

            # Corner 3: between faces 2 and 3
            edge_across_3 = corners[0] - corners[3]  # Edge across face 3
            normal_3 = cross(edge_across_3, edge_along)
            normal_3 = normalize(normal_3)

            # Store normals (one per vertex)
            vertex_normals.append(normal_0)
            vertex_normals.append(normal_1)
            vertex_normals.append(normal_2)
            vertex_normals.append(normal_3)

        # Generate triangle indices (connect adjacent rectangles)
        # Each rectangle has 4 corners, so we connect rectangles with quads
        # Ribbons draws quads as quad strips, connecting adjacent corners
        for i in range(n_points - 1):
            base1 = i * 4
            base2 = (i + 1) * 4

            # Create 4 faces (sides) of the rectangular block
            # Each face is a quad strip connecting two adjacent corners
            # Face 0: corners 0-1 (top face)
            indices.extend([base1 + 0, base2 + 0, base1 + 1])
            indices.extend([base1 + 1, base2 + 0, base2 + 1])

            # Face 1: corners 1-2 (right face)
            indices.extend([base1 + 1, base2 + 1, base1 + 2])
            indices.extend([base1 + 2, base2 + 1, base2 + 2])

            # Face 2: corners 2-3 (bottom face)
            indices.extend([base1 + 2, base2 + 2, base1 + 3])
            indices.extend([base1 + 3, base2 + 2, base2 + 3])

            # Face 3: corners 3-0 (left face)
            indices.extend([base1 + 3, base2 + 3, base1 + 0])
            indices.extend([base1 + 0, base2 + 3, base2 + 0])"""


    elif style == RibbonStyle.ELLIPSE:
        # --- VECTORIZE ELLIPSE STYLE ---

        n = n_points
        depth = width * 0.4

        # Ensure widths shape
        if len(widths) != n:
            widths = np.full(n, width, dtype=np.float32)

        # Radii
        rmaj = 0.5 * widths  # (n,)
        rmin = 0.5 * depth  # scalar

        # --- Precompute circle angles ---
        angles = np.linspace(0.0, 2.0 * np.pi, num_threads, endpoint=False, dtype=np.float32)
        cos_a = np.cos(angles)  # (t,)
        sin_a = np.sin(angles)  # (t,)

        # --- Scale basis vectors ---
        # binormals = major axis, normals = minor axis
        a = binormals * rmaj[:, None]  # (n, 3)
        b_scaled = normals * rmin  # (n, 3)

        # --- Compute all vertices ---
        # (n, t, 3)
        offsets = (
                cos_a[None, :, None] * a[:, None, :] +
                sin_a[None, :, None] * b_scaled[:, None, :]
        )

        vertices = (centerline[:, None, :] + offsets).reshape(-1, 3)

        # --- Compute normals (same as normalized offsets) ---
        normals_all = offsets.copy()

        norms = np.linalg.norm(normals_all, axis=2, keepdims=True)
        norms[norms < 1e-8] = 1.0
        normals_all /= norms

        vertex_normals = normals_all.reshape(-1, 3)

        # --- Build indices (vectorized) ---
        n_segments = n - 1
        t = num_threads

        i = np.arange(n_segments)
        j = np.arange(t)

        base1 = (i[:, None] * t)
        base2 = ((i[:, None] + 1) * t)

        j_next = (j + 1) % t

        # Expand for broadcasting
        j = j[None, :]
        j_next = j_next[None, :]

        # Build triangles
        tri1 = np.stack([
            base1 + j,
            base2 + j,
            base1 + j_next
        ], axis=-1)

        tri2 = np.stack([
            base1 + j_next,
            base2 + j,
            base2 + j_next
        ], axis=-1)

        indices = np.concatenate([tri1, tri2], axis=-1).reshape(-1).astype(np.uint32)
        """# Elliptical tube - like CIRCLE but with major (width) and minor (depth) axes
        depth = width * 0.4

        for i in range(n_points):
            center = centerline[i]
            t = tangents[i]
            n = normals[i]  # minor axis (depth direction)
            b = binormals[i]  # major axis (width direction)

            actual_width = widths[i] if i < len(widths) else width
            actual_depth = depth
            rmaj = 0.5 * actual_width
            rmin = 0.5 * actual_depth

            # Generate ellipse of points around the centerline
            for j in range(num_threads):
                angle = 2.0 * np.pi * j / num_threads
                # Parametric ellipse: offset = rmaj*cos*binormal + rmin*sin*normal
                offset = b * (rmaj * np.cos(angle)) + n * (rmin * np.sin(angle))
                vertex = center + offset
                vertex_normal = normalize(offset)

                vertices.append(vertex)
                vertex_normals.append(vertex_normal)

        # Generate triangle indices (connect adjacent ellipses)
        for i in range(n_points - 1):
            base1 = i * num_threads
            base2 = (i + 1) * num_threads

            for j in range(num_threads):
                j_next = (j + 1) % num_threads

                indices.extend([base1 + j, base2 + j, base1 + j_next])
                indices.extend([base1 + j_next, base2 + j, base2 + j_next])
        """
    else:  # default to square
        # Default to square for 3D blocks
        return generate_ribbon_geometry_ribbons_style(
            ca_coords,
            o_coords,
            cb_coords,
            ss_types,
            width,
            samples_per_segment,
            RibbonStyle.SQUARE,
            num_threads,
        )

    vertices = np.array(vertices, dtype=np.float32)
    vertex_normals = np.array(vertex_normals, dtype=np.float32)
    indices = np.array(indices, dtype=np.uint32)

    # Colors (default to white, can be customized)
    colors = np.ones((len(vertices), 3), dtype=np.float32)

    # Return ribbon edge information for arrow generation (Ribbons-style)
    # Extract the last cross-section edges for arrow base
    ribbon_edges = None
    ribbon_frenet = None

    if style == RibbonStyle.SQUARE and len(vertices) >= 4:
        # For square style, get the last 4 corners (last cross-section)
        last_corners = vertices[-4:]
        # Calculate left and right edges (average of opposite corners)
        left_edge = (last_corners[0] + last_corners[3]) * 0.5
        right_edge = (last_corners[1] + last_corners[2]) * 0.5
        ribbon_edges = (left_edge, right_edge)

        # Also return Frenet frame at the end for arrow orientation
        if len(centerline) > 0:
            last_idx = len(centerline) - 1
            if (
                last_idx < len(tangents)
                and last_idx < len(normals)
                and last_idx < len(binormals)
            ):
                ribbon_frenet = (
                    tangents[last_idx],
                    normals[last_idx],
                    binormals[last_idx],
                )

    elif style == RibbonStyle.ELLIPSE and len(centerline) > 0:
        # For ellipse, use last cross-section extremes along major axis
        last_idx = len(centerline) - 1
        if (
            last_idx < len(tangents)
            and last_idx < len(normals)
            and last_idx < len(binormals)
        ):
            center = centerline[last_idx]
            b = binormals[last_idx]
            rmaj = 0.5 * (widths[last_idx] if last_idx < len(widths) else width)
            depth = width * 0.4
            left_edge = center + b * rmaj
            right_edge = center - b * rmaj
            ribbon_edges = (left_edge, right_edge)
            ribbon_frenet = (
                tangents[last_idx],
                normals[last_idx],
                binormals[last_idx],
            )

    elif style == RibbonStyle.CIRCLE and len(centerline) > 0:
        # For circle, use last cross-section extremes along major axis
        last_idx = len(centerline) - 1
        if (
            last_idx < len(tangents)
            and last_idx < len(normals)
            and last_idx < len(binormals)
        ):
            center = centerline[last_idx]
            b = binormals[last_idx]
            tube_radius = widths[last_idx] if last_idx < len(widths) else width
            left_edge = center + b * tube_radius
            right_edge = center - b * tube_radius
            ribbon_edges = (left_edge, right_edge)
            ribbon_frenet = (
                tangents[last_idx],
                normals[last_idx],
                binormals[last_idx],
            )

    elif style == RibbonStyle.FLAT and len(vertices) >= 2:
        # For flat style, last two vertices are the edges
        ribbon_edges = (vertices[-2], vertices[-1])

        # Return Frenet frame
        if len(centerline) > 0:
            last_idx = len(centerline) - 1
            if (
                last_idx < len(tangents)
                and last_idx < len(normals)
                and last_idx < len(binormals)
            ):
                ribbon_frenet = (
                    tangents[last_idx],
                    normals[last_idx],
                    binormals[last_idx],
                )
    geo_data = GeometryData(vertices=vertices, normals=vertex_normals, indices=indices, colors=colors)
    return geo_data, ribbon_edges, ribbon_frenet


def generate_resgeom_flat(
    p_guide_points: np.ndarray,
    q_guide_points: np.ndarray,
    num_threads: int = 2,
    num_samples: int = 8,
    arrow_base_width: Optional[float | np.floating[Any]] = None,
    arrow_head_width: Optional[float] = None,
    has_arrow: bool = False,
    force_thru_ca: bool = False,
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    Generate flat ribbon meshdata using Ribbons' ResGeomFlat approach.

    This ports the ResGeomFlat function from Ribbons C++ code, including:
    - SetGuidePoints: Interpolates guide points for multiple threads
    - SetGuideLines: Evaluates B-spline curves
    - SetLineNormals: Calculates normals from cross products
    - ArrowLines: Tapers ribbon width for arrowhead (if has_arrow=True)
    - FlipLineNormals: Creates double-sided meshdata with thickness

    This is the "lovely" meshdata generation that creates beautiful beta sheet
    ribbons with proper arrowheads that follow the ribbon plane.

    Args:
        p_guide_points: (N, 3) array of p guide points (left edge)
        q_guide_points: (N, 3) array of q guide points (right edge)
        num_threads: Number of threads (ribbon width subdivisions), typically 2 for flat
        num_samples: Number of samples per B-spline segment
        arrow_base_width: Width at arrow base (if None, calculated from guide points)
        arrow_head_width: Width at arrow head (defaults to 0.0 for point)
        has_arrow: Whether to apply arrowhead tapering
        force_thru_ca: Whether to force ribbon through CA atoms (not implemented yet)

    Returns:
        Tuple of (vertices, normals, indices, colors) arrays
        Vertices are organized as: front face (all threads), back face (all threads)
        Each face has vertices for all threads and all samples
    """
    n_guides = len(p_guide_points)
    if n_guides != len(q_guide_points):
        raise ValueError("p_guide_points and q_guide_points must have same length")
    if n_guides < 4:
        raise ValueError("Need at least 4 guide points for B-spline")

    # Step 1: SetGuidePoints - Interpolate guide points for multiple threads
    # This matches Ribbons' SetGuidePoints function (lines 1097-1121)
    # xg[thread][guide_point][xyz] stores interpolated guide points
    xg = np.zeros((num_threads, 4, 3), dtype=np.float32)

    if num_threads > 1:
        f = 1.0 / (num_threads - 1)
        for k in range(num_threads):
            for j in range(min(4, n_guides)):
                # Interpolate between p and q guide points
                # xg[k][j] = p[j] + k*f * (q[j] - p[j])
                xg[k, j] = p_guide_points[j] + k * f * (
                    q_guide_points[j] - p_guide_points[j]
                )
    else:
        # Single thread: use average of p and q
        for j in range(min(4, n_guides)):
            xg[0, j] = 0.5 * (p_guide_points[j] + q_guide_points[j])

    # Step 2: SetGuideLines - Evaluate B-spline curves for each thread
    # This matches Ribbons' SetGuideLines function (lines 1128-1190)
    # We need to evaluate B-splines for segments starting at each guide point
    # Estimate max samples: (n_guides - 3) segments * num_samples + 1
    max_samples = (n_guides - 3) * num_samples + 1
    xv = np.zeros(
        (num_threads + 2, max_samples + 4, 3), dtype=np.float32
    )  # +2 for edge wrapping, +4 for padding

    # Evaluate B-spline for each thread
    for k in range(num_threads):
        sample_idx = 1  # Start at 1 (0 is for previous residue)
        t_vals = np.linspace(0, 1, num_samples, endpoint=False)

        # Evaluate segments through all guide points
        for i in range(n_guides - 3):
            if i + 3 >= n_guides:
                break

            # Get 4 consecutive guide points for this thread
            if num_threads > 1:
                f = 1.0 / (num_threads - 1)
                g0 = p_guide_points[i] + k * f * (q_guide_points[i] - p_guide_points[i])
                g1 = p_guide_points[i + 1] + k * f * (
                    q_guide_points[i + 1] - p_guide_points[i + 1]
                )
                g2 = p_guide_points[i + 2] + k * f * (
                    q_guide_points[i + 2] - p_guide_points[i + 2]
                )
                g3 = p_guide_points[i + 3] + k * f * (
                    q_guide_points[i + 3] - p_guide_points[i + 3]
                )
            else:
                g0 = 0.5 * (p_guide_points[i] + q_guide_points[i])
                g1 = 0.5 * (p_guide_points[i + 1] + q_guide_points[i + 1])
                g2 = 0.5 * (p_guide_points[i + 2] + q_guide_points[i + 2])
                g3 = 0.5 * (p_guide_points[i + 3] + q_guide_points[i + 3])

            # Choose matrix based on segment position (matching Ribbons bs_line)
            if i == 0:
                matrix = BS_MAT_A
            elif i == 1:
                matrix = BS_MAT_B
            elif i == n_guides - 4:
                matrix = BS_MAT_Y
            elif i == n_guides - 3:
                matrix = BS_MAT_Z
            else:
                matrix = BS_MAT

            # Evaluate B-spline segment
            for t in t_vals:
                pt = evaluate_bspline_segment(g0, g1, g2, g3, t, matrix)
                if sample_idx < max_samples:
                    xv[k + 1, sample_idx] = pt
                    sample_idx += 1

        # Add last point
        if sample_idx < max_samples and n_guides >= 2:
            if num_threads > 1:
                f = 1.0 / (num_threads - 1)
                last_pt = p_guide_points[-2] + k * f * (
                    q_guide_points[-2] - p_guide_points[-2]
                )
            else:
                last_pt = 0.5 * (p_guide_points[-2] + q_guide_points[-2])
            xv[k + 1, sample_idx] = last_pt
            sample_idx += 1

    # Determine actual number of samples
    ns = sample_idx - 1

    # Step 3: ArrowLines - Taper ribbon width for arrowhead (if requested)
    # This matches Ribbons' ArrowLines function (lines 1524-1555)
    if has_arrow and num_threads >= 2:
        # Calculate arrow parameters
        wah = arrow_head_width if arrow_head_width is not None else 0.0
        wab = arrow_base_width if arrow_base_width is not None else 1.0

        # Calculate width decrement per sample
        wah_increment = (wab - wah) / ns if ns > 0 else 0.0

        # Apply tapering to each sample
        for j in range(1, ns + 2):  # j from 1 to ns+1 (matching Ribbons indexing)
            # Calculate current width
            current_wab = wab - wah_increment * (j - 1)

            # Get left and right edges (threads 1 and nt)
            d = xv[num_threads, j] - xv[1, j]  # Right - left
            c = 0.5 * (xv[num_threads, j] + xv[1, j])  # Center

            # Calculate scale factor
            s = np.linalg.norm(d)
            if s > MoLibConstant.EPSILON:
                s = 0.5 * (current_wab / s)
            else:
                s = 0.0

            # Modify edges (matching Ribbons ArrowLines logic)
            xv[1, j] = c - s * d  # Left edge
            xv[num_threads, j] = c + s * d  # Right edge

        # Interpolate intermediate threads (if nt > 3)
        # This matches Ribbons lines 1545-1554
        if num_threads > 3:
            s = 1.0 / (num_threads - 1.0)
            for k in range(2, num_threads):
                for j in range(1, ns + 2):
                    # Interpolate between thread 1 and thread nt
                    xv[k, j] = xv[1, j] + (k - 1) * s * (xv[num_threads, j] - xv[1, j])

    # Step 4: SetLineNormals - Calculate normals from cross products
    # This matches Ribbons' SetLineNormals function for RIB_FLAT (lines 1465-1501)
    # Set edge wrapping for flat style
    for j in range(ns + 3):  # 0 to ns+2
        if j < max_samples + 4:
            xv[0, j] = xv[1, j]  # Wrap left edge
            xv[num_threads + 1, j] = xv[num_threads, j]  # Wrap right edge

    # Calculate normals
    xn = np.zeros(
        (num_threads + 1, ns + 2, 3), dtype=np.float32
    )  # [thread][sample][xyz]

    for k in range(1, num_threads + 1):
        for j in range(1, ns + 2):  # j from 1 to ns+1
            if j + 1 < max_samples + 4 and k + 1 < num_threads + 2:
                # Calculate vectors for cross product
                # a = difference along thread direction (k+1 - k-1)
                # b = difference along sample direction (j+1 - j-1)
                a = xv[k + 1, j] - xv[k - 1, j]
                b = xv[k, j + 1] - xv[k, j - 1]

                # Cross product gives normal
                c = cross(a, b)
                c = normalize(c)
                xn[k, j] = c

    # Step 5: FlipLineNormals - Create backface with thickness
    # This matches Ribbons' FlipLineNormals function (lines 1707-1714)
    FLAT_FUDGE = 0.12  # Thickness for backface

    # Flip normals for backface
    xn_back = -xn.copy()

    # Offset vertices for backface (add thickness along normal)
    xv_back = xv.copy()
    for k in range(2, num_threads + 1):  # Skip edge threads (1 and nt)
        for j in range(1, ns + 2):
            if j < max_samples + 4:
                xv_back[k, j] = xv[k, j] + FLAT_FUDGE * xn[k, j]

    # Step 6: Build output arrays (matching ResGeomFlat structure)
    # Front face: all threads, all samples
    # Back face: all threads (reversed), all samples
    # This matches ResGeomFlat lines 1741-1753
    vertices = []
    normals = []

    # Front face vertices and normals
    for k in range(num_threads):
        for j in range(ns + 1):  # j from 0 to ns
            if j + 1 < max_samples + 4:
                vertices.append(xv[k + 1, j + 1])  # +1 for 1-based indexing
                normals.append(xn[k + 1, j + 1])

    # Back face vertices and normals (reversed thread order)
    for k in range(num_threads):
        k_rev = num_threads - 1 - k  # Reverse order
        for j in range(ns + 1):
            if j + 1 < max_samples + 4:
                vertices.append(xv_back[k_rev + 1, j + 1])
                normals.append(xn_back[k_rev + 1, j + 1])

    vertices = np.array(vertices, dtype=np.float32)
    normals = np.array(normals, dtype=np.float32)

    # Generate triangle indices (QUAD_STRIP converted to triangles)
    # Front face: quads between adjacent threads
    # This matches DrawRibnFlat's QUAD_STRIP logic
    indices = []
    front_base = 0
    back_base = num_threads * (ns + 1)

    for k in range(num_threads - 1):
        for j in range(ns):
            # Front face quad
            v0 = front_base + k * (ns + 1) + j
            v1 = front_base + k * (ns + 1) + j + 1
            v2 = front_base + (k + 1) * (ns + 1) + j
            v3 = front_base + (k + 1) * (ns + 1) + j + 1

            # Two triangles per quad (matching DrawRibnFlat QUAD_STRIP)
            indices.extend([v0, v1, v2])
            indices.extend([v1, v3, v2])

            # Back face quad (reversed winding)
            v0_b = back_base + k * (ns + 1) + j
            v1_b = back_base + k * (ns + 1) + j + 1
            v2_b = back_base + (k + 1) * (ns + 1) + j
            v3_b = back_base + (k + 1) * (ns + 1) + j + 1

            # Two triangles per quad (reversed for backface)
            indices.extend([v0_b, v2_b, v1_b])
            indices.extend([v1_b, v2_b, v3_b])

    indices = np.array(indices, dtype=np.uint32)

    # Colors (default to white)
    colors = np.ones((len(vertices), 3), dtype=np.float32)

    return vertices, normals, indices, colors


def create_resgeom_flat_from_context(
    p_guide_points: np.ndarray[Any, np.dtype[Any]] | ndarray[Any, np.dtype[np.generic]],
    q_guide_points: np.ndarray[Any, np.dtype[Any]] | ndarray[Any, np.dtype[np.generic]],
    ctx: _ResgeomContext,
) -> tuple[ndarray, ndarray]:
    """create resgeomm from context"""
    vertices, normals, _indices, _colors_flat = generate_resgeom_flat(
        p_guide_points=p_guide_points,
        q_guide_points=q_guide_points,
        num_threads=ctx.num_threads,
        num_samples=ctx.num_samples,
        arrow_base_width=ctx.arrow_base_width,
        arrow_head_width=ctx.arrow_head_width,
        has_arrow=ctx.has_arrow,
        force_thru_ca=ctx.force_thru_ca,
    )
    return normals, vertices

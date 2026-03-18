import numpy as np
from molib.calc.geometry.angle import (
    dihedral_angle,
    rotate_vector_around_axis,
    smooth_torsion_angles,
)
from scipy import ndimage
from scipy.signal import savgol_filter


def apply_torsion_smoothing(backbone_points, normal_vectors, window=5):
    """
    Given backbone points and normal vectors, smooth the torsion (dihedral) angles,
    then rotate each normal so its plane follows the smoothed torsion.
    Returns new normals array.

    Args:
        backbone_points: Array of backbone point coordinates
        normal_vectors: Array of normal vectors
        window: Window size for torsion angle smoothing (default 5)
    """
    n = len(backbone_points)
    # Calculate torsion angles for each 4-point segment
    torsion_angles = []
    for i in range(n - 3):
        a = dihedral_angle(
            backbone_points[i],
            backbone_points[i + 1],
            backbone_points[i + 2],
            backbone_points[i + 3],
        )
        torsion_angles.append(a)
    torsion_angles = np.array(torsion_angles)

    # Filter out NaN values and replace with 0
    torsion_angles = np.nan_to_num(torsion_angles, nan=0.0)

    # Smooth torsion angles with configurable window size
    smoothed_angles = smooth_torsion_angles(torsion_angles, window=window)

    # Filter out NaN values from smoothed angles as well
    smoothed_angles = np.nan_to_num(smoothed_angles, nan=0.0)

    # Apply torsion smoothing to normals
    new_normals = np.copy(normal_vectors)
    for i in range(1, n - 2):
        axis = backbone_points[i + 1] - backbone_points[i]
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-8:
            continue
        axis = axis / axis_norm
        # Rotate the normal at i+1 by the difference between smoothed and raw torsion
        raw_angle = torsion_angles[i - 1] if i - 1 < len(torsion_angles) else 0
        smooth_angle = smoothed_angles[i - 1] if i - 1 < len(smoothed_angles) else 0
        delta_angle = smooth_angle - raw_angle
        # Rotate the normal
        new_normals[i + 1] = rotate_vector_around_axis(
            normal_vectors[i + 1], axis, delta_angle
        )
    return new_normals


def apply_face_dihedral_smoothing(face_positions, face_normals, window=5):
    """
    Smooth the dihedral angles between adjacent faces to reduce abrupt angularity.

    The dihedral angle between two adjacent faces is the angle between their planes
    when viewed from the side. This controls how much the faces "bend" or "twist"
    relative to each other.

    Args:
        face_positions: Array of face center positions (N, 3)
        face_normals: Array of face normal vectors (N, 3)
        window: Window size for dihedral angle smoothing (default 5)

    Returns:
        smoothed_face_normals: Array of smoothed face normal vectors (N, 3)
    """
    n = len(face_positions)
    if n < 3:
        return face_normals.copy()

    # Calculate dihedral angles between adjacent faces
    dihedral_angles = []
    for i in range(n - 1):
        # Get the two adjacent faces
        face1_center = face_positions[i]
        face2_center = face_positions[i + 1]
        face1_normal = face_normals[i]
        face2_normal = face_normals[i + 1]

        # Calculate the dihedral angle between the two face planes
        # This is the angle between the normals projected onto the plane perpendicular
        # to the line connecting the face centers
        face_connection = face2_center - face1_center
        face_connection_norm = np.linalg.norm(face_connection)

        if face_connection_norm < 1e-8:
            dihedral_angles.append(0.0)
            continue

        face_connection = face_connection / face_connection_norm

        # Project normals onto plane perpendicular to face connection
        n1_proj = face1_normal - np.dot(face1_normal, face_connection) * face_connection
        n2_proj = face2_normal - np.dot(face2_normal, face_connection) * face_connection

        n1_norm = np.linalg.norm(n1_proj)
        n2_norm = np.linalg.norm(n2_proj)

        if n1_norm < 1e-8 or n2_norm < 1e-8:
            dihedral_angles.append(0.0)
            continue

        n1_proj = n1_proj / n1_norm
        n2_proj = n2_proj / n2_norm

        # Calculate dihedral angle
        cos_angle = np.dot(n1_proj, n2_proj)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)

        # Determine sign using cross product
        cross_product = np.cross(n1_proj, n2_proj)
        sign = np.sign(np.dot(cross_product, face_connection))
        dihedral_angles.append(sign * angle)

    dihedral_angles = np.array(dihedral_angles)

    # Smooth the dihedral angles
    smoothed_angles = smooth_torsion_angles(dihedral_angles, window=window)

    # Filter out NaN values from smoothed angles
    smoothed_angles = np.nan_to_num(smoothed_angles, nan=0.0)

    # Apply the smoothed dihedral angles to adjust face normals
    smoothed_normals = face_normals.copy()

    for i in range(1, n - 1):
        # Calculate the rotation axis (perpendicular to both face connection and current normal)
        face_connection = face_positions[i + 1] - face_positions[i - 1]
        face_connection_norm = np.linalg.norm(face_connection)

        if face_connection_norm < 1e-8:
            continue

        face_connection = face_connection / face_connection_norm

        # Get the rotation axis
        rotation_axis = np.cross(face_connection, face_normals[i])
        rotation_axis_norm = np.linalg.norm(rotation_axis)

        if rotation_axis_norm < 1e-8:
            continue

        rotation_axis = rotation_axis / rotation_axis_norm

        # Calculate the angle difference to apply
        if i - 1 < len(smoothed_angles):
            raw_angle = dihedral_angles[i - 1]
            smooth_angle = smoothed_angles[i - 1]
            delta_angle = smooth_angle - raw_angle

            # Apply rotation to the normal
            smoothed_normals[i] = rotate_vector_around_axis(
                face_normals[i], rotation_axis, delta_angle
            )

    return smoothed_normals


def apply_arrow_dihedral_smoothing(
    arrow_face_center,
    arrow_face_normal,
    prev_face_center,
    prev_face_normal,
    smoothing_factor=0.7,
):
    """
    Smooth the dihedral angle between an arrow face and the previous face to reduce abrupt transitions.
    The angle change is constrained to a maximum of 10 degrees to prevent direction reversal.

    Args:
        arrow_face_center: Center position of the arrow face (3D array)
        arrow_face_normal: Normal vector of the arrow face (3D array)
        prev_face_center: Center position of the previous face (3D array)
        prev_face_normal: Normal vector of the previous face (3D array)
        smoothing_factor: How much to smooth the dihedral angle (0.0 = no change, 1.0 = fully aligned)

    Returns:
        smoothed_arrow_normal: Smoothed normal vector for the arrow face (max 10° change from original)
    """
    # Calculate the connection vector between faces
    face_connection = arrow_face_center - prev_face_center
    face_connection_norm = np.linalg.norm(face_connection)

    if face_connection_norm < 1e-8:
        return arrow_face_normal

    face_connection = face_connection / face_connection_norm

    # Calculate the angle between the normals directly
    # This is more robust than projecting onto a plane
    cos_angle = np.dot(prev_face_normal, arrow_face_normal)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    dihedral_angle = np.arccos(cos_angle)

    # Determine sign using cross product
    cross_product = np.cross(prev_face_normal, arrow_face_normal)
    dot_product = np.dot(cross_product, face_connection)

    # Handle numerical precision issues
    if abs(dot_product) < 1e-8:
        # If cross product is nearly zero, check if normals are similar
        if np.dot(prev_face_normal, arrow_face_normal) > 0:
            sign = 1.0
        else:
            sign = -1.0
    else:
        sign = np.sign(dot_product)

    dihedral_angle = sign * dihedral_angle

    # Apply smoothing to the dihedral angle
    # When smoothing_factor = 0.0: no change (smoothed_angle = 0)
    # When smoothing_factor = 1.0: full alignment (smoothed_angle = dihedral_angle)
    smoothed_angle = dihedral_angle * smoothing_factor

    # Constrain the angle change to maximum 10 degrees to prevent direction reversal
    max_angle_change = np.radians(10.0)  # 10 degrees in radians
    if abs(smoothed_angle) > max_angle_change:
        smoothed_angle = np.sign(smoothed_angle) * max_angle_change

    # Calculate the rotation axis as the cross product of the two normals
    # This gives us the axis perpendicular to both normals
    rotation_axis = np.cross(arrow_face_normal, prev_face_normal)
    rotation_axis_norm = np.linalg.norm(rotation_axis)

    if rotation_axis_norm < 1e-8:
        # If normals are parallel, use a perpendicular vector
        # Find any vector perpendicular to the arrow normal
        if abs(arrow_face_normal[0]) < 0.9:
            rotation_axis = np.cross(arrow_face_normal, np.array([1.0, 0.0, 0.0]))
        else:
            rotation_axis = np.cross(arrow_face_normal, np.array([0.0, 1.0, 0.0]))
        rotation_axis_norm = np.linalg.norm(rotation_axis)

        if rotation_axis_norm < 1e-8:
            return arrow_face_normal

    rotation_axis = rotation_axis / rotation_axis_norm

    # Apply the smoothed rotation to the arrow normal
    # We want to rotate from the current arrow normal towards the previous normal
    # The rotation angle should be the smoothed angle
    smoothed_arrow_normal = rotate_vector_around_axis(
        arrow_face_normal, rotation_axis, smoothed_angle
    )

    return smoothed_arrow_normal


def apply_gaussian_smoothing(vectors, sigma=2.0):
    """
    Apply Gaussian smoothing to a set of vectors for ultra-smooth transitions.

    Args:
        vectors: Array of vectors to smooth (N, 3)
        sigma: Standard deviation for Gaussian kernel (default: 2.0)

    Returns:
        Smoothed vectors array (N, 3)
    """
    if len(vectors) < 3:
        return vectors.copy()

    # Apply Gaussian smoothing to each component
    smoothed_vectors = np.zeros_like(vectors)
    for i in range(3):  # x, y, z components
        smoothed_vectors[:, i] = ndimage.gaussian_filter1d(vectors[:, i], sigma=sigma)

    # Normalize the smoothed vectors
    norms = np.linalg.norm(smoothed_vectors, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0  # Avoid division by zero
    smoothed_vectors = smoothed_vectors / norms

    return smoothed_vectors


def apply_gaussian_smoothing_positions(points, sigma=1.0):
    """
    Apply Gaussian smoothing along the sequence to position arrays (N, 3).
    Unlike vector smoothing, this does NOT normalize outputs.

    Args:
        points: np.ndarray of shape (N, 3)
        sigma: Standard deviation for the Gaussian kernel

    Returns:
        np.ndarray of shape (N, 3) with smoothed positions
    """
    if len(points) < 3:
        return points.copy()

    smoothed = np.zeros_like(points)
    for i in range(3):
        smoothed[:, i] = ndimage.gaussian_filter1d(points[:, i], sigma=sigma)
    return smoothed


def apply_savitzky_golay_smoothing(vectors, window_length=15, polyorder=3):
    """
    Apply Savitzky-Golay smoothing to vectors for preserving features while smoothing.

    Args:
        vectors: Array of vectors to smooth (N, 3)
        window_length: Window length for Savitzky-Golay filter (must be odd)
        polyorder: Polynomial order (must be less than window_length)

    Returns:
        Smoothed vectors array (N, 3)
    """
    if len(vectors) < window_length:
        return vectors.copy()

    # Ensure window_length is odd
    if window_length % 2 == 0:
        window_length += 1

    # Ensure polyorder is valid
    polyorder = min(polyorder, window_length - 1)

    smoothed_vectors = np.zeros_like(vectors)
    for i in range(3):  # x, y, z components
        smoothed_vectors[:, i] = savgol_filter(vectors[:, i], window_length, polyorder)

    # Normalize the smoothed vectors
    norms = np.linalg.norm(smoothed_vectors, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0  # Avoid division by zero
    smoothed_vectors = smoothed_vectors / norms

    return smoothed_vectors


def apply_adaptive_smoothing(vectors, curvature_threshold=0.1):
    """
    Apply adaptive smoothing based on local curvature - more smoothing in high-curvature regions.

    Args:
        vectors: Array of vectors to smooth (N, 3)
        curvature_threshold: Threshold for high curvature detection

    Returns:
        Smoothed vectors array (N, 3)
    """
    if len(vectors) < 5:
        return vectors.copy()

    # Calculate local curvature
    curvatures = []
    for i in range(1, len(vectors) - 1):
        # Calculate curvature as the angle between consecutive segments
        v1 = vectors[i] - vectors[i - 1]
        v2 = vectors[i + 1] - vectors[i]

        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)

        if v1_norm > 1e-8 and v2_norm > 1e-8:
            cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            curvature = 1.0 - cos_angle  # Higher values = more curved
        else:
            curvature = 0.0

        curvatures.append(curvature)

    # Pad curvatures for first and last points
    curvatures = [0.0] + curvatures + [0.0]
    curvatures = np.array(curvatures)

    # Apply adaptive smoothing
    smoothed_vectors = vectors.copy()
    for i in range(1, len(vectors) - 1):
        if curvatures[i] > curvature_threshold:
            # High curvature region - apply more aggressive smoothing
            window = min(5, len(vectors) - 1)
            start_idx = max(0, i - window // 2)
            end_idx = min(len(vectors), i + window // 2 + 1)

            # Average nearby vectors
            local_vectors = vectors[start_idx:end_idx]
            smoothed_vectors[i] = np.mean(local_vectors, axis=0)

            # Normalize
            norm = np.linalg.norm(smoothed_vectors[i])
            if norm > 1e-8:
                smoothed_vectors[i] /= norm

    return smoothed_vectors


def apply_multi_pass_smoothing(
    vectors, num_passes=3, methods=["gaussian", "savitzky", "adaptive"]
):
    """
    Apply multiple passes of different smoothing methods for ultra-smooth results.

    Args:
        vectors: Array of vectors to smooth (N, 3)
        num_passes: Number of smoothing passes
        methods: List of smoothing methods to apply

    Returns:
        Smoothed vectors array (N, 3)
    """
    smoothed_vectors = vectors.copy()

    for pass_num in range(num_passes):
        for method in methods:
            if method == "gaussian":
                smoothed_vectors = apply_gaussian_smoothing(smoothed_vectors, sigma=1.5)
            elif method == "savitzky":
                smoothed_vectors = apply_savitzky_golay_smoothing(
                    smoothed_vectors, window_length=11, polyorder=3
                )
            elif method == "adaptive":
                smoothed_vectors = apply_adaptive_smoothing(
                    smoothed_vectors, curvature_threshold=0.05
                )

    return smoothed_vectors

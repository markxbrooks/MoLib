from typing import Optional

import numpy as np

from molib.calc.math.matrix_util import cross, cross_normalize
from molib.calc.math.normal import normalize
from molib.calc.math.numpy_util import get_np_array, to_up_vec3
from molib.core.constants import MoLibConstant
from molib.pdb.structure.ribbons.calc import use_ribbon_edges_to_determine_arrow_plane, \
    calculate_normals_along_binormal_and_direction, calculate_normals_along_direction, calculate_normals_along_binormal



def generate_arrow_geometry_from_context(config, context, p1: Any, p2: Any, ribbon_geom: RibbonGeometryContext) -> \
tuple[ndarray, ndarray, ndarray, ndarray]:
    av, an, ai, ac = generate_arrow_geometry(
        p1, p2,
        width=config.width_scale * 0.35,
        color=tuple(context.colors[-1]),
        ribbon_plane_normal=ribbon_geom.plane_normal,
        ribbon_binormal=ribbon_geom.binormal,
        ribbon_left_edge=ribbon_geom.left_edge,
        ribbon_right_edge=ribbon_geom.right_edge,
    )
    return ac, ai, an, av

def generate_arrow_geometry(
    p1: np.ndarray,
    p2: np.ndarray,
    width: float = 0.3,
    color: tuple = (1.0, 1.0, 1.0),
    ribbon_plane_normal: Optional[np.ndarray] = None,
    ribbon_binormal: Optional[np.ndarray] = None,
    ribbon_left_edge: Optional[np.ndarray] = None,
    ribbon_right_edge: Optional[np.ndarray] = None,
    arrow_base_width: Optional[float] = None,
    arrow_head_width: Optional[float] = None,
    num_samples: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate vertex/normal/index/colour arrays for a beta-sheet arrow
    that follows the ribbon plane (Ribbons-style).

    This implements Ribbons' ArrowLines approach, which modifies the ribbon
    edges to create an arrowhead that naturally follows the ribbon plane.

    Args:
        p1: Start point of arrow (base, typically end of ribbon)
        p2: End point of arrow (tip direction)
        width: Base width of arrow (if ribbon edges not provided)
        color: RGB color tuple
        ribbon_plane_normal: Normal to ribbon plane (for proper orientation)
        ribbon_binormal: Binormal vector (width direction in ribbon plane)
        ribbon_left_edge: Left edge point of ribbon at arrow base (preferred)
        ribbon_right_edge: Right edge point of ribbon at arrow base (preferred)
        arrow_base_width: Width at arrow base (defaults to width)
        arrow_head_width: Width at arrow head (defaults to 0.0 for point)
        num_samples: Number of samples along arrow length

    Returns:
        Tuple of (vertices, normals, indices, colors) arrays
    """
    p1 = get_np_array(p1)
    p2 = get_np_array(p2)

    # Direction vector (arrow direction)
    direction = p2 - p1
    length = np.linalg.norm(direction)
    if length < MoLibConstant.EPSILON:
        return (
            np.zeros((0, 3)),
            np.zeros((0, 3)),
            np.zeros((0,), dtype=np.uint32),
            np.zeros((0, 3)),
        )

    direction = direction / length

    # If ribbon edges are provided, use them (Ribbons approach)
    if ribbon_left_edge is not None and ribbon_right_edge is not None:
        base_width, binormal, head_width = use_ribbon_edges_to_determine_arrow_plane(arrow_base_width,
                                                                                     arrow_head_width, direction,
                                                                                     ribbon_binormal, ribbon_left_edge,
                                                                                     ribbon_plane_normal,
                                                                                     ribbon_right_edge)

    else:
        # Fallback: calculate plane from direction and provided vectors
        if ribbon_binormal is not None:
            binormal = normalize(get_np_array(ribbon_binormal))
        else:
            # Calculate binormal from direction and a hint vector
            up = to_up_vec3()
            binormal = cross(direction, up)
            if np.linalg.norm(binormal) < MoLibConstant.EPSILON:
                up = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                binormal = cross(direction, up)
            binormal = normalize(binormal)

        if ribbon_plane_normal is not None:
            normal = normalize(get_np_array(ribbon_plane_normal))
        else:
            cross_normalize(binormal, direction)

        base_width = arrow_base_width if arrow_base_width is not None else width
        head_width = arrow_head_width if arrow_head_width is not None else 0.0

    # Generate arrow meshdata using Ribbons' approach
    # Create samples along arrow length, tapering from base_width to head_width
    vertices = []
    vertex_normals = []
    indices = []

    # Ribbons' ArrowLines approach: taper width along the arrow
    for i in range(num_samples + 1):
        t = i / num_samples  # Parameter from 0 (base) to 1 (tip)

        # Taper width from base to head
        current_width = base_width * (1.0 - t) + head_width * t

        # Position along arrow
        pos = p1 + direction * (length * t)

        # Calculate left and right edges at this position
        # Using binormal for width direction
        half_width = current_width * 0.5
        left = pos - binormal * half_width
        right = pos + binormal * half_width

        vertices.append(left)
        vertices.append(right)

        # Calculate normals (pointing outward from arrow center)
        # Normal should be perpendicular to arrow surface
        if i == 0:
            calculate_normals_along_binormal(binormal, vertex_normals)
        elif i == num_samples:
            calculate_normals_along_direction(left, normalize, pos, right, vertex_normals)
        else:
            calculate_normals_along_binormal_and_direction(left, normalize, pos, right, vertex_normals)

    # Add arrow tip (point)
    tip = p2
    vertices.append(tip)
    # Tip normal points along direction
    vertex_normals.append(direction)

    vertices = np.array(vertices, dtype=np.float32)
    vertex_normals = np.array(vertex_normals, dtype=np.float32)

    # Generate triangle indices
    # Body: quad strips connecting adjacent samples
    for i in range(num_samples):
        base = i * 2
        # Quad as two triangles
        indices.extend([base, base + 1, base + 2])  # Triangle 1
        indices.extend([base + 1, base + 3, base + 2])  # Triangle 2

    # Arrow head: triangle connecting last sample to tip
    tip_idx = len(vertices) - 1
    last_base = num_samples * 2
    indices.extend([last_base, last_base + 1, tip_idx])

    indices = np.array(indices, dtype=np.uint32)

    # Color buffer
    colors = np.tile(color, (len(vertices), 1)).astype(np.float32)

    return vertices, vertex_normals, indices, colors

from typing import Any

import numpy as np

from molib.calc.math.matrix_util import cross, cross_normalize
from molib.calc.math.normal import normalize
from molib.calc.math.numpy_util import get_np_array, to_up_vec3
from molib.core.constants import MoLibConstant
from molib.entities.ribbon.build_context import RibbonBuildContext
from molib.pdb.structure.ribbons.arrows.config import ArrowConfig
from molib.pdb.structure.ribbons.calc import use_ribbon_edges_to_determine_arrow_plane, \
    calculate_normals_along_binormal_and_direction, calculate_normals_along_direction, calculate_normals_along_binormal
from molib.pdb.structure.ribbons.ribbon_geometry import RibbonGeometryContext
from picogl.buffers.helper import as_meshdata
from picogl.renderer import MeshData


def generate_arrow_geometry_from_context(config, context: RibbonBuildContext, p1: Any, p2: Any, ribbon_geom: RibbonGeometryContext, arrow_config: ArrowConfig) -> MeshData:
    """generate arrow geometry"""
    color=tuple(context.colors[-1])
    ribbon_plane_normal=ribbon_geom.plane_normal
    ribbon_binormal=ribbon_geom.binormal
    ribbon_left_edge=ribbon_geom.left_edge
    ribbon_right_edge=ribbon_geom.right_edge
    p1 = get_np_array(p1)
    p2 = get_np_array(p2)

    # Direction vector (arrow direction)
    direction = p2 - p1
    length = np.linalg.norm(direction)
    if length < MoLibConstant.EPSILON:
        return as_meshdata(positions=np.zeros((0, 3)),
            colors=np.zeros((0, 3)),
            normals=np.zeros((0,), dtype=np.uint32),
            indices=np.zeros((0, 3)),
        )

    direction = direction / length

    # If ribbon edges are provided, use them (Ribbons approach)
    if ribbon_left_edge is not None and ribbon_right_edge is not None:
        base_width, binormal, head_width = use_ribbon_edges_to_determine_arrow_plane(arrow_config.base_width,
                                                                                     arrow_config.head_width, direction,
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

        base_width = arrow_config.base_width if arrow_config.base_width is not None else config.width_scale
        head_width = arrow_config.head_width if arrow_config.head_width is not None else 0.0

    # Generate arrow meshdata using Ribbons' approach
    # Create samples along arrow length, tapering from base_width to head_width
    vertices = []
    vertex_normals = []
    indices = []

    # Ribbons' ArrowLines approach: taper width along the arrow
    for i in range(arrow_config.num_samples + 1):
        t = i / arrow_config.num_samples  # Parameter from 0 (base) to 1 (tip)

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
        elif i == arrow_config.num_samples:
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
    last_base = arrow_config.num_samples * 2
    indices.extend([last_base, last_base + 1, tip_idx])

    indices = np.array(indices, dtype=np.uint32)

    # Color buffer
    colors = np.tile(color, (len(vertices), 1)).astype(np.float32)

    return as_meshdata(positions=vertices, normals=vertex_normals, indices=indices, colors=colors)

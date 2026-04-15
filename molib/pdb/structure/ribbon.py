"""
Ribbon Geometry

Example Usage:
==============
>>>ca_coords = np.data([res.ca_position for res in residues])  # shape (N, 3)
...mesh_data = generate_ribbon_geometry_per_chain(ca_coords)

This module supports two ribbon generation methods:
1. Catmull-Rom splines (original Elmo/Molscript approach) - fast but less accurate
2. B-splines (Ribbons approach) - more accurate, uses peptide plane meshdata

"""

from dataclasses import dataclass
from typing import Any, Optional, Callable

import numpy as np
from numpy import bool_, dtype, ndarray, floating, generic

from decologr import Decologr as log
from molib.core.constants import MoLibConstant
from molib.entities.ribbon.build_context import RibbonBuildContext
from molib.calc.geometry.ribbons_bspline import (
    generate_ribbon_geometry_ribbons_style, RibbonStyle,
)
from molib.calc.geometry.spline import catmull_rom_chain

from picogl.buffers.geometry import GeometryData
from picogl.buffers.vertex.data import VertexData
from picogl.buffers.vertex.meta_data import VertexMetadata
from picogl.renderer import MeshData

# B-spline ribbon effective half-width is 0.5 * get_width(ss) * width (guide-point factor).
# Legacy ribbons use constant half-width 0.5. To match, use width so 0.5*0.6*width ≈ 0.5 → width ≈ 1.67.
# RIBBON_WIDTH_SCALE =
RIBBON_WIDTH_SCALE = 2.7


@dataclass
class RibbonStyleConfig:
    """Ribbon cross-section style and B-spline width scale (context-based API)."""

    style: str
    width_scale: float = RIBBON_WIDTH_SCALE
    use_ribbons_style: bool = True
    #: Append a Ribbons-style arrowhead at the C-terminus when geometry exposes ribbon edges.
    has_arrow: bool = False


def generate_ribbon_geometry_per_chain_color_by_ca(
    all_ca_coords: np.ndarray,
    all_chain_ids: list,
    all_ca_colors: np.ndarray,
    use_ribbons_style: bool = True,
    all_o_coords: Optional[np.ndarray] = None,
    all_ss_types: Optional[np.ndarray] = None,
    style: str = RibbonStyle.SQUARE,
    ribbon_width_scale: float = RIBBON_WIDTH_SCALE,
    has_arrow: bool = False,
) -> dict[Any, MeshData]:
    """
    Generate ribbon meshdata for each chain separately, with per-CA colors.

    Uses Ribbons-style B-splines by default for better visual accuracy.

    Args:
        all_ca_coords: (N, 3) array of all CA coordinates
        all_chain_ids: list of chain IDs, length N
        all_ca_colors: (N, 3) array of RGB colors per CA
        use_ribbons_style: If True, use B-spline approach (default), else Catmull-Rom
        all_o_coords: Optional (N, 3) array of O coordinates for better accuracy
        all_ss_types: Optional (N,) array of secondary structure types
        style: Ribbon cross-section style ("flat", "circle", "square", "ellipse")
        ribbon_width_scale: B-spline ribbon width factor (see RIBBON_WIDTH_SCALE)
        has_arrow: If True, append a C-terminal arrow when B-spline geometry exposes edges.
    """
    from collections import defaultdict

    # Group CA coordinates, colors, O coords, and SS types by chain
    coords_by_chain = defaultdict(list)
    colors_by_chain = defaultdict(list)
    o_coords_by_chain = defaultdict(list)
    ss_types_by_chain = defaultdict(list)

    for i, (coord, color, chain_id) in enumerate(
        zip(all_ca_coords, all_ca_colors, all_chain_ids)
    ):
        coords_by_chain[chain_id].append(coord)
        colors_by_chain[chain_id].append(color)
        if all_o_coords is not None:
            o_coords_by_chain[chain_id].append(all_o_coords[i])
        if all_ss_types is not None:
            ss_types_by_chain[chain_id].append(all_ss_types[i])

    ribbon_data = {}

    for chain_id, coords in coords_by_chain.items():
        ca_array = np.array(coords, dtype=np.float32)
        color_array = np.array(colors_by_chain[chain_id], dtype=np.float32)
        chain_id_list = [chain_id] * len(ca_array)

        o_array = None
        if all_o_coords is not None and chain_id in o_coords_by_chain:
            o_array = np.array(o_coords_by_chain[chain_id], dtype=np.float32)

        ss_array = None
        if all_ss_types is not None and chain_id in ss_types_by_chain:
            ss_array = np.array(ss_types_by_chain[chain_id])

        vertex_data = generate_ribbon_geometry_with_colors(
            ca_array,
            color_array,
            chain_id_list,
            use_ribbons_style=use_ribbons_style,
            o_coords=o_array,
            ss_types=ss_array,
            style=style,
            ribbon_width_scale=ribbon_width_scale,
            has_arrow=has_arrow,
        )

        ribbon_data[chain_id] = MeshData(
            vertices=vertex_data.geom_data.vertices,
            normals=vertex_data.geom_data.normals,
            indices=vertex_data.geom_data.indices,
            colors=vertex_data.geom_data.colors,
        )

    return ribbon_data


def generate_ribbon_geometry_per_chain_color_by_ca_from_context(
    context: RibbonBuildContext,
    config: RibbonStyleConfig) -> dict[Any, MeshData]:
    """generate ribbon geometry per chain by ca"""
    return generate_ribbon_geometry_per_chain_color_by_ca(
        context.coords,
        context.chain_ids,
        context.colors,
        style=config.style,
        ribbon_width_scale=config.width_scale,
        has_arrow=config.has_arrow,
    )


def normalize(v) -> float:
    """normalize"""
    # Helper function to normalize vectors
    norm = np.linalg.norm(v)
    if norm < MoLibConstant.EPSILON:
        return v
    return v / norm


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
        base_width, binormal, head_width = _use_ribbon_edges_to_determine_arrow_plane(arrow_base_width,
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
            _calculate_normals_along_binormal(binormal, vertex_normals)
        elif i == num_samples:
            _calculate_normals_along_direction(left, normalize, pos, right, vertex_normals)
        else:
            _calculate_normals_along_binormal_and_direction(left, normalize, pos, right, vertex_normals)

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


def to_up_vec3() -> ndarray[Any, dtype[Any]] | ndarray[Any, dtype[generic]]:
    return np.array([0.0, 0.0, 1.0], dtype=np.float32)


def cross_normalize(binormal: float, direction: ndarray[Any, dtype[floating[Any]]]) -> np.ndarray:
    """Calculate normal from cross product"""
    normal = cross(binormal, direction)
    normal = normalize(normal)
    return normal


def cross(binormal: float, direction: ndarray[Any, dtype[floating[Any]]]) -> ndarray[Any, dtype[floating[Any]]]:
    """cross helper"""
    return np.cross(binormal, direction)


def get_np_array(p1: ndarray) -> ndarray[Any, dtype[Any]] | ndarray[Any, dtype[generic]]:
    """get as numpy array"""
    return np.asarray(p1, dtype=np.float32)


def _use_ribbon_edges_to_determine_arrow_plane(arrow_base_width: float | None, arrow_head_width: float | None,
                                               direction: ndarray[Any, dtype[floating[Any]]],
                                               ribbon_binormal: ndarray | None, ribbon_left_edge: ndarray,
                                               ribbon_plane_normal: ndarray | None, ribbon_right_edge: ndarray) -> \
tuple[float | ndarray[Any, dtype[floating[Any]]], float, Any]:
    """Use actual ribbon edges to determine arrow plane"""
    left_edge = np.asarray(ribbon_left_edge, dtype=np.float32)
    right_edge = np.asarray(ribbon_right_edge, dtype=np.float32)

    ribbon_width, ribbon_width_vec = _calculate_ribbon_width_attrs(left_edge, right_edge)

    binormal = _calculate_binormal_from_ribbon_edges(normalize, ribbon_binormal, ribbon_width_vec)

    _calculate_normal_perpendicular_to_ribbon_plane(binormal, direction, ribbon_plane_normal)

    # Arrow base width from ribbon width
    base_width = (
        arrow_base_width if arrow_base_width is not None else ribbon_width * 0.5
    )
    head_width = arrow_head_width if arrow_head_width is not None else 0.0
    return base_width, binormal, head_width


def _calculate_ribbon_width_attrs(left_edge: ndarray[Any, dtype[Any]], right_edge: ndarray[Any, dtype[Any]]) -> tuple[
    floating[Any], ndarray[Any, dtype[Any]]]:
    """Calculate ribbon center and width from edges"""
    ribbon_center = 0.5 * (left_edge + right_edge)
    ribbon_width_vec = right_edge - left_edge
    ribbon_width = np.linalg.norm(ribbon_width_vec)
    return ribbon_width, ribbon_width_vec


def _calculate_binormal_from_ribbon_edges(normalize: Callable[..., Any],
                                          ribbon_binormal: ndarray | None,
                                          ribbon_width_vec: ndarray[Any, dtype[Any]]) -> Any:
    """Calculate binormal (width direction) from ribbon edges"""
    if ribbon_binormal is not None:
        binormal = normalize(np.asarray(ribbon_binormal, dtype=np.float32))
    else:
        binormal = normalize(ribbon_width_vec)
    return binormal


def _calculate_normal_perpendicular_to_ribbon_plane(binormal: Any,
                                                    direction: ndarray[Any, dtype[floating[Any]]],
                                                    ribbon_plane_normal: ndarray | None):
    """Calculate normal (perpendicular to ribbon plane)"""
    if ribbon_plane_normal is not None:
        normal = normalize(np.asarray(ribbon_plane_normal, dtype=np.float32))
    else:
        # Calculate normal from cross product of direction and binormal
        normal = np.cross(direction, binormal)
        if np.linalg.norm(normal) < MoLibConstant.EPSILON:
            # Fallback: use cross product of binormal and direction
            normal = np.cross(binormal, direction)
        normal = normalize(normal)


def _calculate_normals_along_binormal_and_direction(left: float | Any, normalize: Callable[..., Any],
                                                    pos: ndarray[Any, dtype[bool_]] | Any,
                                                    right: ndarray[Any, dtype[bool_]] | float | Any,
                                                    vertex_normals: list[Any]):
    """Middle: average of binormal and direction"""
    left_normal = normalize(left - pos)
    right_normal = normalize(right - pos)
    vertex_normals.append(left_normal)
    vertex_normals.append(right_normal)


def _calculate_normals_along_direction(left: float | Any, normalize: Callable[..., Any],
                                       pos: ndarray[Any, dtype[bool_]] | Any,
                                       right: ndarray[Any, dtype[bool_]] | float | Any, vertex_normals: list[Any]):
    """Tip: normals point along direction"""
    vertex_normals.append(normalize(left - pos))
    vertex_normals.append(normalize(right - pos))


def _calculate_normals_along_binormal(binormal: Any, vertex_normals: list[Any]):
    """Base: normals point along binormal"""
    vertex_normals.append(-binormal)
    vertex_normals.append(binormal)


def generate_ribbon_geometry_per_chain(
    all_ca_coords: np.ndarray,
    all_chain_ids: list,
    chain_colors: dict[str, tuple],
    style: str = RibbonStyle.SQUARE,
    ribbon_width_scale: float = RIBBON_WIDTH_SCALE,
) -> dict[str, MeshData]:
    """
    Generate ribbon meshdata for each chain separately.
    """
    from collections import defaultdict

    # Group CA coordinates by chain
    coords_by_chain = defaultdict(list)
    for coord, chain_id in zip(all_ca_coords, all_chain_ids):
        coords_by_chain[chain_id].append(coord)

    ribbon_mesh_by_chain: dict[str, MeshData] = {}

    for chain_id, coords in coords_by_chain.items():
        # Skip chains with insufficient C-alpha atoms for spline interpolation
        if len(coords) < 4:
            print(
                f"⚠️ Skipping chain {chain_id}: insufficient C-alpha atoms ({len(coords)} < 4 required for spline)"
            )
            continue

        ca_array = np.array(coords, dtype=np.float32)
        chain_id_list = [chain_id] * len(ca_array)

        try:
            # Use generate_ribbon_geometry_with_colors for B-spline/Ribbons-style meshdata
            # with per-chain flat color (tile chain color for each CA)
            color = chain_colors.get(chain_id, (1.0, 1.0, 1.0))
            ca_colors = np.tile(color, (len(ca_array), 1)).astype(np.float32)

            context = RibbonBuildContext(coords=ca_array, colors=ca_colors, chain_ids=chain_id_list)
            config = RibbonStyleConfig(style=style,width_scale=ribbon_width_scale, use_ribbons_style=True)

            vertex_data = generate_ribbon_geometry_with_colors_from_context(
                context,
                config
            )
            verts = vertex_data.geom_data.vertices
            norms = vertex_data.geom_data.normals
            inds = vertex_data.geom_data.indices
            colors = vertex_data.geom_data.colors

            ribbon_mesh_by_chain[chain_id] = MeshData(
                vertices=verts,
                normals=norms,
                indices=inds,
                colors=colors,
            )
        except Exception as ex:
            log.message(f"⚠️ Error generating ribbon for chain {chain_id}: {ex}",
                        scope="generate_ribbon_geometry_per_chain")
            continue

    return ribbon_mesh_by_chain

def generate_ribbon_geometry_with_colors_from_context(
    context: RibbonBuildContext,
    config: RibbonStyleConfig) -> VertexData:
    """generate ribbon geometry per chain by ca"""
    return generate_ribbon_geometry_with_colors(
        ca_coords=context.coords,
        ca_colors=context.colors,
        chain_ids=context.chain_ids,
        o_coords=context.o_coords,
        ss_types=context.ss_types,
        style=config.style,
        ribbon_width_scale=config.width_scale,
        has_arrow=config.has_arrow,
    )


def generate_ribbon_geometry_with_colors(
    ca_coords: np.ndarray,
    ca_colors: np.ndarray,
    chain_ids: list[str],
    width: float = 0.5,
    use_ribbons_style: bool = True,
    o_coords: Optional[np.ndarray] = None,
    ss_types: Optional[np.ndarray] = None,
    style: str = RibbonStyle.SQUARE,
    ribbon_width_scale: float = RIBBON_WIDTH_SCALE,
    has_arrow: bool = False,
) -> VertexData:
    """
    Generate ribbon meshdata with per-CA colors. @@@@@ RIBBON_PATH

    Uses Ribbons-style B-splines by default for better accuracy, or falls back
    to Catmull-Rom for compatibility.

    :param ca_coords: np.ndarray of shape (N, 3)
    :param ca_colors: np.ndarray of shape (N, 3) - RGB colors per CA
    :param chain_ids: list[str], len N
    :param width: float, ribbon half-width
    :param style: RibbonStyle
    :param use_ribbons_style: bool, if True use B-spline approach (default), else Catmull-Rom
    :param o_coords: Optional (N, 3) array of O (oxygen) coordinates for better accuracy
    :param ss_types: Optional (N,) array of secondary structure types ('H', 'S', 'T', etc.)
    :param ribbon_width_scale: B-spline width factor passed to Ribbons-style generator
    :param has_arrow: If True, append a C-terminal arrow (B-spline path only, when edges exist).
    """
    if use_ribbons_style:
        # Use Ribbons-style B-spline approach for better accuracy.
        # B-spline uses get_width(ss)*width and a 0.5 factor in guide points, so effective
        # half-width is smaller than legacy (constant 0.5). Use ribbon_width_scale
        # so modern ribbons match legacy visibility (helix half-width ~0.5).
        try:

            geo_data, ribbon_edges, ribbon_frenet = (
                generate_ribbon_geometry_ribbons_style(
                    ca_coords,
                    o_coords=o_coords,
                    ss_types=ss_types,
                    width=ribbon_width_scale,
                    samples_per_segment=4,  # Fewer samples = less smoothing (helix stays tighter)
                    style=style,
                    num_threads=8,
                )
            )
            vertices = geo_data.vertices
            normals = geo_data.normals
            indices = geo_data.indices

            # Map colors to vertices (interpolate along the ribbon)
            # For now, use the color of the nearest CA atom
            n_vertices = len(vertices)
            colors = np.zeros((n_vertices, 3), dtype=np.float32)
            vertex_chain_ids = []

            # Calculate which CA each vertex is closest to
            for i, vertex in enumerate(vertices):
                # Find nearest CA
                distances = np.linalg.norm(ca_coords - vertex, axis=1)
                nearest_ca_idx = np.argmin(distances)
                colors[i] = ca_colors[nearest_ca_idx]
                vertex_chain_ids.append(chain_ids[nearest_ca_idx])

            if has_arrow and ribbon_edges is not None and ribbon_frenet is not None:
                try:
                    left_edge, right_edge = ribbon_edges
                    tangent, plane_normal, ribbon_binormal = ribbon_frenet
                    p1 = 0.5 * (
                        np.asarray(left_edge, dtype=np.float32)
                        + np.asarray(right_edge, dtype=np.float32)
                    )
                    t = np.asarray(tangent, dtype=np.float32)
                    t_len = float(np.linalg.norm(t))
                    if t_len > 1e-8:
                        t = t / t_len
                    if len(ca_coords) >= 2:
                        step = float(
                            np.linalg.norm(ca_coords[-1] - ca_coords[-2])
                        )
                    else:
                        step = float(ribbon_width_scale) * 0.25
                    arrow_len = max(step * 0.75, float(ribbon_width_scale) * 0.15)
                    p2 = p1 + t * arrow_len
                    last_color = (
                        float(ca_colors[-1, 0]),
                        float(ca_colors[-1, 1]),
                        float(ca_colors[-1, 2]),
                    )
                    av, an, ai, ac = generate_arrow_geometry(
                        p1,
                        p2,
                        width=float(ribbon_width_scale) * 0.35,
                        color=last_color,
                        ribbon_plane_normal=np.asarray(
                            plane_normal, dtype=np.float32
                        ),
                        ribbon_binormal=np.asarray(
                            ribbon_binormal, dtype=np.float32
                        ),
                        ribbon_left_edge=np.asarray(left_edge, dtype=np.float32),
                        ribbon_right_edge=np.asarray(right_edge, dtype=np.float32),
                    )
                    if len(av) > 0:
                        offset = np.uint32(len(vertices))
                        vertices = np.vstack([vertices, av])
                        normals = np.vstack([normals, an])
                        colors = np.vstack([colors, ac])
                        chain_tail = chain_ids[-1] if chain_ids else ""
                        vertex_chain_ids.extend(
                            [chain_tail] * len(av)
                        )
                        indices = np.concatenate(
                            [indices, ai.astype(np.uint32) + offset]
                        )
                except Exception as arrow_ex:
                    log.message(
                        f"Ribbon end arrow: skipped ({arrow_ex})",
                        scope="generate_ribbon_geometry_with_colors",
                    )

            return VertexData(
                geom_data=GeometryData(vertices=vertices, normals=normals, indices=indices, colors=colors),
                meta_data=VertexMetadata(chain_ids=vertex_chain_ids))

        except Exception as e:
            log.error(f"Warning: Ribbons-style ribbon generation failed: {e}")
            log.error("Falling back to Catmull-Rom approach")
            use_ribbons_style = False

    # Fallback to original Catmull-Rom approach
    from numpy import cross, gradient, linalg

    # Check if we have enough points for Catmull-Rom spline
    if len(ca_coords) < 4:
        raise ValueError(
            f"At least 4 C-alpha atoms required for ribbon generation, got {len(ca_coords)}"
        )

    spline = catmull_rom_chain(ca_coords)  # (M, 3)
    spline_colors = catmull_rom_chain(ca_colors)  # interpolate colors (M, 3)
    spline_chain_ids = [
        chain_ids[min(i, len(chain_ids) - 1)] for i in range(len(spline))
    ]

    tangents = gradient(spline, axis=0)
    tangents = tangents / linalg.norm(tangents, axis=1, keepdims=True)

    up_hint = np.array([0, 0, 1], dtype=np.float32)
    vertices, normals, indices, colors, vertex_chain_ids = [], [], [], [], []

    for i, (p, t, col, chain_id) in enumerate(
        zip(spline, tangents, spline_colors, spline_chain_ids)
    ):
        n = cross(t, up_hint)
        if linalg.norm(n) < 1e-3:
            up_hint = np.array([1, 0, 0], dtype=np.float32)
            n = cross(t, up_hint)
        n = n / linalg.norm(n)

        offset = n * width
        left = p - offset
        right = p + offset

        # Add vertices
        vertices.extend([left, right])
        normals.extend([n, n])

        # Add colors for both sides of the ribbon at this spline point
        colors.extend([col, col])
        vertex_chain_ids.extend([chain_id, chain_id])

    for i in range(len(spline) - 1):
        base = i * 2
        indices.extend([base, base + 1, base + 2, base + 1, base + 3, base + 2])

    return VertexData(geom_data=GeometryData(vertices=vertices,
                                             normals=normals,
                                             indices=indices,
                                             colors=colors),
                      meta_data=VertexMetadata(chain_ids=vertex_chain_ids))


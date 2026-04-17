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

from typing import Any, Optional

import numpy as np

from decologr import Decologr as log
from molib.entities.ribbon.build_context import RibbonBuildContext
from molib.calc.geometry.ribbons_bspline import (
    generate_ribbon_geometry_ribbons_style, RibbonStyle,
)
from molib.calc.geometry.spline import catmull_rom_chain
from molib.pdb.structure.ribbons.arrow import generate_arrow_geometry
from molib.pdb.structure.ribbons.style import RIBBON_WIDTH_SCALE, RibbonStyleConfig

from picogl.buffers.geometry import GeometryData
from picogl.buffers.vertex.data import VertexData
from picogl.buffers.vertex.meta_data import VertexMetadata
from picogl.renderer import MeshData


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
            config = RibbonStyleConfig(style=style, width_scale=ribbon_width_scale, use_ribbons_style=True)

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
    """generate ribbon geometry from context"""
    if config.use_ribbons_style:
        # Use Ribbons-style B-spline approach for better accuracy.
        # B-spline uses get_width(ss)*width and a 0.5 factor in guide points, so effective
        # half-width is smaller than legacy (constant 0.5). Use ribbon_width_scale
        # so modern ribbons match legacy visibility (helix half-width ~0.5).
        try:
            return generate_ribbon_ribbons_style(config, context)

        except Exception as e:
            log.error(f"Warning: Ribbons-style ribbon generation failed: {e}")
            log.error("Falling back to Catmull-Rom approach")
            return generate_ribbon_catmull_rom(context)

    return generate_ribbon_catmull_rom(context)


def generate_ribbon_ribbons_style(config: RibbonStyleConfig, context: RibbonBuildContext) -> VertexData:
    """generate ribbon ribbons style"""
    geo_data, ribbon_edges, ribbon_frenet = (
        generate_ribbon_geometry_ribbons_style(
            context.coords,
            o_coords=context.o_coords,
            ss_types=context.ss_types,
            width=config.width_scale,
            samples_per_segment=4,  # Fewer samples = less smoothing (helix stays tighter)
            style=config.style,
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
        distances = np.linalg.norm(context.coords - vertex, axis=1)
        nearest_ca_idx = np.argmin(distances)
        colors[i] = context.colors[nearest_ca_idx]
        vertex_chain_ids.append(context.chain_ids[nearest_ca_idx])

    if config.has_arrow and ribbon_edges is not None and ribbon_frenet is not None:
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
            if len(context.coords) >= 2:
                step = float(
                    np.linalg.norm(context.coords[-1] - context.coords[-2])
                )
            else:
                step = float(config.width_scale) * 0.25
            arrow_len = max(step * 0.75, float(config.width_scale) * 0.15)
            p2 = p1 + t * arrow_len
            last_color = (
                float(context.colors[-1, 0]),
                float(context.colors[-1, 1]),
                float(context.colors[-1, 2]),
            )
            av, an, ai, ac = generate_arrow_geometry(
                p1,
                p2,
                width=float(config.width_scale) * 0.35,
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
                chain_tail = context.chain_ids[-1] if context.chain_ids else ""
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


def generate_ribbon_catmull_rom(context):
    """Fallback to original Catmull-Rom approach"""
    from numpy import cross, gradient, linalg

    # Check if we have enough points for Catmull-Rom spline
    if len(context.coords) < 4:
        raise ValueError(
            f"At least 4 C-alpha atoms required for ribbon generation, got {len(context.coords)}"
        )

    spline = catmull_rom_chain(context.coords)  # (M, 3)
    spline_colors = catmull_rom_chain(context.colors)  # interpolate colors (M, 3)
    spline_chain_ids = [
        context.chain_ids[min(i, len(context.chain_ids) - 1)] for i in range(len(spline))
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


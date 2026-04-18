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
from typing import Any, Tuple
import numpy as np

import numpy as np
from scipy.spatial import cKDTree

from collections import defaultdict

from decologr import Decologr as log
from molib.calc.math.numpy_util import generate_colors_from_positions
from molib.core.constants import MoLibConstant
from molib.entities.ribbon.build_context import RibbonBuildContext
from molib.calc.geometry.ribbons_bspline import (
    generate_ribbon_geometry_ribbons_style_from_context)
from molib.calc.geometry.spline import catmull_rom_chain
from molib.pdb.structure.ribbons.arrows.arrow import  generate_arrow_geometry_from_context
from molib.pdb.structure.ribbons.arrows.config import ArrowConfig
from molib.pdb.structure.ribbons.ribbon_geometry import RibbonGeometryContext
from molib.pdb.structure.ribbons.style import RibbonStyleConfig

from picogl.buffers.helper import as_meshdata
from picogl.renderer import MeshData


@dataclass
class MeshLayout:
    """Mesh Layout"""
    vertices: np.ndarray
    normals: np.ndarray
    colors: np.ndarray
    indices: Optional[np.ndarray]
    
    
def empty_vertex(n_points: int, components: int) -> np.ndarray:
    rows, cols = _buffer_shape(n_points, components)
    return np.empty((rows, cols), dtype=np.float32)

def empty_ribbon_buffers(n_points: int, with_indices: bool = False) -> MeshLayout:
    if n_points < 1:
        raise ValueError("n_points must be at least 1")
    components = 3
    vertices = empty_vertex(n_points, components)
    normals  = empty_vertex(n_points, components)
    colors   = empty_vertex(n_points, components)
    indices  = (np.empty(((n_points - 1) * 6,), dtype=np.uint32)
                if with_indices and n_points > 1 else None)

    return MeshLayout(
        vertices=vertices,
        normals=normals,
        colors=colors,
        indices=indices,
    )


def generate_ribbon_geometry_per_chain_color_by_ca_from_context(
    context: RibbonBuildContext,
    config: RibbonStyleConfig) -> dict[Any, MeshData]:
    """generate ribbon geometry per chain by ca"""

    # Group CA coordinates, colors, O coords, and SS types by chain
    coords_by_chain = defaultdict(list)
    colors_by_chain = defaultdict(list)
    o_coords_by_chain = defaultdict(list)
    ss_types_by_chain = defaultdict(list)

    for i, (coord, color, chain_id) in enumerate(
        zip(context.coords, context.colors, context.chain_ids)
    ):
        coords_by_chain[chain_id].append(coord)
        colors_by_chain[chain_id].append(color)
        if context.o_coords is not None:
            o_coords_by_chain[chain_id].append(context.o_coords[i])
        if context.ss_types is not None:
            ss_types_by_chain[chain_id].append(context.ss_types[i])

    ribbon_mesh_by_chain = {}

    for chain_id, coords in coords_by_chain.items():
        ca_array = np.array(coords, dtype=np.float32)
        color_array = np.array(colors_by_chain[chain_id], dtype=np.float32)
        chain_id_list = [chain_id] * len(ca_array)

        o_array = None
        if context.o_coords is not None and chain_id in o_coords_by_chain:
            o_array = np.array(o_coords_by_chain[chain_id], dtype=np.float32)

        ss_array = None
        if context.ss_types is not None and chain_id in ss_types_by_chain:
            ss_array = np.array(ss_types_by_chain[chain_id])

        colored_context = RibbonBuildContext(coords=ca_array,
                                             o_coords=o_array,
                                             ss_types=ss_array,
                                             colors=color_array,
                                             chain_ids=chain_id_list)

        ribbon_mesh_by_chain[chain_id] = generate_ribbon_geometry_with_colors_from_context(context=colored_context, config=config)

    return ribbon_mesh_by_chain


def generate_ribbon_geometry_per_chain_from_context(config: RibbonStyleConfig, context: RibbonBuildContext) -> dict:
    """generate ribbon geometry per chain from context"""


    # Group CA coordinates by chain
    coords_by_chain = defaultdict(list)
    for coord, chain_id in zip(context.coords, context.chain_ids):
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
            color = context.colors.get(chain_id, (1.0, 1.0, 1.0))
            ca_colors = generate_colors_from_positions(positions=ca_array, r=color[0], g=color[1], b=color[2])

            context = RibbonBuildContext(coords=ca_array, colors=ca_colors, chain_ids=chain_id_list)

            ribbon_mesh_by_chain[chain_id] = generate_ribbon_geometry_with_colors_from_context(
                context,
                config
            )
        except Exception as ex:
            log.message(f"⚠️ Error generating ribbon for chain {chain_id}: {ex}",
                        scope="generate_ribbon_geometry_per_chain")
            continue

    return ribbon_mesh_by_chain


def generate_ribbon_geometry_with_colors_from_context(
    context: RibbonBuildContext,
    config: RibbonStyleConfig) -> MeshData:
    """generate ribbon geometry from context"""
    if config.use_ribbons_style:
        try:
            return generate_ribbon_ribbons_style(config, context)

        except Exception as e:
            log.error(f"Warning: Ribbons-style ribbon generation failed: {e}")
            log.error("Falling back to Catmull-Rom approach")
            return generate_ribbon_catmull_rom(context)

    return generate_ribbon_catmull_rom(context)

def _append_arrow(
    mesh_data: MeshData,
    vertex_chain_ids,
    context: RibbonBuildContext,
    config: ArrowConfig,
    ribbon_edges,
    ribbon_frenet,
) -> tuple[MeshData, list[str]]:
    if not (config.has_arrow and ribbon_edges and ribbon_frenet):
        return mesh_data, vertex_chain_ids

    try:
        left_edge, right_edge = ribbon_edges
        tangent, plane_normal, ribbon_binormal = ribbon_frenet

        p1 = 0.5 * (left_edge + right_edge)

        t = tangent / (np.linalg.norm(tangent) +  MoLibConstant.EPSILON_SMALL)

        step = (
            np.linalg.norm(context.coords[-1] - context.coords[-2])
            if len(context.coords) >= 2
            else config.width_scale * 0.25
        )

        arrow_len = max(step * 0.75, config.width_scale * 0.15)
        p2 = p1 + t * arrow_len

        ribbon_geom = RibbonGeometryContext(plane_normal=plane_normal,
            binormal=ribbon_binormal,
            left_edge=left_edge,
            right_edge=right_edge,
        )
        arrow_config = ArrowConfig(base_width=config.width_scale * 0.5)
        arrow_mesh = generate_arrow_geometry_from_context(config, context, p1=p1, p2=p2, ribbon_geom=ribbon_geom, arrow_config=arrow_config)
        if arrow_mesh.vertices is not None:
            if len(arrow_mesh.vertices) == 0:
                return arrow_mesh, vertex_chain_ids

        if arrow_mesh.vertices is None or len(arrow_mesh.vertices) == 0:
            return mesh_data, vertex_chain_ids

        mesh_data.append_mesh(arrow_mesh)

        chain_tail = context.chain_ids[-1] if context.chain_ids else ""
        vertex_chain_ids.extend([chain_tail] * len(arrow_mesh.vertices))

    except Exception as e:
        log.message(f"Ribbon end arrow skipped: {e}", scope="ribbon")

    return mesh_data, vertex_chain_ids


def generate_ribbon_ribbons_style(config: RibbonStyleConfig, context: RibbonBuildContext) -> MeshData:
    """generate ribbon ribbons style"""
    # Use Ribbons-style B-spline approach for better accuracy.
    # B-spline uses get_width(ss)*width and a 0.5 factor in guide points, so effective
    # half-width is smaller than legacy (constant 0.5). Use ribbon_width_scale
    # so modern ribbons match legacy visibility (helix half-width ~0.5).
    mesh_data, ribbon_edges, ribbon_frenet = generate_ribbon_geometry_ribbons_style_from_context(config, context)

    # Efficient color mapping
    tree = cKDTree(context.coords)
    # _, nearest = tree.query(vertices)
    nearest = tree.query(mesh_data.vertices, workers=-1)[1]
    colors = context.colors[nearest]
    color_updated_mesh = as_meshdata(positions=mesh_data.vertices, normals=mesh_data.normals, colors=colors, indices=mesh_data.indices)
    vertex_chain_ids = [context.chain_ids[i] for i in nearest]

    # Arrow
    mesh_data, _ = _append_arrow(
        color_updated_mesh,
        vertex_chain_ids,
        context,
        config,
        ribbon_edges,
        ribbon_frenet,
    )
    return mesh_data


def generate_ribbon_catmull_rom(context: RibbonBuildContext, width: float = 0.5) -> MeshData:
    if len(context.coords) < 4:
        raise ValueError(f"Need ≥4 CA atoms, got {len(context.coords)}")

    spline = catmull_rom_chain(context.coords)
    spline_colors = catmull_rom_chain(context.colors)

    tangents = np.gradient(spline, axis=0)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / (norms + 1e-8)

    up_hint = np.array([0, 0, 1], dtype=np.float32)

    n_points = len(spline)

    vertices = _vec3_points(n_points)
    normals = _vec3_points(n_points)
    colors = _vec3_points(n_points)
    indices = _vec3_empty_indices(n_points)

    vertex_chain_ids = []

    for i in range(n_points):
        p = spline[i]
        t = tangents[i]

        n = np.cross(t, up_hint)
        if np.linalg.norm(n) < 1e-3:
            up_hint = np.array([1, 0, 0], dtype=np.float32)
            n = np.cross(t, up_hint)

        n = n / (np.linalg.norm(n) + 1e-8)

        offset = n * width

        vertices[2*i]     = p - offset
        vertices[2*i + 1] = p + offset

        normals[2*i:2*i+2] = n
        colors[2*i:2*i+2] = spline_colors[i]

        chain_id = context.chain_ids[min(i, len(context.chain_ids) - 1)]
        vertex_chain_ids.extend([chain_id, chain_id])

    # Indices (vectorizable but keeping readable)
    idx = 0
    for i in range(n_points - 1):
        base = 2 * i
        indices[idx:idx+6] = [
            base, base+1, base+2,
            base+1, base+3, base+2
        ]
        idx += 6
    return as_meshdata(positions=vertices, normals=normals, colors=colors, indices=indices)


# def _vec3_empty_indices(n_points: int) -> np.ndarray:
#.   return np.empty(((n_points - 1) * 6,), dtype=np.uint32)


# def _vec3_points(n_points: int) -> np.ndarray:
#.   return np.empty((n_points * 2, 3), dtype=np.float32)
    

def _buffer_shape(n_points: int, components: int = 3) -> Tuple[int, int]:
    """Generic shape for a 3-component (or n-component) per-point buffer.
       For ribbons: 2 points per control point -> rows = n_points * 2, cols = components."""
    return (n_points * 2, components)

def _vec3_points(n_points: int) -> np.ndarray:
    """Allocate a 3-component per-vertex buffer for ribbon points."""
    rows, cols = _buffer_shape(n_points, components=3)
    return np.empty((rows, cols), dtype=np.float32)

def _vec3_empty_indices(n_points: int) -> np.ndarray:
    """Allocate indices buffer for a strip between consecutive points (3 components implied)."""
    # If your indexing uses 6 indices per segment for a double-sided quad strip:
    return np.empty(((n_points - 1) * 6,), dtype=np.uint32)


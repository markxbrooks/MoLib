from typing import Any, Callable

import numpy as np
from numpy import ndarray, dtype, floating, bool_

from molib.calc.math.normal import normalize
from molib.core.constants import MoLibConstant


def use_ribbon_edges_to_determine_arrow_plane(arrow_base_width: float | None, arrow_head_width: float | None,
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


def calculate_normals_along_binormal_and_direction(left: float | Any, normalize: Callable[..., Any],
                                                   pos: ndarray[Any, dtype[bool_]] | Any,
                                                   right: ndarray[Any, dtype[bool_]] | float | Any,
                                                   vertex_normals: list[Any]):
    """Middle: average of binormal and direction"""
    left_normal = normalize(left - pos)
    right_normal = normalize(right - pos)
    vertex_normals.append(left_normal)
    vertex_normals.append(right_normal)


def calculate_normals_along_direction(left: float | Any, normalize: Callable[..., Any],
                                      pos: ndarray[Any, dtype[bool_]] | Any,
                                      right: ndarray[Any, dtype[bool_]] | float | Any, vertex_normals: list[Any]):
    """Tip: normals point along direction"""
    vertex_normals.append(normalize(left - pos))
    vertex_normals.append(normalize(right - pos))


def calculate_normals_along_binormal(binormal: Any, vertex_normals: list[Any]):
    """Base: normals point along binormal"""
    vertex_normals.append(-binormal)
    vertex_normals.append(binormal)

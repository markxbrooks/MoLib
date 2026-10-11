"""Crystallographic symmetry axes for reciprocal-space zone plots."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from molib.xtal.map.unit_cell import UnitCell
from molib.xtal.reflections.zone import ZoneSpec, project_to_zone_plane

_SUBSCRIPT = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


@dataclass(frozen=True, slots=True)
class SymmetryAxis:
    """One crystallographic rotation / screw axis projected into a zone plot.

    :param label: Hermann–Mauguin style symbol (``2``, ``2₁``, ``4₃``, …)
    :param order: Rotational order (2, 3, 4, or 6)
    :param direction: Approximate integer axis direction in crystal fractions
    :param orientation: ``in_plane`` draws a line through the origin along
        ``(u, v)``; ``normal`` draws a point symbol at the origin (axis along
        the zone normal)
    :param u: Plot-plane u component of the axis direction (0 for normal)
    :param v: Plot-plane v component of the axis direction (0 for normal)
    :param screw_fraction: Translation component along the axis in [0, 1)
    """

    label: str
    order: int
    direction: tuple[int, int, int]
    orientation: Literal["in_plane", "normal"]
    u: float
    v: float
    screw_fraction: float = 0.0


def resolve_space_group(space_group: Any) -> Any | None:
    """Return a ``gemmi.SpaceGroup`` from a name, object, or ``str(sg)`` repr."""
    if space_group is None:
        return None
    import gemmi

    if isinstance(space_group, gemmi.SpaceGroup):
        return space_group
    text = str(space_group).strip()
    if not text:
        return None
    match = re.search(r'SpaceGroup\("([^"]+)"\)', text)
    if match:
        text = match.group(1)
    else:
        match = re.search(r"SpaceGroup\('([^']+)'\)", text)
        if match:
            text = match.group(1)
    try:
        return gemmi.find_spacegroup_by_name(text)
    except RuntimeError:
        try:
            return gemmi.SpaceGroup(text)
        except (RuntimeError, ValueError, TypeError):
            return None


def _canonicalize_direction(axis: np.ndarray) -> tuple[int, int, int]:
    """Map a float axis direction to a small integer triple."""
    a = np.asarray(axis, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(a))
    if n < 1e-12:
        return (0, 0, 1)
    a = a / n
    # Scale so the largest component is near an integer ≤ 3.
    best: tuple[int, int, int] | None = None
    best_err = 1e9
    for scale in (1.0, 2.0, 3.0, np.sqrt(2.0), np.sqrt(3.0)):
        cand = np.rint(a * scale).astype(int)
        if np.all(cand == 0):
            continue
        # gcd reduce
        g = int(np.gcd.reduce(np.abs(cand)))
        if g > 1:
            cand = cand // g
        err = float(np.linalg.norm(a * scale - cand))
        if err < best_err:
            best_err = err
            best = (int(cand[0]), int(cand[1]), int(cand[2]))
    if best is None:
        return (0, 0, 1)
    # Flip so the first non-zero component is positive.
    for c in best:
        if c != 0:
            if c < 0:
                best = (-best[0], -best[1], -best[2])
            break
    return best


def _screw_label(order: int, screw_fraction: float) -> str:
    """Build a rotation / screw symbol such as ``2₁`` or ``4``."""
    s = float(screw_fraction) % 1.0
    if s > 0.95:
        s = 0.0
    if s < 0.05:
        return str(order)
    k = int(round(s * order))
    if k <= 0 or k >= order:
        return str(order)
    return f"{order}{str(k).translate(_SUBSCRIPT)}"


def _axis_direction_from_rotation(R: np.ndarray) -> np.ndarray:
    """Return the rotation axis (eigenvalue +1) for a proper rotation matrix."""
    m = R - np.eye(3)
    _, _, vh = np.linalg.svd(m)
    return vh[-1].astype(np.float64)


def zone_symmetry_axes(
    space_group: Any,
    zone: ZoneSpec,
    unit_cell: UnitCell | None = None,
) -> list[SymmetryAxis]:
    """Return symmetry axes that are visible in *zone*.

    Proper rotations / screws whose axis is parallel to the zone normal become
    origin symbols; axes lying in the zone plane become directed lines through
    the origin in plot coordinates. Oblique axes are omitted.

    :param space_group: Gemmi space group, HM name, or ``str(sg)``
    :param zone: Current reciprocal-space zone
    :param unit_cell: Optional cell for metric projection of in-plane axes
    :returns: Deduplicated list of :class:`SymmetryAxis`
    """
    sg = resolve_space_group(space_group)
    if sg is None:
        return []

    normal = np.asarray(zone.normal, dtype=np.float64)
    n_norm = float(np.linalg.norm(normal))
    if n_norm < 1e-12:
        return []
    normal = normal / n_norm

    # key → (order, screw, direction, orientation, raw_axis)
    found: dict[tuple[str, tuple[int, int, int]], tuple[int, float, np.ndarray]] = {}

    for op in sg.operations():
        order = int(op.rot_type())
        if order not in (2, 3, 4, 6):
            continue
        den = float(op.DEN)
        R = np.array(op.rot, dtype=np.float64) / den
        t = np.array(op.tran, dtype=np.float64) / den
        axis = _axis_direction_from_rotation(R)
        direction = _canonicalize_direction(axis)
        dir_vec = np.asarray(direction, dtype=np.float64)
        d_norm = float(np.linalg.norm(dir_vec))
        if d_norm < 1e-12:
            continue
        dir_unit = dir_vec / d_norm
        screw = float(np.dot(t, dir_unit))
        # Keep the representative with the smallest positive screw for labelling
        # (e.g. prefer 4₁ over 4₃ = 4₁³).
        screw_mod = screw % 1.0
        if screw_mod > 0.95:
            screw_mod = 0.0
        key = (str(order), direction)
        prev = found.get(key)
        if prev is None or (
            screw_mod > 0.05
            and (prev[1] < 0.05 or screw_mod < prev[1] - 1e-9)
        ):
            found[key] = (order, screw_mod, dir_vec)

    axes: list[SymmetryAxis] = []
    for (_order_key, direction), (order, screw, dir_vec) in found.items():
        d_norm = float(np.linalg.norm(dir_vec))
        dir_unit = dir_vec / d_norm
        alignment = abs(float(np.dot(dir_unit, normal)))
        label = _screw_label(order, screw)

        if alignment > 0.92:
            axes.append(
                SymmetryAxis(
                    label=label,
                    order=order,
                    direction=direction,
                    orientation="normal",
                    u=0.0,
                    v=0.0,
                    screw_fraction=screw,
                )
            )
            continue
        if alignment > 0.25:
            # Oblique to the zone — not drawable as a clean 2D element.
            continue

        uv = project_to_zone_plane(
            np.asarray([direction], dtype=np.float64),
            zone,
            unit_cell=unit_cell,
        )[0]
        length = float(np.hypot(uv[0], uv[1]))
        if length < 1e-9:
            continue
        u = float(uv[0] / length)
        v = float(uv[1] / length)
        # Stable sign for drawing.
        if u < -1e-9 or (abs(u) <= 1e-9 and v < 0.0):
            u, v = -u, -v
        axes.append(
            SymmetryAxis(
                label=label,
                order=order,
                direction=direction,
                orientation="in_plane",
                u=u,
                v=v,
                screw_fraction=screw,
            )
        )

    # One symbol per crystal direction: keep highest rotational order.
    by_direction: dict[tuple[int, int, int], SymmetryAxis] = {}
    for ax in axes:
        prev = by_direction.get(ax.direction)
        if prev is None or ax.order > prev.order:
            by_direction[ax.direction] = ax

    def sort_key(ax: SymmetryAxis) -> tuple:
        orient = 0 if ax.orientation == "normal" else 1
        return (orient, -ax.order, ax.direction, ax.label)

    return sorted(by_direction.values(), key=sort_key)

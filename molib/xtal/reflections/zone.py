"""Reciprocal-space zone membership and plane projection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from molib.xtal.map.unit_cell import UnitCell


class ZonePreset(str, Enum):
    """Named reciprocal-space zone presets."""

    HK0 = "hk0"
    H0L = "h0l"
    OKL = "0kl"
    HHL = "hhl"
    CUSTOM = "custom"


@dataclass(frozen=True, slots=True)
class ZoneSpec:
    """Zone defined by reciprocal normal ``z`` and integer level."""

    normal: tuple[int, int, int]
    level: int = 0
    preset: ZonePreset = ZonePreset.CUSTOM
    tolerance: int = 0

    @classmethod
    def from_preset(cls, preset: ZonePreset, *, level: int = 0) -> "ZoneSpec":
        """Build a :class:`ZoneSpec` from a named preset."""
        normals = {
            ZonePreset.HK0: (0, 0, 1),
            ZonePreset.H0L: (0, 1, 0),
            ZonePreset.OKL: (1, 0, 0),
            ZonePreset.HHL: (1, -1, 0),
        }
        if preset == ZonePreset.CUSTOM:
            raise ValueError("CUSTOM preset requires an explicit normal")
        return cls(normal=normals[preset], level=level, preset=preset)


def zone_membership(
    miller: np.ndarray,
    zone: ZoneSpec,
) -> np.ndarray:
    """Return boolean mask of reflections in the zone.

    Membership is ``|h·z - level| <= tolerance`` with integer arithmetic when
    ``tolerance == 0``.

    :param miller: ``(N, 3)`` integer Miller indices
    :param zone: Zone specification
    :returns: Boolean mask of length ``N``
    """
    hkl = np.asarray(miller, dtype=np.int32).reshape(-1, 3)
    z = np.asarray(zone.normal, dtype=np.int32).reshape(3)
    dot = hkl @ z
    if zone.tolerance <= 0:
        return dot == int(zone.level)
    return np.abs(dot - int(zone.level)) <= int(zone.tolerance)


def _reciprocal_metric(unit_cell: UnitCell | None) -> np.ndarray:
    """Return 3×3 reciprocal metric tensor G* (Å⁻²)."""
    if unit_cell is None:
        return np.eye(3, dtype=np.float64)
    import gemmi

    cell = gemmi.UnitCell(
        unit_cell.a,
        unit_cell.b,
        unit_cell.c,
        unit_cell.alpha,
        unit_cell.beta,
        unit_cell.gamma,
    )
    # gemmi ``orth.mat`` maps fractional → orthogonal Å.
    # Reciprocal Cartesian: h_cart = orth^{-T} @ h ; G* = A*^T A*.
    m = np.array(cell.orth.mat.tolist(), dtype=np.float64)
    inv_t = np.linalg.inv(m).T
    return inv_t.T @ inv_t


def plane_basis(zone: ZoneSpec) -> tuple[np.ndarray, np.ndarray]:
    """Return two integer lattice vectors spanning the zone plane.

    The vectors ``e1``, ``e2`` are chosen so that for standard presets the
    plot axes match common crystallographic labelling (e.g. hk0 → h along u,
    k along v).
    """
    z = np.asarray(zone.normal, dtype=np.int32)
    if zone.preset == ZonePreset.HK0 or tuple(z.tolist()) == (0, 0, 1):
        return np.array([1, 0, 0], dtype=np.float64), np.array(
            [0, 1, 0], dtype=np.float64
        )
    if zone.preset == ZonePreset.H0L or tuple(z.tolist()) == (0, 1, 0):
        return np.array([1, 0, 0], dtype=np.float64), np.array(
            [0, 0, 1], dtype=np.float64
        )
    if zone.preset == ZonePreset.OKL or tuple(z.tolist()) == (1, 0, 0):
        return np.array([0, 1, 0], dtype=np.float64), np.array(
            [0, 0, 1], dtype=np.float64
        )
    if zone.preset == ZonePreset.HHL or tuple(z.tolist()) == (1, -1, 0):
        return np.array([1, 1, 0], dtype=np.float64), np.array(
            [0, 0, 1], dtype=np.float64
        )

    # General: pick two vectors orthogonal to z with integer components.
    if z[0] == 0 and z[1] == 0:
        e1 = np.array([1.0, 0.0, 0.0])
        e2 = np.array([0.0, 1.0, 0.0])
    elif z[0] == 0 and z[2] == 0:
        e1 = np.array([1.0, 0.0, 0.0])
        e2 = np.array([0.0, 0.0, 1.0])
    elif z[1] == 0 and z[2] == 0:
        e1 = np.array([0.0, 1.0, 0.0])
        e2 = np.array([0.0, 0.0, 1.0])
    else:
        # Cross with a unit axis not parallel to z.
        axis = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(z, axis))) > 0.9 * np.linalg.norm(z):
            axis = np.array([0.0, 1.0, 0.0])
        e1 = np.cross(z.astype(np.float64), axis)
        e2 = np.cross(z.astype(np.float64), e1)
    return e1, e2


def project_to_zone_plane(
    miller: np.ndarray,
    zone: ZoneSpec,
    unit_cell: UnitCell | None = None,
) -> np.ndarray:
    """Project Miller indices into 2D zone-plane coordinates ``(u, v)``.

    Without a cell, ``u,v`` are integer coefficients along the plane basis
    (e.g. hk0 → ``(h, k)``). With a cell, coordinates are in reciprocal
    Cartesian Å⁻¹ along an orthonormalization of that basis using G*.

    :param miller: ``(N, 3)`` Miller indices
    :param zone: Zone specification
    :param unit_cell: Optional cell for metric scaling
    :returns: ``(N, 2)`` float array of plot coordinates
    """
    hkl = np.asarray(miller, dtype=np.float64).reshape(-1, 3)
    e1, e2 = plane_basis(zone)

    if unit_cell is None:
        # Solve h ≈ a*e1 + b*e2 in least squares for each reflection.
        basis = np.column_stack((e1, e2))  # 3x2
        # For orthogonal integer presets this recovers (h,k) etc. exactly.
        uv, _, _, _ = np.linalg.lstsq(basis, hkl.T, rcond=None)
        return uv.T

    gstar = _reciprocal_metric(unit_cell)
    # Orthonormalize e1,e2 in the reciprocal metric.
    g11 = float(e1 @ gstar @ e1)
    g12 = float(e1 @ gstar @ e2)
    g22 = float(e2 @ gstar @ e2)
    # Coefficients of h in the {e1,e2} basis via metric inner products.
    # h·e1* etc.
    rhs1 = hkl @ (gstar @ e1)
    rhs2 = hkl @ (gstar @ e2)
    det = g11 * g22 - g12 * g12
    if abs(det) < 1e-18:
        basis = np.column_stack((e1, e2))
        uv, _, _, _ = np.linalg.lstsq(basis, hkl.T, rcond=None)
        return uv.T
    a = (g22 * rhs1 - g12 * rhs2) / det
    b = (-g12 * rhs1 + g11 * rhs2) / det
    # Scale axes to Å⁻¹ lengths of e1, e2.
    len1 = np.sqrt(max(g11, 0.0))
    len2 = np.sqrt(max(g22, 0.0))
    return np.column_stack((a * len1, b * len2))


def axis_labels(zone: ZoneSpec) -> tuple[str, str]:
    """Return short axis labels for the zone plot."""
    if zone.preset == ZonePreset.HK0:
        return "h", "k"
    if zone.preset == ZonePreset.H0L:
        return "h", "l"
    if zone.preset == ZonePreset.OKL:
        return "k", "l"
    if zone.preset == ZonePreset.HHL:
        return "h(=k)", "l"
    return "u", "v"

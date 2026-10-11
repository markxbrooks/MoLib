"""Resolution filtering and constant-d contours in a zone plane."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from molib.xtal.map.unit_cell import UnitCell
from molib.xtal.reflections.zone import ZoneSpec, plane_basis, project_to_zone_plane


@dataclass(frozen=True, slots=True)
class ResolutionFilter:
    """High / low resolution cutoffs in Å (``None`` = no bound)."""

    d_min: float | None = None
    d_max: float | None = None

    def mask(self, d_spacing: np.ndarray | None) -> np.ndarray:
        """Return keep-mask for reflections with spacing *d_spacing*."""
        if d_spacing is None:
            return np.ones(0, dtype=bool)
        d = np.asarray(d_spacing, dtype=np.float64)
        keep = np.isfinite(d)
        if self.d_min is not None:
            keep &= d >= float(self.d_min)
        if self.d_max is not None:
            keep &= d <= float(self.d_max)
        return keep


def resolution_contour_uv(
    d_angstrom: float,
    zone: ZoneSpec,
    unit_cell: UnitCell | None,
    *,
    n_points: int = 128,
) -> np.ndarray | None:
    """Sample the constant-resolution contour in zone-plane ``(u,v)`` coords.

    For an orthogonal cell and axial zones this is a circle in Å⁻¹ space.
    For general cells the contour is the intersection of the resolution
    sphere with the zone plane, sampled in the same ``(u,v)`` coordinates used
    by :func:`project_to_zone_plane`.

    :returns: ``(N, 2)`` polyline, or ``None`` if geometry is unavailable
    """
    if d_angstrom <= 0 or unit_cell is None:
        return None
    import gemmi

    cell = gemmi.UnitCell(
        unit_cell.a,
        unit_cell.b,
        unit_cell.c,
        unit_cell.alpha,
        unit_cell.beta,
        unit_cell.gamma,
    )
    m = np.array(cell.orth.mat.tolist(), dtype=np.float64)
    a_star_basis = np.linalg.inv(m).T  # columns ≈ a*,b*,c* in Cartesian Å⁻¹

    e1, e2 = plane_basis(zone)
    # Cartesian directions of plot axes in Å⁻¹
    v1 = a_star_basis @ e1
    v2 = a_star_basis @ e2
    # Orthonormalize in Cartesian for sampling a circle of radius 1/d
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-18 or n2 < 1e-18:
        return None
    u_hat = v1 / n1
    # Remove component along u_hat from v2
    v2p = v2 - u_hat * float(np.dot(v2, u_hat))
    n2p = np.linalg.norm(v2p)
    if n2p < 1e-18:
        return None
    v_hat = v2p / n2p

    radius = 1.0 / float(d_angstrom)
    angles = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=True)
    cart = np.outer(np.cos(angles), u_hat) * radius + np.outer(
        np.sin(angles), v_hat
    ) * radius
    # Convert Cartesian Å⁻¹ back to Miller-like then to plot uv via projection
    # of synthetic fractional reciprocal coordinates.
    # h_frac = A*^-1 @ cart = m.T @ cart
    h_frac = (m.T @ cart.T).T
    return project_to_zone_plane(h_frac, zone, unit_cell=unit_cell)

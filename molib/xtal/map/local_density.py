"""Local density sampling with crystallographic continuity.

``continuous_local`` keeps a single unit-cell volume in memory. When the
focus approaches or crosses a cell face, neighbouring density is obtained
by sampling that volume under the crystal's translation lattice (periodic
boundary conditions). Space-group rotational/screw operators map within
the unit cell and are used to fold lookups when an optional space-group
name is supplied (helps ASU-only maps).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from decologr import Decologr as log

try:
    import gemmi
except ImportError:  # pragma: no cover - gemmi is a hard dep in practice
    gemmi = None


def _space_group_ops(space_group: str | object | None) -> list:
    """Return gemmi symmetry Ops for *space_group*, or ``[]`` if unknown."""
    if gemmi is None or space_group is None:
        return []
    try:
        if hasattr(space_group, "operations"):
            return list(space_group.operations())
        name = str(space_group).strip()
        if not name or name.lower() in {"", "none", "unknown", "p 1", "p1"}:
            # P1: lattice translations only (handled by wrap).
            if name.lower() in {"p 1", "p1"}:
                return []
            if not name:
                return []
        sg = gemmi.find_spacegroup_by_name(name)
        if sg is None:
            # gemmi sometimes stores "<gemmi.SpaceGroup(\"P 43\")>"-like strings
            if "SpaceGroup" in name and '"' in name:
                inner = name.split('"')[1]
                sg = gemmi.find_spacegroup_by_name(inner)
        if sg is None:
            return []
        return list(sg.operations())
    except Exception as ex:
        log.debug(f"Could not resolve space group ops from {space_group!r}: {ex}")
        return []


def crop_local_density_with_symmetry(
    volume: np.ndarray,
    center: Tuple[int, int, int],
    radius_grid: int,
    *,
    space_group: str | object | None = None,
) -> tuple[Optional[np.ndarray], np.ndarray]:
    """Crop a local brick around *center*, filling out-of-cell voxels via symmetry.

    Absolute grid indices may lie outside ``[0, n)`` (adjacent unit cells).
    Those voxels are filled by:
      1. Lattice wrap into the stored unit-cell array (always).
      2. Optional space-group fold: if *space_group* is set, try each Op so
         ASU-sparse maps still yield density at symmetry-equivalent sites.

    The returned *origin* is the unwrapped lower corner ``center - radius``
    so marching-cubes vertices transform into the correct neighbouring cell.

    :param volume: Unit-cell density, shape ``(nx, ny, nz)``
    :param center: Focus grid index (may be outside the array)
    :param radius_grid: Half-width of the brick in voxels
    :param space_group: Hermann–Mauguin name, gemmi SpaceGroup, or None
    :return: ``(sub_volume, origin_xyz)`` or ``(None, zeros)`` if empty
    """
    if volume is None or volume.size == 0:
        return None, np.zeros(3, dtype=np.float32)

    vol = np.asarray(volume)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3-D volume, got shape {vol.shape}")

    nx, ny, nz = (int(vol.shape[0]), int(vol.shape[1]), int(vol.shape[2]))
    if nx < 1 or ny < 1 or nz < 1:
        return None, np.zeros(3, dtype=np.float32)

    radius = max(1, int(radius_grid))
    cx, cy, cz = (int(center[0]), int(center[1]), int(center[2]))
    x0, y0, z0 = cx - radius, cy - radius, cz - radius
    size = 2 * radius + 1

    xs = np.arange(x0, x0 + size, dtype=np.int64)
    ys = np.arange(y0, y0 + size, dtype=np.int64)
    zs = np.arange(z0, z0 + size, dtype=np.int64)
    ix, iy, iz = np.meshgrid(xs, ys, zs, indexing="ij")

    # Lattice translations: adjacent cells map back into the stored unit cell.
    sx = np.mod(ix, nx)
    sy = np.mod(iy, ny)
    sz = np.mod(iz, nz)
    sub = vol[sx, sy, sz].astype(vol.dtype, copy=True)

    ops = _space_group_ops(space_group)
    if len(ops) > 1:
        # Fold through non-identity ops: for voxels that are still ~empty after
        # wrap (typical of ASU-only maps), take the strongest symmetry image.
        sub = _fill_via_space_group_ops(vol, sub, ix, iy, iz, ops)

    origin = np.array([x0, y0, z0], dtype=np.float32)
    return sub, origin


def _fill_via_space_group_ops(
    volume: np.ndarray,
    sub: np.ndarray,
    ix: np.ndarray,
    iy: np.ndarray,
    iz: np.ndarray,
    ops: Sequence,
) -> np.ndarray:
    """Overwrite near-zero samples with density from non-identity space-group ops."""
    nx, ny, nz = volume.shape
    dims = np.array([nx, ny, nz], dtype=np.float64)
    # Fractional coords of absolute (possibly out-of-cell) grid points.
    frac = np.stack(
        [ix / dims[0], iy / dims[1], iz / dims[2]],
        axis=-1,
    ).astype(np.float64)

    # Only bother where wrap left little signal.
    abs_sub = np.abs(sub)
    threshold = float(np.nanmax(abs_sub)) * 1e-4 if abs_sub.size else 0.0
    if threshold <= 0.0:
        threshold = 1e-8
    empty = abs_sub <= threshold
    if not np.any(empty):
        return sub

    out = sub
    for op in ops:
        # Skip identity (already covered by lattice wrap).
        try:
            if op.triplet().replace(" ", "") in {"x,y,z", "X,Y,Z"}:
                continue
        except Exception:
            pass

        # Apply op to fractional coordinates of empty voxels only.
        empty_frac = frac[empty]
        if empty_frac.size == 0:
            break
        try:
            # gemmi Op.apply_to_xyz expects a sequence of 3 floats.
            transformed = np.empty_like(empty_frac)
            for i, f in enumerate(empty_frac):
                transformed[i] = op.apply_to_xyz(f.tolist())
        except Exception:
            continue

        # Lattice-wrap into the unit cell and sample.
        wrapped = transformed - np.floor(transformed)
        gx = np.clip(np.floor(wrapped[:, 0] * nx).astype(np.int64), 0, nx - 1)
        gy = np.clip(np.floor(wrapped[:, 1] * ny).astype(np.int64), 0, ny - 1)
        gz = np.clip(np.floor(wrapped[:, 2] * nz).astype(np.int64), 0, nz - 1)
        sampled = volume[gx, gy, gz]

        # Prefer the stronger |density| image for empty sites.
        stronger = np.abs(sampled) > abs_sub[empty]
        if not np.any(stronger):
            continue
        if out is sub:
            out = sub.copy()
        empty_idx = np.where(empty)
        sel = stronger
        out[
            empty_idx[0][sel],
            empty_idx[1][sel],
            empty_idx[2][sel],
        ] = sampled[sel]
        abs_sub = np.abs(out)
        empty = abs_sub <= threshold
        if not np.any(empty):
            break

    return out

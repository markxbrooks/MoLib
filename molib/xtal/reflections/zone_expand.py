"""Expand reflections into a reciprocal-space zone via crystal symmetry."""

from __future__ import annotations

from typing import Any

import numpy as np

from molib.xtal.reflections.symmetry_axes import resolve_space_group
from molib.xtal.reflections.zone import ZoneSpec, zone_membership


def _operation_matrices(space_group: Any) -> list[np.ndarray]:
    """Return integer 3×3 reciprocal rotation matrices for *space_group*."""
    import gemmi

    sg = resolve_space_group(space_group)
    if sg is None:
        return [np.eye(3, dtype=np.int32)]
    den = float(gemmi.Op.DEN)
    mats: list[np.ndarray] = []
    for op in sg.operations():
        rot = np.asarray(op.rot, dtype=np.float64) / den
        mats.append(np.rint(rot).astype(np.int32))
    if not mats:
        mats.append(np.eye(3, dtype=np.int32))
    return mats


def expand_to_zone(
    miller: np.ndarray,
    zone: ZoneSpec,
    space_group: Any | None = None,
    *,
    include_friedel: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Map reflections into *zone* using space-group (and optional Friedel) images.

    Without a resolvable space group this is equivalent to
    :func:`zone_membership` filtering of the input indices.

    With a space group, every observation is transformed by each reciprocal
    rotation; images that fall in the zone are kept. This fills sparsely
    observed zones (e.g. ``h0l`` on a spindle-aligned ``0kl``-rich scan) from
    symmetry-equivalent reflections.

    :param miller: ``(N, 3)`` Miller indices
    :param zone: Target zone
    :param space_group: gemmi space group, name string, or ``None``
    :param include_friedel: Also include ``(-h,-k,-l)`` images when the group
        is not already centrosymmetric
    :returns: ``(miller_out, source_indices)`` where ``source_indices[j]`` is
        the row of *miller* that produced ``miller_out[j]``
    """
    hkl = np.asarray(miller, dtype=np.int32).reshape(-1, 3)
    n = hkl.shape[0]
    if n == 0:
        return (
            np.zeros((0, 3), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
        )

    sg = resolve_space_group(space_group)
    if sg is None:
        mask = zone_membership(hkl, zone)
        idx = np.nonzero(mask)[0].astype(np.int32)
        return hkl[idx].copy(), idx

    mats = _operation_matrices(sg)
    if include_friedel:
        try:
            import gemmi

            ops = sg.operations()
            if not bool(ops.is_centrosymmetric):
                mats.extend([-m for m in list(mats)])
        except Exception:
            mats.extend([-m for m in list(mats)])

    out_hkl: list[np.ndarray] = []
    out_src: list[np.ndarray] = []
    src_all = np.arange(n, dtype=np.int32)
    for rot in mats:
        # gemmi apply_to_hkl matches ``hkl @ R`` for reciprocal rotations.
        transformed = np.asarray(hkl @ rot, dtype=np.int32)
        mask = zone_membership(transformed, zone)
        if not np.any(mask):
            continue
        out_hkl.append(transformed[mask])
        out_src.append(src_all[mask])

    if not out_hkl:
        return (
            np.zeros((0, 3), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
        )
    return np.concatenate(out_hkl, axis=0), np.concatenate(out_src, axis=0)

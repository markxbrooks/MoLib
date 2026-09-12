"""Coordinate helpers for molecular entities (independent of GL mesh types)."""

from __future__ import annotations

from typing import Any

import numpy as np


def atom_xyz(atom: Any) -> tuple[float, float, float]:
    """Return ``(x, y, z)`` from ``atom.coords`` or ``atom.x`` / ``y`` / ``z``.

    :param atom: ``Atom3D``, ``Vector3``, or any object with ``coords`` or
        ``x``/``y``/``z``
    :return: World-space Cartesian triple
    """
    coords = getattr(atom, "coords", None)
    if coords is not None:
        return float(coords[0]), float(coords[1]), float(coords[2])
    if isinstance(atom, np.ndarray):
        return float(atom[0]), float(atom[1]), float(atom[2])
    return float(atom.x), float(atom.y), float(atom.z)

"""Coordinate helpers for molecular entities (independent of GL mesh types)."""

from __future__ import annotations

from typing import Any

import numpy as np


def atom_radius(atom: Any, fallback: float) -> float:
    """Return ``atom.radius`` in Å when it is finite and positive.

    :class:`~molib.entities.atom.Atom3D` always stores a resolved radius
    (explicit PQR value, otherwise van der Waals). This helper still accepts
    duck-typed objects that may leave ``radius`` unset.

    :param atom: Object that may expose ``radius`` (e.g. ``Atom3D``)
    :param fallback: World-space radius used when *atom* has no usable radius
    :return: Radius in Ångstroms
    """
    raw = getattr(atom, "radius", None)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float(fallback)
    if value != value or value <= 0.0:
        return float(fallback)
    return value


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

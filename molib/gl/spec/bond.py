"""Bond mesh specifications (line sticks vs cylinder shafts)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from molib.gl.spec.atom import AtomColorFn


@dataclass(frozen=True, slots=True)
class BondMeshSpec:
    """Shared parameters for bond line-stick geometry.

    :param atoms: Atoms (``Atom3D`` or compatible) whose coordinates define endpoints
    :param indices: Flat or ``(N, 2)`` atom-index bond pairs
    :param color_fn: Per-atom RGB; mesh default when omitted
    :param color_bonds: When false, every vertex/shaft uses *bond_color*
    :param bond_color: Uniform RGB when *color_bonds* is false
    """

    atoms: list
    indices: np.ndarray | None = None
    color_fn: AtomColorFn | None = None
    color_bonds: bool = False
    bond_color: tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass(frozen=True, slots=True)
class BondCylinderSpec(BondMeshSpec):
    """Cylinder-shaft parameters on top of :class:`BondMeshSpec`.

    :param radius: Cylinder radius in Å
    :param segments: Radial tessellation of each shaft
    """

    radius: float = 0.06
    segments: int = 8

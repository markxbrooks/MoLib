"""
Defines a specification for bonds mesh, including cylinder parameters
and visualization preferences.

This specification encapsulates details required for rendering bonds as
cylinders in molecular visualization. It includes atom references,
bond definitions, coloring options, and geometric parameters.

Classes:
    BondsMeshSpec: A data class for specifying bond mesh parameters.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from molib.gl.spec.atom import AtomColorFn


@dataclass
class BondsMeshSpec:
    """Parameters for :class:`~molib.gl.mesh.bond.cylinder.BondCylindersMesh`.

    :param atoms: Atoms whose coordinates define cylinder endpoints
    :param indices: Flat or ``(N, 2)`` atom-index bond pairs
    :param color_fn: Per-atom RGB sampled from the first atom of each pair
    :param color_bonds: When false, every shaft uses *bond_color*
    :param bond_color: Uniform RGB when *color_bonds* is false
    :param radius: Cylinder radius in Å
    :param segments: Radial tessellation of each shaft
    """

    atoms: list
    indices: np.ndarray | None = None
    color_fn: Optional[AtomColorFn] = None
    color_bonds: bool = False
    bond_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    radius: float = 0.06
    segments: int = 8

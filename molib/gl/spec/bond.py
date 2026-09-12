from dataclasses import dataclass
from typing import Optional

from molib.gl.spec.atom import AtomColorFn


@dataclass
class BondsMeshSpec:
    """Parameters for a PicoGL :class:`~picogl.renderer.molecular.bonds.BondsMesh`.

    :param bond_pairs: ``(atom1, atom2)`` pairs to instance as cylinders
    :param color_fn: Per-bond RGB sampled from the first atom
    :param radius: Cylinder radius in Å
    :param segments: Radial tessellation of each shaft
    """

    bond_pairs: list
    color_fn: Optional[AtomColorFn] = None
    radius: float = 0.06
    segments: int = 8

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, Any, Optional

AtomColorFn = Callable[[Any], tuple[float, float, float]]
_ATOM_POINT_SIZE_TO_RADIUS = 0.025
_ATOM_SPHERE_MIN_RADIUS = 0.05
_ATOM_BUFFER_SPHERE_SLICES = 12
_ATOM_BUFFER_SPHERE_STACKS = 12


@dataclass
class AtomsSphereSpec:
    """Parameters for a PicoGL :class:`~picogl.renderer.molecular.atoms.AtomsMesh`.

    :param atoms: Atoms to instance (``Atom3D`` or PicoGL-compatible)
    :param color_fn: Per-atom RGB; chain palette when omitted
    :param radius: Uniform world-space sphere radius in Å
    :param slices: Longitudinal sphere tessellation
    :param stacks: Latitudinal sphere tessellation
    :param radii: Optional per-atom radii in Å; overrides *radius* per atom
    """

    atoms: list
    color_fn: Optional[AtomColorFn] = None
    radius: float = 0.2
    slices: int = 8
    stacks: int = 8
    radii: Optional[Sequence[float]] = None

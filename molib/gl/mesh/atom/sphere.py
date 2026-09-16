"""Sphere-instanced atom mesh for molecular visualization."""

from __future__ import annotations


from typing import Sequence, Callable, Any

import numpy as np

from picogl.backend.gl.enums import GLDrawMode
from picogl.core.geometry.sphere import SphereGeometrySpec, SphereMesh
from molib.entities.coords import atom_xyz
from molib.gl.mesh.molecule import MolecularMesh
from molib.pdb.color import make_chain_color_fn
from picogl.renderer.meshdata import MeshData
from picogl.renderer.molecular.pnc_buffer import PNCBuffer

__all__ = ["AtomSpheresMesh", "atom_xyz", "make_chain_color_fn"]


class AtomSpheresMesh(MolecularMesh):
    """
    Build triangle meshes by instancing a sphere template at each atom position.

    Atoms must expose either ``x``, ``y``, ``z`` or a ``coords`` sequence
    (as in MoLib ``Atom3D``). The default color function colors by ``chain_id``;
    pass a custom ``color_fn(atom)`` for other schemes.

    ``radius``, ``slices``, and ``stacks`` construct a
    :class:`~picogl.core.geometry.sphere.SphereGeometrySpec` when
    ``geometry`` is omitted. Optional *radii* scales each instance so
    ``Atom3D.radius`` can override the uniform template radius.
    """

    draw_mode = GLDrawMode.TRIANGLES

    def __init__(
        self,
        atoms: Sequence[Any],
        *,
        color_fn: Callable[[Any], tuple[float, float, float]] = None,
        radius: float = 0.2,
        slices: int = 16,
        stacks: int = 16,
        geometry: SphereGeometrySpec | None = None,
        radii: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.atoms = atoms
        self.color_fn = color_fn
        self.geometry = geometry or SphereGeometrySpec(
            radius=radius, slices=slices, stacks=stacks
        )
        self._sphere = SphereMesh(self.geometry)
        self.radii = None if radii is None else np.asarray(radii, dtype=np.float32)

    @property
    def radius(self) -> float:
        """Sphere radius from :attr:`geometry`."""
        return self.geometry.radius

    @property
    def slices(self) -> int:
        """Longitudinal subdivisions from :attr:`geometry`."""
        return self.geometry.slices

    @property
    def stacks(self) -> int:
        """Latitudinal subdivisions from :attr:`geometry`."""
        return self.geometry.stacks

    def _resolved_color_fn(self) -> Callable[[Any], tuple[float, float, float]]:
        """Return *color_fn*, or a chain-palette sampler built from ``self.atoms``."""
        if self.color_fn is not None:
            return self.color_fn
        return make_chain_color_fn([atom.chain_id for atom in self.atoms])

    def build_mesh_data(self) -> MeshData:
        """Instance sphere geometry at each atom and assign per-atom colors.

        Expands one :class:`~picogl.core.geometry.sphere.SphereMesh` template
        through :class:`~picogl.renderer.molecular.pnc_buffer.PNCBuffer`. The
        result is a fully expanded :class:`~picogl.renderer.meshdata.MeshData`
        (one sphere per atom) so existing VAO / ``first_item`` draw paths stay
        unchanged.
        """
        if not self.atoms:
            return self._empty_mesh_data(
                elements_per_item=self._sphere.elements_per_item,
                vertices_per_item=self._sphere.vertices_per_item,
            )

        template = self._sphere.build()
        positions = np.asarray(
            [atom_xyz(atom) for atom in self.atoms],
            dtype=np.float32,
        )
        color_fn = self._resolved_color_fn()
        colors = np.asarray(
            [color_fn(atom) for atom in self.atoms],
            dtype=np.float32,
        ).reshape(-1, 3)

        scales = None
        if self.radii is not None:
            radii = np.asarray(self.radii, dtype=np.float32).reshape(-1)
            if radii.shape[0] != positions.shape[0]:
                raise ValueError("radii must have one value per atom")
            template_r = float(self.geometry.radius) or 1.0
            scales = radii / np.float32(template_r)

        buf = PNCBuffer()
        buf.add_instances(template, positions, colors, scales=scales)
        return buf.to_mesh_arrays().as_meshdata(
            mode=GLDrawMode.TRIANGLES,
            indexed=True,
            elements_per_item=self._sphere.elements_per_item,
            vertices_per_item=self._sphere.vertices_per_item,
        )

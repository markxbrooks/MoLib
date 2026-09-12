"""Line-instanced atom mesh for molecular visualization."""

from __future__ import annotations


from typing import Sequence, Callable, Any

import numpy as np

from picogl.backend.gl.enums import GLDrawMode
from molib.calc.math.vector import Vector3
from molib.entities.atom import Atom3D
from molib.gl.mesh.molecule import MolecularMesh
from molib.pdb.color import palette_rgb_at
from picogl.renderer.draw_spec import MeshDrawInfo
from picogl.renderer.mesh_arrays import MeshArrays
from picogl.renderer.meshdata import MeshData
from molib.gl.mesh.atom.sphere_geometry import AtomSphereGeometry


class AtomlinesMesh(MolecularMesh):
    """
    Build triangle meshes by instancing a sphere template at each atom position.

    Atoms must expose either ``x``, ``y``, ``z`` or a ``coords`` sequence
    (as in MoLib ``Atom3D``). The default color function colors by ``chain_id``;
    pass a custom ``color_fn(atom)`` for other schemes.

    ``radius``, ``slices``, and ``stacks`` construct an :class:`AtomGeometry`
    when ``geometry`` is omitted.
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
        geometry: AtomSphereGeometry | None = None,
    ) -> None:
        super().__init__()
        self.atoms = atoms
        self.color_fn = color_fn
        self.geometry = geometry or AtomSphereGeometry(
            radius=radius, slices=slices, stacks=stacks
        )

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

        Expands one :class:`~picogl.renderer.molecular.atom_geometry.AtomGeometry`
        template with NumPy broadcasting. The result is a fully expanded
        :class:`~picogl.renderer.meshdata.MeshData` (one sphere per atom) so
        existing VAO / ``first_item`` draw paths stay unchanged.
        """
        if not self.atoms:
            return self._empty_mesh_data(
                elements_per_item=self.geometry.elements_per_item,
                vertices_per_item=self.geometry.vertices_per_item,
            )

        template = self.geometry.build()
        vertices = template.positions
        normals = template.normals
        indices = np.asarray(template.indices, dtype=np.uint32).ravel()

        positions = np.asarray(
            [atom_xyz(atom) for atom in self.atoms],
            dtype=np.float32,
        )
        color_fn = self._resolved_color_fn()
        colors = np.asarray(
            [color_fn(atom) for atom in self.atoms],
            dtype=np.float32,
        ).reshape(-1, 3)

        n_atoms = int(positions.shape[0])
        n_vertices = int(vertices.shape[0])
        offsets = np.arange(n_atoms, dtype=np.uint32) * np.uint32(n_vertices)

        out_vertices = (vertices[None, :, :] + positions[:, None, :]).reshape(-1, 3)
        out_normals = np.tile(normals, (n_atoms, 1))
        out_colors = np.repeat(colors, n_vertices, axis=0)
        out_indices = (indices[None, :] + offsets[:, None]).reshape(-1)

        expanded = MeshArrays(
            positions=out_vertices,
            normals=out_normals,
            colors=out_colors,
            indices=out_indices,
        )
        mesh_data = expanded.as_meshdata()
        mesh_data.draw_info = MeshDrawInfo(
            mode=GLDrawMode.TRIANGLES,
            indexed=True,
            elements_per_item=self.geometry.elements_per_item,
            vertices_per_item=self.geometry.vertices_per_item,
        )
        return mesh_data
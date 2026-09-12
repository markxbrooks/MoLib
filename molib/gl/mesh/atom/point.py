"""Unindexed GL_POINTS mesh for molecular atom positions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from molib.entities.coords import atom_xyz
from molib.gl.mesh.atom.point_geometry import AtomPointGeometry
from molib.gl.mesh.molecule import MolecularMesh
from molib.pdb.color import make_chain_color_fn
from picogl.backend.gl.enums import GLDrawMode
from picogl.renderer.draw_spec import MeshDrawInfo
from picogl.renderer.mesh_arrays import MeshArrays
from picogl.renderer.meshdata import MeshData


class AtomPointsMesh(MolecularMesh):
    """Render atoms as OpenGL points.

    Each atom becomes one vertex. There is no element buffer; drawing uses
    ``glDrawArrays(GL_POINTS, ...)``. ``radius``, ``slices``, and ``stacks``
    are not sphere tessellation parameters: only ``radius`` is accepted, as
    metadata on :class:`AtomPointGeometry`.
    """

    draw_mode = GLDrawMode.POINTS

    def __init__(
        self,
        atoms: Sequence[Any],
        *,
        color_fn: Callable[[Any], tuple[float, float, float]] | None = None,
        radius: float = 0.2,
        geometry: AtomPointGeometry | None = None,
    ) -> None:
        super().__init__()
        self.atoms = atoms
        self.color_fn = color_fn
        self.geometry = geometry or AtomPointGeometry(radius=radius)

    @property
    def radius(self) -> float:
        """Point radius metadata from :attr:`geometry`."""
        return self.geometry.radius

    def _resolved_color_fn(
        self,
    ) -> Callable[[Any], tuple[float, float, float]]:
        """Return *color_fn*, or a chain-palette sampler built from ``self.atoms``."""
        if self.color_fn is not None:
            return self.color_fn
        return make_chain_color_fn([atom.chain_id for atom in self.atoms])

    def build_mesh_data(self) -> MeshData:
        """Place one GL point at each atom coordinate.

        The origin template from :class:`AtomPointGeometry` is a single vertex
        at the origin, so atom positions are used directly. The mesh is
        unindexed ``POINTS``. Dummy zero normals satisfy
        :class:`~picogl.renderer.mesh_arrays.MeshArrays`.

        Returns
        -------
        MeshData
            One vertex and color per atom, ``draw_info`` in ``POINTS`` mode.
        """
        if not self.atoms:
            return self._empty_mesh_data(
                elements_per_item=self.geometry.elements_per_item,
                vertices_per_item=self.geometry.vertices_per_item,
                indexed=False,
            )

        positions = np.asarray(
            [atom_xyz(atom) for atom in self.atoms],
            dtype=np.float32,
        ).reshape(-1, 3)
        color_fn = self._resolved_color_fn()
        colors = np.asarray(
            [color_fn(atom) for atom in self.atoms],
            dtype=np.float32,
        ).reshape(-1, 3)

        arrays = MeshArrays(
            positions=positions,
            normals=np.zeros_like(positions),
            colors=colors,
        )
        mesh_data = arrays.as_meshdata(mode=GLDrawMode.POINTS)
        mesh_data.draw_info = MeshDrawInfo(
            mode=GLDrawMode.POINTS,
            indexed=False,
            elements_per_item=self.geometry.elements_per_item,
            vertices_per_item=self.geometry.vertices_per_item,
        )
        return mesh_data

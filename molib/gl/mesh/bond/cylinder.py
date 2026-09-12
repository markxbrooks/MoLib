"""Cylinder bond mesh for molecular visualization."""

from __future__ import annotations

from typing import Sequence, Callable, Any

import numpy as np

from picogl.backend.gl.enums import GLDrawMode
from picogl.renderer.draw_spec import MeshDrawInfo
from picogl.renderer.meshdata import MeshData
from picogl.renderer.molecular import BondGeometry
from molib.gl.mesh.molecule import MolecularMesh
from molib.gl.mesh.atom.sphere import make_chain_color_fn, atom_xyz


class BondCylindersMesh(MolecularMesh):
    """
    Build cylinder meshes connecting pairs of atoms.

    Each bond is a pair ``(atom1, atom2)`` with ``x``, ``y``, ``z``, and
    ``chain_id`` on each atom. Color is taken from the first atom, mirroring
    :class:`~picogl.renderer.molecular.atoms.AtomsMesh` semantics.

    ``radius`` and ``segments`` construct a :class:`BondGeometry` when
    ``geometry`` is omitted.
    """

    draw_mode = GLDrawMode.TRIANGLES

    def __init__(
        self,
        bonds: Sequence[tuple["Atom3D", "Atom3D"]],
        *,
        color_fn: Callable[[Any], tuple[float, float, float]] = None,
        radius: float = 0.06,
        segments: int = 8,
        geometry: BondGeometry | None = None,
    ) -> None:
        super().__init__()
        self.bonds = bonds
        self.color_fn = color_fn
        self.geometry = geometry or BondGeometry(radius=radius, segments=segments)

    @property
    def radius(self) -> float:
        """Cylinder radius from :attr:`geometry`."""
        return self.geometry.radius

    @property
    def segments(self) -> int:
        """Radial segment count from :attr:`geometry`."""
        return self.geometry.segments

    def _resolved_color_fn(self) -> Callable[[Any], tuple[float, float, float]]:
        """Return *color_fn*, or a chain-palette sampler from the first atoms."""
        if self.color_fn is not None:
            return self.color_fn
        return make_chain_color_fn([atom1.chain_id for atom1, _atom2 in self.bonds])

    def build_mesh_data(self) -> MeshData:
        """Build oriented cylinder shafts for all bonds.

        Geometry is generated in one :meth:`BondGeometry.build_many` call.
        Colors are repeated only for shafts that survive the zero-length filter.
        """
        if not self.bonds:
            return self._empty_mesh_data(
                elements_per_item=self.geometry.elements_per_item,
                vertices_per_item=self.geometry.vertices_per_item,
            )

        starts = np.asarray(
            [atom_xyz(atom1) for atom1, _atom2 in self.bonds],
            dtype=np.float64,
        )
        ends = np.asarray(
            [atom_xyz(atom2) for _atom1, atom2 in self.bonds],
            dtype=np.float64,
        )
        geometry, valid = self.geometry.build_many(starts, ends)
        if geometry.positions.shape[0] == 0:
            return self._empty_mesh_data(
                elements_per_item=self.geometry.elements_per_item,
                vertices_per_item=self.geometry.vertices_per_item,
            )

        color_fn = self._resolved_color_fn()
        colors = np.asarray(
            [color_fn(atom1) for atom1, _atom2 in self.bonds],
            dtype=np.float32,
        ).reshape(-1, 3)
        colors = np.repeat(
            colors[valid],
            self.geometry.vertices_per_item,
            axis=0,
        )

        mesh_data = geometry.with_colors(colors).as_meshdata()
        mesh_data.draw_info = MeshDrawInfo(
            mode=GLDrawMode.TRIANGLES,
            indexed=True,
            elements_per_item=self.geometry.elements_per_item,
            vertices_per_item=self.geometry.vertices_per_item,
        )
        return mesh_data

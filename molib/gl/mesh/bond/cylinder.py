"""Cylinder bond mesh for molecular visualization."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from molib.gl.mesh.atom.sphere import atom_xyz, make_chain_color_fn
from molib.gl.mesh.molecule import MolecularMesh
from picogl.backend.gl.enums import GLDrawMode
from picogl.renderer.meshdata import MeshData
from picogl.renderer.molecular import BondGeometry

_WHITE = (1.0, 1.0, 1.0)


class BondCylindersMesh(MolecularMesh):
    """Build cylinder shafts from atom positions and pair indices.

    Constructor arguments match :class:`~molib.gl.mesh.bond.line.BondLinesMesh`
    (``atoms``, ``indices``, ``color_fn``, ``bond_color``, ``color_bonds``).
    Cylinder-only extras are ``geometry``, or ``radius`` / ``segments`` when
    *geometry* is omitted.

    Color is one RGB per shaft, taken from the first atom of each pair
    (uniform *bond_color* when *color_bonds* is false). Zero-length pairs
    are omitted by :meth:`BondGeometry.build_many`.
    """

    draw_mode = GLDrawMode.TRIANGLES

    def __init__(
        self,
        atoms: Sequence[Any],
        indices: np.ndarray | Sequence[int] | None = None,
        *,
        color_fn: Callable[[Any], tuple[float, float, float]] | None = None,
        bond_color: tuple[float, float, float] = _WHITE,
        color_bonds: bool = False,
        geometry: BondGeometry | None = None,
        radius: float = 0.06,
        segments: int = 8,
    ) -> None:
        super().__init__()
        self.atoms = atoms
        self.indices = (
            np.zeros((0,), dtype=np.uint32)
            if indices is None
            else np.asarray(indices, dtype=np.uint32).ravel()
        )
        self.color_fn = color_fn
        self.bond_color = bond_color
        self.color_bonds = color_bonds
        self.geometry = geometry or BondGeometry(radius=radius, segments=segments)

    @property
    def radius(self) -> float:
        """Cylinder radius from :attr:`geometry`."""
        return self.geometry.radius

    @property
    def segments(self) -> int:
        """Radial segment count from :attr:`geometry`."""
        return self.geometry.segments

    def _resolved_color_fn(
        self,
    ) -> Callable[[Any], tuple[float, float, float]]:
        """Return *color_fn*, ``atom.color``, or a chain-palette sampler."""
        if self.color_fn is not None:
            return self.color_fn
        chain_ids = [
            atom.chain_id for atom in self.atoms if getattr(atom, "chain_id", None)
        ]
        chain_fn = make_chain_color_fn(chain_ids) if chain_ids else None

        def color_fn(atom: Any) -> tuple[float, float, float]:
            color = getattr(atom, "color", None)
            if color is not None:
                return (float(color[0]), float(color[1]), float(color[2]))
            if chain_fn is not None:
                return chain_fn(atom)
            return _WHITE

        return color_fn

    def _pair_colors(self, pairs: np.ndarray) -> np.ndarray:
        """Return ``(N, 3)`` RGB, one row per input pair (atom A)."""
        n_pairs = int(pairs.shape[0])
        if not self.color_bonds:
            return np.broadcast_to(
                np.asarray(self.bond_color[:3], dtype=np.float32),
                (n_pairs, 3),
            ).copy()
        color_fn = self._resolved_color_fn()
        return np.asarray(
            [color_fn(self.atoms[int(a)]) for a, _b in pairs],
            dtype=np.float32,
        ).reshape(-1, 3)

    def build_mesh_data(self) -> MeshData:
        """Build oriented cylinder shafts for all finite bond pairs.

        Geometry is generated in one :meth:`BondGeometry.build_many` call.
        Colors are repeated only for shafts that survive the zero-length filter.

        Returns
        -------
        MeshData
            Indexed triangle cylinders with per-shaft colors.
        """
        if not self.atoms or self.indices.size == 0:
            return self._empty_mesh_data(
                elements_per_item=self.geometry.elements_per_item,
                vertices_per_item=self.geometry.vertices_per_item,
            )

        pairs = self.indices.reshape(-1, 2)
        starts = np.asarray(
            [atom_xyz(self.atoms[int(a)]) for a, _b in pairs],
            dtype=np.float64,
        )
        ends = np.asarray(
            [atom_xyz(self.atoms[int(b)]) for _a, b in pairs],
            dtype=np.float64,
        )
        geometry, valid = self.geometry.build_many(starts, ends)
        if geometry.positions.shape[0] == 0:
            return self._empty_mesh_data(
                elements_per_item=self.geometry.elements_per_item,
                vertices_per_item=self.geometry.vertices_per_item,
            )

        colors = np.repeat(
            self._pair_colors(pairs)[valid],
            self.geometry.vertices_per_item,
            axis=0,
        )
        return geometry.with_colors(colors).as_meshdata(
            mode=GLDrawMode.TRIANGLES,
            indexed=True,
            elements_per_item=self.geometry.elements_per_item,
            vertices_per_item=self.geometry.vertices_per_item,
        )

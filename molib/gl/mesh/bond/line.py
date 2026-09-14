"""Indexed GL_LINES mesh for molecular bonds (atom vertices + pair indices)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from molib.entities.coords import atom_xyz
from molib.gl.mesh.molecule import MolecularMesh
from molib.pdb.color import make_chain_color_fn
from picogl.backend.gl.enums import GLDrawMode
from picogl.renderer.mesh_arrays import MeshArrays
from picogl.renderer.meshdata import MeshData

_WHITE = (1.0, 1.0, 1.0)


class BondLinesMesh(MolecularMesh):
    """Build line-stick bonds from atom positions and pair indices.

    Vertices are one per atom. The element buffer stores ``GL_LINES`` pairs
    that index those atoms. Color is per-atom (uniform *bond_color*, or
    *color_fn* / ``atom.color`` when *color_bonds* is true).
    """

    draw_mode = GLDrawMode.LINES

    def __init__(
        self,
        atoms: Sequence[Any],
        indices: np.ndarray | Sequence[int] | None = None,
        *,
        color_fn: Callable[[Any], tuple[float, float, float]] | None = None,
        bond_color: tuple[float, float, float] = _WHITE,
        color_bonds: bool = False,
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
                return float(color[0]), float(color[1]), float(color[2])
            if chain_fn is not None:
                return chain_fn(atom)
            return _WHITE

        return color_fn

    def _build_colors(self, n_atoms: int) -> np.ndarray:
        """Return ``(N, 3)`` RGB, one row per atom."""
        if not self.color_bonds:
            return np.broadcast_to(
                np.asarray(self.bond_color[:3], dtype=np.float32),
                (n_atoms, 3),
            ).copy()
        color_fn = self._resolved_color_fn()
        return np.asarray(
            [color_fn(atom) for atom in self.atoms],
            dtype=np.float32,
        ).reshape(-1, 3)

    def build_mesh_data(self) -> MeshData:
        """Assemble indexed line geometry for the stored atoms and pairs.

        Returns
        -------
        MeshData
            One vertex per atom, ``GL_LINES`` indices, dummy zero normals.
        """
        if not self.atoms:
            return self._empty_mesh_data(
                elements_per_item=2,
                vertices_per_item=1,
                indexed=True,
            )

        positions = np.asarray(
            [atom_xyz(atom) for atom in self.atoms],
            dtype=np.float32,
        ).reshape(-1, 3)
        colors = self._build_colors(int(positions.shape[0]))
        arrays = MeshArrays(
            positions=positions,
            normals=np.zeros_like(positions),
            colors=colors,
            indices=self.indices,
        )
        return arrays.as_meshdata(
            mode=GLDrawMode.LINES,
            indexed=True,
            elements_per_item=2,
        )

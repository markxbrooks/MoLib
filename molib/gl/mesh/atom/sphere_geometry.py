"""Sphere template geometry for a single atom."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from picogl.core.geometry.sphere import unit_sphere_mesh
from picogl.renderer.mesh_arrays import MeshArrays


@dataclass(frozen=True, slots=True)
class AtomSphereGeometry:
    """Geometry parameters for one atom sphere.

    Builds a reusable unit-sphere template. Instancing across atoms is the
    responsibility of :class:`~picogl.renderer.molecular.atoms.AtomsMesh`.
    """

    radius: float = 0.2
    slices: int = 16
    stacks: int = 16

    def build(self) -> MeshArrays:
        """Build a sphere template centered at the origin.

        Returns
        -------
        MeshArrays
            Positions, normals, and indices with no per-atom colors.
        """
        vertices, normals, indices = self._sphere()
        return MeshArrays(
            positions=vertices,
            normals=normals,
            indices=np.asarray(indices, dtype=np.uint32).ravel(),
        )

    @property
    def vertices_per_item(self) -> int:
        """Number of vertices in one sphere instance."""
        return (self.stacks + 1) * (self.slices + 1)

    @property
    def elements_per_item(self) -> int:
        """Number of triangle indices in one sphere instance."""
        return 6 * self.stacks * self.slices

    def _sphere(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(vertices, normals, indices)`` for this tessellation."""
        return unit_sphere_mesh(
            radius=self.radius,
            slices=self.slices,
            stacks=self.stacks,
        )

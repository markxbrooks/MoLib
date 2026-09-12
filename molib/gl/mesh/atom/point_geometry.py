"""One-vertex origin template for rendering an atom as a GL point."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from picogl.renderer.mesh_arrays import MeshArrays


@dataclass(frozen=True, slots=True)
class AtomPointGeometry:
    """Geometry template for rendering an atom as an OpenGL point.

    The template is a single vertex at the origin. World-space placement is
    the responsibility of :class:`~molib.gl.mesh.atom.point.AtomPointsMesh`.
    ``radius`` is metadata for point sizing; it does not affect coordinates.
    """

    radius: float = 0.2

    @property
    def vertices_per_item(self) -> int:
        """Number of vertices in one point instance."""
        return 1

    @property
    def elements_per_item(self) -> int:
        """Number of draw elements in one point instance."""
        return 1

    def build(self) -> MeshArrays:
        """Build the reusable point template centered at the origin.

        Returns
        -------
        MeshArrays
            One zero position and a matching zero normal. No indices.
        """
        return MeshArrays(
            positions=np.zeros((1, 3), dtype=np.float32),
            normals=np.zeros((1, 3), dtype=np.float32),
        )

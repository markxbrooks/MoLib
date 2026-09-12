"""Base class for molecular mesh data and GPU backend adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Any

if TYPE_CHECKING:
    pass

from abc import ABC, abstractmethod

import numpy as np

from picogl.backend.gl.enums import GLDrawMode
from picogl.renderer.draw_spec import MeshDrawInfo
from picogl.renderer.glmesh import GLMesh
from picogl.renderer.legacy_glmesh import LegacyGLMesh
from picogl.renderer.meshdata import MeshData


class MolecularMesh(ABC):
    """
    Domain mesh that builds :class:`~picogl.renderer.meshdata.MeshData` and can
    materialize legacy or modern GPU wrappers.
    """

    draw_mode: GLDrawMode = GLDrawMode.TRIANGLES

    def __init__(self) -> None:
        self._mesh_data: Optional[MeshData] = None
        self._legacy_glmesh: Optional[LegacyGLMesh] = None

    @abstractmethod
    def build_mesh_data(self) -> MeshData:
        """Construct mesh arrays for this molecular primitive."""

    def _empty_mesh_data(
        self,
        *,
        elements_per_item: int,
        vertices_per_item: int,
        indexed: bool = True,
    ) -> MeshData:
        """Return an empty mesh with per-item draw strides.

        Parameters
        ----------
        elements_per_item
            Indices (or vertices, when unindexed) per logical item.
        vertices_per_item
            Vertices per logical item.
        indexed
            When ``True``, include an empty element buffer.
        """
        if indexed:
            data = MeshData.from_raw(
                vertices=np.zeros((0, 3), dtype=np.float32),
                indices=np.zeros((0,), dtype=np.uint32),
            )
        else:
            data = MeshData.from_raw(
                vertices=np.zeros((0, 3), dtype=np.float32),
            )
        data.draw_info = MeshDrawInfo(
            mode=self.draw_mode,
            indexed=indexed,
            elements_per_item=elements_per_item,
            vertices_per_item=vertices_per_item,
        )
        return data

    def to_mesh_data(self) -> MeshData:
        """Return cached :class:`MeshData`, building on first access."""
        if self._mesh_data is None:
            self._mesh_data = self.build_mesh_data()
        return self._mesh_data

    def to_legacy_glmesh(self, *, upload: bool = True) -> LegacyGLMesh:
        """
        Build or return a cached :class:`~picogl.renderer.legacy_glmesh.LegacyGLMesh`.

        Parameters
        ----------
        upload :
            When ``True``, upload GPU buffers immediately.
        """
        if self._legacy_glmesh is None:
            from picogl.renderer.legacy_glmesh import LegacyGLMesh

            self._legacy_glmesh = LegacyGLMesh.from_mesh_data(self.to_mesh_data())
        if upload:
            self._legacy_glmesh.upload()
        return self._legacy_glmesh

    def to_glmesh(self, *, upload: bool = True, **kwargs: Any) -> GLMesh:
        """
        Build a :class:`~picogl.renderer.glmesh.GLMesh` from this mesh data.

        Extra keyword arguments are forwarded to
        :meth:`~picogl.renderer.glmesh.GLMesh.from_mesh_data`.
        """
        from picogl.renderer.glmesh import GLMesh

        mesh = GLMesh.from_mesh_data(self.to_mesh_data(), **kwargs)
        if upload:
            mesh.upload()
        return mesh

    def draw_legacy(self, mode: Optional[GLDrawMode] = None) -> None:
        """Draw via a cached legacy GL mesh using this mesh's default mode."""
        self.to_legacy_glmesh(upload=True).draw(mode or self.draw_mode)

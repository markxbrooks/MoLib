from __future__ import annotations

from typing import Protocol


class Vector3Like(Protocol):
    """3D point or direction with scalar components (e.g. molib ``Point3D``)."""

    x: float
    y: float
    z: float

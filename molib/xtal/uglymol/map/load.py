"""Unified entry point for uglymol density-map loaders.

CCP4/DSN6 uglymol → :class:`~molib.xtal.uglymol.map.elmap.ElMap`.
Gemmi MTZ/CCP4 → :class:`~molib.xtal.map.helper.DensityMapData` → MapInfo.
"""

from __future__ import annotations

from typing import Any

from molib.xtal.uglymol.map.ccp4_loader import Ccp4MapLoader
from molib.xtal.uglymol.map.dsn6_loader import Dsn6MapLoader
from molib.xtal.uglymol.map.elmap import ElMap


def load_map(buffer: Any, format: str = "ccp4", **kwargs) -> ElMap:
    """Load a density map buffer into an :class:`ElMap`.

    :param buffer: Raw map file bytes
    :param format: ``\"ccp4\"`` or ``\"dsn6\"``
    :param kwargs: Forwarded to the format loader (e.g. ``expand_symmetry``)
    :return: Populated ElMap
    """
    fmt = str(format).strip().lower()
    if fmt in {"ccp4", "mrc", "map"}:
        return Ccp4MapLoader().load(buffer, **kwargs)
    if fmt == "dsn6":
        return Dsn6MapLoader().load(buffer)
    raise ValueError(f"Unsupported map format: {format!r}")

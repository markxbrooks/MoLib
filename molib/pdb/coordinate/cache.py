"""
CoordinateData cache for fast reload of the same PDB.

When the same PDB file is loaded again with the same coordinate parameters,
returns cached CoordinateData instead of regenerating from PandasPdb.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from decologr import Decologr as log
from molib.pdb.coordinate.data import CoordinateData
from molib.pdb.coordinate.generator import generate_coordinate_data

# In-memory cache: key -> CoordinateData. Key = (resolved_path, params_key).
_COORD_CACHE: dict[tuple[str, str], CoordinateData] = {}
_CACHE_MAX_SIZE = 32
_CACHE_ORDER: list[tuple[str, str]] = []  # FIFO order for eviction


def _cache_key(path: str, **coord_params: Any) -> tuple[str, str]:
    """Build a cache key from resolved path and coordinate params."""
    resolved = os.path.realpath(os.path.abspath(path))
    params_key = ",".join(f"{k}={coord_params.get(k)!r}" for k in sorted(coord_params))
    return (resolved, params_key)


def get_coordinate_data_cached(
    path: str,
    pdb_pandas,
    **coord_params: Any,
) -> Optional[CoordinateData]:
    """
    Return CoordinateData for the given path and params, using cache when possible.

    On cache miss, generates via generate_coordinate_data(pdb_pandas, **coord_params),
    stores in cache, and returns. On cache hit, returns the cached instance.

    Args:
        path: PDB file path (used for cache key; should match the source of pdb_pandas).
        pdb_pandas: PandasPdb (or compatible) instance.
        **coord_params: Arguments passed to generate_coordinate_data
            (include_atom, include_hetatm, chain_from, atom_filter, etc.).

    Returns:
        CoordinateData or None if generation returns None.
    """
    key = _cache_key(path, **coord_params)
    if key in _COORD_CACHE:
        log.message("CoordinateData cache hit", level=logging.DEBUG)
        return _COORD_CACHE[key]

    coord = generate_coordinate_data(pdb_pandas, **coord_params)
    if coord is None:
        return None

    # Evict oldest if at capacity
    while len(_COORD_CACHE) >= _CACHE_MAX_SIZE and _CACHE_ORDER:
        old_key = _CACHE_ORDER.pop(0)
        _COORD_CACHE.pop(old_key, None)

    _COORD_CACHE[key] = coord
    _CACHE_ORDER.append(key)
    return coord


def clear_coordinate_cache() -> None:
    """Clear the CoordinateData cache (e.g. when closing all structures)."""
    _COORD_CACHE.clear()
    _CACHE_ORDER.clear()


def get_cache_size() -> int:
    """Return current number of cached CoordinateData entries."""
    return len(_COORD_CACHE)

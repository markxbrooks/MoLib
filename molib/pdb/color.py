"""
Module for accessing colors from color_array maps
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from decologr import Decologr as log
from molib.core.color import ColorMap
from molib.entities.atom import Atom3D


def get_atom_color(name: str) -> tuple[float, float, float]:
    """
    get_atom_color

    :param name: str
    :return: tuple[float, float, float] color_array
    """
    color = ColorMap.atom_colors.get(name, (0.5, 0.5, 0.5))
    return color


def get_ss_color(ss: str) -> tuple[float, float, float]:
    """
    get_ss_color

    :param ss: str
    :return: tuple[float, float, float] color_array
    """
    color = ColorMap.secondary_structure_color_map.get(
        ss, ColorMap.secondary_structure_color_map[" "]
    )
    return color


def palette_rgb_at(chain_index: int) -> tuple[float, float, float]:
    """
    RGB for the *chain_index*-th slot in the shared chain palette.

    Uses :attr:`ColorMap.colors` (same source as :func:`generate_chain_colors`).
    Indices wrap so more chains than palette entries stay supported.
    """
    palette = ColorMap.colors
    if not palette:
        return 1.0, 1.0, 1.0
    row = palette[chain_index % len(palette)]
    return float(row[0]), float(row[1]), float(row[2])


def rgb_for_chain_id_among(
    chain_id: str, chain_ids: list[str]
) -> tuple[float, float, float]:
    """
    Chain color for *chain_id* using **sorted unique** *chain_ids* ordering.

    Matches :func:`generate_chain_colors` when *chain_ids* lists every chain in
    the structure. If *chain_id* is missing from *chain_ids*, it is included so
    the result is still defined.
    """
    if not chain_ids:
        return palette_rgb_at(0)
    unique_sorted = sorted(set(chain_ids) | {chain_id})
    idx = unique_sorted.index(chain_id)
    return palette_rgb_at(idx)


def make_chain_color_fn(
    chain_ids: Sequence[str],
) -> Callable[[Any], tuple[float, float, float]]:
    """Create a per-atom color function that assigns colors by chain ID.

    Chain IDs are sorted and deduplicated so that the resulting colors are
    deterministic and match :func:`generate_chain_colors` when ``chain_ids``
    contains every chain in the structure.
    """
    unique_sorted = sorted(set(chain_ids))
    color_map = {
        chain_id: palette_rgb_at(index)
        for index, chain_id in enumerate(unique_sorted)
    }

    def color_fn(atom: Atom3D) -> tuple[float, float, float]:
        """Return the palette color corresponding to ``atom.chain_id``."""
        chain_id = atom.chain_id
        if chain_id not in color_map:
            color_map[chain_id] = palette_rgb_at(len(color_map))
        return color_map[chain_id]

    return color_fn


def generate_chain_colors(chain_ids: list) -> dict:
    """
    Build ``chain_id -> (r, g, b)`` using :func:`palette_rgb_at` on **sorted**
    unique chain IDs (stable, matches :meth:`Atom3D._color_by_chain`).
    """
    unique_sorted = sorted(set(chain_ids))
    chain_colors = {cid: palette_rgb_at(i) for i, cid in enumerate(unique_sorted)}
    log.message(f"self.chain_colors: {chain_colors}", silent=True)
    return chain_colors

"""
This module provides multiple implementations of color providers for molecular
modeling purposes. The color providers can assign colors at various levels:
per atom, per chain, or based on secondary structure.

Classes:
- ColorProvider: Protocol defining the interface for color providers.
- PerCAColorProvider: Assigns colors on a per-atom basis using a given array of colors.
- PerChainColorProvider: Provides colors based on chain identifiers.
- SecondaryStructureColorProvider: Assigns colors based on secondary structure information.
"""

from typing import Protocol
import numpy as np

from molib.core import ColorMap


class ColorProvider(Protocol):
    """Color Provider"""
    def get_color(self, index: int, chain_id: str) -> tuple[float, float, float]:
        ...

class PerCAColorProvider:
    """Per CA Color Provider"""
    def __init__(self, colors: np.ndarray):
        self.colors = colors

    def get_color(self, index: int, chain_id: str):
        return tuple(self.colors[index])

class PerChainColorProvider:
    """Per chain color provider"""
    def __init__(self, chain_colors: dict[str, tuple[float, float, float]]):
        self.chain_colors = chain_colors

    def get_color(self, index: int, chain_id: str):
        cid = str(chain_id).strip()
        return self.chain_colors.get(cid, (1.0, 1.0, 1.0))


class SecondaryStructureColorProvider:
    """Secondary Structure Provider"""
    def __init__(self, mol3d):
        """constructor"""
        self.residues = list(mol3d.get_ribbon_guide_residues())

    def get_color(self, index: int, chain_id: str):
        res = self.residues[index]
        ss = getattr(res, "secstruc", " ")
        return ColorMap.secondary_structure_color_map.get(
            ss,
            ColorMap.secondary_structure_color_map[" "],
        )

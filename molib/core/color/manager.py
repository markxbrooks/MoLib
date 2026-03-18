"""
Color Manager
"""

from __future__ import annotations

import numpy as np


class ChainColorManager:
    """
    Singleton managing chain → colour mapping.
    Provides lookup and update methods.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, default_color=(0.6, 0.6, 0.6)):
        if not hasattr(self, "_initialized"):  # avoid reinitializing singleton
            self._colors: dict[str, tuple[float, float, float]] = {}
            self.default_color = default_color
            self._initialized = True

    def set_color_map(self, color_map: dict[str, tuple[float, float, float]]):
        """Assign a colour mapping."""
        self._colors = color_map

    def set_color(self, chain_id: str, color: tuple[float, float, float]):
        """Assign a specific colour to a chain."""
        self._colors[chain_id] = color

    def get_color(self, chain_id: str) -> "np.ndarray":
        """Get colour for a chain, or default if not set."""
        return np.array(
            self._colors.get(chain_id, self.default_color), dtype=np.float32
        )

    def get_all(self) -> dict[str, tuple[float, float, float]]:
        """Return the full mapping."""
        return dict(self._colors)

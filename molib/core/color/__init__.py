"""
Provides functionality for handling color mapping, color schemes, and color management.

This module imports and integrates classes from map, strategy, and manager modules to
facilitate operations involving color mapping, applying color schemes, and managing
color chains.

Imported Classes:
- ColorMap: Used for handling color mapping functionality.
- ColorScheme: Provides strategies for applying specific color schemes.
- ChainColorManager: Manages chains of colors to coordinate various elements.
"""

from .manager import ChainColorManager
from .map import ColorMap
from .strategy import ColorScheme

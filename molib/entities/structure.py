"""
Structure3D class
"""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np

from molib.entities.secondary_structure_type import SecondaryStructureType


class Structure3D:
    """Base class for 3D structural elements - Pure Python class for performance."""

    def __init__(
        self,
        name: str = "",
        type: str = "",
        parent: Optional["Structure3D"] = None,
        color: Optional["np.ndarray"] = None,
        selected: bool = False,
        visible: bool = True,
        secstruc: SecondaryStructureType = SecondaryStructureType.COIL,
        chain_id: str = "A",
        coords: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        next: Optional["Structure3D"] = None,
        prev: Optional["Structure3D"] = None,
    ):
        self.name = name
        self.type = type
        self.parent = parent
        self.color = color
        self.selected = selected
        self.visible = visible
        self.secstruc = secstruc
        self.chain_id = chain_id
        self.coords = coords
        self.next = next
        self.prev = prev

        # Lazy initialization for performance
        if self.color is None:
            self.color = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    @property
    def pos(self) -> "np.ndarray":
        """Return coordinates as a NumPy array."""
        return np.array(self.coords, dtype=np.float32)

    def __iter__(self) -> Iterator["Structure3D"]:
        current = self
        while current is not None:
            yield current
            current = current.next

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __getitem__(self, index: int) -> "Structure3D":
        if index < 0:
            index = len(self) + index
        for i, item in enumerate(self):
            if i == index:
                return item
        raise IndexError("Structure3D index out of range")

    def __reversed__(self) -> Iterator["Structure3D"]:
        return reversed(list(self))

    def set_color(self, r: float, g: float, b: float) -> None:
        """Set colour in-place."""
        self.color[:] = (r, g, b)

    def as_serializable_dict(self) -> dict:
        """Return a dict safe for JSON serialization."""
        return {
            "color_scheme": self.name,
            "type": self.type,
            "colour": self.color.tolist(),
            "visible": self.visible,
            "secstruc": self.secstruc,
            "chain_id": self.chain_id,
            "coords": self.coords,
        }

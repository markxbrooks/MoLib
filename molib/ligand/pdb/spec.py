"""
This module provides a class to define and parse specifications
for lines in a PDB (Protein Data Bank) format. It enables parsing
specific substrings from a line based on defined start and stop
positions.

The module contains a single class, `PDBLineSpec`, which facilitates
the creation of such specifications and provides functionality to
extract substrings using these specifications.
"""

from dataclasses import dataclass
from typing import Callable, Any


@dataclass(frozen=True)
class PDBLineSpec:
    name: str
    start_pos: int
    stop_pos: int
    converter: Callable[[str], Any] | None = None

    def parse(self, line: str):
        raw = line[self.start_pos:self.stop_pos].strip()
        if self.converter and raw:
            return self.converter(raw)
        return raw

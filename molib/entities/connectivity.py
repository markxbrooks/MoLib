"""
Connectivity helpers for Bond3D.

These utilities operate on Bond3D objects to provide:
- adjacency information (atom -> bonds)
- fragment detection (connected components)

They are written in pure Python as a first step; the logic can later be moved
to a compiled core while preserving the same public API.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Set

from molib.entities.atom import Atom3D
from molib.entities.bond import Bond3D


def build_bond_adjacency(bonds: Iterable[Bond3D]) -> Dict[Atom3D, List[Bond3D]]:
    """
    Build an adjacency mapping: Atom3D -> list of incident Bond3D objects.
    """
    adjacency: Dict[Atom3D, List[Bond3D]] = {}
    for bond in bonds:
        adjacency.setdefault(bond.atom1, []).append(bond)
        adjacency.setdefault(bond.atom2, []).append(bond)
    return adjacency


def find_fragments(bonds: Iterable[Bond3D]) -> List[Set[Atom3D]]:
    """
    Find connected components (fragments) in the bond graph.

    Returns a list of fragments, where each fragment is a set of Atom3D objects.
    """
    adjacency = build_bond_adjacency(bonds)
    visited: Set[Atom3D] = set()
    fragments: List[Set[Atom3D]] = []

    for atom in adjacency.keys():
        if atom in visited:
            continue
        # Depth-first search to collect a fragment
        stack = [atom]
        fragment: Set[Atom3D] = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            fragment.add(current)
            for bond in adjacency.get(current, []):
                neighbor = bond.other(current)
                if neighbor not in visited:
                    stack.append(neighbor)
        if fragment:
            fragments.append(fragment)

    return fragments


__all__ = [
    "build_bond_adjacency",
    "find_fragments",
]

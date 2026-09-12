"""Per-chain Cα CPU geometry (unindexed line-strip traces)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from molib.core.constants import MoLibConstant
from picogl.renderer.mesh_arrays import MeshArrays

CalphaColorFn = Callable[
    [Sequence[Any], dict[str, tuple[float, float, float]], str],
    tuple[np.ndarray, np.ndarray],
]


class CalphaMeshBuilder:
    """Build per-chain Cα :class:`~picogl.renderer.mesh_arrays.MeshArrays`.

    Grouping, Cα selection, and ordering live here. Coloring is injected via
    *color_fn* so ElMo display policy (chain vs domain, selection white)
    stays outside MoLib.
    """

    def __init__(
        self,
        chain_colors: dict[str, tuple[float, float, float]],
        color_fn: CalphaColorFn,
    ) -> None:
        self.chain_colors = chain_colors
        self.color_fn = color_fn

    def build(self, atoms: Sequence[Any]) -> dict[str, MeshArrays]:
        """Return one unindexed mesh per chain that has at least two Cα atoms.

        :param atoms: Mixed-name atoms; non-Cα residues are ignored
        :return: ``chain_id -> MeshArrays`` with dummy zero normals
        """
        meshes: dict[str, MeshArrays] = {}
        for chain_id, chain_atoms in self._atoms_by_chain(atoms).items():
            ca_atoms = self._ca_atoms(chain_atoms)
            if len(ca_atoms) < 2:
                continue
            colors, positions = self.color_fn(
                ca_atoms,
                self.chain_colors,
                chain_id,
            )
            positions_a = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
            colors_a = np.asarray(colors, dtype=np.float32).reshape(-1, 3)
            meshes[chain_id] = MeshArrays(
                positions=positions_a,
                normals=np.zeros_like(positions_a),
                colors=colors_a,
            )
        return meshes

    @staticmethod
    def _atoms_by_chain(atoms: Sequence[Any]) -> dict[str, list[Any]]:
        """Group *atoms* by ``chain_id``."""
        atoms_by_chain: dict[str, list[Any]] = defaultdict(list)
        for atom in atoms:
            atoms_by_chain[atom.chain_id].append(atom)
        return atoms_by_chain

    @staticmethod
    def _ca_atoms(atoms: Sequence[Any]) -> list[Any]:
        """Keep Cα atoms and sort by parent residue number."""
        ca_name = MoLibConstant.PEPTIDE_CHAIN_ATOMNAME
        ca_atoms = [
            atom for atom in atoms if getattr(atom, "name", "") == ca_name
        ]
        ca_atoms.sort(
            key=lambda atom: int(
                getattr(getattr(atom, "parent", None), "residue_number", 0) or 0
            )
        )
        return ca_atoms

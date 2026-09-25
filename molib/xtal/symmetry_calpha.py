"""Closest crystallographic symmetry mates as Cα traces.

Generates packing-related Cα coordinates near the asymmetric unit by applying
space-group operations and nearby lattice translations (unlike the older mate
generator that folded fractional coords into ``[0, 1)`` and missed adjacent cells).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gemmi
import numpy as np
from decologr import Decologr as log

SYMMETRY_MATES_CONTACT_DISTANCE = 8.0

SYMMETRY_MATES_MAXIMUM_MATES = 50


@dataclass(frozen=True)
class SymmetryCalphaTrace:
    """One chain-level Cα trace for a contacting symmetry mate."""

    mate_id: int
    chain_id: str
    positions: np.ndarray  # shape (N, 3), float32, ordered by residue
    min_distance: float
    operation_index: int
    lattice: Tuple[int, int, int]


def _extract_ca_by_chain(model: gemmi.Model) -> Dict[str, np.ndarray]:
    """Return Cα orthogonal coordinates keyed by chain name."""
    by_chain: Dict[str, list[np.ndarray]] = {}
    for chain in model:
        coords: list[np.ndarray] = []
        for residue in chain:
            ca = residue.find_atom("CA", "*")
            if ca is None:
                continue
            coords.append(np.array([ca.pos.x, ca.pos.y, ca.pos.z], dtype=np.float64))
        if coords:
            by_chain[chain.name] = np.asarray(coords, dtype=np.float64)
    return by_chain


def _min_pairwise_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Minimum Euclidean distance between two CA clouds."""
    if a.size == 0 or b.size == 0:
        return float("inf")
    # Chunk to keep memory bounded for large chains.
    chunk = 256
    best = float("inf")
    for i in range(0, a.shape[0], chunk):
        block = a[i : i + chunk]
        diff = block[:, None, :] - b[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=-1)).min()
        if dist < best:
            best = float(dist)
    return best


def _transform_ca_cloud(
    coords: np.ndarray,
    cell: gemmi.UnitCell,
    op: gemmi.Op,
    lattice: Tuple[int, int, int],
) -> np.ndarray:
    """Apply symmetry op + lattice translation to a ``(N, 3)`` orthogonal CA array."""
    tx, ty, tz = lattice
    out = np.empty_like(coords)
    for i, (x, y, z) in enumerate(coords):
        frac = cell.fractionalize(gemmi.Position(float(x), float(y), float(z)))
        sym = op.apply_to_xyz([frac.x, frac.y, frac.z])
        orth = cell.orthogonalize(
            gemmi.Fractional(sym[0] + tx, sym[1] + ty, sym[2] + tz)
        )
        out[i] = (orth.x, orth.y, orth.z)
    return out


def generate_closest_calpha_mates(
    structure: gemmi.Structure,
    *,
    contact_distance: float = SYMMETRY_MATES_CONTACT_DISTANCE,
    max_mates: int = SYMMETRY_MATES_MAXIMUM_MATES,
    lattice_range: int = 1,
) -> List[SymmetryCalphaTrace]:
    """Generate contacting symmetry-mate Cα traces closest to the ASU.

    :param structure: gemmi structure with cell and space group
    :param contact_distance: Keep mates whose nearest CA is within this distance (Å)
    :param max_mates: Maximum number of mate *molecules* (all chains of a mate share
        one mate_id); traces are returned per chain
    :param lattice_range: Inclusive lattice search radius (``(-R..R)³``)
    :return: Ordered list of Cα traces (closest mates first)
    """
    if not structure or len(structure) == 0:
        return []

    cell = structure.cell
    space_group = structure.spacegroup_hm
    sg = gemmi.find_spacegroup_by_name(space_group) if space_group else None
    if sg is None:
        log.warning(f"No space group for closest Cα mates: {space_group!r}")
        return []

    original = structure[0]
    ca_by_chain = _extract_ca_by_chain(original)
    if not ca_by_chain:
        log.warning("No Cα atoms found for closest symmetry mates")
        return []

    original_all = np.concatenate(list(ca_by_chain.values()), axis=0)
    ops = list(sg.operations())
    lattices = [
        (tx, ty, tz)
        for tx in range(-lattice_range, lattice_range + 1)
        for ty in range(-lattice_range, lattice_range + 1)
        for tz in range(-lattice_range, lattice_range + 1)
    ]

    candidates: list[tuple[float, int, Tuple[int, int, int], Dict[str, np.ndarray]]] = []

    for op_index, op in enumerate(ops):
        for lattice in lattices:
            if op_index == 0 and lattice == (0, 0, 0):
                continue

            transformed: Dict[str, np.ndarray] = {}
            for chain_id, coords in ca_by_chain.items():
                transformed[chain_id] = _transform_ca_cloud(coords, cell, op, lattice)

            mate_all = np.concatenate(list(transformed.values()), axis=0)
            min_dist = _min_pairwise_distance(mate_all, original_all)
            if min_dist <= float(contact_distance):
                candidates.append((min_dist, op_index, lattice, transformed))

    candidates.sort(key=lambda item: item[0])
    selected = candidates[: max(0, int(max_mates))]

    traces: List[SymmetryCalphaTrace] = []
    for mate_id, (min_dist, op_index, lattice, transformed) in enumerate(
        selected, start=1
    ):
        for chain_id, positions in transformed.items():
            if positions.shape[0] < 2:
                continue
            traces.append(
                SymmetryCalphaTrace(
                    mate_id=mate_id,
                    chain_id=chain_id,
                    positions=np.asarray(positions, dtype=np.float32),
                    min_distance=float(min_dist),
                    operation_index=op_index,
                    lattice=lattice,
                )
            )

    log.info(
        f"Closest Cα mates: {len(selected)} molecules / {len(traces)} chain traces "
        f"(contact ≤ {contact_distance:.1f} Å, SG {space_group})"
    )
    return traces


def generate_closest_calpha_mates_from_pdb(
    pdb_path: str,
    *,
    contact_distance: float = SYMMETRY_MATES_CONTACT_DISTANCE,
    max_mates: int = SYMMETRY_MATES_MAXIMUM_MATES,
    lattice_range: int = 1,
) -> Tuple[List[SymmetryCalphaTrace], Dict[str, Any]]:
    """Load *pdb_path* and return closest Cα mate traces plus symmetry metadata.

    :param pdb_path: Path to a PDB/mmCIF file readable by gemmi
    :param contact_distance: Contact cutoff in Å
    :param max_mates: Maximum contacting mate molecules
    :param lattice_range: Lattice search radius
    :return: ``(traces, symmetry_info)``
    """
    structure = gemmi.read_structure(pdb_path)
    from molib.xtal.symmetry import SymmetryMatesGenerator

    info = SymmetryMatesGenerator().get_symmetry_info(structure)
    traces = generate_closest_calpha_mates(
        structure,
        contact_distance=contact_distance,
        max_mates=max_mates,
        lattice_range=lattice_range,
    )
    info["calpha_mate_count"] = len({t.mate_id for t in traces})
    info["calpha_trace_count"] = len(traces)
    return traces, info


def traces_to_atom_like(
    traces: Sequence[SymmetryCalphaTrace],
    *,
    color: Tuple[float, float, float] = (0.55, 0.75, 0.95),
) -> list[Any]:
    """Convert traces to lightweight atom-like objects for Cα mesh builders.

    Each atom has ``name``, ``chain_id``, ``coords``/``pos``, ``color``, and
    ``parent`` with ``residue_number`` so
    :class:`~molib.gl.mesh.calpha.CalphaMeshBuilder` and ElMo's
    ``calpha_color_by_chain`` work.

    :param traces: Symmetry Cα traces
    :param color: RGB colour shared by all mate Cα atoms
    :return: Flat list of atom-like objects
    """
    from types import SimpleNamespace

    atoms: list[Any] = []
    for trace in traces:
        key = f"sym{trace.mate_id}_{trace.chain_id}"
        for i, pos in enumerate(trace.positions):
            xyz = np.asarray(pos, dtype=np.float32).reshape(3)
            atoms.append(
                SimpleNamespace(
                    name="CA",
                    chain_id=key,
                    coords=tuple(float(v) for v in xyz),
                    pos=xyz,
                    color=color,
                    selected=False,
                    parent=SimpleNamespace(residue_number=i + 1, selected=False),
                )
            )
    return atoms

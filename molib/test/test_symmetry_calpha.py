"""Tests for closest crystallographic Cα symmetry mates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gemmi")
pytest.importorskip("molib")

from molib.xtal.symmetry_calpha import (
    generate_closest_calpha_mates,
    generate_closest_calpha_mates_from_pdb,
    traces_to_atom_like,
)


def _tiny_p212121_structure():
    """Build a tiny P212121 structure with a short CA trace."""
    import gemmi

    st = gemmi.Structure()
    st.cell = gemmi.UnitCell(40.0, 50.0, 60.0, 90.0, 90.0, 90.0)
    st.spacegroup_hm = "P 21 21 21"
    model = gemmi.Model(0)
    chain = gemmi.Chain("A")
    for i, (x, y, z) in enumerate(
        [(5.0, 5.0, 5.0), (8.8, 5.0, 5.0), (12.6, 5.0, 5.0)], start=1
    ):
        res = gemmi.Residue()
        res.name = "ALA"
        res.seqid = gemmi.SeqId(str(i), " ")
        atom = gemmi.Atom()
        atom.name = "CA"
        atom.element = gemmi.Element("C")
        atom.pos = gemmi.Position(x, y, z)
        res.add_atom(atom)
        chain.add_residue(res)
    model.add_chain(chain)
    st.add_model(model)
    return st


def test_generate_closest_calpha_mates_finds_packing_neighbors():
    traces = generate_closest_calpha_mates(
        _tiny_p212121_structure(),
        contact_distance=25.0,
        max_mates=20,
        lattice_range=1,
    )
    assert traces
    assert all(t.positions.ndim == 2 and t.positions.shape[1] == 3 for t in traces)
    assert all(t.positions.shape[0] == 3 for t in traces)
    # Identity at (0,0,0) must not appear
    assert all(not (t.operation_index == 0 and t.lattice == (0, 0, 0)) for t in traces)
    # Sorted by distance
    mate_dists = []
    seen = set()
    for t in traces:
        if t.mate_id not in seen:
            mate_dists.append(t.min_distance)
            seen.add(t.mate_id)
    assert mate_dists == sorted(mate_dists)


def test_traces_to_atom_like_uses_calpha_mesh_contract():
    traces = generate_closest_calpha_mates(
        _tiny_p212121_structure(),
        contact_distance=25.0,
        max_mates=2,
    )
    atoms = traces_to_atom_like(traces)
    assert atoms
    assert all(a.name == "CA" for a in atoms)
    assert all(str(a.chain_id).startswith("sym") for a in atoms)
    assert all(hasattr(a.parent, "residue_number") for a in atoms)
    assert all(len(a.coords) == 3 for a in atoms)
    positions = np.array([a.coords for a in atoms], dtype=np.float32)
    assert positions.ndim == 2 and positions.shape[1] == 3


def test_generate_from_2vug_pdb_smoke():
    pdb = Path("/Users/brooks/projects/ElMo/elmo/test_data/2VUG_final.pdb")
    if not pdb.is_file():
        pytest.skip("2VUG_final.pdb not available")
    traces, info = generate_closest_calpha_mates_from_pdb(
        str(pdb),
        contact_distance=8.0,
        max_mates=12,
    )
    assert "space_group" in info
    assert info.get("calpha_trace_count", 0) == len(traces)
    if traces:
        assert np.isfinite(traces[0].positions).all()

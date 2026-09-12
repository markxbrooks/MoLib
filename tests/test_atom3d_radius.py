"""Tests for Atom3D radius population (explicit PQR vs van der Waals)."""

from __future__ import annotations

import pandas as pd
from molib.entities.atom import Atom3D
from molib.ligand.element import DEFAULT_VDW_RADIUS, vdw_radius_for_element
from molib.pdb.coordinate.data import CoordinateData
from molib.pdb.molscript.parser import parse_pdb_atoms_to_mol3d


def test_atom3d_uses_vdw_radius_from_element() -> None:
    atom = Atom3D(name="CA", element="C", coords=(0.0, 0.0, 0.0))
    assert atom.radius == vdw_radius_for_element("C")
    assert atom.radius == 1.70


def test_atom3d_infers_element_from_atom_name() -> None:
    atom = Atom3D(name="CA", coords=(0.0, 0.0, 0.0))
    assert atom.radius == vdw_radius_for_element("C")


def test_atom3d_keeps_explicit_pqr_radius() -> None:
    atom = Atom3D(name="N", element="N", radius=1.824, coords=(0.0, 0.0, 0.0))
    assert atom.radius == 1.824


def test_atom3d_invalid_radius_falls_back_to_vdw() -> None:
    atom = Atom3D(name="O", element="O", radius=0.0, coords=(0.0, 0.0, 0.0))
    assert atom.radius == vdw_radius_for_element("O")
    unknown = Atom3D(name="Xx", element="Xx", coords=(0.0, 0.0, 0.0))
    assert unknown.radius == DEFAULT_VDW_RADIUS


def test_from_pdb_line_sets_element_and_vdw_radius() -> None:
    line = (
        "ATOM      1  N   MET A   1      27.340  24.430   2.614  1.00 11.26"
        "           N  "
    )
    atom = Atom3D().from_pdb_line(line)
    assert atom.name.strip() == "N"
    assert (atom.element or "").strip() == "N"
    assert atom.radius == vdw_radius_for_element("N")
    assert atom.b_factor == 11.26


def test_from_pdb_line_reads_pqr_radius() -> None:
    line = (
        "ATOM      1  N   MET     1      27.360  24.230   2.614  0.1592 1.8240"
    )
    atom = Atom3D().from_pdb_line(line)
    assert atom.radius == 1.8240


def _atom_df(**extra) -> pd.DataFrame:
    data = {
        "chain_id": ["A", "A"],
        "residue_number": [1, 1],
        "atom_name": ["N", "CA"],
        "residue_name": ["ALA", "ALA"],
        "x_coord": [0.0, 1.5],
        "y_coord": [0.0, 0.0],
        "z_coord": [0.0, 0.0],
        "element_symbol": ["N", "C"],
        "b_factor": [10.0, 12.0],
        "alt_loc": ["", ""],
        "segment_id": ["", ""],
    }
    data.update(extra)
    return pd.DataFrame(data)


def test_parse_pdb_atoms_to_mol3d_populates_vdw_radius() -> None:
    mol = parse_pdb_atoms_to_mol3d(_atom_df(), CoordinateData())
    residue = mol.models[0].chains["A"].residues[0]
    assert residue.atoms["N"].radius == vdw_radius_for_element("N")
    assert residue.atoms["CA"].radius == vdw_radius_for_element("C")


def test_parse_pdb_atoms_to_mol3d_uses_radius_column() -> None:
    mol = parse_pdb_atoms_to_mol3d(
        _atom_df(radius=[1.1, 2.2], charge=[0.1, 0.0]),
        CoordinateData(),
    )
    residue = mol.models[0].chains["A"].residues[0]
    assert residue.atoms["N"].radius == 1.1
    assert residue.atoms["CA"].radius == 2.2
    assert residue.atoms["N"].pqr_charge == 0.1

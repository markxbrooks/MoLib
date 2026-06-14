"""Tests for Connolly surface resource behavior."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from molib.calc.geometry import connolly_surface as cs


def test_connolly_resource_key_is_deterministic_for_normalized_arrays():
    spec = cs.ConnollySurfaceSpec(grid_spacing=0.8, smoothing=False)
    positions = [[0, 0, 0], [1, 2, 3]]
    radii = [1.5, 1.7]

    r1 = cs.ConnollySurfaceResource(positions, radii, spec)
    r2 = cs.ConnollySurfaceResource(
        np.asarray(positions, dtype=np.float64),
        np.asarray(radii, dtype=np.float64),
        spec,
    )

    assert r1.compute_key() == r2.compute_key()


def test_connolly_resource_key_changes_when_spec_changes():
    positions = np.asarray([[0, 0, 0], [1, 2, 3]], dtype=np.float32)
    radii = np.asarray([1.5, 1.7], dtype=np.float32)

    r1 = cs.ConnollySurfaceResource(
        positions, radii, cs.ConnollySurfaceSpec(grid_spacing=0.8)
    )
    r2 = cs.ConnollySurfaceResource(
        positions, radii, cs.ConnollySurfaceSpec(grid_spacing=1.0)
    )

    assert r1.compute_key() != r2.compute_key()


def test_connolly_resource_uses_lazy_global_geometry_cache():
    cs.clear_connolly_surface_cache()
    spec = cs.ConnollySurfaceSpec(grid_spacing=1.0, smoothing=False)
    positions = np.asarray([[0, 0, 0], [3, 0, 0]], dtype=np.float32)
    radii = np.asarray([1.7, 1.7], dtype=np.float32)
    vertices = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.asarray([[0, 1, 2]], dtype=np.uint32)
    normals = np.asarray([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32)

    with patch.object(
        cs,
        "calculate_connolly_surface",
        return_value=(vertices, faces, normals),
    ) as calculate:
        mesh1 = cs.ConnollySurfaceResource(positions, radii, spec).get_mesh()
        mesh2 = cs.ConnollySurfaceResource(positions, radii, spec).get_mesh()

    assert calculate.call_count == 1
    assert mesh1 is mesh2
    np.testing.assert_array_equal(mesh1.vertices, vertices)
    unpacked_vertices, unpacked_faces, unpacked_normals = mesh1
    assert unpacked_vertices is mesh1.vertices
    assert unpacked_faces is mesh1.faces
    assert unpacked_normals is mesh1.normals


def test_connolly_resource_from_molecule_extracts_positions_and_radii():
    atom_c = SimpleNamespace(pos=np.asarray([0.0, 0.0, 0.0]), element="C")
    atom_o = SimpleNamespace(pos=np.asarray([1.0, 0.0, 0.0]), element="O")
    residue = SimpleNamespace(atoms={"C": atom_c, "O": atom_o})
    chain = SimpleNamespace(residues=[residue])
    model = SimpleNamespace(chains={"A": chain})
    molecule = SimpleNamespace(models=[model])

    resource = cs.ConnollySurfaceResource.from_molecule(
        molecule, cs.ConnollySurfaceSpec()
    )

    assert resource.positions.shape == (2, 3)
    np.testing.assert_allclose(resource.radii, [1.70, 1.52], rtol=1e-6)

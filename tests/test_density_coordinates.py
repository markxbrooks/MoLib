"""Regression tests for the electron-density coordinate pipeline.

These pin down the canonical voxel -> Cartesian convention used by the
viewer pipeline:

    cart = (i, j, k) / dims @ frac_to_orth + origin

and verify the exact inverse (Cartesian -> grid) used for world->grid
lookups, against gemmi's ``grid.get_position`` ground truth, for both
orthorhombic and monoclinic (non-orthogonal) cells.
"""

import numpy as np
import gemmi
from unittest import TestCase

from molib.xtal.map.density import (
    AxisOrder,
    crystallographic_info_from_grid,
    transform_grid_vertices_to_cartesian,
)


def _make_grid(a, b, c, alpha, beta, gamma, nx, ny, nz):
    grid = gemmi.FloatGrid(
        np.zeros((nx, ny, nz), dtype=np.float32),
        cell=gemmi.UnitCell(a, b, c, alpha, beta, gamma),
    )
    return grid


class TestDensityCoordinateConvention(TestCase):

    def _assert_convention(self, grid):
        info = crystallographic_info_from_grid(grid)
        dims = tuple(info.grid.dimensions)
        origin = (info.grid.origin.x, info.grid.origin.y, info.grid.origin.z)
        f2o = np.asarray(info.transforms.frac_to_orth, dtype=np.float64)
        f2i = np.asarray(info.transforms.orth_to_frac, dtype=np.float64)

        # Map grid covers the full unit cell starting at the fractional origin.
        self.assertEqual(origin, (0.0, 0.0, 0.0))

        # Spacing is the extent of each axis (lattice vector) per voxel.
        spacing = (
            info.grid.spacing.x,
            info.grid.spacing.y,
            info.grid.spacing.z,
        )
        self.assertTrue(
            np.allclose(
                np.asarray(spacing),
                np.linalg.norm(f2o, axis=0) / np.asarray(dims),
                atol=1e-8,
            )
        )

        # Transforms are exact inverses.
        self.assertTrue(np.allclose(f2o @ f2i, np.eye(3), atol=1e-8))

        # Axis order info is normalized to XYZ.
        self.assertEqual(info.grid.axis_order, AxisOrder.XYZ)

        # Integer grid points must agree with gemmi's ground truth.
        rng = np.random.default_rng(3)
        pts = rng.integers(0, np.asarray(dims), size=(200, 3)).astype(np.float64)
        ours = transform_grid_vertices_to_cartesian(pts, dims, f2o, origin)
        for i, (ix, iy, iz) in enumerate(np.round(pts).astype(int)):
            g = grid.get_position(ix, iy, iz)
            self.assertTrue(
                np.allclose(ours[i], (g.x, g.y, g.z), atol=1e-8),
                f"grid->cart mismatch at {(ix, iy, iz)}: {ours[i]} vs {(g.x, g.y, g.z)}",
            )

        # Continuous round-trip: cart -> frac -> grid is exact.
        grid_pts = rng.uniform(0, np.asarray(dims), size=(1000, 3))
        cart = transform_grid_vertices_to_cartesian(
            grid_pts, dims, f2o, origin
        )
        back = (
            (cart - np.asarray(origin)) @ np.linalg.inv(f2o)
        ) * np.asarray(dims)
        self.assertTrue(np.allclose(back, grid_pts, atol=1e-9))

    def test_orthorhombic_p2121(self):
        grid = _make_grid(58.96, 103.26, 150.93, 90, 90, 90, 48, 72, 108)
        self._assert_convention(grid)

    def test_monoclinic_c2_like(self):
        grid = _make_grid(50.0, 90.0, 70.0, 90.0, 108.0, 90.0, 44, 78, 60)
        self._assert_convention(grid)

    def test_triclinic(self):
        grid = _make_grid(40.0, 45.0, 50.0, 95.0, 102.0, 88.0, 30, 34, 38)
        self._assert_convention(grid)

    def test_origin_offset_survives_roundtrip(self):
        # Even with a translated sub-grid the inverse must map back exactly.
        grid = _make_grid(50.0, 60.0, 70.0, 90, 90, 90, 32, 40, 48)
        f2o = np.asarray(
            crystallographic_info_from_grid(grid).transforms.frac_to_orth,
            dtype=np.float64,
        )
        dims = (32, 40, 48)
        origin = (2.5, -3.0, 7.25)
        pts = np.random.default_rng(5).uniform(0, dims, (300, 3))
        cart = transform_grid_vertices_to_cartesian(pts, dims, f2o, origin)
        back = ((cart - np.asarray(origin)) @ np.linalg.inv(f2o)) * np.asarray(dims)
        self.assertTrue(np.allclose(back, pts, atol=1e-9))
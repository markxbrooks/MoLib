"""Regression tests for the electron-density coordinate pipeline.

These pin down the canonical voxel -> Cartesian convention used by the
viewer pipeline:

    cart = (i, j, k) / dims @ frac_to_orth.T + origin

and verify the exact inverse (Cartesian -> grid) used for world->grid
lookups, against gemmi's ``grid.get_position`` ground truth, for both
orthorhombic and monoclinic (non-orthogonal) cells.
"""

import numpy as np
import gemmi
import os
import tempfile
from unittest import TestCase

from molib.xtal.ccp4.mtz.column_pair import MtzColumnPair
from molib.xtal.ccp4.mtz.errors import MtzColumnNotFoundError
from molib.xtal.ccp4.mtz.filespec import MtzFileSpec
from molib.xtal.map.density import (
    AxisOrder,
    crystallographic_info_from_grid,
    transform_grid_vertices_to_cartesian,
)
from molib.xtal.map.helper import (
    MapType,
    DensityMapData,
    load_ccp4_map,
    load_ccp4_map_optimized,
    load_density_map_auto_mtz,
    load_density_map_from_columns,
    load_maps_from_mtz_file_spec,
    _load_spec_map,
    _select_map_columns,
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
            (cart - np.asarray(origin)) @ np.linalg.inv(f2o).T
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
        back = ((cart - np.asarray(origin)) @ np.linalg.inv(f2o).T) * np.asarray(dims)
        self.assertTrue(np.allclose(back, pts, atol=1e-9))


class TestCentredCellFrame(TestCase):
    """Centred space groups must pin the conventional-cell frame.

    unit_cell.orth.mat is the conventional fractional -> Cartesian matrix,
    i.e. the frame of PDB coordinates. gemmi's primitive_orth_matrix() maps
    into a primitive-cell basis that differs for centred lattices, so these
    tests assert frac_to_orth == orth.mat and explicitly != primitive --
    which fails if the primitive path is ever reintroduced.
    """

    CENTRED = [
        # (label, spacegroup, (a, b, c, alpha, beta, gamma), dims)
        ("C2", "C 1 2 1", (50.0, 40.0, 60.0, 90.0, 101.2, 90.0), (24, 20, 30)),
        ("H3", "H 3", (50.0, 50.0, 120.0, 90.0, 90.0, 120.0), (22, 22, 28)),
        ("I222", "I 2 2 2", (50.0, 40.0, 60.0, 90.0, 90.0, 90.0), (25, 21, 31)),
    ]

    @staticmethod
    def _make_centred_grid(sg, cell_params, dims):
        cell = gemmi.UnitCell(*cell_params)
        grid = gemmi.FloatGrid(np.zeros(dims, dtype=np.float32), cell=cell)
        grid.spacegroup = gemmi.find_spacegroup_by_name(sg)
        return grid

    def test_info_uses_conventional_cell_not_primitive(self):
        for label, sg, cell_params, dims in self.CENTRED:
            with self.subTest(spacegroup=label):
                grid = self._make_centred_grid(sg, cell_params, dims)
                info = crystallographic_info_from_grid(grid)
                f2o = np.asarray(info.transforms.frac_to_orth, dtype=np.float64)
                f2i = np.asarray(info.transforms.orth_to_frac, dtype=np.float64)
                orth = np.array(grid.unit_cell.orth.mat, dtype=np.float64)
                prim = np.array(
                    grid.unit_cell.primitive_orth_matrix(
                        grid.spacegroup.centring_type()
                    ),
                    dtype=np.float64,
                )

                # Conventional-cell matrix, not the primitive-cell one.
                self.assertTrue(np.allclose(f2o, orth, atol=1e-8))
                self.assertFalse(
                    np.allclose(orth, prim, atol=1e-6),
                    "test lacks teeth: orth.mat differs from "
                    "primitive_orth_matrix for this centred cell",
                )

                # Transforms are exact inverses; cart <-> grid round-trips.
                self.assertTrue(np.allclose(f2o @ f2i, np.eye(3), atol=1e-8))
                dims_t = tuple(info.grid.dimensions)
                pts = np.random.default_rng(11).uniform(
                    0, np.asarray(dims_t), size=(200, 3)
                )
                cart = transform_grid_vertices_to_cartesian(
                    pts, dims_t, f2o, (0.0, 0.0, 0.0)
                )
                back = (cart @ np.linalg.inv(f2o).T) * np.asarray(dims_t)
                self.assertTrue(np.allclose(back, pts, atol=1e-9))

                # Spacing is the extent of each lattice vector per voxel.
                spacing = (
                    info.grid.spacing.x,
                    info.grid.spacing.y,
                    info.grid.spacing.z,
                )
                self.assertTrue(
                    np.allclose(
                        np.asarray(spacing),
                        np.linalg.norm(f2o, axis=0) / np.asarray(dims_t),
                        atol=1e-8,
                    )
                )


class TestLoaderConventionalFrame(TestCase):
    """CCP4 loaders must return the conventional-cell frac_to_orth (orth.mat)
    with no primitive-cell override, and metadata consistent with the XYZ
    volume."""

    REAL_MAPS = [
        "/home/brooks/projects/ElMo/elmo/test_data/2VUG.ccp4",
        "/home/brooks/projects/ElMo/elmo/test_data/1mru.map",
    ]

    @staticmethod
    def _loaders(path):
        return [
            ("load_ccp4_map", load_ccp4_map(
                path, expand_symmetry=False, convert_to_cartesian=True,
                carve_density=False,
            )),
            ("load_ccp4_map_optimized", load_ccp4_map_optimized(
                path, expand_symmetry=False, convert_to_cartesian=True,
                carve_density=False,
            )),
        ]

    def _check_loader(self, result, map_path, centring_type=None):
        self.assertIsInstance(
            result, DensityMapData, f"{map_path} failed to load"
        )
        data = result
        info = data.crystallographic_info
        f2o = np.asarray(info.transforms.frac_to_orth, dtype=np.float64)
        f2i = np.asarray(info.transforms.orth_to_frac, dtype=np.float64)
        grid = gemmi.read_ccp4_map(map_path).grid
        orth = np.array(grid.unit_cell.orth.mat, dtype=np.float64)

        self.assertEqual(f2o.shape, (3, 3))
        self.assertTrue(
            np.allclose(f2o, orth, atol=1e-6),
            f"{map_path}: frac_to_orth is not the conventional orth.mat",
        )
        self.assertTrue(
            np.allclose(f2i, np.linalg.inv(f2o), atol=1e-8),
            f"{map_path}: orth_to_frac is not the exact inverse",
        )

        # Metadata must describe the XYZ volume consistently.
        self.assertEqual(tuple(info.grid.dimensions), data.volume.shape)
        self.assertEqual(info.grid.axis_order, AxisOrder.XYZ)
        origin = (info.grid.origin.x, info.grid.origin.y, info.grid.origin.z)
        self.assertEqual(origin, (0.0, 0.0, 0.0))
        dims = np.asarray(info.grid.dimensions, dtype=np.float64)
        spacing = np.array(
            [
                info.grid.spacing.x,
                info.grid.spacing.y,
                info.grid.spacing.z,
            ]
        )
        self.assertTrue(
            np.allclose(spacing, np.linalg.norm(f2o, axis=0) / dims, atol=1e-8)
        )

        if centring_type is not None:
            prim = np.array(
                grid.unit_cell.primitive_orth_matrix(centring_type),
                dtype=np.float64,
            )
            self.assertFalse(
                np.allclose(orth, prim, atol=1e-6),
                "teeth: orth.mat must differ from primitive for centred cell",
            )

    def test_load_ccp4_map_centred_synthetic(self):
        sg = "C 1 2 1"
        cell_params = (50.0, 40.0, 60.0, 90.0, 101.2, 90.0)
        dims = (24, 20, 30)
        cell = gemmi.UnitCell(*cell_params)
        temp_grid = gemmi.FloatGrid(np.zeros(dims, dtype=np.float32), cell=cell)
        temp_grid.spacegroup = gemmi.find_spacegroup_by_name(sg)
        ccp4 = gemmi.Ccp4Map()
        ccp4.grid = temp_grid
        ccp4.update_ccp4_header()
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "centred.ccp4")
            ccp4.write_ccp4_map(path)
            for name, result in self._loaders(path):
                with self.subTest(loader=name):
                    self._check_loader(result, path, centring_type="C")

    def test_real_ccp4_loaders(self):
        for path in self.REAL_MAPS:
            if not os.path.exists(path):
                self.skipTest(f"missing test map: {path}")
            with self.subTest(map=os.path.basename(path)):
                for name, result in self._loaders(path):
                    with self.subTest(loader=name):
                        self._check_loader(result, path)


class TestLoadMapsFromMtzFileSpec(TestCase):
    """load_maps_from_mtz_file_spec loads the map and difference maps."""

    MTZ_PATH = "/home/brooks/projects/ElMo/elmo/test_data/2VUG_final.mtz"

    def _spec(self, file_path=None):
        return MtzFileSpec(
            file_path=file_path or self.MTZ_PATH,
            map_coefficients=MtzColumnPair.map(
                f_label="FWT",
                phi_label="PHWT",
            ),
            difference_coefficients=MtzColumnPair.difference(
                f_label="DELFWT",
                phi_label="PHDELWT",
            ),
        )

    def test_loads_map_and_difference_from_spec(self):
        if not os.path.exists(self.MTZ_PATH):
            self.skipTest(f"missing test mtz: {self.MTZ_PATH}")

        map_data, difference_data = load_maps_from_mtz_file_spec(self._spec())

        self.assertIsInstance(map_data, DensityMapData)
        self.assertIsInstance(difference_data, DensityMapData)
        self.assertEqual(map_data.map_type, MapType.TWO_FO_FC)
        self.assertEqual(difference_data.map_type, MapType.FO_FC)
        self.assertEqual(map_data.crystallographic_info.map_type, MapType.TWO_FO_FC.value)
        self.assertEqual(
            difference_data.crystallographic_info.map_type, MapType.FO_FC.value
        )
        self.assertEqual(map_data.volume.shape, difference_data.volume.shape)
        self.assertEqual(
            map_data.crystallographic_info.grid.dimensions,
            difference_data.crystallographic_info.grid.dimensions,
        )
        self.assertGreater(np.abs(map_data.volume).max(), 0.0)
        self.assertFalse(np.array_equal(map_data.volume, difference_data.volume))

    def test_return_none_for_missing_difference_columns(self):
        if not os.path.exists(self.MTZ_PATH):
            self.skipTest(f"missing test mtz: {self.MTZ_PATH}")

        spec = MtzFileSpec(
            file_path=self.MTZ_PATH,
            map_coefficients=MtzColumnPair.map(
                f_label="FWT",
                phi_label="PHWT",
            ),
            difference_coefficients=MtzColumnPair.difference(
                f_label="NOT_A_COLUMN",
                phi_label="ALSO_NOT_PHI",
            ),
        )

        map_data, difference_data = load_maps_from_mtz_file_spec(spec)
        self.assertIsInstance(map_data, DensityMapData)
        self.assertIsNone(difference_data)

    def test_exported_at_package_level(self):
        from molib.xtal.map import load_maps_from_mtz_file_spec as exported

        self.assertIs(exported, load_maps_from_mtz_file_spec)


def _make_mtz_with_columns(mtz_path, columns):
    """Write a minimal gemmi MTZ with the given (label, type) columns."""
    mtz = gemmi.Mtz()
    mtz.cell = gemmi.UnitCell(58.962, 103.257, 150.930, 90, 90, 90)
    mtz.spacegroup = gemmi.SpaceGroup("P 21 21 21")
    mtz.add_dataset("synthetic")
    for label in ("H", "K", "L"):
        mtz.add_column(label, "H")
    for label, col_type in columns:
        mtz.add_column(label, col_type)
    n = 16
    array = np.zeros((n, 3 + len(columns)), dtype="f4")
    array[:, 0] = np.arange(1, n + 1)
    array[:, 1] = (np.arange(1, n + 1) * 2) % 7 + 1
    array[:, 2] = (np.arange(1, n + 1) * 3) % 13 + 1
    for i, (label, _) in enumerate(columns):
        array[:, 3 + i] = np.arange(1, n + 1) * (i + 1) + 1.5
    mtz.set_data(array)
    mtz.write_to_file(str(mtz_path))
    return mtz_path


class TestMapTypeAwareMtzSelection(TestCase):
    """Map-type-aware deterministic coefficient selection."""

    MTZ_PATH = "/home/brooks/projects/ElMo/elmo/test_data/2VUG_final.mtz"

    def test_selects_2fofc_columns_from_fwt(self):
        if not os.path.exists(self.MTZ_PATH):
            self.skipTest(f"missing test mtz: {self.MTZ_PATH}")
        self.assertEqual(
            _select_map_columns(self.MTZ_PATH, MapType.TWO_FO_FC),
            ("FWT", "PHWT"),
        )

    def test_selects_fofc_columns_from_delfwt(self):
        if not os.path.exists(self.MTZ_PATH):
            self.skipTest(f"missing test mtz: {self.MTZ_PATH}")
        self.assertEqual(
            _select_map_columns(self.MTZ_PATH, MapType.FO_FC),
            ("DELFWT", "PHDELWT"),
        )

    def test_auto_mtz_map_types_produce_distinct_non_constant_maps(self):
        if not os.path.exists(self.MTZ_PATH):
            self.skipTest(f"missing test mtz: {self.MTZ_PATH}")

        two = load_density_map_auto_mtz(self.MTZ_PATH, map_type=MapType.TWO_FO_FC)
        one = load_density_map_auto_mtz(self.MTZ_PATH, map_type=MapType.FO_FC)

        self.assertIsInstance(two, DensityMapData)
        self.assertIsInstance(one, DensityMapData)
        self.assertEqual(two.map_type, MapType.TWO_FO_FC)
        self.assertEqual(one.map_type, MapType.FO_FC)
        self.assertEqual(two.crystallographic_info.map_type, MapType.TWO_FO_FC.value)
        self.assertEqual(one.crystallographic_info.map_type, MapType.FO_FC.value)
        self.assertEqual(two.volume.shape, one.volume.shape)
        # Guard against the constant-volume regression: a gridded 2Fo-Fc map
        # must contain real electron-density variation.
        self.assertGreater(np.std(two.volume), 1e-6)
        self.assertGreater(np.std(one.volume), 1e-6)
        self.assertFalse(np.array_equal(two.volume, one.volume))

    def test_unrelated_coefficients_fail_explicitly(self):
        out_dir = tempfile.mkdtemp(prefix="mtz_no_map_columns_")
        self.addCleanup(self._rmtree, out_dir)
        mtz_path = _make_mtz_with_columns(
            os.path.join(out_dir, "observed_only.mtz"),
            [("FP", "F"), ("PHIC", "P")],
        )

        with self.assertRaises(ValueError):
            _select_map_columns(mtz_path, MapType.TWO_FO_FC)
        with self.assertRaises(ValueError):
            _select_map_columns(mtz_path, MapType.FO_FC)
        # The public loader never silently substitutes observed FP/PHIC.
        self.assertIsNone(
            load_density_map_auto_mtz(mtz_path, map_type=MapType.TWO_FO_FC)
        )

    def test_coerce_accepts_strings_and_rejects_unknown(self):
        self.assertIs(MapType.coerce("2Fo-Fc"), MapType.TWO_FO_FC)
        self.assertIs(MapType.coerce("Fo-Fc"), MapType.FO_FC)
        self.assertIs(MapType.coerce(MapType.TWO_FO_FC), MapType.TWO_FO_FC)
        with self.assertRaises(ValueError):
            MapType.coerce("unknown")

    def test_missing_columns_raise_mtz_column_not_found(self):
        out_dir = tempfile.mkdtemp(prefix="mtz_missing_cols_")
        self.addCleanup(self._rmtree, out_dir)
        mtz_path = _make_mtz_with_columns(
            os.path.join(out_dir, "fwt_only.mtz"),
            [("FWT", "F"), ("PHWT", "P")],
        )
        with self.assertRaises(MtzColumnNotFoundError):
            load_density_map_from_columns(
                mtz_path,
                "NOT_F",
                "PHWT",
                map_type=MapType.TWO_FO_FC,
            )
        with self.assertRaises(MtzColumnNotFoundError):
            load_density_map_from_columns(
                mtz_path,
                "FWT",
                "NOT_PHI",
                map_type=MapType.TWO_FO_FC,
            )

    def test_load_spec_map_swallows_only_column_not_found(self):
        out_dir = tempfile.mkdtemp(prefix="mtz_spec_missing_")
        self.addCleanup(self._rmtree, out_dir)
        mtz_path = _make_mtz_with_columns(
            os.path.join(out_dir, "fwt_only.mtz"),
            [("FWT", "F"), ("PHWT", "P")],
        )
        missing = MtzColumnPair.difference("NOPE", "ALSO_NOPE")
        self.assertIsNone(_load_spec_map(mtz_path, missing))

        # Non-column ValueErrors must not be treated as absent columns.
        from unittest.mock import patch

        with patch(
            "molib.xtal.map.helper.load_density_map_from_columns",
            side_effect=ValueError("malformed spec"),
        ):
            with self.assertRaises(ValueError):
                _load_spec_map(
                    mtz_path,
                    MtzColumnPair.map("FWT", "PHWT"),
                )

    @staticmethod
    def _rmtree(path):
        import shutil

        shutil.rmtree(path, ignore_errors=True)
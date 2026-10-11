"""Unit tests for molib.xtal.reflections (Gemmi fixtures, no CCP4)."""

from __future__ import annotations

from pathlib import Path

import gemmi
import numpy as np
import pytest

from molib.xtal.reflections.formats import ReflectionFileFormat, sniff_reflection_format
from molib.xtal.reflections.loader import ReflectionFileLoader
from molib.xtal.reflections.errors import (
    ReflectionLoadError,
    UnsupportedReflectionFormatError,
)
from molib.xtal.reflections.quantity import (
    DisplayQuantity,
    available_quantities,
    compute_display_values,
    resolve_quantity,
)
from molib.xtal.reflections.resolution import ResolutionFilter
from molib.xtal.reflections.zone import (
    ZonePreset,
    ZoneSpec,
    project_to_zone_plane,
    zone_membership,
)


def _write_test_mtz(path: Path) -> None:
    mtz = gemmi.Mtz(with_base=True)
    mtz.spacegroup = gemmi.SpaceGroup("P 21 21 21")
    mtz.set_cell_for_all(gemmi.UnitCell(50, 60, 70, 90, 90, 90))
    mtz.add_dataset("HKL_main")
    mtz.add_column("F", "F")
    mtz.add_column("SIGF", "Q")
    mtz.add_column("I", "J")
    mtz.add_column("SIGI", "Q")
    rows = []
    for h in range(-2, 3):
        for k in range(-2, 3):
            for l in range(-2, 3):
                if h == 0 and k == 0 and l == 0:
                    continue
                f = float(abs(h) + abs(k) + abs(l) + 1)
                rows.append([h, k, l, f, 0.5, f * f, 1.0])
    mtz.set_data(np.asarray(rows, dtype=np.float32))
    mtz.write_to_file(str(path))


def _write_test_sfcif(path: Path) -> None:
    path.write_text(
        """
data_test
_cell.length_a 50
_cell.length_b 60
_cell.length_c 70
_cell.angle_alpha 90
_cell.angle_beta 90
_cell.angle_gamma 90
_symmetry.space_group_name_H-M 'P 21 21 21'
loop_
_refln.index_h
_refln.index_k
_refln.index_l
_refln.F_meas_au
_refln.F_meas_sigma_au
1 0 0 10.0 0.5
0 1 0 12.0 0.6
0 0 1 11.0 0.4
1 1 0 9.0 0.5
0 0 2 8.0 0.4
""",
        encoding="utf-8",
    )


@pytest.fixture()
def mtz_path(tmp_path: Path) -> Path:
    path = tmp_path / "test.mtz"
    _write_test_mtz(path)
    return path


@pytest.fixture()
def sfcif_path(tmp_path: Path) -> Path:
    path = tmp_path / "test.cif"
    _write_test_sfcif(path)
    return path


def test_sniff_mtz(mtz_path: Path) -> None:
    assert sniff_reflection_format(mtz_path) == ReflectionFileFormat.MTZ


def test_sniff_sfcif(sfcif_path: Path) -> None:
    assert sniff_reflection_format(sfcif_path) == ReflectionFileFormat.SF_CIF


def test_load_mtz(mtz_path: Path) -> None:
    data = ReflectionFileLoader().load(mtz_path)
    assert data.format == ReflectionFileFormat.MTZ
    assert data.n_reflections > 0
    assert data.unit_cell is not None
    assert data.space_group is not None
    assert "F" in data.column_labels()
    assert data.h.dtype == np.int32
    assert data.d_spacing is not None
    assert data.d_spacing.shape[0] == data.n_reflections


def test_load_sfcif(sfcif_path: Path) -> None:
    data = ReflectionFileLoader().load(sfcif_path)
    assert data.format == ReflectionFileFormat.SF_CIF
    assert data.n_reflections == 5
    assert data.unit_cell is not None
    assert any("F_meas" in lab for lab in data.column_labels())


def test_unsupported_ascii_hkl(tmp_path: Path) -> None:
    path = tmp_path / "plain.hkl"
    path.write_text("1 0 0 10.0\n0 1 0 11.0\n", encoding="utf-8")
    with pytest.raises(UnsupportedReflectionFormatError):
        ReflectionFileLoader().load(path)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ReflectionLoadError):
        ReflectionFileLoader().load(tmp_path / "nope.mtz")


def test_zone_membership_presets() -> None:
    miller = np.array(
        [
            [1, 2, 0],
            [1, 0, 3],
            [0, 2, 3],
            [2, 2, 1],
            [1, 1, 0],
        ],
        dtype=np.int32,
    )
    assert zone_membership(miller, ZoneSpec.from_preset(ZonePreset.HK0)).tolist() == [
        True,
        False,
        False,
        False,
        True,
    ]
    assert zone_membership(miller, ZoneSpec.from_preset(ZonePreset.H0L)).tolist() == [
        False,
        True,
        False,
        False,
        False,
    ]
    assert zone_membership(miller, ZoneSpec.from_preset(ZonePreset.OKL)).tolist() == [
        False,
        False,
        True,
        False,
        False,
    ]
    # (2,2,1)·(1,-1,0)=0 and (1,1,0)·(1,-1,0)=0 are both in hhl.
    assert zone_membership(miller, ZoneSpec.from_preset(ZonePreset.HHL)).tolist() == [
        False,
        False,
        False,
        True,
        True,
    ]


def test_project_hk0_without_cell() -> None:
    miller = np.array([[1, 2, 0], [3, -1, 0]], dtype=np.int32)
    uv = project_to_zone_plane(miller, ZoneSpec.from_preset(ZonePreset.HK0))
    np.testing.assert_allclose(uv, [[1, 2], [3, -1]], atol=1e-6)


def test_project_non_orthogonal_smoke() -> None:
    from molib.xtal.map.unit_cell import UnitCell

    cell = UnitCell(a=50, b=60, c=70, alpha=90, beta=105, gamma=90, source="test")
    miller = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.int32)
    uv = project_to_zone_plane(
        miller, ZoneSpec.from_preset(ZonePreset.HK0), unit_cell=cell
    )
    assert uv.shape == (3, 2)
    assert np.all(np.isfinite(uv))


def test_quantity_f_and_i_over_sig(mtz_path: Path) -> None:
    data = ReflectionFileLoader().load(mtz_path)
    avail = available_quantities(data)
    assert DisplayQuantity.F in avail
    assert DisplayQuantity.I in avail
    sel = resolve_quantity(data, DisplayQuantity.F_OVER_SIGF)
    assert sel is not None
    values, missing = compute_display_values(data, sel)
    assert values.shape[0] == data.n_reflections
    assert missing.shape[0] == data.n_reflections
    assert np.isfinite(values[~missing]).all()


def test_resolution_filter(mtz_path: Path) -> None:
    data = ReflectionFileLoader().load(mtz_path)
    filt = ResolutionFilter(d_min=5.0, d_max=30.0)
    mask = filt.mask(data.d_spacing)
    assert mask.dtype == bool
    assert mask.shape[0] == data.n_reflections
    kept = data.d_spacing[mask]
    assert np.all(kept >= 5.0 - 1e-5)
    assert np.all(kept <= 30.0 + 1e-5)

"""Gemmi-based reflection file loader (no Qt)."""

from __future__ import annotations

from pathlib import Path

import gemmi
import numpy as np

from molib.xtal.map.unit_cell import UnitCell
from molib.xtal.reflections.errors import (
    ReflectionLoadError,
    UnsupportedReflectionFormatError,
)
from molib.xtal.reflections.formats import ReflectionFileFormat, sniff_reflection_format
from molib.xtal.reflections.model import ColumnDescriptor, ReflectionData


class ReflectionFileLoader:
    """Load MTZ or SF-CIF reflection files into :class:`ReflectionData`."""

    def load(self, path: Path | str) -> ReflectionData:
        """Load and validate a reflection file.

        :param path: Filesystem path
        :returns: Populated :class:`ReflectionData`
        :raises UnsupportedReflectionFormatError: Ambiguous / unsupported format
        :raises ReflectionLoadError: Read or validation failure
        """
        p = Path(path)
        if not p.is_file():
            raise ReflectionLoadError(f"Reflection file not found: {p}")

        fmt = sniff_reflection_format(p)
        if fmt == ReflectionFileFormat.MTZ:
            return self._load_mtz(p)
        if fmt == ReflectionFileFormat.SF_CIF:
            return self._load_sf_cif(p)
        raise UnsupportedReflectionFormatError(
            f"Unsupported or ambiguous reflection format for {p.name!r}. "
            "Supported: binary MTZ (.mtz) and structure-factor CIF (.cif/.fcf "
            "or CIF-formatted .hkl). Plain ASCII .hkl lists are not loaded "
            "as MTZ."
        )

    def _load_mtz(self, path: Path) -> ReflectionData:
        try:
            mtz = gemmi.read_mtz_file(str(path))
        except Exception as ex:
            raise ReflectionLoadError(f"Failed to read MTZ {path}: {ex}") from ex

        miller = np.asarray(mtz.make_miller_array(), dtype=np.int32)
        if miller.ndim != 2 or miller.shape[1] != 3:
            raise ReflectionLoadError("MTZ Miller array has unexpected shape")
        if miller.shape[0] == 0:
            raise ReflectionLoadError(f"MTZ file has no reflections: {path}")

        h = miller[:, 0].copy()
        k = miller[:, 1].copy()
        l = miller[:, 2].copy()

        columns: list[ColumnDescriptor] = []
        values: dict[str, np.ndarray] = {}
        missing: dict[str, np.ndarray] = {}
        for col in mtz.columns:
            label = str(col.label)
            if label in {"H", "K", "L"}:
                continue
            col_type = str(getattr(col, "type", "") or "")
            dataset_id = getattr(col, "dataset_id", None)
            dataset_name = ""
            try:
                if dataset_id is not None and 0 <= int(dataset_id) < len(mtz.datasets):
                    dataset_name = str(mtz.datasets[int(dataset_id)].dataset_name)
            except Exception:
                dataset_name = ""
            columns.append(
                ColumnDescriptor(
                    label=label,
                    column_type=col_type,
                    dataset_id=int(dataset_id) if dataset_id is not None else None,
                    dataset_name=dataset_name,
                )
            )
            arr = np.asarray(col.array, dtype=np.float32).copy()
            if arr.shape[0] != h.shape[0]:
                raise ReflectionLoadError(
                    f"Column {label!r} length {arr.shape[0]} != {h.shape[0]}"
                )
            miss = ~np.isfinite(arr)
            values[label] = arr
            missing[label] = miss

        unit_cell = None
        try:
            cell = mtz.cell
            unit_cell = UnitCell(
                a=float(cell.a),
                b=float(cell.b),
                c=float(cell.c),
                alpha=float(cell.alpha),
                beta=float(cell.beta),
                gamma=float(cell.gamma),
                source="MTZ",
                space_group=str(mtz.spacegroup) if mtz.spacegroup else "",
            )
        except Exception:
            unit_cell = None

        space_group = str(mtz.spacegroup) if mtz.spacegroup else None
        datasets: list[dict[str, str | float]] = []
        for ds in mtz.datasets:
            info: dict[str, str | float] = {
                "name": str(ds.dataset_name),
                "project_name": str(ds.project_name),
                "crystal_name": str(ds.crystal_name),
            }
            try:
                if getattr(ds, "wavelength", None) is not None:
                    info["wavelength"] = float(ds.wavelength)
            except Exception:
                pass
            datasets.append(info)

        d_spacing = None
        try:
            d_spacing = np.asarray(mtz.make_d_array(), dtype=np.float32).copy()
        except Exception:
            d_spacing = None

        file_d_min = file_d_max = None
        try:
            if mtz.resolution_high is not None:
                file_d_min = float(mtz.resolution_high)
            if mtz.resolution_low is not None:
                file_d_max = float(mtz.resolution_low)
        except Exception:
            pass

        history = [str(x) for x in getattr(mtz, "history", []) or []]
        is_merged = None
        # Heuristic: BATCH column implies unmerged
        if any(c.label.upper() == "BATCH" or c.column_type.upper() == "B" for c in columns):
            is_merged = False
        elif history:
            is_merged = True

        return ReflectionData(
            path=path.resolve(),
            format=ReflectionFileFormat.MTZ,
            h=h,
            k=k,
            l=l,
            columns=columns,
            values=values,
            missing=missing,
            unit_cell=unit_cell,
            space_group=space_group,
            datasets=datasets,
            d_spacing=d_spacing,
            is_merged=is_merged,
            file_resolution_high=file_d_min,
            file_resolution_low=file_d_max,
            history=history,
        )

    def _load_sf_cif(self, path: Path) -> ReflectionData:
        try:
            doc = gemmi.cif.read(str(path))
            blocks = gemmi.as_refln_blocks(doc)
        except Exception as ex:
            raise ReflectionLoadError(
                f"Failed to read structure-factor CIF {path}: {ex}"
            ) from ex

        if not blocks:
            raise UnsupportedReflectionFormatError(
                f"No reflection loop found in CIF file: {path}"
            )

        rb = blocks[0]
        try:
            miller = np.asarray(rb.make_miller_array(), dtype=np.int32)
        except Exception as ex:
            raise ReflectionLoadError(
                f"Could not extract Miller indices from {path}: {ex}"
            ) from ex
        if miller.shape[0] == 0:
            raise ReflectionLoadError(f"SF-CIF file has no reflections: {path}")

        h = miller[:, 0].copy()
        k = miller[:, 1].copy()
        l = miller[:, 2].copy()

        labels = list(rb.column_labels())
        skip = {"index_h", "index_k", "index_l", "H", "K", "L"}
        columns: list[ColumnDescriptor] = []
        values: dict[str, np.ndarray] = {}
        missing: dict[str, np.ndarray] = {}
        for label in labels:
            if label in skip:
                continue
            try:
                arr = np.asarray(rb.make_float_array(label), dtype=np.float32).copy()
            except Exception:
                try:
                    arr = np.asarray(rb.get_float(label), dtype=np.float32).copy()
                except Exception:
                    continue
            if arr.shape[0] != h.shape[0]:
                continue
            # Infer a loose CCP4-like type for quantity discovery
            col_type = "R"
            up = label.upper()
            if "SIG" in up or "SIGMA" in up:
                col_type = "Q"
            elif up.startswith("I") or "INTENSITY" in up:
                col_type = "J"
            elif up.startswith("F") or "F_MEAS" in up or "AMPL" in up:
                col_type = "F"
            columns.append(
                ColumnDescriptor(label=label, column_type=col_type, dataset_name="")
            )
            values[label] = arr
            missing[label] = ~np.isfinite(arr)

        unit_cell = None
        try:
            cell = rb.cell
            if cell is not None and cell.a > 0:
                unit_cell = UnitCell(
                    a=float(cell.a),
                    b=float(cell.b),
                    c=float(cell.c),
                    alpha=float(cell.alpha),
                    beta=float(cell.beta),
                    gamma=float(cell.gamma),
                    source="SF_CIF",
                    space_group=str(rb.spacegroup) if rb.spacegroup else "",
                )
        except Exception:
            unit_cell = None

        space_group = str(rb.spacegroup) if rb.spacegroup else None
        is_merged = None
        try:
            if rb.is_unmerged():
                is_merged = False
            elif rb.is_merged():
                is_merged = True
        except Exception:
            pass

        d_spacing = None
        try:
            d_spacing = np.asarray(rb.make_d_array(), dtype=np.float32).copy()
        except Exception:
            d_spacing = None

        return ReflectionData(
            path=path.resolve(),
            format=ReflectionFileFormat.SF_CIF,
            h=h,
            k=k,
            l=l,
            columns=columns,
            values=values,
            missing=missing,
            unit_cell=unit_cell,
            space_group=space_group,
            datasets=[],
            d_spacing=d_spacing,
            is_merged=is_merged,
        )

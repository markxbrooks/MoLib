"""
Utilities for loading and processing electron density maps from MTZ and CCP4 files.
"""
from pathlib import Path

from dataclasses import dataclass

import faulthandler
import os
import pathlib
import re
from numpy import dtype, ndarray
from typing import Callable, Optional, Tuple, Any

import gemmi
import numpy as np
from decologr import Decologr as log
from molib.xtal.ccp4.mtz.filespec import MtzFileSpec, MtzDensitySpec
from molib.xtal.ccp4.mtz.column_pair import MtzColumnPair
from molib.xtal.ccp4.mtz.errors import MtzColumnNotFoundError
from molib.xtal.info.resolve import normalize_crystallographic_info_from_dict
from molib.xtal.uglymol.map.helpers import (
    extract_symop_text,
    parse_symmetry_operator_to_matrix,
)
from molib.xtal.map.density import (
    AxisOrder,
    CrystallographicInfo,
    MapType,
    crystallographic_info_from_grid,
)

# Enable faulthandler for debugging SIGBUS crashes on macOS
faulthandler.enable()

ORIGIN = (0.0, 0.0, 0.0)


@dataclass(slots=True)
class DensityMapSpec:
    """Specification for loading and processing a density map."""

    map_path: str | pathlib.Path  # mtz or ccp4map path
    pdb_path: str | pathlib.Path | None = None  # pdb file path

    mtz: MtzDensitySpec | None = None
    expand_symmetry: bool = True

    carve_density: bool = True
    carve_cutoff: float = 4.0

    carve_density_centroid: bool = False
    centroid: tuple[float, float, float] | None = None
    centroid_cutoff: float = 15.0

    progress_callback: Callable | None = None


_TWO_FO_FC_CANDIDATES = (
    ("FWT", "PHWT"),
    ("2FOFCWT", "PH2FOFCWT"),
)

_FO_FC_CANDIDATES = (
    ("DELFWT", "PHDELWT"),
    ("DELF", "PHDEL"),
)


def _mtz_f_phi_label_sets(mtz_path: str) -> tuple[dict[str, str], dict[str, str]]:
    """Return {UPPER: original-case-label} lookups for F and PHI columns."""
    mtz = gemmi.read_mtz_file(mtz_path)
    f_map = {col.label.upper(): col.label for col in mtz.columns if col.type == "F"}
    p_map = {col.label.upper(): col.label for col in mtz.columns if col.type == "P"}
    return f_map, p_map


def _select_map_columns(mtz_path: str, map_type: MapType) -> tuple[str, str]:
    """Deterministically choose the F/PHI columns for a requested map type.

    Unlike guessing through arbitrary combinations, this checks the labels
    actually present in the file and only ever returns coefficient pairs that
    represent the requested map type. It fails explicitly (``ValueError``)
    rather than silently gridding an unrelated pair (e.g. observed FP/PHIC).
    """
    map_type = MapType.coerce(map_type)
    f_map, p_map = _mtz_f_phi_label_sets(mtz_path)

    if map_type is MapType.TWO_FO_FC:
        candidates = _TWO_FO_FC_CANDIDATES
    elif map_type is MapType.FO_FC:
        candidates = _FO_FC_CANDIDATES
    else:
        raise ValueError(f"Unsupported map type: {map_type!r}")

    for f_label, phi_label in candidates:
        if f_label in f_map and phi_label in p_map:
            return f_map[f_label], p_map[phi_label]

    raise ValueError(
        f"No {map_type.value} coefficients found in {mtz_path}. "
        f"Available F columns: {sorted(f_map)}; "
        f"available PHI columns: {sorted(p_map)}"
    )


def _log_density_statistics(volume: np.ndarray) -> None:
    """Log density statistics immediately after gridding to catch stale/constant data."""
    if volume is None or volume.size == 0:
        log.error("❌ Generated density volume is empty")
        return
    finite = np.isfinite(volume)
    finite_count = int(finite.sum())
    log.info(
        "📊 Density statistics: "
        f"shape={volume.shape}, "
        f"finite={finite_count}/{volume.size}, "
        f"min={np.min(volume[finite]):.6f}, "
        f"max={np.max(volume[finite]):.6f}, "
        f"mean={np.mean(volume[finite]):.6f}, "
        f"std={np.std(volume[finite]):.6f}"
    )
    if finite_count == 0:
        log.error("❌ Generated density volume contains no finite values")
    elif np.all(volume[finite] == volume[finite].flat[0]):
        log.error(
            "❌ Generated density volume has no variation: "
            f"value={volume[finite].flat[0]}"
        )

@dataclass
class DensityMapData:
    """Loaded electron-density map and its crystallographic metadata.

    ``volume.shape`` is always kept in lock-step with
    ``crystallographic_info.grid.dimensions`` (see :func:`_log_grid_consistency`).
    """

    volume: np.ndarray
    crystallographic_info: CrystallographicInfo
    source: str = ""
    map_type: MapType | None = None


@dataclass
class SimpleHeader:
    """Minimal CCP4 header view used by the symmetry-expansion helpers."""

    nsymbt: int
    nx: int = 0
    ny: int = 0
    nz: int = 0
    nxstart: int = 0
    nystart: int = 0
    nzstart: int = 0


def _build_header_from_ccp4_map(ccp4_map: gemmi.Ccp4Map) -> SimpleHeader:
    """Build a ``SimpleHeader`` from a gemmi Ccp4Map (or test mock).

    Prefers parsing the raw CCP4 header bytes, then falls back to the
    ``header.nsymbt`` / ``grid.shape`` attributes used by mocks.
    """
    header = None
    try:
        import struct

        if hasattr(ccp4_map, "ccp4_header") and isinstance(
            ccp4_map.ccp4_header, (bytes, bytearray)
        ):
            header_ints = struct.unpack("<256i", ccp4_map.ccp4_header[:1024])
            header = SimpleHeader(
                nsymbt=header_ints[23],
                nx=header_ints[7],
                ny=header_ints[8],
                nz=header_ints[9],
                nxstart=header_ints[4],
                nystart=header_ints[5],
                nzstart=header_ints[6],
            )
    except Exception:
        header = None

    if header is None:
        nsymbt = 0
        if hasattr(ccp4_map, "header") and hasattr(ccp4_map.header, "nsymbt"):
            try:
                nsymbt = int(ccp4_map.header.nsymbt)
            except Exception:
                nsymbt = 0
        shape = getattr(ccp4_map.grid, "shape", ORIGIN)
        nx, ny, nz = (
            (int(shape[0]), int(shape[1]), int(shape[2]))
            if len(shape) == 3
            else ORIGIN
        )
        header = SimpleHeader(nsymbt=nsymbt, nx=nx, ny=ny, nz=nz)
    return header


def _read_symmetry_ops(
    map_path: str, header
) -> tuple[bytes, int, list, list, list]:
    """Read the raw CCP4 bytes and derive (buffer, nsymbt, n_grid, start, end)."""
    with open(map_path, "rb") as f:
        map_buffer = f.read()
    nsymbt = header.nsymbt
    n_grid = [header.nx, header.ny, header.nz]
    start = [header.nxstart, header.nystart, header.nzstart]
    end = [start[0] + n_grid[0], start[1] + n_grid[1], start[2] + n_grid[2]]
    return map_buffer, nsymbt, n_grid, start, end


def _parse_symmetry_matrix(symop: str, n_grid: list) -> list:
    """Parse a CCP4 symmetry text line into a grid-scaled 4x4 matrix."""
    symop_matrix = parse_symmetry_operator_to_matrix(symop)
    for j in range(3):
        symop_matrix[j][3] = round(symop_matrix[j][3] * n_grid[j])
    return symop_matrix


def _apply_symmetry_mates(
    volume: np.ndarray,
    expanded_volume: np.ndarray,
    map_buffer: bytes,
    nsymbt: int,
    n_grid: list,
    start: list,
    end: list,
    center_offset: list,
) -> int:
    """Apply every symmetry mate from ``map_buffer`` into ``expanded_volume``.

    Returns the number of symmetry operations applied.
    """
    symmetry_count = 0
    for i in range(0, nsymbt, 80):
        symop = extract_symop_text(map_buffer, i)
        symop = symop.strip()

        # Skip identity operation
        if re.match(r"^\s*x\s*,\s*y\s*,\s*z\s*$", symop, re.I):
            continue

        try:
            # Parse symmetry operation
            symop_matrix = _parse_symmetry_matrix(symop, n_grid)

            log.info(f"🔄 Applying symmetry: {symop}")

            # Apply symmetry operation to create symmetry mate
            apply_symmetry_to_volume(
                volume,
                expanded_volume,
                symop_matrix,
                0,
                1,
                2,  # Default axis order
                start,
                end,
                center_offset,
            )
            symmetry_count += 1

        except Exception as e:
            log.warning(f"⚠️ Failed to apply symmetry operation '{symop}': {e}")
            continue

    return symmetry_count


def _collect_atom_coordinates(pdb) -> list[list[float]]:
    """Collect ``[x, y, z]`` coordinates for every atom in a gemmi structure."""
    atoms = []
    for model in pdb:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atoms.append([atom.pos.x, atom.pos.y, atom.pos.z])
    return atoms


def _xyz_to_dict(value) -> dict:
    """Accept ``GridOrigin``/``GridSpacing`` dataclasses or ``{"x", "y", "z"}`` dicts.

    Returns a plain ``{"x": float, "y": float, "z": float}`` dict so carving
    code is agnostic to whether the crystallographic metadata came from the
    normalized dataclasses or a legacy dict.
    """
    if isinstance(value, dict):
        return {k: float(value[k]) for k in ("x", "y", "z")}
    return {
        "x": float(value.x),
        "y": float(value.y),
        "z": float(value.z),
    }


def _apply_mask_and_finalize(
    density_map: np.ndarray,
    mask: np.ndarray,
    progress_callback,
    cutoff_distance: float,
    label: str,
    extra_log_fn=None,
    guard_division: bool = True,
) -> np.ndarray:
    """Zero masked-out voxels, then log carve statistics and finalize."""
    carved_density = density_map.copy()
    carved_density[~mask] = 0.0

    if progress_callback:
        progress_callback(90, 100, "Finalizing carved density...")

    original_nonzero = np.count_nonzero(density_map)
    carved_nonzero = np.count_nonzero(carved_density)
    if guard_division and original_nonzero == 0:
        reduction_factor = 0
    else:
        reduction_factor = (
            (original_nonzero - carved_nonzero) / original_nonzero * 100
        )

    log.info(f"✅ {label}:")
    if extra_log_fn:
        extra_log_fn()
    log.info(f"   Original non-zero voxels: {original_nonzero:,}")
    log.info(f"   Carved non-zero voxels: {carved_nonzero:,}")
    log.info(f"   Reduction: {reduction_factor:.1f}%")
    log.info(f"   Cutoff distance: {cutoff_distance}Å")

    if progress_callback:
        progress_callback(100, 100, f"{label} complete")

    return carved_density


def _grid_to_xyz_array(
    grid: gemmi.FloatGrid,
    crystallographic_info: CrystallographicInfo | None = None,
) -> np.ndarray:
    """Return a NumPy array of the grid in the canonical X, Y, Z axis order.

    Gemmi stores the array with shape (nu, nv, nw); when the grid's axis
    order is not XYZ the array must be transposed so that ``array[i, j, k]``
    is the density at (X, Y, Z). Grids whose axis order is unknown are left
    in their native order (assumed XYZ).

    When ``crystallographic_info`` is given, its grid dimensions and axis
    order are synced to the returned XYZ array so the metadata always
    describes the canonical volume.
    """
    array = np.array(grid, copy=True)
    try:
        axis_order = AxisOrder.from_gemmi(grid.axis_order)
    except ValueError:
        log.warning(
            "⚠️ Unknown grid axis order %r; assuming XYZ",
            grid.axis_order,
        )
        return array
    if axis_order is not AxisOrder.XYZ:
        array = axis_order.transpose_to_xyz(array)
    if crystallographic_info is not None:
        crystallographic_info.grid.axis_order = AxisOrder.XYZ
        crystallographic_info.grid.dimensions = tuple(array.shape)
    return array


def load_density_map_from_columns(
    mtz_path: str,
    f_label: str,
    phi_label: str,
    *,
    map_type: MapType | str | None = None,
    sample_rate: float = 0.0,
) -> DensityMapData | None:
    """Grid explicit F/PHI columns into a canonical :class:`DensityMapData`.

    The F/PHI labels determine how the MTZ is transformed. ``map_type``
    records the semantic meaning of those coefficients and is never
    inferred from the column labels.

    Missing labels raise :exc:`MtzColumnNotFoundError` listing the available
    F/PHI columns so the caller can decide the fallback behavior.
    """
    mtz = gemmi.read_mtz_file(mtz_path)

    requested_f = f_label.strip().upper()
    requested_phi = phi_label.strip().upper()

    f_set = {col.label.upper(): col.label for col in mtz.columns if col.type == "F"}
    p_set = {col.label.upper(): col.label for col in mtz.columns if col.type == "P"}

    resolved_f = f_set.get(requested_f)
    if resolved_f is None:
        raise MtzColumnNotFoundError(
            f"Requested F label '{requested_f}' not found in {mtz_path}. "
            f"Available F labels: {sorted(f_set)}"
        )

    resolved_phi = p_set.get(requested_phi)
    if resolved_phi is None:
        raise MtzColumnNotFoundError(
            f"Requested PHI label '{requested_phi}' not found in {mtz_path}. "
            f"Available PHI labels: {sorted(p_set)}"
        )

    resolved_map_type = (
        MapType.coerce(map_type) if map_type is not None else None
    )
    xtal_map_type = (
        resolved_map_type.value if resolved_map_type is not None else "CCP4_MAP"
    )

    # Fourier boundary: everything up to here only inspects reflection
    # headers; the volume materializes at this transform.
    log.info(
        f"⚙️ Gridding MTZ coefficients: {resolved_f} + {resolved_phi} "
        f"(map_type={resolved_map_type}, sample_rate={sample_rate}) "
        f"from {mtz_path}"
    )
    grid = mtz.transform_f_phi_to_map(
        resolved_f, resolved_phi, sample_rate=sample_rate
    )
    log.info(f"ℹ️ Gemmi grid nu/nv/nw: {grid.nu}/{grid.nv}/{grid.nw}")

    crystallographic_info = normalize_crystallographic_info_from_dict(
        crystallographic_info_from_grid(grid, map_type=xtal_map_type)
    )
    crystallographic_info.log_summary()

    # Convert to a NumPy array in the canonical XYZ axis order. The
    # spacing, origin, and transformations come from
    # crystallographic_info_from_grid() -- no manual overrides here.
    np_array = _grid_to_xyz_array(grid, crystallographic_info)
    log.info(f"ℹ️ Loaded MTZ map shape: {np_array.shape}")
    _log_density_statistics(np_array)

    return DensityMapData(
        volume=np_array,
        crystallographic_info=crystallographic_info,
        source=mtz_path,
        map_type=resolved_map_type,
    )


def load_mtz_file(
    spec: MtzFileSpec,
) -> tuple[DensityMapData | None, DensityMapData | None]:
    """Load the 2Fo-Fc (map) and Fo-Fc (difference) maps from an MTZ file spec.

    Args:
        spec: MTZ file spec with the map (2Fo-Fc) and difference (Fo-Fc)
            coefficient column pairs.

    Returns:
        tuple of (map_data, difference_data); either element is None when
        the requested columns are not present in the file.
    """
    log.info(
        f"Loading maps from MTZ spec: {spec.file_path} "
        f"({spec.map_coefficients.f_label}/{spec.map_coefficients.phi_label}, "
        f"{spec.difference_coefficients.f_label}/{spec.difference_coefficients.phi_label})"
    )
    map_data = _load_spec_map(spec.file_path, spec.map_coefficients)
    difference_data = _load_spec_map(spec.file_path, spec.difference_coefficients)
    return map_data, difference_data


load_maps_from_mtz_file_spec = load_mtz_file


def _load_spec_map(
    mtz_path: str,
    coefficients: MtzColumnPair,
) -> DensityMapData | None:
    """Grid one map from an :class:`MtzColumnPair`, returning None when absent.

    ``load_density_map_from_columns`` raises :exc:`MtzColumnNotFoundError` for
    missing labels; the spec-based loader treats that as "column pair not
    present" (None) so a file with only 2Fo-Fc still yields a usable tuple.
    Other errors propagate.
    """
    try:
        return load_density_map_from_columns(
            mtz_path,
            coefficients.f_label,
            coefficients.phi_label,
            map_type=coefficients.map_type,
        )
    except MtzColumnNotFoundError:
        return None


def _resolve_centroid(
    centroid: tuple[float, float, float] | None,
    crystallographic_info: CrystallographicInfo,
) -> tuple[float, float, float]:
    """Return the caller-provided centroid or fall back to the unit-cell center.

    The centroid is in Cartesian Å orthogonal coordinates. The default
    centroid is the fractional cell center ``(0.5, 0.5, 0.5)`` mapped to
    orth via ``frac_to_orth`` -- correct for non-orthogonal cells, where
    ``a/2, b/2, c/2`` would be wrong.
    """
    if centroid is not None:
        return tuple(float(v) for v in centroid)

    frac_center = crystallographic_info.unit_cell.fractional_center
    frac_to_orth = crystallographic_info.transforms.frac_to_orth
    orth = np.asarray(frac_to_orth, dtype=np.float64) @ np.asarray(frac_center)
    centroid = tuple(float(c) for c in orth)
    log.info(f"🧬 Using unit cell center as default centroid: {centroid}")
    return centroid


def _log_grid_consistency(map_data: DensityMapData, label: str) -> DensityMapData:
    """Enforce and log the ``volume.shape == grid.dimensions`` invariant.

    The canonical pipeline must keep the metadata grid dimensions in lock-step
    with the volume array: the renderer computes ``cart = (vertex / dims) @
    frac_to_orth.T + origin``, so a desync (e.g. after symmetry expansion)
    silently misplaces the density.
    """
    volume_shape = tuple(map_data.volume.shape)
    grid_dims = tuple(map_data.crystallographic_info.grid.dimensions)
    if volume_shape != grid_dims:
        log.info(
            f"🔗 Grid dims invariant: {label} changed volume shape {grid_dims} -> "
            f"{volume_shape}; syncing metadata"
        )
        map_data.crystallographic_info.grid.dimensions = volume_shape
    else:
        log.info(f"🔗 Grid dims invariant: {label} consistent ({volume_shape})")
    return map_data


def _load_mtz_density_map(
    map_path: pathlib.Path,
    *,
    mtz_spec: MtzDensitySpec | None = None,
) -> DensityMapData | None:
    """Grid an MTZ file into a canonical :class:`DensityMapData`.

    ``mtz_spec`` selects the coefficients: explicit ``f_label``/``phi_label``
    when both are given, otherwise deterministic map-type selection. A plain
    ``map_type`` alone is honored too (``MtzDensitySpec`` defaulting to
    2Fo-Fc).
    """
    try:
        log.info(
            "MTZ detected — gridding density from reflections "
            "(not a CCP4 pre-computed density map)."
        )

        if mtz_spec is None:
            mtz_spec = MtzDensitySpec()

        if mtz_spec.f_label is not None and mtz_spec.phi_label is not None:
            return load_density_map_from_columns(
                str(map_path),
                mtz_spec.f_label,
                mtz_spec.phi_label,
                map_type=mtz_spec.map_type,
                sample_rate=mtz_spec.sample_rate,
            )

        return load_density_map_auto_mtz(
            str(map_path),
            map_type=mtz_spec.map_type,
            sample_rate=mtz_spec.sample_rate,
        )

    except Exception as e:
        log.error(f"❌ Could not load MTZ density map from {map_path}: {e}")
        return None


def _load_ccp4_density_map(
    map_path: pathlib.Path,
    *,
    pdb_path: pathlib.Path | None = None,
    expand_symmetry: bool,
    convert_to_cartesian: bool,
    carve_density: bool,
    carve_cutoff: float,
    carve_density_centroid: bool,
    centroid: tuple[float, float, float] | None,
    centroid_cutoff: float,
    progress_callback: Callable | None,
) -> DensityMapData | None:
    """Load a CCP4 map (``.map``/``.ccp4``) into a canonical :class:`DensityMapData`.

    Pipeline (each stage operates on the object, never unpacked):
        read -> :func:`_density_map_data_from_ccp4` -> [symmetry expansion]
        -> :func:`_carve_protein_density` -> :func:`_carve_centroid_density`.

    Coordinate semantics:
        Carving is a grid operation measured in Cartesian Å. Each grid point
        (i, j, k) has Cartesian position ``origin + index * spacing``, and
        ``centroid`` is in the same Cartesian Å frame. ``pdb_path``, when the
        corresponding structure exists, is used for coordinate-based symmetry
        expansion and protein carving. ``convert_to_cartesian`` is deprecated
        and currently a no-op: the canonical pipeline keeps the native-cell
        volume and applies ``frac_to_orth`` at render time.
    """
    try:
        log.info(f"Loading CCP4 map: {map_path}")

        if pdb_path is None:
            pdb_path = map_path.with_suffix(".pdb")
        else:
            pdb_path = pathlib.Path(pdb_path)
        if pdb_path.exists():
            log.info(f"Loading corresponding PDB file: {pdb_path}")
        else:
            log.warning(f"⚠️ Corresponding PDB file not found: {pdb_path}")

        ccp4_map = gemmi.read_ccp4_map(str(map_path))

        map_data = _density_map_data_from_ccp4(ccp4_map, str(map_path))

        if expand_symmetry:
            map_data = _expand_ccp4_symmetry_op(
                map_data,
                str(map_path),
                str(pdb_path),
                _build_header_from_ccp4_map(ccp4_map),
            )

        if convert_to_cartesian:
            log.warning(
                "⚠️ convert_to_cartesian is deprecated; density maps remain in "
                "canonical grid coordinates (frac_to_orth is applied at render time)"
            )

        map_data = _carve_protein_density(
            map_data,
            pdb_path,
            carve_density,
            carve_cutoff,
            progress_callback,
        )

        if carve_density_centroid:
            map_data = _carve_centroid_density(
                map_data,
                centroid,
                centroid_cutoff,
                progress_callback,
            )

        return map_data

    except FileNotFoundError:
        log.error(f"❌ File not found: {map_path}")
        return None
    except Exception as e:
        log.error(f"❌ Could not load CCP4 map from {map_path}: {e}")
        return None


def _density_map_data_from_ccp4(
    ccp4_map: gemmi.Ccp4Map,
    source: str,
) -> DensityMapData:
    """Build the canonical :class:`DensityMapData` from a CCP4 grid."""
    grid = ccp4_map.grid

    crystallographic_info = crystallographic_info_from_grid(grid)
    crystallographic_info = normalize_crystallographic_info_from_dict(
        crystallographic_info
    )

    crystallographic_info.log_grid_metadata()

    volume = _grid_to_xyz_array(grid, crystallographic_info)
    log.info(f"ℹ️ Loaded CCP4 map shape: {volume.shape}")
    _log_density_statistics(volume)

    return DensityMapData(
        volume=volume,
        crystallographic_info=crystallographic_info,
        source=source,
    )


def _expand_ccp4_symmetry_op(
    map_data: DensityMapData,
    map_path: str,
    pdb_path: str,
    header,
) -> DensityMapData:
    """Apply symmetry expansion to the map object when operations exist."""
    if header.nsymbt <= 0:
        log.info("ℹ️ No symmetry operations found in map header")
        return map_data

    log.info(f"🔄 Expanding symmetry operations (NSYMBT: {header.nsymbt})")
    expanded = expand_ccp4_symmetry_optimized(
        map_data.volume,
        map_path,
        header,
        pdb_path,
    )
    map_data.volume = expanded
    log.info(f"✅ Symmetry expanded - new shape: {expanded.shape}")
    return _log_grid_consistency(map_data, "symmetry expansion")


def _carve_protein_density(
    map_data: DensityMapData,
    pdb_path,
    carve_density: bool,
    carve_cutoff: float,
    progress_callback,
) -> DensityMapData:
    """Carve density around the protein structure when requested."""
    log.parameter("carve_density", carve_density)
    if carve_density and pdb_path and os.path.exists(pdb_path):
        log.info(f"🔪 Carving density within {carve_cutoff}Å of protein structure...")
        map_data.volume = carve_density_around_protein(
            map_data.volume,
            pdb_path,
            map_data.crystallographic_info.grid.origin,
            map_data.crystallographic_info.grid.spacing,
            carve_cutoff,
            progress_callback,
        )
        log.info(
            f"✅ Density carving complete - new shape: {map_data.volume.shape}"
        )
    elif carve_density and not pdb_path:
        log.warning("⚠️ Density carving requested but no PDB file provided")
    return map_data


def _carve_centroid_density(
    map_data: DensityMapData,
    centroid: tuple[float, float, float] | None,
    centroid_cutoff: float,
    progress_callback,
) -> DensityMapData:
    """Carve density around the (resolved) centroid."""
    centroid = _resolve_centroid(centroid, map_data.crystallographic_info)
    log.info(
        f"🔪 Carving density within {centroid_cutoff}Å of centroid {centroid}..."
    )
    map_data.volume = carve_density_around_position(
        map_data.volume,
        centroid,
        map_data.crystallographic_info.grid.origin,
        map_data.crystallographic_info.grid.spacing,
        centroid_cutoff,
        progress_callback,
    )
    log.info(f"✅ Centroid carving complete - new shape: {map_data.volume.shape}")
    return map_data


def load_density_map(spec: DensityMapSpec) -> DensityMapData | None:
    """Canonical density-map loader: dispatch on ``spec.map_path`` suffix.

    - ``.mtz`` grids reflections via :func:`_load_mtz_density_map` using
      ``spec.mtz`` (:class:`MtzDensitySpec`).
    - ``.map`` / ``.ccp4`` / ``.omap`` load via :func:`_load_ccp4_density_map`
      using the CCP4-relevant spec fields.

    ``pdb_path`` defaults to the map path with a ``.pdb`` suffix. The returned
    volume is always in canonical XYZ axis order; ``centroid`` is in Cartesian
    Å. Symmetry expansion and carving apply to CCP4 maps.

    Returns:
        DensityMapData (volume + crystallographic_info) or None if loading fails
    """
    map_path = pathlib.Path(spec.map_path)
    suffix = map_path.suffix.lower()

    if suffix == ".mtz":
        return _load_mtz_density_map(map_path, mtz_spec=spec.mtz)

    if suffix in (".map", ".ccp4", ".omap"):
        return _load_ccp4_density_map(
            map_path,
            pdb_path=resolve_pdb_path(map_path, spec.pdb_path),
            expand_symmetry=spec.expand_symmetry,
            convert_to_cartesian=False,
            carve_density=spec.carve_density,
            carve_cutoff=spec.carve_cutoff,
            carve_density_centroid=spec.carve_density_centroid,
            centroid=spec.centroid,
            centroid_cutoff=spec.centroid_cutoff,
            progress_callback=spec.progress_callback,
        )

    raise ValueError(
        f"Unsupported density map format: {spec.map_path!r} "
        f"(suffix {suffix!r})"
    )


def resolve_pdb_path(map_path: Path, pdb_path: str | Path | None) -> Path:
    if pdb_path is None:
        pdb_path = map_path.with_suffix(".pdb")
    else:
        pdb_path = pathlib.Path(pdb_path)
    return pdb_path


def load_ccp4_map_optimized(
    map_path: str,
    pdb_path: str = None,
    expand_symmetry: bool = True,
    convert_to_cartesian: bool = False,
    carve_density: bool = True,
    carve_cutoff: float = 4.0,
    progress_callback: Callable = None,
    carve_density_centroid: bool = False,
    centroid: tuple[float, float, float] | None = None,
    centroid_cutoff: float = 15.0,
) -> DensityMapData | None:
    """
    Load a CCP4 map file using Gemmi with optimized symmetry expansion and optional density carving.

    Backward-compatible wrapper around :func:`load_density_map`.

    Args:
        map_path: Path to CCP4 map file
        pdb_path: Optional path to PDB file for coordinate-based optimization
        expand_symmetry: Whether to expand symmetry operations (default: True)
        convert_to_cartesian: Whether to convert from fractional to cartesian coordinates (default: False)
        carve_density: Whether to carve density within cutoff distance of protein (default: False)
        carve_cutoff: Distance in Ångströms for density carving (default: 4.0)
        progress_callback: Callback function for progress updates
        carve_density_centroid: Whether to carve density around centroid (default: False)
        centroid: Tuple of (x, y, z) coordinates for centroid carving (default: None)
        centroid_cutoff: Distance cutoff for centroid carving in Å (default: 15.0)

    Returns:
        DensityMapData (volume + crystallographic_info) or None if loading fails
    """
    return load_density_map(
        DensityMapSpec(
            map_path=map_path,
            pdb_path=pdb_path,
            expand_symmetry=expand_symmetry,
            carve_density=carve_density,
            carve_cutoff=carve_cutoff,
            carve_density_centroid=carve_density_centroid,
            centroid=centroid,
            centroid_cutoff=centroid_cutoff,
            progress_callback=progress_callback,
        )
    )


def carve_density_with_gemmi(
    ccp4_map: gemmi.Ccp4Map, pdb_path: str, cutoff: float, progress_callback=None
):
    """
    Carve density: keep voxels within `cutoff` Å of any atom, zero others.
    Uses gemmi.FloatGrid.mask_points_in_constant_radius (fast, C++ backend).
    """
    try:
        if progress_callback:
            progress_callback(10, 100, "Loading PDB structure...")

        st = gemmi.read_structure(pdb_path)
        st.remove_hydrogens()
        st.setup_entities()

        if progress_callback:
            progress_callback(20, 100, "Preparing density grid...")

        grid = ccp4_map.grid
        orig = grid.clone()

        # Create a mask grid (same size)
        mask = grid.clone()
        mask.fill(0.0)

        if progress_callback:
            progress_callback(30, 100, "Creating atom mask...")

        # Mark voxels near atoms
        mask.mask_points_in_constant_radius(
            st[0],
            cutoff,
            1.0,
            ignore_hydrogen=True,
            ignore_zero_occupancy_atoms=True,
        )

        if progress_callback:
            progress_callback(50, 100, "Analyzing mask region...")

        # Get bounding box of non-zero mask region
        nz_box = mask.get_nonzero_extent()
        size = nz_box.get_size()
        if size.x == 0 or size.y == 0 or size.z == 0:
            log.warning("No atoms found within cutoff distance, returning original map")
            if progress_callback:
                progress_callback(100, 100, "No atoms found - no carving needed")
            return ccp4_map  # no region to carve

        if progress_callback:
            progress_callback(60, 100, "Converting coordinates...")

        # Convert fractional box corners to grid indices
        # Use the correct gemmi API for coordinate conversion
        uc = grid.unit_cell

        # Convert fractional coordinates to orthogonal coordinates
        min_orth = uc.orthogonalize(nz_box.minimum)
        max_orth = uc.orthogonalize(nz_box.maximum)

        # Convert orthogonal coordinates to fractional coordinates
        min_frac = uc.fractionalize(min_orth)
        max_frac = uc.fractionalize(max_orth)

        # Convert fractional coordinates to grid indices
        # Get grid dimensions
        grid_shape = grid.shape

        # Convert fractional coordinates to grid indices
        lower_idx = [
            int(min_frac.x * grid_shape[0]),
            int(min_frac.y * grid_shape[1]),
            int(min_frac.z * grid_shape[2]),
        ]

        upper_idx = [
            int(max_frac.x * grid_shape[0]),
            int(max_frac.y * grid_shape[1]),
            int(max_frac.z * grid_shape[2]),
        ]

        # Ensure indices are within bounds
        lower_idx[0] = max(0, min(lower_idx[0], grid_shape[0] - 1))
        lower_idx[1] = max(0, min(lower_idx[1], grid_shape[1] - 1))
        lower_idx[2] = max(0, min(lower_idx[2], grid_shape[2] - 1))

        upper_idx[0] = max(0, min(upper_idx[0], grid_shape[0] - 1))
        upper_idx[1] = max(0, min(upper_idx[1], grid_shape[1] - 1))
        upper_idx[2] = max(0, min(upper_idx[2], grid_shape[2] - 1))

        # Compute shape (inclusive range)
        shape = [upper_idx[i] - lower_idx[i] + 1 for i in range(3)]

        # Extract subarrays (NumPy views)
        mask_sub = mask.get_subarray(lower_idx, shape)
        orig_sub = orig.get_subarray(lower_idx, shape)
        grid_sub = grid.get_subarray(lower_idx, shape)

        # Check if mask has any non-zero values before proceeding
        mask_nonzero_count = np.count_nonzero(mask_sub)
        if mask_nonzero_count == 0:
            log.warning("⚠️ Mask is empty - no atoms within cutoff distance")
            log.warning("Returning original map without carving")
            return ccp4_map

        # Debug: Check original density range
        orig_array = np.array(orig)
        orig_min, orig_max = orig_array.min(), orig_array.max()
        orig_nonzero = np.count_nonzero(orig_array)
        log.info(f"Original density range: {orig_min:.6f} to {orig_max:.6f}")
        log.info(f"Original non-zero voxels: {orig_nonzero:,}")

        if progress_callback:
            progress_callback(80, 100, "Applying density mask...")

        # Apply mask - preserve original density where mask is non-zero
        grid.fill(0.0)
        grid_sub[mask_sub != 0.0] = orig_sub[mask_sub != 0.0]

        if progress_callback:
            progress_callback(90, 100, "Finalizing carved density...")

        # Verify that we have preserved some density
        carved_array = np.array(grid)
        carved_min, carved_max = carved_array.min(), carved_array.max()
        non_zero_count = np.count_nonzero(carved_array)

        log.info("✅ Density carving complete using gemmi")
        log.info(f"   Mask had {mask_nonzero_count:,} non-zero voxels")
        log.info(f"   Carved density range: {carved_min:.6f} to {carved_max:.6f}")
        log.info(f"   Preserved {non_zero_count:,} non-zero voxels")

        # Check if the carved density has meaningful values for isosurface extraction
        if carved_max <= 0.0:
            log.warning(
                "⚠️ Carved density has no positive values - isosurface extraction may fail"
            )
        elif carved_max < 0.1:
            log.warning(
                f"⚠️ Carved density max value ({carved_max:.6f}) is very small - consider adjusting isosurface level"
            )

        if progress_callback:
            progress_callback(100, 100, "Density carving complete")

        return ccp4_map

    except Exception as e:
        log.error(f"❌ Error in carve_density_with_gemmi: {e}")
        log.warning("Returning original map without carving")
        return ccp4_map


def load_ccp4_map(
    map_path: str,
    expand_symmetry: bool = True,
    convert_to_cartesian: bool = False,
    carve_density: bool = True,
    carve_cutoff: float = 4.0,
    progress_callback=None,
    carve_density_centroid: bool = False,
    pdb_centroid_or_clicked_position: tuple[float, float, float] | None = None,
    centroid_cutoff: float = 15.0,
) -> DensityMapData | None:
    """
    Load a CCP4 map using Gemmi.

    Backward-compatible wrapper around :func:`load_density_map` (CCP4 branch).
    The centroid argument keeps its historical name
    ``pdb_centroid_or_clicked_position``; the loader itself only needs the
    Cartesian-Å coordinate, and the caller decides whether it came from a PDB
    centroid, a mouse click, or the unit-cell center.

    Args:
        map_path: Path to a CCP4 map (``.map``, ``.ccp4``, …).
        expand_symmetry: Whether to expand symmetry operations (default: True)
        convert_to_cartesian: Whether to convert from fractional to cartesian coordinates (default: False)
        carve_density: Whether to carve density around protein structure (default: True)
        carve_cutoff: Distance cutoff for protein carving in Å (default: 4.0)
        progress_callback: Callback function for progress updates
        carve_density_centroid: Whether to carve density around centroid (default: False)
        pdb_centroid_or_clicked_position: Tuple of (x, y, z) Cartesian Å coordinates for centroid carving (default: None — falls back to unit-cell center)
        centroid_cutoff: Distance cutoff for centroid carving in Å (default: 15.0)

    Returns:
        DensityMapData (volume + crystallographic_info) or None if loading fails
    """
    return load_density_map(
        DensityMapSpec(
            map_path=map_path,
            pdb_path=None,
            expand_symmetry=expand_symmetry,
            carve_density=carve_density,
            carve_cutoff=carve_cutoff,
            carve_density_centroid=carve_density_centroid,
            centroid=pdb_centroid_or_clicked_position,
            centroid_cutoff=centroid_cutoff,
            progress_callback=progress_callback,
        )
    )


def load_ccp4_maps(
    *map_paths: str,
    expand_symmetry: bool = False,
    pdb_paths: list[str] | None = None,
) -> list["DensityMapData"] | None:
    """
    Load multiple CCP4 map files at once with optimized symmetry expansion.

    Args:
        *map_paths: Variable number of paths to CCP4 map files
        expand_symmetry: Whether to expand symmetry operations (default: True)
        pdb_paths: Optional list of PDB file paths for coordinate-based optimization

    Returns:
        List of tuples (numpy array, crystallographic_info) or None if any loading fails

    Example:
        # Load multiple maps like UglyMol
        maps = load_ccp4_maps("data/1mru.map", "data/1mru_diff.map")
        if maps:
            main_map, diff_map = maps
            main_volume, main_info = main_map
            diff_volume, diff_info = diff_map
    """
    try:
        import os

        if not map_paths:
            log.error("❌ No map paths provided to load_ccp4_maps")
            return None

        log.info(f"🔄 Loading {len(map_paths)} CCP4 maps: {map_paths}")

        loaded_maps = []
        for i, map_path in enumerate(map_paths):
            log.info(f"📁 Loading map {i+1}/{len(map_paths)}: {map_path}")

            # Try to find corresponding PDB file
            pdb_path = None
            if pdb_paths and i < len(pdb_paths):
                pdb_path = pdb_paths[i]
            else:
                pdb_path = _find_corresponding_pdb_file(map_path)

            # Use optimized expansion if PDB file is available
            if pdb_path and os.path.exists(pdb_path):
                log.info(f"Using optimized expansion with PDB: {pdb_path}")
                result = load_ccp4_map_optimized(
                    map_path,
                    pdb_path,
                    expand_symmetry=expand_symmetry,
                    carve_density=True,
                )
            else:
                log.info(f"Using standard expansion (no PDB file found)")
                result = load_ccp4_map(map_path, expand_symmetry=expand_symmetry)

            if result is None:
                log.error(f"❌ Failed to load map {i+1}: {map_path}")
                return None

            loaded_maps.append(result)
            log.info(f"✅ Successfully loaded map {i+1}: {map_path}")

        log.info(f"🎉 Successfully loaded all {len(map_paths)} maps")
        return loaded_maps

    except Exception as e:
        log.error(f"❌ Error in load_ccp4_maps: {e}")
        return None


def load_mtz_maps(
    *mtz_paths: str, sample_rate=0.0
) -> list["DensityMapData"] | None:
    """
    Load multiple MTZ files at once (similar to UglyMol's V.load_ccp4_maps).

    Args:
        *mtz_paths: Variable number of paths to MTZ map files
        sample_rate: Sampling rate for map generation (0.0 = full resolution)

    Returns:
        List of tuples (numpy array, crystallographic_info) or None if any loading fails

    Example:
        # Load multiple MTZ maps like UglyMol
        maps = load_mtz_maps("data/2fofc.mtz", "data/fofc.mtz")
        if maps:
            main_map, diff_map = maps
            main_volume, main_info = main_map
            diff_volume, diff_info = diff_map
    """
    try:
        if not mtz_paths:
            log.error("❌ No MTZ paths provided to load_mtz_maps")
            return None

        log.info(f"🔄 Loading {len(mtz_paths)} MTZ maps: {mtz_paths}")

        loaded_maps = []
        for i, mtz_path in enumerate(mtz_paths):
            log.info(f"📁 Loading MTZ map {i+1}/{len(mtz_paths)}: {mtz_path}")

            result = load_density_map_auto_mtz(mtz_path, sample_rate=sample_rate)
            if result is None:
                log.error(f"❌ Failed to load MTZ map {i+1}: {mtz_path}")
                return None

            loaded_maps.append(result)
            log.info(f"✅ Successfully loaded MTZ map {i+1}: {mtz_path}")

        log.info(f"🎉 Successfully loaded all {len(mtz_paths)} MTZ maps")
        return loaded_maps

    except Exception as e:
        log.error(f"❌ Error in load_mtz_maps: {e}")
        return None


def load_density_map_auto_mtz(
    mtz_path: str,
    *,
    map_type: MapType | str = MapType.TWO_FO_FC,
    sample_rate: float = 0.0,
) -> DensityMapData | None:
    """
    Load a density map from an MTZ file for a specific map type.

    Coefficient selection is deterministic: the requested ``map_type``
    (``"2Fo-Fc"`` by default) selects only coefficient pairs that represent
    that map. No arbitrary fallbacks (FP/PHIC, etc.) are tried — if the
    relevant columns are absent the load fails explicitly.

    Args:
        mtz_path: Path to MTZ file
        map_type: Requested map type ("2Fo-Fc" or "Fo-Fc"), as a string or
            MapType/ElMo MapType enum value
        sample_rate: Sampling rate for map generation (0.0 = full resolution)

    Returns:
        DensityMapData or None if loading fails
    """
    try:
        resolved_map_type = MapType.coerce(map_type)
        f_label, phi_label = _select_map_columns(mtz_path, resolved_map_type)
        log.info(
            f"Auto-loading MTZ file: {mtz_path} "
            f"(map type: {resolved_map_type.value}, columns: {f_label}/{phi_label})"
        )
        return load_density_map_from_columns(
            mtz_path,
            f_label,
            phi_label,
            map_type=resolved_map_type,
            sample_rate=sample_rate,
        )

    except Exception as e:
        log.error(f"❌ Error in auto-loading MTZ file {mtz_path}: {e}")
        return None


def load_density_map_with_columns(
    mtz_path: str,
    f_column: str,
    phi_column: str,
    sample_rate: float = 0.0,
    *,
    map_type: MapType | str | None = None,
) -> DensityMapData | None:
    """
    Load density map from MTZ file with specific F and PHI column selections.

    Args:
        mtz_path: Path to MTZ file
        f_column: F column label
        phi_column: PHI column label
        sample_rate: Sampling rate for map generation (0.0 = full resolution)
        map_type: Semantic map identity (not inferred from labels)

    Returns:
        DensityMapData or None if loading fails
    """
    try:
        log.info(f"Loading MTZ file with specific columns: {mtz_path}")
        log.info(f"📊 F column: {f_column}, PHI column: {phi_column}")

        result = load_density_map_from_columns(
            mtz_path,
            f_column,
            phi_column,
            map_type=map_type,
            sample_rate=sample_rate,
        )

        if result is not None:
            log.info(f"✅ Successfully loaded MTZ with {f_column}/{phi_column}")
            return result
        else:
            log.error(
                f"❌ Failed to load MTZ file with columns {f_column}/{phi_column}"
            )
            return None

    except Exception as e:
        log.error(
            f"❌ Error loading MTZ file {mtz_path} with columns {f_column}/{phi_column}: {e}"
        )
        return None


def _find_corresponding_pdb_file(map_path: str) -> str | None:
    """
    Try to find a corresponding PDB file for the given map file.

    Args:
        map_path: Path to the map file

    Returns:
        Path to corresponding PDB file if found, None otherwise
    """
    try:
        from pathlib import Path

        map_file = Path(map_path)
        map_dir = map_file.parent
        map_stem = map_file.stem

        # Common PDB file patterns to try
        pdb_patterns = [
            f"{map_stem}.pdb",
            f"{map_stem}.ent",
            f"{map_stem.upper()}.pdb",
            f"{map_stem.upper()}.ent",
            f"{map_stem.lower()}.pdb",
            f"{map_stem.lower()}.ent",
        ]

        # Try patterns in the same directory
        for pattern in pdb_patterns:
            pdb_path = map_dir / pattern
            if pdb_path.exists():
                log.info(f"Found corresponding PDB file: {pdb_path}")
                return str(pdb_path)

        # Try patterns with common prefixes/suffixes
        additional_patterns = [
            f"{map_stem}_final.pdb",
            f"{map_stem}_model.pdb",
            f"{map_stem}_structure.pdb",
            f"pdb{map_stem}.ent",
            f"{map_stem}.cif",  # Sometimes structures are in CIF format
        ]

        for pattern in additional_patterns:
            pdb_path = map_dir / pattern
            if pdb_path.exists():
                log.info(f"Found corresponding structure file: {pdb_path}")
                return str(pdb_path)

        log.info(f"ℹ️ No corresponding PDB file found for {map_path}")
        return None

    except Exception as e:
        log.warning(f"⚠️ Error searching for PDB file: {e}")
        return None


def expand_ccp4_symmetry_optimized(
    volume: np.ndarray, map_path: str, header, pdb_path: str = None
) -> np.ndarray:
    """
    Optimized CCP4 map expansion using symmetry operations.
    Only expands to cover actual molecular coordinates, avoiding empty unit cells.

    Args:
        volume: The original volume data
        map_path: Path to the CCP4 map file
        header: The CCP4 header object containing symmetry information
        pdb_path: Optional path to PDB file for coordinate-based optimization

    Returns:
        Optimized expanded volume with symmetry operations applied
    """
    try:
        log.info(f"🔄 Optimized symmetry expansion for {map_path}")

        # Read the raw file to access symmetry operations
        map_buffer, nsymbt, n_grid, start, end = _read_symmetry_ops(map_path, header)
        if nsymbt == 0:
            log.info("ℹ️ No symmetry operations to expand")
            return volume

        log.info(f"📐 Found {nsymbt} bytes of symmetry operations")

        # Calculate optimal expansion bounds based on molecular coordinates
        if pdb_path and os.path.exists(pdb_path):
            optimal_bounds = _calculate_optimal_expansion_bounds(
                volume, header, map_path, pdb_path, n_grid, start
            )
        else:
            # Fallback to original 2x expansion
            optimal_bounds = {
                "shape": [n * 2 for n in volume.shape],
                "offset": [n // 2 for n in [n * 2 for n in volume.shape]],
            }

        # Create optimized expanded volume
        expanded_shape = optimal_bounds["shape"]
        expanded_volume = np.zeros(expanded_shape, dtype=volume.dtype)
        center_offset = optimal_bounds["offset"]

        # Copy original volume to center of expanded volume
        expanded_volume[
            center_offset[0] : center_offset[0] + volume.shape[0],
            center_offset[1] : center_offset[1] + volume.shape[1],
            center_offset[2] : center_offset[2] + volume.shape[2],
        ] = volume

        log.info(f"📊 Original volume shape: {volume.shape}")
        log.info(f"📊 Optimized expanded volume shape: {expanded_shape}")
        log.info(
            f"📊 Expansion factor: {expanded_shape[0] * expanded_shape[1] * expanded_shape[2] / (volume.shape[0] * volume.shape[1] * volume.shape[2]):.1f}x"
        )

        # Process each symmetry operation
        symmetry_count = _apply_symmetry_mates(
            volume,
            expanded_volume,
            map_buffer,
            nsymbt,
            n_grid,
            start,
            end,
            center_offset,
        )

        log.info(f"✅ Applied {symmetry_count} symmetry operations")
        return expanded_volume

    except Exception as e:
        log.error(f"❌ Error in optimized symmetry expansion: {e}")
        log.info("ℹ️ Returning original volume without symmetry expansion")
        return volume


def expand_ccp4_symmetry(volume: np.ndarray, map_path: str, header) -> np.ndarray:
    """
    Expand CCP4 map using symmetry operations.

    Args:
        volume: The original volume data
        map_path: Path to the CCP4 map file
        header: The CCP4 header object containing symmetry information

    Returns:
        Expanded volume with symmetry operations applied
    """
    try:
        import re

        from molib.xtal.uglymol.map.helpers import (
            extract_symop_text,
            parse_symmetry_operator_to_matrix,
        )

        log.info(f"🔄 Expanding symmetry for {map_path}")

        # Read the raw file to access symmetry operations
        map_buffer, nsymbt, n_grid, start, end = _read_symmetry_ops(map_path, header)
        if nsymbt == 0:
            log.info("ℹ️ No symmetry operations to expand")
            return volume

        log.info(f"📐 Found {nsymbt} bytes of symmetry operations")

        # Get axis mapping (assuming standard order)
        ax, ay, az = 0, 1, 2  # Default axis order

        # Create expanded volume (2x larger to accommodate symmetry mates)
        expanded_shape = [n * 2 for n in volume.shape]
        expanded_volume = np.zeros(expanded_shape, dtype=volume.dtype)

        # Copy original volume to center of expanded volume
        center_offset = [n // 2 for n in expanded_shape]
        expanded_volume[
            center_offset[0] : center_offset[0] + volume.shape[0],
            center_offset[1] : center_offset[1] + volume.shape[1],
            center_offset[2] : center_offset[2] + volume.shape[2],
        ] = volume

        log.info(f"📊 Original volume shape: {volume.shape}")
        log.info(f"📊 Expanded volume shape: {expanded_shape}")

        # Process each symmetry operation
        symmetry_count = _apply_symmetry_mates(
            volume,
            expanded_volume,
            map_buffer,
            nsymbt,
            n_grid,
            start,
            end,
            center_offset,
        )

        log.info(f"✅ Applied {symmetry_count} symmetry operations")
        return expanded_volume

    except Exception as e:
        log.error(f"❌ Error expanding symmetry: {e}")
        log.info("ℹ️ Returning original volume without symmetry expansion")
        return volume


def apply_symmetry_to_volume(
    source_volume: np.ndarray,
    target_volume: np.ndarray,
    mat: list,
    ax: int,
    ay: int,
    az: int,
    start: list,
    end: list,
    center_offset: list,
):
    """
    Apply a symmetry operation to create a symmetry mate in the target volume.

    Args:
        source_volume: Source volume data
        target_volume: Target volume to fill with symmetry mate
        mat: 4x4 transformation matrix
        ax, ay, az: Axis mapping
        start, end: Grid boundaries
        center_offset: Offset to center of target volume
    """
    # Get source volume dimensions
    src_shape = source_volume.shape

    # Iterate through source volume coordinates
    for z in range(src_shape[2]):
        for y in range(src_shape[1]):
            for x in range(src_shape[0]):
                # Get original grid coordinates
                it = [x + start[0], y + start[1], z + start[2]]

                # Apply symmetry transformation
                xyz = [0, 0, 0]
                for j in range(3):
                    xyz[j] = (
                        it[ax] * mat[j][0]
                        + it[ay] * mat[j][1]
                        + it[az] * mat[j][2]
                        + mat[j][3]
                    )

                # Convert to target volume coordinates
                target_x = int(xyz[0]) + center_offset[0]
                target_y = int(xyz[1]) + center_offset[1]
                target_z = int(xyz[2]) + center_offset[2]

                # Check bounds and copy value
                if (
                    0 <= target_x < target_volume.shape[0]
                    and 0 <= target_y < target_volume.shape[1]
                    and 0 <= target_z < target_volume.shape[2]
                ):
                    target_volume[target_x, target_y, target_z] = source_volume[x, y, z]


def _calculate_optimal_expansion_bounds(
    volume: np.ndarray, header, map_path: str, pdb_path: str, n_grid: list, start: list
) -> dict:
    """
    Calculate optimal expansion bounds based on molecular coordinates and symmetry operations.

    Args:
        volume: Original volume data
        header: CCP4 header object
        map_path: Path to map file
        pdb_path: Path to PDB file
        n_grid: Grid dimensions
        start: Grid start coordinates

    Returns:
        Dictionary with optimal shape and offset for expansion
    """
    try:
        # Load PDB structure
        pdb = gemmi.read_structure(pdb_path)

        # Get all atomic coordinates
        atoms = _collect_atom_coordinates(pdb)

        if not atoms:
            log.warning("No atoms found in PDB, using default expansion")
            return {
                "shape": [n * 2 for n in volume.shape],
                "offset": [n // 2 for n in [n * 2 for n in volume.shape]],
            }

        coords = np.array(atoms)
        log.info(f"Found {len(atoms)} atoms in PDB structure")

        # Get symmetry operations
        map_buffer, nsymbt, n_grid, _, _ = _read_symmetry_ops(map_path, header)
        symmetry_operations = []

        for i in range(0, nsymbt, 80):
            symop = extract_symop_text(map_buffer, i)
            symop = symop.strip()

            # Skip identity operation
            if re.match(r"^\s*x\s*,\s*y\s*,\s*z\s*$", symop, re.I):
                continue

            try:
                symmetry_operations.append(_parse_symmetry_matrix(symop, n_grid))
            except Exception as e:
                log.warning(f"Failed to parse symmetry operation '{symop}': {e}")
                continue

        # Apply symmetry operations to get all symmetry-related coordinates
        all_coords = [coords]  # Start with original coordinates

        for symop_matrix in symmetry_operations:
            # Apply symmetry operation to coordinates
            sym_coords = np.zeros_like(coords)
            for i, coord in enumerate(coords):
                # Apply transformation matrix
                x, y, z = coord
                new_x = (
                    symop_matrix[0][0] * x
                    + symop_matrix[0][1] * y
                    + symop_matrix[0][2] * z
                    + symop_matrix[0][3]
                )
                new_y = (
                    symop_matrix[1][0] * x
                    + symop_matrix[1][1] * y
                    + symop_matrix[1][2] * z
                    + symop_matrix[1][3]
                )
                new_z = (
                    symop_matrix[2][0] * x
                    + symop_matrix[2][1] * y
                    + symop_matrix[2][2] * z
                    + symop_matrix[2][3]
                )
                sym_coords[i] = [new_x, new_y, new_z]

            all_coords.append(sym_coords)

        # Combine all coordinates
        all_coords = np.vstack(all_coords)

        # Calculate bounding box of all coordinates
        min_coords = all_coords.min(axis=0)
        max_coords = all_coords.max(axis=0)

        log.info(
            f"Coordinate bounds: X({min_coords[0]:.1f}, {max_coords[0]:.1f}), Y({min_coords[1]:.1f}, {max_coords[1]:.1f}), Z({min_coords[2]:.1f}, {max_coords[2]:.1f})"
        )

        # Convert to grid coordinates
        # Assuming the map is in the same coordinate system as the PDB
        grid_spacing = [1.0, 1.0, 1.0]  # This should be calculated from the map

        # Calculate grid bounds with some padding
        padding = 10  # Grid points of padding
        min_grid = np.floor(min_coords / np.array(grid_spacing)).astype(int) - padding
        max_grid = np.ceil(max_coords / np.array(grid_spacing)).astype(int) + padding

        # Ensure bounds are within reasonable limits
        min_grid = np.maximum(min_grid, [0, 0, 0])
        max_grid = np.minimum(
            max_grid, [n * 3 for n in volume.shape]
        )  # Max 3x expansion

        # Calculate optimal shape and offset
        optimal_shape = (max_grid - min_grid).tolist()
        optimal_offset = (-min_grid).tolist()

        # Ensure minimum size
        optimal_shape = [max(s, volume.shape[i]) for i, s in enumerate(optimal_shape)]

        log.info(f"Optimal expansion: shape={optimal_shape}, offset={optimal_offset}")
        log.info(
            f"Expansion factor: {np.prod(optimal_shape) / np.prod(volume.shape):.1f}x"
        )

        return {"shape": optimal_shape, "offset": optimal_offset}

    except Exception as e:
        log.error(f"Error calculating optimal expansion bounds: {e}")
        # Fallback to default expansion
        return {
            "shape": [n * 2 for n in volume.shape],
            "offset": [n // 2 for n in [n * 2 for n in volume.shape]],
        }


def carve_density_around_position(
    density_map: np.ndarray,
    position: tuple[float, float, float],
    grid_origin: dict,  # Dictionary with 'x', 'y', 'z' keys for grid origin in Å
    grid_spacing: dict,  # Dictionary with 'x', 'y', 'z' keys for grid spacing in Å
    cutoff_distance: float = 15.0,  # Distance in Ångströms to include around the centroid
    progress_callback=None,  # Callback function for progress tracking
) -> np.ndarray:
    """
    Carve out electron density within a specified distance of a centroid.

    Args:
        density_map: 3D numpy array of electron density
        position: Tuple of (x, y, z) coordinates of the centroid in Å
        grid_origin: Dictionary with 'x', 'y', 'z' keys for grid origin in Å
        grid_spacing: Dictionary with 'x', 'y', 'z' keys for grid spacing in Å
        cutoff_distance: Distance in Ångströms to include around centroid (default: 15.0)
        progress_callback: Callback function for progress tracking
    Returns:
        Carved density map with zeros outside the cutoff distance
    """
    try:
        import numpy as np
        from scipy.spatial.distance import cdist

        log.info(
            f"🔪 Carving density within {cutoff_distance}Å of centroid at {position}"
        )

        # Normalize origin/spacing to dicts (accepts GridOrigin/GridSpacing or dicts)
        grid_origin = _xyz_to_dict(grid_origin)
        grid_spacing = _xyz_to_dict(grid_spacing)

        if progress_callback:
            progress_callback(10, 100, "Processing centroid coordinates...")
        # Validate centroid coordinates
        if len(position) != 3:
            log.error(
                f"❌ Invalid centroid coordinates: {position}. Expected 3 values (x, y, z)"
            )
            if progress_callback:
                progress_callback(100, 100, "Invalid centroid - no carving needed")
            return density_map

        # Convert centroid to numpy array
        # position = (-46.82485, 38.996235, 55.17638)  # @@@
        centroid_coords = np.array([position])
        log.info(f"Centroid coordinates: {position}")

        if progress_callback:
            progress_callback(20, 100, "Creating coordinate grid...")

        # Create coordinate grid for the density map
        grid_shape = density_map.shape
        log.info(f"Grid shape: {grid_shape}")

        # Generate grid coordinates
        x = grid_origin["x"] + np.arange(grid_shape[0]) * grid_spacing["x"]
        y = grid_origin["y"] + np.arange(grid_shape[1]) * grid_spacing["y"]
        z = grid_origin["z"] + np.arange(grid_shape[2]) * grid_spacing["z"]
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        grid_coords = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

        if progress_callback:
            progress_callback(40, 100, "Calculating distances to centroid...")

        # Calculate distances from each grid point to centroid
        log.info("Calculating distances from grid points to centroid...")
        distances = cdist(grid_coords, centroid_coords)
        min_distances = np.min(distances, axis=1)

        if progress_callback:
            progress_callback(60, 100, "Creating density mask...")

        # Create mask for points within cutoff distance
        mask = min_distances <= cutoff_distance
        mask = mask.reshape(grid_shape)

        if progress_callback:
            progress_callback(80, 100, "Applying density mask...")

        # Apply mask to density map and finalize
        return _apply_mask_and_finalize(
            density_map,
            mask,
            progress_callback,
            cutoff_distance,
            "Density carving around centroid complete",
            extra_log_fn=lambda: log.info(f"   Centroid: {position}"),
        )

    except Exception as e:
        log.error(f"❌ Error carving density around centroid: {e}")
        log.warning("Returning original density map")
        return density_map


def carve_density_around_protein(
    density_map: np.ndarray,
    pdb_path: str,
    grid_origin: dict,
    grid_spacing: dict,
    cutoff_distance: float = 4.0,
    progress_callback=None,
) -> np.ndarray:
    """
    Carve out electron density within a specified distance of protein atoms.

    Args:
        density_map: 3D numpy array of electron density
        pdb_path: Path to PDB file containing protein structure
        grid_origin: Dictionary with 'x', 'y', 'z' keys for grid origin in Å
        grid_spacing: Dictionary with 'x', 'y', 'z' keys for grid spacing in Å
        cutoff_distance: Distance in Ångströms to include around protein (default: 4.0)
        progress_callback: Callback function for progress tracking

    Returns:
        Carved density map with zeros outside the cutoff distance
    """
    try:
        import gemmi
        import numpy as np
        from scipy.spatial.distance import cdist

        log.info(f"🔪 Carving density within {cutoff_distance}Å of protein structure")

        # Normalize origin/spacing to dicts (accepts GridOrigin/GridSpacing or dicts)
        grid_origin = _xyz_to_dict(grid_origin)
        grid_spacing = _xyz_to_dict(grid_spacing)

        if progress_callback:
            progress_callback(10, 100, "Loading PDB structure...")

        # Load PDB structure
        pdb = gemmi.read_structure(pdb_path)

        if progress_callback:
            progress_callback(20, 100, "Extracting atomic coordinates...")

        # Get all atomic coordinates
        atoms = _collect_atom_coordinates(pdb)

        if not atoms:
            log.warning("No atoms found in PDB, returning original density map")
            if progress_callback:
                progress_callback(100, 100, "No atoms found - no carving needed")
            return density_map

        coords = np.array(atoms, dtype=np.float64)
        log.info(f"Found {len(atoms)} atoms in protein structure")

        if progress_callback:
            progress_callback(40, 100, "Creating coordinate grid...")

        # Create coordinate grid for the density map
        grid_shape = density_map.shape

        # Defensive programming: ensure all values are proper Python floats
        # This prevents SIGBUS error from numpy scalar type issues
        try:
            log.info("DEBUG: Starting coordinate generation...")
            log.info(f"DEBUG: grid_shape = {grid_shape}")
            log.info(f"DEBUG: grid_origin = {grid_origin}")
            log.info(f"DEBUG: grid_spacing = {grid_spacing}")

            origin_x = float(grid_origin["x"])
            origin_y = float(grid_origin["y"])
            origin_z = float(grid_origin["z"])
            spacing_x = float(grid_spacing["x"])
            spacing_y = float(grid_spacing["y"])
            spacing_z = float(grid_spacing["z"])

            log.info(
                f"DEBUG: Converted values - origin: ({origin_x}, {origin_y}, {origin_z}), spacing: ({spacing_x}, {spacing_y}, {spacing_z})"
            )

        except (KeyError, TypeError, ValueError) as e:
            log.error(f"❌ Error accessing grid origin/spacing values: {e}")
            log.error(
                f"   grid_origin keys: {list(grid_origin.keys()) if isinstance(grid_origin, dict) else 'Not a dict'}"
            )
            log.error(
                f"   grid_spacing keys: {list(grid_spacing.keys()) if isinstance(grid_spacing, dict) else 'Not a dict'}"
            )
            raise ValueError(f"Invalid grid origin or spacing data: {e}")

        # Generate coordinate arrays with explicit float64 dtype for memory alignment
        log.info("DEBUG: Creating coordinate arrays...")
        try:
            x = origin_x + np.arange(grid_shape[0], dtype=np.float64) * spacing_x
            log.info(f"DEBUG: Created x array with shape {x.shape}, dtype {x.dtype}")
        except Exception as e:
            log.error(f"❌ Error creating x array: {e}")
            raise

        try:
            y = origin_y + np.arange(grid_shape[1], dtype=np.float64) * spacing_y
            log.info(f"DEBUG: Created y array with shape {y.shape}, dtype {y.dtype}")
        except Exception as e:
            log.error(f"❌ Error creating y array: {e}")
            raise

        try:
            z = origin_z + np.arange(grid_shape[2], dtype=np.float64) * spacing_z
            log.info(f"DEBUG: Created z array with shape {z.shape}, dtype {z.dtype}")
        except Exception as e:
            log.error(f"❌ Error creating z array: {e}")
            raise

        # Create meshgrid with explicit dtype
        log.info("DEBUG: Creating meshgrid...")
        try:
            X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
            log.info(
                f"DEBUG: Created meshgrid - X: {X.shape}, Y: {Y.shape}, Z: {Z.shape}"
            )
        except Exception as e:
            log.error(f"❌ Error creating meshgrid: {e}")
            raise

        # Stack coordinates with explicit dtype to prevent memory alignment issues
        log.info("DEBUG: Stacking coordinates...")
        try:
            grid_coords = np.column_stack((X.ravel(), Y.ravel(), Z.ravel())).astype(
                np.float64
            )
            log.info(
                f"DEBUG: Created grid_coords with shape {grid_coords.shape}, dtype {grid_coords.dtype}"
            )
            log.info(
                f"DEBUG: grid_coords is contiguous: {grid_coords.flags.c_contiguous}"
            )
        except Exception as e:
            log.error(f"❌ Error stacking coordinates: {e}")
            raise

        if progress_callback:
            progress_callback(60, 100, "Calculating distances to atoms...")

        # Calculate distances from each grid point to nearest atom using spatial indexing
        log.info(
            "Calculating distances from grid points to protein atoms using spatial indexing..."
        )

        try:
            from scipy.spatial import cKDTree

            # Build KDTree for fast nearest neighbor queries
            log.info("DEBUG: Building KDTree for atoms...")
            atom_tree = cKDTree(coords)
            log.info(f"DEBUG: KDTree built for {len(coords)} atoms")

            # Query nearest neighbors for all grid points
            log.info("DEBUG: Querying nearest neighbors...")
            min_distances, _ = atom_tree.query(grid_coords, k=1)
            log.info(
                f"DEBUG: Nearest neighbor query completed - shape: {min_distances.shape}, dtype: {min_distances.dtype}"
            )

        except ImportError:
            log.warning(
                "⚠️ scipy.spatial.cKDTree not available, falling back to cdist (slower)"
            )
            # Fallback to original cdist approach
            try:
                grid_coords_contiguous = np.ascontiguousarray(
                    grid_coords, dtype=np.float64
                )
                coords_contiguous = np.ascontiguousarray(coords, dtype=np.float64)

                log.info("DEBUG: Using cdist fallback...")
                distances = cdist(grid_coords_contiguous, coords_contiguous)
                min_distances = np.min(distances, axis=1)
                log.info(
                    f"DEBUG: cdist fallback completed - shape: {min_distances.shape}"
                )
            except Exception as e:
                log.error(f"❌ Error in cdist fallback: {e}")
                raise
        except Exception as e:
            log.error(f"❌ Error in spatial indexing: {e}")
            raise

        if progress_callback:
            progress_callback(80, 100, "Applying density mask...")

        # Create mask for points within cutoff distance
        mask = min_distances <= cutoff_distance
        mask = mask.reshape(grid_shape)

        # Apply mask to density map and finalize
        return _apply_mask_and_finalize(
            density_map,
            mask,
            progress_callback,
            cutoff_distance,
            "Density carving complete",
            guard_division=False,
        )

    except Exception as e:
        log.error(f"❌ Error carving density around protein: {e}")
        log.warning("Returning original density map")
        return density_map


def load_density_map_with_extent(
    mtz_path: str,
    pdb_path: str,
    margin: float = 13.0,
    f_label="FWT",
    phi_label="PHWT",
    sample_rate=0.0,
) -> tuple[ndarray[Any, dtype[Any]], CrystallographicInfo] | None:
    """
    Load density map using Gemmi's set_extent() to cover structure with margin.
    This is much more efficient than post-processing filtering.

    Args:
        mtz_path: Path to MTZ file
        pdb_path: Path to PDB file for structure
        margin: Margin in Ångströms around structure (default: 13.0)
        f_label: F column label (default: "FWT")
        phi_label: PHI column label (default: "PHWT")
        sample_rate: Sampling rate for map generation (0.0 = full resolution)

    Returns:
        tuple of (numpy array, crystallographic_info) or None if loading fails
    """
    try:
        import os

        if not os.path.exists(pdb_path):
            log.error(f"❌ PDB file not found: {pdb_path}")
            return None

        log.info(
            f"🔮 Loading map with {margin}Å margin around structure using Gemmi set_extent()"
        )
        log.info(f"📁 MTZ file: {mtz_path}")
        log.info(f"📁 PDB file: {pdb_path}")

        # Load structure
        structure = gemmi.read_structure(pdb_path)
        log.info(f"✅ Loaded structure with {len(list(structure[0]))} chains")

        # Load MTZ file
        mtz = gemmi.read_mtz_file(mtz_path)

        # Check if requested labels exist
        f_labels = [col.label for col in mtz.columns if col.type == "F"]
        phi_labels = [col.label for col in mtz.columns if col.type == "P"]

        if f_label not in f_labels:
            log.error(f"❌ Requested F label '{f_label}' not found in MTZ file")
            log.error(f"Available F labels: {f_labels}")
            return None

        if phi_label not in phi_labels:
            log.error(f"❌ Requested PHI label '{phi_label}' not found in MTZ file")
            log.error(f"Available PHI labels: {phi_labels}")
            return None

        # Create map with extent set to structure + margin
        log.info(f"🎯 Setting map extent to structure + {margin}Å margin")

        # First create the map without extent
        grid = mtz.transform_f_phi_to_map(f_label, phi_label, sample_rate=sample_rate)

        # Create a Ccp4Map object to use set_extent
        ccp4_map = gemmi.Ccp4Map()
        ccp4_map.grid = grid
        ccp4_map.update_ccp4_header()  # Required before set_extent

        # Set extent to cover structure with margin
        ccp4_map.set_extent(structure.calculate_fractional_box(margin=margin))

        # Get the modified grid
        grid = ccp4_map.grid

        # Extract crystallographic information
        crystallographic_info = crystallographic_info_from_grid(grid)

        # Convert to NumPy array
        np_array = _grid_to_xyz_array(grid, crystallographic_info)

        log.info("✅ Map loaded with extent:")
        log.info(f"   Shape: {np_array.shape}")
        log.info(f"   Non-zero voxels: {np.count_nonzero(np_array):,}")
        log.info(f"   Margin: {margin}Å")
        log.info(f"   Grid origin: {crystallographic_info.grid.origin}")
        log.info(f"   Grid spacing: {crystallographic_info.grid.spacing}")

        return np_array, crystallographic_info

    except Exception as e:
        log.error(f"❌ Error loading map with extent: {e}")
        import traceback

        traceback.print_exc()
        return None


def filter_density_sphere(
    density_map: np.ndarray,
    grid_origin: dict,
    grid_spacing: dict,
    center_coords: tuple[float, float, float] = (0.0, 0.0, 0.0),
    radius: float = 13.0,
) -> np.ndarray:
    """
    Filter density map to show only values within a sphere of specified radius around center coordinates.
    Uses vectorized operations for efficiency.

    Args:
        density_map: 3D numpy array of electron density
        grid_origin: Dictionary with 'x', 'y', 'z' keys for grid origin in Å
        grid_spacing: Dictionary with 'x', 'y', 'z' keys for grid spacing in Å
        center_coords: Tuple of (x, y, z) center coordinates in Å (default: origin)
        radius: Radius of sphere in Ångströms (default: 13.0)

    Returns:
        Filtered density map with zeros outside the sphere
    """
    try:
        log.info(
            f"🔮 Filtering density within {radius}Å sphere around center {center_coords}"
        )

        grid_shape = density_map.shape
        log.info(f"   Processing map of shape: {grid_shape}")

        # Create coordinate grids using vectorized operations
        i, j, k = np.ogrid[: grid_shape[0], : grid_shape[1], : grid_shape[2]]

        # Calculate real-space coordinates for all grid points at once
        x_coords = grid_origin["x"] + i * grid_spacing["x"]
        y_coords = grid_origin["y"] + j * grid_spacing["y"]
        z_coords = grid_origin["z"] + k * grid_spacing["z"]

        # Calculate distances from center for all points at once
        center_x, center_y, center_z = center_coords
        distances_squared = (
            (x_coords - center_x) ** 2
            + (y_coords - center_y) ** 2
            + (z_coords - center_z) ** 2
        )
        distances = np.sqrt(distances_squared)

        # Create mask for points within sphere radius
        mask = distances <= radius

        # Apply mask to create filtered density map
        filtered_density = np.where(mask, density_map, 0.0)

        # Calculate statistics
        original_nonzero = np.count_nonzero(density_map)
        filtered_nonzero = np.count_nonzero(filtered_density)
        points_in_sphere = np.sum(mask)

        if original_nonzero > 0:
            reduction_factor = (
                (original_nonzero - filtered_nonzero) / original_nonzero * 100
            )
        else:
            reduction_factor = 0.0

        log.info("✅ Sphere filtering complete:")
        log.info(f"   Original non-zero voxels: {original_nonzero:,}")
        log.info(f"   Filtered non-zero voxels: {filtered_nonzero:,}")
        log.info(f"   Points in sphere: {points_in_sphere:,}")
        log.info(f"   Data reduction: {reduction_factor:.1f}%")
        log.info(f"   Sphere radius: {radius}Å")
        log.info(f"   Center: {center_coords}")

        return filtered_density

    except Exception as e:
        log.error(f"❌ Error filtering density sphere: {e}")
        log.warning("Returning original density map")
        return density_map

"""
Utilities for loading and processing electron density maps from MTZ and CCP4 files.
"""
from pathlib import Path

from collections.abc import Mapping
from dataclasses import dataclass, field

import faulthandler
import os
import pathlib
import re
import struct

from numpy import dtype, ndarray
from numpy.typing import NDArray
from typing import Callable, Any

import gemmi
import numpy as np
from decologr import Decologr as log
from molib.xtal.ccp4.map.globals import CCP4_HEADER_SIZE
from molib.xtal.ccp4.map.volume import VolumeStatistics, VolumeData
from molib.xtal.ccp4.mtz.filespec import MtzFileSpec, MtzDensitySpec
from molib.xtal.ccp4.mtz.column_pair import MtzColumnPair
from molib.xtal.ccp4.mtz.errors import MtzColumnNotFoundError
from molib.xtal.info.resolve import normalize_crystallographic_info_from_dict
from molib.xtal.map.builders.processing import build_map_processing_settings
from molib.xtal.map.builders.density import build_density_map_spec
from molib.xtal.map.info import MapProcessingSettings
from molib.xtal.map.spec import DensityMapSpec
from molib.xtal.uglymol.map.helpers import (
    extract_symop_text,
    parse_symmetry_operator_to_matrix,
)
from molib.xtal.map.crystal import CrystallographicInfo
from molib.xtal.map.axis import AxisOrder
from molib.xtal.map.map_type import MapType

# Enable faulthandler for debugging SIGBUS crashes on macOS
faulthandler.enable()

ORIGIN = (0.0, 0.0, 0.0)

# Z-slab thickness for carve distance queries (avoids full-volume meshgrid).
_CARVE_SLAB_SIZE = 32

ProgressCallback = Callable[[int, int, str], None]

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

    if map_type is MapType.NORMAL:
        candidates = _TWO_FO_FC_CANDIDATES
    elif map_type is MapType.DIFFERENCE:
        candidates = _FO_FC_CANDIDATES
    else:
        raise MtzColumnNotFoundError(f"Unsupported map type: {map_type!r}")

    for f_label, phi_label in candidates:
        if f_label in f_map and phi_label in p_map:
            return f_map[f_label], p_map[phi_label]

    raise MtzColumnNotFoundError(
        f"No {map_type.value} coefficients found in {mtz_path}. "
        f"Available F columns: {sorted(f_map)}; "
        f"available PHI columns: {sorted(p_map)}"
    )


def log_density_statistics(volume: np.ndarray) -> None:
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

    ``volume`` is the sole canonical density array. Statistics are cached in
    ``_volume_stats`` and exposed via :attr:`volume_stats` /
    :attr:`volume_data` (read-only derived views).

    When ``crystallographic_info`` is present, ``volume.shape`` must match
    ``crystallographic_info.grid.dimensions`` (XYZ convention). Pipeline steps
    that intentionally reshape must sync grid metadata before constructing or
    replacing the volume (see :func:`_log_grid_consistency`).

    ``margin`` is the Gemmi ``set_extent`` margin in Å when extent-based loading
    was used; otherwise ``None``.
    """

    volume: np.ndarray
    crystallographic_info: CrystallographicInfo | None = None
    source: str = ""
    map_type: MapType | None = None
    margin: float | None = None
    _volume_stats: VolumeStatistics = field(init=False, repr=False)
    _volume_data: VolumeData | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate volume/grid and cache derived statistics."""
        self._validate_volume(self.volume)
        self._volume_stats = VolumeStatistics.from_array(self.volume)
        self._volume_data = None
        self.validate_grid_consistency()

    @property
    def volume_stats(self) -> VolumeStatistics:
        """Statistics snapshot for the current :attr:`volume`."""
        return self._volume_stats

    @property
    def volume_data(self) -> VolumeData:
        """:class:`VolumeData` view sharing :attr:`volume` and cached stats."""
        cached = self._volume_data
        if cached is None or cached.volume is not self.volume:
            self._volume_data = VolumeData(
                self.volume,
                statistics=self._volume_stats,
            )
        return self._volume_data

    def replace_volume(
        self,
        volume: np.ndarray,
        crystallographic_info: CrystallographicInfo | None = None,
    ) -> None:
        """Replace the volume in place and refresh derived statistics.

        :param volume: Replacement density array
        :param crystallographic_info: Optional updated metadata. Required when
            ``volume.shape`` differs from the current grid dimensions.
        :raises ValueError: If the replacement is incompatible with retained
            crystallographic metadata
        """
        self._validate_volume(volume)
        info = (
            self.crystallographic_info
            if crystallographic_info is None
            else crystallographic_info
        )
        self._validate_replacement_grid(volume, info)
        self.volume = volume
        self.crystallographic_info = info
        self._volume_stats = VolumeStatistics.from_array(volume)
        self._volume_data = None

    def with_volume(
        self,
        volume: np.ndarray,
        crystallographic_info: CrystallographicInfo | None = None,
    ) -> "DensityMapData":
        """Return a copy with a replaced volume and refreshed stats.

        :param volume: Replacement density array
        :param crystallographic_info: Optional updated metadata. When omitted,
            the current crystallographic info is retained only if compatible
            with ``volume.shape``.
        :return: New :class:`DensityMapData`
        :raises ValueError: If shape/frame is incompatible without new metadata
        """
        info = (
            self.crystallographic_info
            if crystallographic_info is None
            else crystallographic_info
        )
        self._validate_volume(volume)
        self._validate_replacement_grid(volume, info)
        return DensityMapData(
            volume=volume,
            crystallographic_info=info,
            source=self.source,
            map_type=self.map_type,
            margin=self.margin,
        )

    @staticmethod
    def _validate_volume(volume: np.ndarray) -> None:
        """Require a non-empty finite 3D density array."""
        if not isinstance(volume, np.ndarray):
            raise TypeError(
                f"Density volume must be a NumPy array; got {type(volume)!r}"
            )
        if volume.ndim != 3:
            raise ValueError(
                f"Expected a 3D density volume; got shape {volume.shape}"
            )
        if volume.size == 0:
            raise ValueError("Density volume must not be empty")
        if not np.isfinite(volume).all():
            raise ValueError("Density volume contains non-finite values")

    def validate_grid_consistency(self) -> None:
        """Validate volume shape against crystallographic grid metadata."""
        self._validate_replacement_grid(
            self.volume,
            self.crystallographic_info,
        )

    def _validate_replacement_grid(
        self,
        volume: np.ndarray,
        crystallographic_info: CrystallographicInfo | None,
    ) -> None:
        """Ensure replacement volume is compatible with grid metadata."""
        if crystallographic_info is None:
            return
        grid = getattr(crystallographic_info, "grid", None)
        if grid is None:
            return
        dimensions = getattr(grid, "dimensions", None)
        if dimensions is None:
            return
        volume_shape = tuple(int(x) for x in volume.shape)
        grid_dims = tuple(int(x) for x in dimensions)
        if volume_shape != grid_dims:
            raise ValueError(
                "Replacement volume shape is incompatible with crystallographic "
                f"grid dimensions (XYZ); volume={volume_shape}, grid={grid_dims}. "
                "Supply updated crystallographic_info when the grid changes."
            )

    def log_map_data(self) -> None:
        """Log a structured summary of this loaded map."""
        stats = self.volume_stats
        log.info("✅ Map loaded with extent:")
        log.info(f"   Shape: {self.volume.shape}")
        log.info(
            f"   mean={stats.mean:.3f}, "
            f"std={stats.std:.3f}, "
            f"range=[{stats.min_value:.3f}, {stats.max_value:.3f}]"
        )
        log.info(f"   Non-zero voxels: {np.count_nonzero(self.volume):,}")
        if self.margin is not None:
            log.info(f"   Margin: {self.margin}Å (Gemmi set_extent)")
        if self.map_type:
            log.info(f"   Map type: {self.map_type}")
        if self.source:
            log.info(f"   Source: {self.source}")
        if self.crystallographic_info is not None:
            log.info(f"   Grid origin: {self.crystallographic_info.grid.origin}")
            log.info(f"   Grid spacing: {self.crystallographic_info.grid.spacing}")


@dataclass(slots=True)
class CCP4MiniHeader:
    """CCP4 header fields required for symmetry expansion.

    Distinguishes CRS storage axes (``nc``/``nr``/``ns``, starts, MAPC/MAPR/MAPS)
    from crystallographic XYZ sampling (``nx``/``ny``/``nz``).

    Parsing assumes a standard little-endian 1024-byte CCP4 header.
    """

    nsymbt: int
    nc: int = 0
    nr: int = 0
    ns: int = 0
    nx: int = 0
    ny: int = 0
    nz: int = 0
    ncstart: int = 0
    nrstart: int = 0
    nsstart: int = 0
    mapc: int = 1
    mapr: int = 2
    maps: int = 3


def _validate_map_crs(mapc: int, mapr: int, maps: int) -> None:
    """Require MAPC/MAPR/MAPS to be a permutation of 1, 2, 3."""
    if sorted((mapc, mapr, maps)) != [1, 2, 3]:
        raise ValueError(
            "Invalid axis mapping: MAPC, MAPR, MAPS must be unique and in [1, 2, 3]"
        )


def _crs_to_xyz_start_size(
    header: CCP4MiniHeader,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Remap CRS starts and storage sizes into crystallographic XYZ.

    :param header: Mini-header with CRS fields and MAPC/MAPR/MAPS
    :return: ``(xyz_start, xyz_size)`` for use with XYZ-ordered volumes
    """
    _validate_map_crs(header.mapc, header.mapr, header.maps)
    xyz_start = [0, 0, 0]
    xyz_size = [0, 0, 0]
    xyz_start[header.mapc - 1] = int(header.ncstart)
    xyz_start[header.mapr - 1] = int(header.nrstart)
    xyz_start[header.maps - 1] = int(header.nsstart)
    xyz_size[header.mapc - 1] = int(header.nc)
    xyz_size[header.mapr - 1] = int(header.nr)
    xyz_size[header.maps - 1] = int(header.ns)
    return (xyz_start[0], xyz_start[1], xyz_start[2]), (
        xyz_size[0],
        xyz_size[1],
        xyz_size[2],
    )


def _build_header_from_ccp4_map(ccp4_map: gemmi.Ccp4Map) -> CCP4MiniHeader:
    """Build a :class:`CCP4MiniHeader` from a gemmi Ccp4Map (or test mock).

    Prefers parsing the raw CCP4 header bytes, then falls back to the
    ``header.nsymbt`` / ``grid.shape`` attributes used by mocks.
    """
    raw = getattr(ccp4_map, "ccp4_header", None)
    if isinstance(raw, (bytes, bytearray)):
        if len(raw) < CCP4_HEADER_SIZE:
            raise ValueError(
                f"CCP4 header is shorter than {CCP4_HEADER_SIZE} bytes "
                f"(got {len(raw)})"
            )
        header_ints = struct.unpack(
            "<256i",
            bytes(raw[:CCP4_HEADER_SIZE]),
        )
        header = CCP4MiniHeader(
            nsymbt=int(header_ints[23]),
            nc=int(header_ints[0]),
            nr=int(header_ints[1]),
            ns=int(header_ints[2]),
            ncstart=int(header_ints[4]),
            nrstart=int(header_ints[5]),
            nsstart=int(header_ints[6]),
            nx=int(header_ints[7]),
            ny=int(header_ints[8]),
            nz=int(header_ints[9]),
            mapc=int(header_ints[16]),
            mapr=int(header_ints[17]),
            maps=int(header_ints[18]),
        )
        _validate_map_crs(header.mapc, header.mapr, header.maps)
        return header

    nsymbt = 0
    if hasattr(ccp4_map, "header") and hasattr(ccp4_map.header, "nsymbt"):
        try:
            nsymbt = int(ccp4_map.header.nsymbt)
        except (TypeError, ValueError):
            nsymbt = 0
    shape = getattr(getattr(ccp4_map, "grid", None), "shape", ORIGIN)
    if len(shape) == 3:
        nc, nr, ns = int(shape[0]), int(shape[1]), int(shape[2])
    else:
        nc, nr, ns = 0, 0, 0
    return CCP4MiniHeader(
        nsymbt=nsymbt,
        nc=nc,
        nr=nr,
        ns=ns,
        nx=nc,
        ny=nr,
        nz=ns,
        mapc=1,
        mapr=2,
        maps=3,
    )


@dataclass(frozen=True, slots=True)
class CCP4Map:
    """Raw CCP4 data and symmetry-expansion geometry in XYZ convention."""

    buffer: bytes
    nsymbt: int
    n_grid: tuple[int, int, int]
    start: tuple[int, int, int]
    end: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class VolumeGeometry:
    """Grid geometry required for symmetry expansion (XYZ convention)."""

    n_grid: tuple[int, int, int]
    start: tuple[int, int, int]
    end: tuple[int, int, int]
    center_offset: tuple[int, int, int]

    @classmethod
    def from_ccp4_map(
        cls,
        ccp4_map: CCP4Map,
        center_offset: tuple[int, int, int] | list[int],
    ) -> "VolumeGeometry":
        """Build geometry from file metadata plus an expansion center offset."""
        offset = tuple(int(x) for x in center_offset)
        if len(offset) != 3:
            raise ValueError(
                f"center_offset must have length 3, got {len(offset)}"
            )
        return cls(
            n_grid=ccp4_map.n_grid,
            start=ccp4_map.start,
            end=ccp4_map.end,
            center_offset=(offset[0], offset[1], offset[2]),
        )


# Helper volumes are canonical XYZ after grid_to_xyz_array.
CANONICAL_AXIS_ORDER = AxisOrder.XYZ


def read_symmetry_ops(
        map_path: str, header: CCP4MiniHeader
) -> CCP4Map:
    """Read raw CCP4 data and return symmetry-expansion metadata.

    ``n_grid`` is XYZ sampling (NX/NY/NZ) for scaling fractional translations.
    ``start``/``end`` are CRS starts/sizes remapped into crystallographic XYZ
    via MAPC/MAPR/MAPS so they match XYZ-ordered NumPy volumes.

    Symmetry-record offsets passed to :func:`extract_symop_text` start at 0;
    that helper adds the 1024-byte main-header offset internally.
    """
    with open(map_path, "rb") as f:
        map_buffer = f.read()
    if header.nsymbt < 0:
        raise ValueError(f"Invalid NSYMBT value: {header.nsymbt}")
    if len(map_buffer) < CCP4_HEADER_SIZE + header.nsymbt:
        raise ValueError(
            "CCP4 file is shorter than its declared header and "
            "symmetry-record region"
        )
    n_grid = (int(header.nx), int(header.ny), int(header.nz))
    xyz_start, xyz_size = _crs_to_xyz_start_size(header)
    end = (
        xyz_start[0] + xyz_size[0],
        xyz_start[1] + xyz_size[1],
        xyz_start[2] + xyz_size[2],
    )
    return CCP4Map(
        buffer=map_buffer,
        nsymbt=int(header.nsymbt),
        n_grid=n_grid,
        start=xyz_start,
        end=end,
    )


def _parse_symmetry_matrix(symop: str, n_grid: list) -> list:
    """Parse a CCP4 symmetry text line into a grid-scaled 3x4 matrix."""
    symop_matrix = parse_symmetry_operator_to_matrix(symop)
    for j in range(3):
        symop_matrix[j][3] = round(symop_matrix[j][3] * n_grid[j])
    return symop_matrix


def _as_4x4_transform(mat: list | np.ndarray) -> np.ndarray:
    """Normalize a 3x4 or 4x4 transform to a float64 4x4 matrix."""
    arr = np.asarray(mat, dtype=np.float64)
    if arr.shape == (4, 4):
        return arr
    if arr.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :] = arr
        return out
    raise ValueError(f"Expected 3x4 or 4x4 transform, got {arr.shape}")


def _apply_symmetry_operation(
    volume: np.ndarray,
    expanded_volume: np.ndarray,
    matrix: list | np.ndarray,
    geometry: VolumeGeometry,
) -> None:
    """Apply a symmetry operation using canonical XYZ coordinates."""
    apply_symmetry_to_volume(
        volume,
        expanded_volume,
        _as_4x4_transform(matrix),
        geometry,
    )


def _apply_symmetry_mates(
    volume: np.ndarray,
    expanded_volume: np.ndarray,
    ccp4_map: CCP4Map,
    geometry: VolumeGeometry,
) -> int:
    """Apply non-identity symmetry operations to the expanded volume.

    :param volume: Source density volume (XYZ)
    :param expanded_volume: Destination volume receiving mates
    :param ccp4_map: Raw CCP4 buffer and ``nsymbt``
    :param geometry: XYZ grid geometry including center offset
    :return: Number of symmetry operations successfully applied
    """
    symmetry_count = 0
    for offset in range(0, ccp4_map.nsymbt, 80):
        symop = extract_symop_text(ccp4_map.buffer, offset).strip()

        if re.fullmatch(r"x\s*,\s*y\s*,\s*z", symop, re.IGNORECASE):
            continue

        try:
            symop_matrix = _parse_symmetry_matrix(
                symop,
                list(geometry.n_grid),
            )
            log.info("Applying symmetry: %s", symop)
            _apply_symmetry_operation(
                volume,
                expanded_volume,
                symop_matrix,
                geometry,
            )
            symmetry_count += 1
        except (ValueError, IndexError, TypeError) as exc:
            log.warning(
                "Failed to apply symmetry operation %r: %s",
                symop,
                exc,
            )

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
    """Accept ``GridOrigin``/``GridSpacing`` dataclasses or ``{"x", "y", "z"}`` mappings.

    Returns a plain ``{"x": float, "y": float, "z": float}`` dict so carving
    code is agnostic to whether the crystallographic metadata came from the
    normalized dataclasses or a legacy dict.
    """
    if isinstance(value, Mapping):
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
    # Always guard: empty maps must not divide by zero (ignore flag).
    _ = guard_division
    if original_nonzero == 0:
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


def _grid_to_xyz_array(grid: gemmi.FloatGrid) -> np.ndarray:
    """Return a copy of the density grid in canonical XYZ order.

    Pure conversion helper: does not mutate crystallographic metadata.
    Call :meth:`CrystallographicInfo.sync_to_xyz_volume` after conversion
    when metadata must match the returned array.
    """
    return CrystallographicInfo.grid_to_xyz_array(grid)


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
        CrystallographicInfo.from_grid(grid, map_type=xtal_map_type)
    )
    crystallographic_info.log_summary()

    # Convert to a NumPy array in the canonical XYZ axis order, then sync
    # metadata so axis_order/dimensions match the returned volume.
    np_array = CrystallographicInfo.grid_to_xyz_array(grid)
    crystallographic_info.sync_to_xyz_volume(np_array)
    log.info(f"ℹ️ Loaded MTZ map shape: {np_array.shape}")
    density_map_data = DensityMapData(
        volume=np_array,
        crystallographic_info=crystallographic_info,
        source=mtz_path,
        map_type=resolved_map_type,
    )
    density_map_data.log_map_data()
    return density_map_data


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
    """Sync grid metadata after an intentional pipeline reshape.

    Pipeline steps that change ``volume.shape`` (e.g. symmetry expansion) must
    update ``grid.dimensions`` here before returning a validated
    :class:`DensityMapData`. Domain-object construction itself raises on
    mismatch rather than silently rewriting metadata.
    """
    if map_data.crystallographic_info is None:
        return map_data
    volume_shape = tuple(int(x) for x in map_data.volume.shape)
    grid_dims = tuple(
        int(x) for x in map_data.crystallographic_info.grid.dimensions
    )
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

    crystallographic_info = CrystallographicInfo.from_grid(grid)
    volume = CrystallographicInfo.grid_to_xyz_array(grid)
    crystallographic_info.sync_to_xyz_volume(volume)
    density_map_data = DensityMapData(
        volume=volume,
        crystallographic_info=crystallographic_info,
        source=source,
    )
    density_map_data.crystallographic_info.log_grid_metadata()
    density_map_data.log_map_data()
    return density_map_data


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
    log.info(f"✅ Symmetry expanded - new shape: {expanded.shape}")
    # Intentional reshape: sync grid dims to the new array, then replace.
    info = map_data.crystallographic_info
    if info is not None:
        volume_shape = tuple(int(x) for x in expanded.shape)
        grid_dims = tuple(int(x) for x in info.grid.dimensions)
        if volume_shape != grid_dims:
            log.info(
                f"🔗 Grid dims invariant: symmetry expansion changed volume "
                f"shape {grid_dims} -> {volume_shape}; syncing metadata"
            )
            info.grid.dimensions = volume_shape
        else:
            log.info(
                f"🔗 Grid dims invariant: symmetry expansion consistent "
                f"({volume_shape})"
            )
    map_data.replace_volume(expanded, crystallographic_info=info)
    return map_data


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
        carved = carve_density_around_protein(
            map_data,
            pdb_path,
            carve_cutoff,
            progress_callback=progress_callback,
        )
        map_data.replace_volume(carved)
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
    carved = carve_density_around_position(
        map_data,
        centroid,
        centroid_cutoff,
        progress_callback=progress_callback,
    )
    map_data.replace_volume(carved)
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
            carve_density=spec.processing.carve_density,
            carve_cutoff=spec.processing.carve_cutoff,
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
            processing=MapProcessingSettings(carve_density=carve_density,
                                             carve_cutoff=carve_cutoff,
                                             carve_density_centroid=carve_density_centroid,
                                             centroid_cutoff=centroid_cutoff),
            centroid=centroid,
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
    """Load a CCP4 map using Gemmi.

    Backward-compatible wrapper around :func:`load_density_map`.
    """
    processing = build_map_processing_settings(carve_cutoff=carve_cutoff,
                                               carve_density=carve_density,
                                               carve_density_centroid=carve_density_centroid,
                                               centroid_cutoff=centroid_cutoff,
                                               convert_to_cartesian=convert_to_cartesian)
    spec = build_density_map_spec(expand_symmetry=expand_symmetry,
                                  map_path=map_path,
                                  progress_callback=progress_callback,
                                  processing=processing,
                                  centroid=pdb_centroid_or_clicked_position)

    return load_density_map(spec)


def load_ccp4_map_from_processing(
    file_path: str,
    processing: MapProcessingSettings,
    progress_callback: Callable[[Any, Any, Any], None] | None = None,
    *,
    pdb_centroid_or_clicked_position: tuple[float, float, float] | None = None,
) -> DensityMapData | None:
    """Load a CCP4 map using :class:`MapProcessingSettings`.

    Thin translator around :func:`load_ccp4_map`. Centroid position is scene
    state and must be passed explicitly (not stored on *processing*).
    """
    return load_ccp4_map(
        file_path,
        convert_to_cartesian=bool(processing.convert_to_cartesian),
        expand_symmetry=False,
        carve_density=bool(processing.carve_density),
        carve_cutoff=float(processing.carve_cutoff or 4.0),
        progress_callback=progress_callback,
        carve_density_centroid=bool(processing.carve_density_centroid),
        pdb_centroid_or_clicked_position=pdb_centroid_or_clicked_position,
        centroid_cutoff=float(processing.centroid_cutoff or 15.0),
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
            log.info(f"📁 Loading map {i + 1}/{len(map_paths)}: {map_path}")

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
                log.info("Using standard expansion (no PDB file found)")
                result = load_ccp4_map(map_path, expand_symmetry=expand_symmetry)

            if result is None:
                log.error(f"❌ Failed to load map {i + 1}: {map_path}")
                return None

            loaded_maps.append(result)
            log.info(f"✅ Successfully loaded map {i + 1}: {map_path}")

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
            log.info(f"📁 Loading MTZ map {i + 1}/{len(mtz_paths)}: {mtz_path}")

            result = load_density_map_auto_mtz(mtz_path, sample_rate=sample_rate)
            if result is None:
                log.error(f"❌ Failed to load MTZ map {i + 1}: {mtz_path}")
                return None

            loaded_maps.append(result)
            log.info(f"✅ Successfully loaded MTZ map {i + 1}: {mtz_path}")

        log.info(f"🎉 Successfully loaded all {len(mtz_paths)} MTZ maps")
        return loaded_maps

    except Exception as e:
        log.error(f"❌ Error in load_mtz_maps: {e}")
        return None


def load_density_map_auto_mtz(
        mtz_path: str,
        *,
        map_type: MapType | str = MapType.NORMAL,
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
        path = pathlib.Path(mtz_path)
        suffix = path.suffix.lower()
        if suffix in (".map", ".ccp4", ".omap"):
            log.error(
                f"❌ Refusing to read {mtz_path} as MTZ "
                f"(suffix {suffix!r}); use load_ccp4_map / load_density_map"
            )
            return None
        if suffix and suffix not in (".mtz", ".hkl"):
            log.warning(
                f"⚠️ Unexpected suffix {suffix!r} for MTZ auto-load of {mtz_path}"
            )

        resolved_map_type = MapType.coerce(map_type)
        f_label, phi_label = _select_map_columns(str(path), resolved_map_type)
        log.info(
            f"Auto-loading MTZ file: {mtz_path} "
            f"(map type: {resolved_map_type.value}, columns: {f_label}/{phi_label})"
        )
        return load_density_map_from_columns(
            str(path),
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
        ccp4_map = read_symmetry_ops(map_path, header)
        if ccp4_map.nsymbt == 0:
            log.info("ℹ️ No symmetry operations to expand")
            return volume

        log.info(f"📐 Found {ccp4_map.nsymbt} bytes of symmetry operations")

        # Calculate optimal expansion bounds based on molecular coordinates
        if pdb_path and os.path.exists(pdb_path):
            optimal_bounds = _calculate_optimal_expansion_bounds(
                volume,
                header,
                map_path,
                pdb_path,
                list(ccp4_map.n_grid),
                list(ccp4_map.start),
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
        geometry = VolumeGeometry.from_ccp4_map(ccp4_map, center_offset)

        # Copy original volume to center of expanded volume
        expanded_volume[
            center_offset[0]: center_offset[0] + volume.shape[0],
            center_offset[1]: center_offset[1] + volume.shape[1],
            center_offset[2]: center_offset[2] + volume.shape[2],
        ] = volume

        log.info(f"📊 Original volume shape: {volume.shape}")
        log.info(f"📊 Optimized expanded volume shape: {expanded_shape}")
        log.info(
            f"📊 Expansion factor: {expanded_shape[0] * expanded_shape[1] * expanded_shape[2] / (volume.shape[0] * volume.shape[1] * volume.shape[2]):.1f}x"
        )

        symmetry_count = _apply_symmetry_mates(
            volume,
            expanded_volume,
            ccp4_map,
            geometry,
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
        log.info(f"🔄 Expanding symmetry for {map_path}")

        # Read the raw file to access symmetry operations
        ccp4_map = read_symmetry_ops(map_path, header)
        if ccp4_map.nsymbt == 0:
            log.info("ℹ️ No symmetry operations to expand")
            return volume

        log.info(f"📐 Found {ccp4_map.nsymbt} bytes of symmetry operations")

        # Create expanded volume (2x larger to accommodate symmetry mates)
        expanded_shape = [n * 2 for n in volume.shape]
        expanded_volume = np.zeros(expanded_shape, dtype=volume.dtype)

        # Copy original volume to center of expanded volume
        center_offset = [n // 2 for n in expanded_shape]
        geometry = VolumeGeometry.from_ccp4_map(ccp4_map, center_offset)
        expanded_volume[
            center_offset[0]: center_offset[0] + volume.shape[0],
            center_offset[1]: center_offset[1] + volume.shape[1],
            center_offset[2]: center_offset[2] + volume.shape[2],
        ] = volume

        log.info(f"📊 Original volume shape: {volume.shape}")
        log.info(f"📊 Expanded volume shape: {expanded_shape}")

        symmetry_count = _apply_symmetry_mates(
            volume,
            expanded_volume,
            ccp4_map,
            geometry,
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
    transform: np.ndarray,
    geometry: VolumeGeometry,
) -> None:
    """Apply a symmetry transform to a source volume in XYZ order.

    Volumes are assumed to use canonical crystallographic XYZ array axes
    (:data:`CANONICAL_AXIS_ORDER`). Fractional coordinates are truncated with
    ``int()`` toward zero (same policy as the prior implementation).

    :param source_volume: Source density (3D)
    :param target_volume: Destination density (3D)
    :param transform: 4x4 symmetry transform
    :param geometry: Grid ``start`` and ``center_offset`` (XYZ)
    """
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(
            f"Expected a 4x4 symmetry transform, got {transform.shape}"
        )
    if source_volume.ndim != 3 or target_volume.ndim != 3:
        raise ValueError("Source and target volumes must be 3D arrays")

    start = geometry.start
    center_offset = geometry.center_offset
    rotation = transform[:3, :3]
    translation = transform[:3, 3]

    for x, y, z in np.ndindex(source_volume.shape):
        grid_xyz = (
            x + start[0],
            y + start[1],
            z + start[2],
        )
        transformed = rotation @ grid_xyz + translation
        target_xyz = (
            int(transformed[0]) + center_offset[0],
            int(transformed[1]) + center_offset[1],
            int(transformed[2]) + center_offset[2],
        )
        if all(
            0 <= target_xyz[i] < target_volume.shape[i] for i in range(3)
        ):
            target_volume[target_xyz] = source_volume[x, y, z]


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
        ccp4_map = read_symmetry_ops(map_path, header)
        map_buffer = ccp4_map.buffer
        nsymbt = ccp4_map.nsymbt
        n_grid = list(ccp4_map.n_grid)
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
                apply_transformation_matrix(coord, i, sym_coords, symop_matrix)

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


def apply_transformation_matrix(coord: tuple[int, Any],
                                i: int,
                                sym_coords: ndarray[Any, dtype[Any]],
                                symop_matrix: Any):
    """ Apply transformation matrix """
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


def _carve_density_near_coordinates(
    map_data: DensityMapData,
    reference_coordinates: NDArray[np.float64],
    cutoff_distance: float,
    *,
    progress_callback: ProgressCallback | None = None,
    finalize_label: str = "Density carving complete",
) -> np.ndarray:
    """Retain density within ``cutoff_distance`` of reference Cartesian points.

    Processes the volume in Z-slabs to avoid allocating a full-volume
    coordinate mesh. Uses :class:`scipy.spatial.cKDTree` for nearest-neighbor
    distances.

    Geometry is taken from ``map_data.crystallographic_info.grid`` so the
    density array and its coordinate system cannot drift apart.

    Geometry contract (canonical XYZ volumes):

    1. ``grid.axis_order`` is :attr:`AxisOrder.XYZ`.
    2. ``grid.dimensions`` matches ``map_data.volume.shape``.
    3. ``grid.origin`` is the Cartesian position of index ``(0, 0, 0)``.
    4. ``grid.spacing`` is the Cartesian step along each array axis under the
       axis-aligned model ``origin + index * spacing``.

    For strongly non-orthogonal unit cells this axis-aligned model is an
    approximation; full-cell / ``frac_to_orth`` voxel placement is not applied
    here.

    :param map_data: Density volume plus crystallographic metadata
    :param reference_coordinates: ``(N, 3)`` Cartesian reference points
    :param cutoff_distance: Retention radius in Å (finite, non-negative)
    :param progress_callback: Optional ``(value, maximum, message)`` callback
    :param finalize_label: Label passed to :func:`_apply_mask_and_finalize`
    :return: New carved density array (does not mutate ``map_data.volume``)
    """
    from scipy.spatial import cKDTree

    density_map = map_data.volume
    grid = map_data.crystallographic_info.grid

    if not np.isfinite(cutoff_distance) or cutoff_distance < 0.0:
        raise ValueError(
            f"cutoff_distance must be finite and >= 0, got {cutoff_distance!r}"
        )
    if density_map.ndim != 3:
        raise ValueError(
            f"density_map must be 3D, got shape {density_map.shape}"
        )
    if tuple(density_map.shape) != tuple(grid.dimensions):
        raise ValueError(
            f"density_map shape {density_map.shape} does not match "
            f"grid.dimensions {grid.dimensions}"
        )
    if grid.axis_order is not AxisOrder.XYZ:
        raise ValueError(
            f"Carving requires canonical XYZ volumes, got axis_order="
            f"{grid.axis_order!r}"
        )

    refs = np.asarray(reference_coordinates, dtype=np.float64)
    if refs.ndim != 2 or refs.shape[1] != 3 or refs.shape[0] < 1:
        raise ValueError(
            f"reference_coordinates must have shape (N, 3) with N>=1, "
            f"got {refs.shape}"
        )

    if progress_callback is not None:
        progress_callback(20, 100, "Building reference spatial index...")

    tree = cKDTree(refs)
    nx, ny, nz = density_map.shape
    origin = np.asarray(grid.origin.to_tuple(), dtype=np.float64)
    spacing = np.asarray(grid.spacing.to_tuple(), dtype=np.float64)

    mask = np.zeros(density_map.shape, dtype=bool)
    x_coords = origin[0] + np.arange(nx, dtype=np.float64) * spacing[0]
    y_coords = origin[1] + np.arange(ny, dtype=np.float64) * spacing[1]

    if progress_callback is not None:
        progress_callback(40, 100, "Carving density slabs...")

    for z0 in range(0, nz, _CARVE_SLAB_SIZE):
        z1 = min(z0 + _CARVE_SLAB_SIZE, nz)
        z_coords = origin[2] + np.arange(z0, z1, dtype=np.float64) * spacing[2]
        X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing="ij")
        slab_coords = np.column_stack(
            (X.ravel(), Y.ravel(), Z.ravel()),
        )
        min_distances, _ = tree.query(slab_coords, k=1)
        mask[:, :, z0:z1] = (
            min_distances.reshape(nx, ny, z1 - z0) <= cutoff_distance
        )

    if progress_callback is not None:
        progress_callback(80, 100, "Applying density mask...")

    return _apply_mask_and_finalize(
        density_map,
        mask,
        progress_callback,
        cutoff_distance,
        finalize_label,
    )


def carve_density_around_position(
    map_data: DensityMapData,
    position: tuple[float, float, float],
    cutoff_distance: float = 15.0,
    *,
    progress_callback: ProgressCallback | None = None,
) -> np.ndarray:
    """Retain density within a cutoff of a Cartesian position.

    :param map_data: Density volume plus crystallographic metadata
    :param position: ``(x, y, z)`` Cartesian point in Å
    :param cutoff_distance: Retention radius in Å
    :param progress_callback: Optional progress callback
    :return: New carved density array
    """
    if len(position) != 3:
        raise ValueError(
            f"Invalid position {position!r}; expected 3 Cartesian coordinates"
        )
    log.info(
        "Carving density within %.3fÅ of position %s",
        cutoff_distance,
        position,
    )
    log.info("   Centroid: %s", position)
    if progress_callback is not None:
        progress_callback(10, 100, "Processing position coordinates...")
    reference_coordinates = np.asarray([position], dtype=np.float64)
    return _carve_density_near_coordinates(
        map_data,
        reference_coordinates,
        cutoff_distance,
        progress_callback=progress_callback,
        finalize_label="Density carving around centroid complete",
    )


def carve_density_around_protein(
    map_data: DensityMapData,
    pdb_path: str,
    cutoff_distance: float = 4.0,
    *,
    progress_callback: ProgressCallback | None = None,
) -> np.ndarray:
    """Retain density within a cutoff of protein atoms.

    :param map_data: Density volume plus crystallographic metadata
    :param pdb_path: Path to a PDB/mmCIF structure
    :param cutoff_distance: Retention radius in Å
    :param progress_callback: Optional progress callback
    :return: New carved density array
    :raises ValueError: If the structure contains no atoms
    """
    log.info(
        "Carving density within %.3fÅ of protein structure %s",
        cutoff_distance,
        pdb_path,
    )
    if progress_callback is not None:
        progress_callback(10, 100, "Loading protein structure...")

    structure = gemmi.read_structure(pdb_path)

    if progress_callback is not None:
        progress_callback(20, 100, "Extracting atomic coordinates...")

    atoms = _collect_atom_coordinates(structure)
    if not atoms:
        raise ValueError(f"No atom coordinates found in {pdb_path}")

    reference_coordinates = np.asarray(atoms, dtype=np.float64)
    log.info("Found %d atoms in protein structure", len(atoms))
    return _carve_density_near_coordinates(
        map_data,
        reference_coordinates,
        cutoff_distance,
        progress_callback=progress_callback,
        finalize_label="Density carving complete",
    )


def load_density_map_with_extent(
        mtz_path: str,
        pdb_path: str,
        margin: float = 13.0,
        f_label="FWT",
        phi_label="PHWT",
        sample_rate=0.0,
) -> DensityMapData | None:
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

        # Extract crystallographic information and XYZ volume
        crystallographic_info = CrystallographicInfo.from_grid(grid)
        volume = CrystallographicInfo.grid_to_xyz_array(grid)
        crystallographic_info.sync_to_xyz_volume(volume)

        density_map_data = DensityMapData(
            volume=volume,
            crystallographic_info=crystallographic_info,
            margin=margin,
        )
        density_map_data.log_map_data()
        return density_map_data

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

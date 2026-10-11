"""UI-independent reflection data model."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

from molib.xtal.map.unit_cell import UnitCell
from molib.xtal.reflections.formats import ReflectionFileFormat


@dataclass(frozen=True, slots=True)
class ColumnDescriptor:
    """Metadata for one reflection value column."""

    label: str
    column_type: str
    dataset_id: int | None = None
    dataset_name: str = ""


@dataclass(slots=True)
class ReflectionData:
    """Typed reflection batch owned by the loader / viewer.

    NumPy arrays are owned by this object. UI code should treat them as
    read-only unless it intentionally copies.
    """

    path: Path
    format: ReflectionFileFormat
    h: np.ndarray
    k: np.ndarray
    l: np.ndarray
    columns: list[ColumnDescriptor]
    values: dict[str, np.ndarray]
    missing: dict[str, np.ndarray]
    unit_cell: UnitCell | None = None
    space_group: str | None = None
    datasets: list[dict[str, str | float]] = field(default_factory=list)
    d_spacing: np.ndarray | None = None
    is_merged: bool | None = None
    file_resolution_high: float | None = None
    file_resolution_low: float | None = None
    history: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        n = int(self.h.shape[0])
        if self.k.shape[0] != n or self.l.shape[0] != n:
            raise ValueError("Miller index arrays must have equal length")
        for label, arr in self.values.items():
            if arr.shape[0] != n:
                raise ValueError(f"Column {label!r} length mismatch")
            mask = self.missing.get(label)
            if mask is not None and mask.shape[0] != n:
                raise ValueError(f"Missing mask for {label!r} length mismatch")
        if self.d_spacing is not None and self.d_spacing.shape[0] != n:
            raise ValueError("d_spacing length mismatch")

    @property
    def n_reflections(self) -> int:
        """Number of observations / reflections."""
        return int(self.h.shape[0])

    @property
    def miller(self) -> np.ndarray:
        """``(N, 3)`` integer Miller indices."""
        return np.column_stack((self.h, self.k, self.l))

    def column_labels(self) -> list[str]:
        """Return value-column labels in descriptor order."""
        return [c.label for c in self.columns]

    def get_values(self, label: str) -> np.ndarray:
        """Return the float array for *label*."""
        return self.values[label]

    def get_missing(self, label: str) -> np.ndarray:
        """Return boolean missing mask for *label* (True = missing)."""
        if label in self.missing:
            return self.missing[label]
        return np.zeros(self.n_reflections, dtype=bool)

    def summary_dict(self) -> dict[str, object]:
        """Compact metadata dict for UI summary panels."""
        cell = self.unit_cell.to_dict() if self.unit_cell is not None else None
        dmin = dmax = None
        if self.d_spacing is not None and self.d_spacing.size:
            finite = self.d_spacing[np.isfinite(self.d_spacing)]
            if finite.size:
                dmin = float(np.min(finite))
                dmax = float(np.max(finite))
        return {
            "path": str(self.path),
            "format": self.format.value,
            "n_reflections": self.n_reflections,
            "space_group": self.space_group,
            "unit_cell": cell,
            "columns": [
                {
                    "label": c.label,
                    "type": c.column_type,
                    "dataset": c.dataset_name,
                }
                for c in self.columns
            ],
            "datasets": list(self.datasets),
            "is_merged": self.is_merged,
            "resolution_high_file": self.file_resolution_high,
            "resolution_low_file": self.file_resolution_low,
            "resolution_high_computed": dmin,
            "resolution_low_computed": dmax,
        }


def empty_reflection_data(path: Path | str = "") -> ReflectionData:
    """Return an empty :class:`ReflectionData` placeholder."""
    return ReflectionData(
        path=Path(path) if path else Path(),
        format=ReflectionFileFormat.UNKNOWN,
        h=np.zeros(0, dtype=np.int32),
        k=np.zeros(0, dtype=np.int32),
        l=np.zeros(0, dtype=np.int32),
        columns=[],
        values={},
        missing={},
    )

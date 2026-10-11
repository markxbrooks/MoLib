"""Display quantity selection from available MTZ / SF-CIF columns."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from molib.xtal.reflections.model import ColumnDescriptor, ReflectionData


class DisplayQuantity(str, Enum):
    """Closed set of display quantities for the reflection plot."""

    I = "I"
    F = "F"
    I_OVER_SIGI = "I_over_SIGI"
    F_OVER_SIGF = "F_over_SIGF"
    COLUMN = "COLUMN"


@dataclass(frozen=True, slots=True)
class QuantitySelection:
    """Resolved columns for a display quantity."""

    quantity: DisplayQuantity
    value_label: str
    sigma_label: str | None = None
    display_name: str = ""

    def __post_init__(self) -> None:
        if not self.display_name:
            object.__setattr__(self, "display_name", self.value_label)


def _labels_by_type(columns: list[ColumnDescriptor]) -> dict[str, list[str]]:
    by_type: dict[str, list[str]] = {}
    for col in columns:
        by_type.setdefault(col.column_type.upper(), []).append(col.label)
    return by_type


def _find_sigma_for(
    columns: list[ColumnDescriptor], value_label: str
) -> str | None:
    """Heuristic sigma partner for a value column."""
    labels = {c.label: c for c in columns}
    candidates = (
        f"SIG{value_label}",
        f"SIG_{value_label}",
        f"{value_label}SIG",
        "SIGI" if value_label.upper().startswith("I") else None,
        "SIGF" if value_label.upper().startswith("F") else None,
        "F_meas_sigma_au" if "F_meas" in value_label else None,
        "intensity_sigma" if "intensity" in value_label.lower() else None,
    )
    for name in candidates:
        if name and name in labels:
            return name
    # CCP4 type Q near the value column
    by_type = _labels_by_type(columns)
    for q in by_type.get("Q", []):
        if value_label.upper() in q.upper() or q.upper() in value_label.upper():
            return q
    if by_type.get("Q"):
        return by_type["Q"][0]
    return None


def discover_column_pairs(
    columns: list[ColumnDescriptor],
) -> list[tuple[str, str | None, str]]:
    """Return ``(value_label, sigma_label|None, kind)`` candidates.

    *kind* is ``I``, ``F``, or ``other``.
    """
    by_type = _labels_by_type(columns)
    pairs: list[tuple[str, str | None, str]] = []
    for label in by_type.get("J", []) + [
        c.label
        for c in columns
        if c.label.upper() in {"I", "IMEAN", "I(+)", "I(-)"}
        or c.label.upper().startswith("I_")
    ]:
        pairs.append((label, _find_sigma_for(columns, label), "I"))
    for label in by_type.get("F", []) + [
        c.label
        for c in columns
        if c.label.upper() in {"F", "FP", "F_MEAS_AU", "F_MEAS"}
        or "F_MEAS" in c.label.upper()
    ]:
        # de-dupe
        if any(p[0] == label for p in pairs):
            continue
        pairs.append((label, _find_sigma_for(columns, label), "F"))
    # unique preserve order
    seen: set[str] = set()
    out: list[tuple[str, str | None, str]] = []
    for item in pairs:
        if item[0] in seen:
            continue
        seen.add(item[0])
        out.append(item)
    return out


def resolve_quantity(
    data: ReflectionData,
    quantity: DisplayQuantity,
    *,
    column_label: str | None = None,
) -> QuantitySelection | None:
    """Resolve a :class:`QuantitySelection` from *data* or return ``None``."""
    pairs = discover_column_pairs(data.columns)
    labels = set(data.column_labels())

    if quantity == DisplayQuantity.COLUMN:
        if column_label is None or column_label not in labels:
            return None
        return QuantitySelection(
            quantity=quantity,
            value_label=column_label,
            sigma_label=_find_sigma_for(data.columns, column_label),
            display_name=column_label,
        )

    kind = {
        DisplayQuantity.I: "I",
        DisplayQuantity.I_OVER_SIGI: "I",
        DisplayQuantity.F: "F",
        DisplayQuantity.F_OVER_SIGF: "F",
    }[quantity]

    for value_label, sigma_label, pkind in pairs:
        if pkind != kind:
            continue
        if quantity in (DisplayQuantity.I_OVER_SIGI, DisplayQuantity.F_OVER_SIGF):
            if sigma_label is None or sigma_label not in labels:
                continue
            return QuantitySelection(
                quantity=quantity,
                value_label=value_label,
                sigma_label=sigma_label,
                display_name=f"{value_label}/{sigma_label}",
            )
        return QuantitySelection(
            quantity=quantity,
            value_label=value_label,
            sigma_label=sigma_label,
            display_name=value_label,
        )
    return None


def available_quantities(data: ReflectionData) -> list[DisplayQuantity]:
    """List quantities that can be resolved from *data*."""
    out: list[DisplayQuantity] = []
    for q in (
        DisplayQuantity.F,
        DisplayQuantity.I,
        DisplayQuantity.F_OVER_SIGF,
        DisplayQuantity.I_OVER_SIGI,
    ):
        if resolve_quantity(data, q) is not None:
            out.append(q)
    if data.columns:
        out.append(DisplayQuantity.COLUMN)
    return out


def compute_display_values(
    data: ReflectionData,
    selection: QuantitySelection,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(values, missing_mask)`` for plotting.

    Does not alter underlying data: negatives are preserved; missing stays
    marked.
    """
    values = np.asarray(data.get_values(selection.value_label), dtype=np.float64)
    missing = np.asarray(data.get_missing(selection.value_label), dtype=bool).copy()

    if selection.quantity in (
        DisplayQuantity.I_OVER_SIGI,
        DisplayQuantity.F_OVER_SIGF,
    ):
        if selection.sigma_label is None:
            raise ValueError("Sigma column required for I/σ or F/σ display")
        sigma = np.asarray(
            data.get_values(selection.sigma_label), dtype=np.float64
        )
        sigma_missing = np.asarray(
            data.get_missing(selection.sigma_label), dtype=bool
        )
        missing = missing | sigma_missing | ~np.isfinite(sigma) | (np.abs(sigma) < 1e-30)
        out = np.full_like(values, np.nan, dtype=np.float64)
        ok = ~missing
        out[ok] = values[ok] / sigma[ok]
        return out, missing

    missing = missing | ~np.isfinite(values)
    return values, missing

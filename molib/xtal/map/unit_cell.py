from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Any

from decologr import LogMixin, Decologr as log


@dataclass(slots=True)
class UnitCell(LogMixin):
    """UnitCell"""
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    source: str = ""
    space_group: str = ""

    @property
    def center(self) -> tuple[float, float, float]:
        return (
            self.a / 2,
            self.b / 2,
            self.c / 2,
        )

    def validate(self) -> bool:
        from molib.xtal.unit_cell import validate_unit_cell
        return validate_unit_cell(self)

    @classmethod
    def from_dict(cls, data: dict) -> "UnitCell":
        """Create a UnitCell from a dictionary."""
        return cls(
            a=float(data["a"]),
            b=float(data["b"]),
            c=float(data["c"]),
            alpha=float(data["alpha"]),
            beta=float(data["beta"]),
            gamma=float(data["gamma"]),
            source=data.get("source", ""),
            space_group=data.get("space_group", ""),
        )

    def to_dict(self) -> dict:
        """Convert the UnitCell to a dictionary."""
        return {
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "space_group": self.space_group,
            "source": self.source,
        }

    @property
    def fractional_center(self) -> tuple[float, float, float]:
        """Return the geometric center in fractional coordinates."""
        return 0.5, 0.5, 0.5

    @property
    def is_orthogonal(self) -> bool:
        """is_orthogonal"""
        return (
            abs(self.alpha - 90.0) < 0.1
            and abs(self.beta - 90.0) < 0.1
            and abs(self.gamma - 90.0) < 0.1
        )

    @property
    def is_monoclinic(self) -> bool:
        """is monoclinic"""
        return (abs(self.beta - 90.0) > 0.1
                or abs(self.alpha - 90.0) > 0.1
                or abs(self.gamma - 90.0) > 0.1)

    def format_display(self) -> str:
        """
        Format unit cell information for display.

        Args:
            unit_cell: Dictionary containing unit cell parameters

        Returns:
            Formatted string for display
        """
        # Lazy import: unit_cell imports UnitCell from this module.
        from molib.xtal.unit_cell import format_unit_cell_display

        display = format_unit_cell_display(self)

        return display

    def check_consistency(
            self,
            pandas_pdb,
            *,
            length_tolerance: float = 0.01,
            angle_tolerance: float = 0.01,
    ) -> bool:
        """Check whether the current unit cell matches the PDB unit cell.

        Lengths are compared in Å; angles are compared in degrees.
        """
        parameters = ("a", "b", "c", "alpha", "beta", "gamma")

        try:
            # Lazy import: unit_cell imports UnitCell from this module.
            from molib.xtal.unit_cell import extract_unit_cell_dict_from_pdb

            extracted_info = extract_unit_cell_dict_from_pdb(pandas_pdb)

            if not isinstance(extracted_info, Mapping):
                self.log_info("Cannot validate unit cell: no cell parameters found")
                return False

            for key in parameters:
                current = getattr(self, key, None)
                extracted = extracted_info.get(key)

                if current is None or extracted is None:
                    self.log_info(
                        f"Cannot validate unit cell: missing parameter '{key}'"
                    )
                    return False

                try:
                    current = float(current)
                    extracted = float(extracted)
                except (TypeError, ValueError):
                    self.log_info(
                        f"Cannot validate unit cell: invalid parameter '{key}'"
                    )
                    return False

                if not math.isfinite(current) or not math.isfinite(extracted):
                    self.log_info(
                        f"Cannot validate unit cell: non-finite parameter '{key}'"
                    )
                    return False

                tolerance = (
                    length_tolerance
                    if key in ("a", "b", "c")
                    else angle_tolerance
                )

                if abs(current - extracted) > tolerance:
                    self.log_info(
                        f"Unit cell parameter '{key}' differs: "
                        f"{current:.4f} vs {extracted:.4f} "
                        f"(tolerance {tolerance})"
                    )
                    return False

            return True

        except Exception:
            log.exception("Error checking unit cell consistency")
            return False

    def log_summary(self):
        self.log_info(
            f"Unit cell set from {self.source}: "
            f"a={self.a:.2f}, "
            f"b={self.b:.2f}, "
            f"c={self.c:.2f} Å"
            f"alpha={self.alpha:.2f}, "
            f"beta={self.beta:.2f}, "
            f"gamma={self.gamma:.2f} Å"
        )


def normalize_unit_cell_from_dict(
        unit_cell_info: dict[Any, Any] | UnitCell
) -> "UnitCell":
    """normalize unit cell from dict"""
    if not isinstance(unit_cell_info, UnitCell):
        if isinstance(unit_cell_info, dict):
            unit_cell_info = UnitCell.from_dict(unit_cell_info)
    return unit_cell_info

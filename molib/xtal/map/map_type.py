from __future__ import annotations

from enum import Enum


class MapType(str, Enum):
    """Requested electron-density map type used to select MTZ coefficients.

    Values match the canonical ElMo ``MapType`` (``"2Fo-Fc"`` / ``"Fo-Fc"``)
    so either enum (or the plain string) can be passed to the loaders.
    """

    TWO_FO_FC = "2Fo-Fc"
    FO_FC = "Fo-Fc"
    UNKNOWN = "unknown"

    @classmethod
    def coerce(cls, map_type: "MapType | str") -> "MapType":
        """Normalize a map type (str or StrEnum) to this enum; raise on unknown.

        Accepts canonical values (``"2Fo-Fc"``, ``"Fo-Fc"``) and common aliases
        such as ``"2fofc"``, ``"fofc"``, ``"delfwt"``, ``"fwt"``.
        """
        if isinstance(map_type, cls):
            return map_type
        if not isinstance(map_type, str):
            raise ValueError(f"Unsupported map type: {map_type!r}")
        raw = map_type.strip()
        try:
            return cls(raw)
        except ValueError:
            pass
        normalized = raw.lower().replace("_", "-")
        if (
            normalized.startswith("2fo")
            or normalized.startswith("2mfo")
            or normalized in {"2fofc", "fwt"}
        ):
            return cls.TWO_FO_FC
        if normalized in {"fo-fc", "fofc", "delfwt", "difference"}:
            return cls.FO_FC
        if normalized in {"unknown", ""}:
            return cls.UNKNOWN
        raise ValueError(
            f"Unsupported map type: {map_type!r}. "
            f"Expected one of: {[m.value for m in cls]}"
        )

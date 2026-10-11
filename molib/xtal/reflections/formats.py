"""Reflection file format detection."""

from __future__ import annotations

from enum import Enum
from pathlib import Path


class ReflectionFileFormat(str, Enum):
    """Detected reflection file formats supported or recognized by the loader."""

    MTZ = "mtz"
    SF_CIF = "sf_cif"
    UNKNOWN = "unknown"


def sniff_reflection_format(path: Path | str) -> ReflectionFileFormat:
    """Detect the reflection format from extension and file content.

    :param path: Path to a reflection file
    :returns: Detected :class:`ReflectionFileFormat`
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".mtz":
        return ReflectionFileFormat.MTZ

    # Peek text for CIF / SF-CIF markers (also covers some .hkl that are CIF).
    try:
        with p.open("rb") as handle:
            head = handle.read(4096)
    except OSError:
        return ReflectionFileFormat.UNKNOWN

    if not head:
        return ReflectionFileFormat.UNKNOWN

    # MTZ binary magic is often "MTZ " at the start of the header block.
    if head[:4] == b"MTZ " or (len(head) > 4 and b"MTZ " in head[:80]):
        return ReflectionFileFormat.MTZ

    try:
        text = head.decode("utf-8", errors="ignore")
    except Exception:
        return ReflectionFileFormat.UNKNOWN

    lowered = text.lstrip().lower()
    if lowered.startswith("data_") or "data_" in lowered[:200]:
        if (
            "_refln." in lowered
            or "_refln_" in lowered
            or "loop_" in lowered
            and "index_h" in lowered
        ):
            return ReflectionFileFormat.SF_CIF
        if "_refln." in text or "index_h" in lowered:
            return ReflectionFileFormat.SF_CIF
        # Generic CIF without reflection loop — still try SF-CIF path later.
        if suffix in {".cif", ".hkl", ".fcf"}:
            return ReflectionFileFormat.SF_CIF

    if suffix in {".cif", ".fcf"}:
        return ReflectionFileFormat.SF_CIF

    return ReflectionFileFormat.UNKNOWN

"""Crystallographic reflection data (MTZ / SF-CIF) independent of map FFT.

Supported formats (v1):

- Binary CCP4 MTZ via :func:`gemmi.read_mtz_file`.
- Small-molecule structure-factor CIF (SF-CIF) via :func:`gemmi.as_refln_blocks`.

Ambiguous ``.hkl`` files that are not SF-CIF are rejected with
:class:`UnsupportedReflectionFormatError` rather than being forced through the
MTZ reader.

Zone convention: a reflection with Miller index vector ``h`` belongs to a zone
with integer normal ``z`` and level ``n`` when ``h · z = n``. Plot coordinates
are the components of ``h`` along an orthonormal basis of the zone plane in
reciprocal Cartesian space (Å⁻¹) derived from the unit cell when available,
otherwise from the integer plane basis alone.
"""

from molib.xtal.reflections.errors import (
    ReflectionLoadError,
    UnsupportedReflectionFormatError,
)
from molib.xtal.reflections.formats import ReflectionFileFormat, sniff_reflection_format
from molib.xtal.reflections.loader import ReflectionFileLoader
from molib.xtal.reflections.model import ColumnDescriptor, ReflectionData
from molib.xtal.reflections.quantity import DisplayQuantity, QuantitySelection
from molib.xtal.reflections.zone import ZonePreset, ZoneSpec

__all__ = [
    "ColumnDescriptor",
    "DisplayQuantity",
    "QuantitySelection",
    "ReflectionData",
    "ReflectionFileFormat",
    "ReflectionFileLoader",
    "ReflectionLoadError",
    "UnsupportedReflectionFormatError",
    "ZonePreset",
    "ZoneSpec",
    "sniff_reflection_format",
]

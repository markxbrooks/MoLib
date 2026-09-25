"""
Map processing modules for crystallographic data.
"""

# Convenience re-exports (e.g. `from molib.xtal.map import load_ccp4_map`).
# Resolved lazily so that importing a leaf submodule (density) never pulls
# in helper -> filespec at package-import time (circular-import hazard).
_LAZY_HELPER_EXPORTS = (
    "load_ccp4_map",
    "load_ccp4_map_optimized",
    "load_ccp4_maps",
    "load_density_map",
    "load_maps_from_mtz_file_spec",
    "load_mtz_file",
)

__all__ = list(_LAZY_HELPER_EXPORTS)


def __getattr__(name: str):
    if name in _LAZY_HELPER_EXPORTS:
        from . import helper

        return getattr(helper, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + __all__)
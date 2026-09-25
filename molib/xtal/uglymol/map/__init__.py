"""Uglymol map package: ElMap domain object and CCP4/DSN6 loaders.

Hierarchy
---------
* CCP4/DSN6 (this package) → :class:`~molib.xtal.uglymol.map.elmap.ElMap`
* Gemmi MTZ/CCP4 → DensityMapData → MapInfo (see ``molib.xtal.map``)
"""

from molib.xtal.uglymol.map.ccp4_loader import Ccp4MapLoader
from molib.xtal.uglymol.map.dsn6_loader import Dsn6MapLoader
from molib.xtal.uglymol.map.elmap import ElMap
from molib.xtal.uglymol.map.load import load_map

__all__ = [
    "Ccp4MapLoader",
    "Dsn6MapLoader",
    "ElMap",
    "load_map",
]

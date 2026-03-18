"""
Log parameters
"""

from biopandas.pdb import PandasPdb
from decologr import Decologr as log
from molib.pdb.coordinate.data import CoordinateData


def log_returned_parameters(coord_data: CoordinateData, scope: str = ""):
    """Inspect the return type & contents"""
    log.message(f"  -> Returned type: {type(coord_data)}", scope=scope)
    if hasattr(coord_data, "__dict__"):
        log.message(
            f"  -> Returned keys: {list(coord_data.__dict__.keys())}", scope=scope
        )
    elif isinstance(coord_data, dict):
        log.message(f"  -> Returned keys: {list(coord_data.keys())}", scope=scope)
        log.message(f"  -> Returned keys: {list(coord_data.values())}", scope=scope)
    else:
        log.message(f"  -> Returned: {coord_data}", scope=scope)


def log_input_parameters(cfg, pdb: PandasPdb, scope: str = ""):
    """Before setting the attribute"""
    log.message(f"Setting scene.{cfg['coord_attr']} ...", scope=scope)
    log.message(f"  coord_params = {cfg['coord_params']}", scope=scope)
    log.message(
        f"  pdb_pandas   = {type(pdb)} (len={len(pdb) if hasattr(pdb, '__len__') else 'n/a'})",
        scope=scope,
    )

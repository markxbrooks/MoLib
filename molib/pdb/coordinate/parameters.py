"""
Log parameters
"""

from biopandas.pdb import PandasPdb
from decologr import Decologr as log
from molib.pdb.coordinate.data import CoordinateData


def log_returned_parameters(
    coord_data: CoordinateData, scope: str = "", *, silent: bool = False
):
    """Inspect the return type & contents"""
    log.message(
        f"  -> Returned type: {type(coord_data)}", scope=scope, silent=silent
    )
    if hasattr(coord_data, "__dict__"):
        log.message(
            f"  -> Returned keys: {list(coord_data.__dict__.keys())}",
            scope=scope,
            silent=silent,
        )
    elif isinstance(coord_data, dict):
        log.message(
            f"  -> Returned keys: {list(coord_data.keys())}", scope=scope, silent=silent
        )
        log.message(
            f"  -> Returned keys: {list(coord_data.values())}",
            scope=scope,
            silent=silent,
        )
    else:
        log.message(f"  -> Returned: {coord_data}", scope=scope, silent=silent)


def log_input_parameters(
    cfg, pdb: PandasPdb, scope: str = "", *, silent: bool = False
):
    """Before setting the attribute"""
    log.message(
        f"Setting scene.{cfg['coord_attr']} ...", scope=scope, silent=silent
    )
    log.message(
        f"  coord_params = {cfg['coord_params']}", scope=scope, silent=silent
    )
    log.message(
        f"  pdb_pandas   = {type(pdb)} (len={len(pdb) if hasattr(pdb, '__len__') else 'n/a'})",
        scope=scope,
        silent=silent,
    )

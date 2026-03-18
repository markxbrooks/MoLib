from __future__ import annotations

from _hashlib import openssl_sha1
from _sha1 import sha1

import numpy as np


def hash_atoms(atoms: list["Atom3D"]) -> str:
    """
    _hash_atoms

    :param atoms: list[Atom3D]
    :return: str
    Generate a hash based on atom coordinate_data_main and element types
    """
    if len(atoms) == 0:
        return ""
    coords = np.concatenate([a.coords for a in atoms])
    data = f"{coords.tobytes()}{''.join(a.element for a in atoms)}"
    return sha1(data.encode()).hexdigest()

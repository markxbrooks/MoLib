"""RCSB Chemical Component Dictionary SMILES lookup."""

from functools import lru_cache
from typing import Any

try:
    import requests
except ImportError:  # pragma: no cover - optional at import time
    requests = None


CHEMCOMP_URL = "https://data.rcsb.org/rest/v1/core/chemcomp/{ligand_id}"

# RCSB REST uses mixed key styles; keep both current and legacy names.
_SMILES_KEYS = (
    "SMILES_stereo",
    "SMILES",
    "canonical_smiles",
    "smiles",
)


def smiles_from_chemcomp_payload(data: dict[str, Any]) -> str | None:
    """Extract a SMILES string from an RCSB ``chemcomp`` JSON payload.

    Parameters
    ----------
    data
        Decoded JSON from ``/rest/v1/core/chemcomp/{id}``.

    Returns
    -------
    str or None
        Stereo SMILES if present, otherwise canonical SMILES.
    """
    desc = data.get("rcsb_chem_comp_descriptor")
    if not isinstance(desc, dict):
        chem_comp = data.get("chem_comp")
        desc = (
            chem_comp.get("rcsb_chem_comp_descriptor")
            if isinstance(chem_comp, dict)
            else None
        )
    if not isinstance(desc, dict):
        return None

    for key in _SMILES_KEYS:
        value = desc.get(key)
        if value:
            return str(value)
    return None


@lru_cache(maxsize=10_000)
def smiles_from_chemcomp(ligand_id: str) -> str | None:
    """Fetch SMILES for a CCD residue name from the RCSB ChemComp service.

    Parameters
    ----------
    ligand_id
        Three-letter (or CCD) component ID such as ``GDP``.

    Returns
    -------
    str or None
        SMILES string, or ``None`` if the lookup fails.
    """
    if requests is None:
        return None

    ligand_id = (ligand_id or "").strip().upper()
    if not ligand_id:
        return None

    try:
        response = requests.get(
            CHEMCOMP_URL.format(ligand_id=ligand_id), timeout=15
        )
        if response.status_code != 200:
            return None
        return smiles_from_chemcomp_payload(response.json())
    except Exception:
        return None

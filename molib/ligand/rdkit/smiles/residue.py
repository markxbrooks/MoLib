"""SMILES lookup by PDB residue / CCD component ID."""

from molib.ligand.rdkit.smiles.chemcomp import smiles_from_chemcomp

# RDKit-parseable SMILES for frequent PDB heterogens. Metals use bracket
# form; bare symbols such as ``Mg`` are not valid SMILES.
COMMON_LIGAND_SMILES: dict[str, str] = {
    "HOH": "O",
    "WAT": "O",
    "SO4": "[O-]S(=O)(=O)[O-]",
    "PO4": "[O-]P(=O)([O-])[O-]",
    "CL": "[Cl-]",
    "NA": "[Na+]",
    "MG": "[Mg+2]",
    "CA": "[Ca+2]",
    "ZN": "[Zn+2]",
    "FE": "[Fe+2]",
    "MN": "[Mn+2]",
    "CU": "[Cu+2]",
    "NI": "[Ni+2]",
    "CO": "[Co+2]",
    "K": "[K+]",
    "EDO": "CCO",
    "GOL": "C(CO)O",
    "ACT": "CC(=O)O",
    "ZN2": "[Zn+2]",
    "CA2": "[Ca+2]",
    "MG2": "[Mg+2]",
    # Nucleotides / cofactors: distance-based bond guessing fails on
    # aromatic rings plus phosphate valences (e.g. GDP in 1HH4).
    "AMP": "c1nc(N)c2ncn(c2n1)C3OC(COP(=O)(O)O)C(O)C3O",
    "ADP": "c1nc(N)c2ncn(c2n1)C3OC(COP(=O)(O)OP(=O)(O)O)C(O)C3O",
    "ATP": "c1nc(N)c2ncn(c2n1)C3OC(COP(=O)(O)OP(=O)(O)OP(=O)(O)O)C(O)C3O",
    "GDP": "c1nc2c(n1C3C(C(C(O3)COP(=O)(O)OP(=O)(O)O)O)O)N=C(NC2=O)N",
    "GTP": "c1nc2c(n1C3C(C(C(O3)COP(=O)(O)OP(=O)(O)OP(=O)(O)O)O)O)N=C(NC2=O)N",
    "NAD": (
        "c1cc(c[n+](c1)C2C(C(C(O2)COP(=O)(O)OP(=O)(O)"
        "OCC3C(C(C(O3)n4cnc5c4ncnc5N)O)O)O)O)C(=O)N"
    ),
}


def smiles_from_residue_name(
    res_name: str, *, lookup_chemcomp: bool = True
) -> str | None:
    """Return SMILES for a PDB residue name.

    Looks up a local table of common heterogens first, then optionally the
    RCSB Chemical Component Dictionary.

    Parameters
    ----------
    res_name
        Residue / CCD component ID (for example ``GDP``).
    lookup_chemcomp
        If True and the residue is not in the local table, query RCSB.

    Returns
    -------
    str or None
        SMILES string, or ``None`` if unknown.
    """
    key = (res_name or "").strip().upper()
    if not key:
        return None

    smiles = COMMON_LIGAND_SMILES.get(key)
    if smiles:
        return smiles
    if lookup_chemcomp:
        return smiles_from_chemcomp(key)
    return None

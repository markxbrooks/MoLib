"""GDP in 1HH4 must yield RDKit-parseable SMILES."""

from pathlib import Path

from rdkit import Chem

from molib.ligand.element import normalize_element_symbol
from molib.ligand.pdb.parser import PDBLigandParser
from molib.ligand.rdkit.smiles.chemcomp import smiles_from_chemcomp_payload
from molib.ligand.rdkit.smiles.residue import smiles_from_residue_name

# Chain-A GDP HETATM block from PDB 1HH4 (28 heavy atoms).
_GDP_1HH4_PDB = """\
HETATM 5785  PB  GDP A1190      37.480  23.358  30.674  1.00 38.10           P  
HETATM 5786  O1B GDP A1190      38.170  24.562  31.191  1.00 38.10           O  
HETATM 5787  O2B GDP A1190      38.223  22.462  29.935  1.00 38.10           O  
HETATM 5788  O3B GDP A1190      36.697  22.633  31.902  1.00 38.10           O  
HETATM 5789  O3A GDP A1190      36.366  23.848  29.668  1.00 38.10           O  
HETATM 5790  PA  GDP A1190      35.997  23.428  28.078  1.00 38.10           P  
HETATM 5791  O1A GDP A1190      37.077  23.758  27.123  1.00 38.10           O  
HETATM 5792  O2A GDP A1190      35.624  21.975  28.159  1.00 38.10           O  
HETATM 5793  O5' GDP A1190      34.812  24.392  27.825  1.00 38.10           O  
HETATM 5794  C5' GDP A1190      33.486  24.223  28.422  1.00 38.10           C  
HETATM 5795  C4' GDP A1190      32.367  24.591  27.551  1.00 38.10           C  
HETATM 5796  O4' GDP A1190      32.588  25.977  27.044  1.00 38.10           O  
HETATM 5797  C3' GDP A1190      32.263  23.769  26.212  1.00 38.10           C  
HETATM 5798  O3' GDP A1190      30.826  23.540  25.973  1.00 38.10           O  
HETATM 5799  C2' GDP A1190      32.841  24.630  25.061  1.00 38.10           C  
HETATM 5800  O2' GDP A1190      32.324  24.472  23.728  1.00 38.10           O  
HETATM 5801  C1' GDP A1190      32.630  26.086  25.555  1.00 38.10           C  
HETATM 5802  N9  GDP A1190      33.713  27.047  25.192  1.00 38.10           N  
HETATM 5803  C8  GDP A1190      35.129  26.683  25.507  1.00 38.10           C  
HETATM 5804  N7  GDP A1190      35.647  27.870  24.978  1.00 38.10           N  
HETATM 5805  C5  GDP A1190      34.867  28.801  24.447  1.00 38.10           C  
HETATM 5806  C6  GDP A1190      34.791  30.059  23.808  1.00 38.10           C  
HETATM 5807  O6  GDP A1190      35.958  30.687  23.627  1.00 38.10           O  
HETATM 5808  N1  GDP A1190      33.668  30.671  23.406  1.00 38.10           N  
HETATM 5809  C2  GDP A1190      32.371  29.953  23.627  1.00 38.10           C  
HETATM 5810  N2  GDP A1190      31.408  30.715  23.152  1.00 38.10           N  
HETATM 5811  N3  GDP A1190      32.356  28.818  24.180  1.00 38.10           N  
HETATM 5812  C4  GDP A1190      33.514  28.246  24.579  1.00 38.10           C  
END
"""


def test_normalize_element_symbol_title_cases_magnesium() -> None:
    """PDB stores Mg as MG; RDKit needs Mg."""
    assert normalize_element_symbol("MG") == "Mg"
    assert normalize_element_symbol(" C") == "C"
    assert normalize_element_symbol("") == ""


def test_smiles_from_residue_name_gdp_is_offline() -> None:
    """GDP SMILES comes from the local table, not RCSB."""
    smiles = smiles_from_residue_name("GDP", lookup_chemcomp=False)
    assert smiles
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    assert mol.GetNumHeavyAtoms() == 28


def test_smiles_from_chemcomp_payload_uses_rcsb_keys() -> None:
    """RCSB ChemComp JSON uses SMILES / SMILES_stereo, not nested chem_comp."""
    payload = {
        "chem_comp": {"id": "GDP", "name": "GUANOSINE-5'-DIPHOSPHATE"},
        "rcsb_chem_comp_descriptor": {
            "SMILES": "c1nc2c(n1C3C(C(C(O3)COP(=O)(O)OP(=O)(O)O)O)O)N=C(NC2=O)N",
            "SMILES_stereo": (
                "c1nc2c(n1[C@H]3[C@@H]([C@@H]([C@H](O3)CO[P@](=O)(O)"
                "OP(=O)(O)O)O)O)N=C(NC2=O)N"
            ),
        },
    }
    smiles = smiles_from_chemcomp_payload(payload)
    assert smiles is not None
    assert smiles.startswith("c1nc2c")
    assert Chem.MolFromSmiles(smiles) is not None


def test_parser_generates_smiles_for_1hh4_gdp(tmp_path: Path) -> None:
    """Reproduce 1HH4 GDP: HETATM coordinates alone used to yield empty SMILES."""
    pdb_path = tmp_path / "1hh4_gdp.pdb"
    pdb_path.write_text(_GDP_1HH4_PDB, encoding="utf-8")

    ligands = PDBLigandParser().parse_pdb_file(pdb_path, deduplicate=False)
    gdp = [lig for lig in ligands if lig.ligand_id == "GDP"]
    assert len(gdp) == 1
    assert gdp[0].atom_count == 28
    assert gdp[0].smiles
    mol = Chem.MolFromSmiles(gdp[0].smiles)
    assert mol is not None
    assert mol.GetNumHeavyAtoms() == 28
    assert "P" in gdp[0].formula
    assert gdp[0].formula.startswith("C10")

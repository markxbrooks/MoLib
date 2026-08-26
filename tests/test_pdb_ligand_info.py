"""Unit tests for PDBLigandInfo descriptors and export."""

from rdkit import Chem

from molib.ligand.pdb.info import PDBLigandInfo


def _ethanol() -> Chem.Mol:
    return Chem.MolFromSmiles("CCO")


def test_pdb_ligand_info_mw_and_tpsa_from_mol() -> None:
    info = PDBLigandInfo(
        ligand_id="LIG1",
        ligand_name="ethanol",
        chain_id="A",
        res_seq=1,
        mol=_ethanol(),
    )

    assert info.mw > 40.0
    assert info.molecular_weight == info.mw
    assert info.tpsa > 0.0
    assert info.logp is not None
    assert info.atom_count == 3


def test_pdb_ligand_info_to_dict_includes_docking_score() -> None:
    info = PDBLigandInfo(
        ligand_id="LIG1",
        ligand_name="ethanol",
        chain_id="A",
        res_seq=1,
        mol=_ethanol(),
        docking_score=-6.25,
    )

    data = info.to_dict()

    assert data["ligand_id"] == "LIG1"
    assert data["ligand_name"] == "ethanol"
    assert data["chain_id"] == "A"
    assert data["res_seq"] == 1
    assert data["smiles"]
    assert data["mw"] == info.mw
    assert data["logp"] == info.logp
    assert data["tpsa"] == info.tpsa
    assert data["rotatable_bonds"] == info.rotatable_bonds
    assert data["aromatic_rings"] == 0
    assert data["heavy_atoms"] == info.heavy_atoms
    assert data["atom_count"] == info.atom_count
    assert data["docking_score"] == -6.25


def test_pdb_ligand_info_tpsa_fallback_without_mol() -> None:
    info = PDBLigandInfo(
        ligand_id="LIG1",
        ligand_name="ethanol",
        chain_id="A",
        res_seq=1,
        tpsa=20.5,
        logp=0.1,
    )

    assert info.tpsa == 20.5
    assert info.logp == 0.1
    assert info.to_dict()["docking_score"] is None

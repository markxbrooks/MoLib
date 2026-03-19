# Import project metadata
# Navigate up from building/apple/ to project root
import time
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Literal

import gemmi
import pandas as pd
import requests
from rdkit.Chem import AllChem

from decologr import Decologr as log
from decologr import setup_logging
from molib.ligand import PDBLigandInfo, PDBLigandParser
from rdkit import Chem


@lru_cache(maxsize=1)
def load_ccd():
    monomer_dir = gemmi.expand_topology_path("$CLIBD_MON")
    return gemmi.read_monomer_lib(
        monomer_dir, [], ignore_missing=True  # empty list = load all on demand
    )


ENTRY_URL = "https://data.rcsb.org/rest/v1/core/entry/{}"
SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
PDB_FTP_BASE = "https://files.rcsb.org/download/"
PDB_INDEX_URL = "https://files.rcsb.org/ligand/download/INDEX_general_PL.ent"
PDB_MIRROR_URLS = [
    "https://files.rcsb.org/download/",
    "https://pdbj.org/rest/download/",
]
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"
PDB_CACHE_DIR = CACHE_DIR / "pdb"
CIF_CACHE_DIR = CACHE_DIR / "cif"
LIGAND_CACHE_DIR = CACHE_DIR / "pdb_ligands"
# BORING_LIGANDS = {"HOH", "DMS", "EDO", "SO4", "PO4", "NA", "CL", "ADP", "ANP", "GOL", "PEG", "NAD"}

# fragment.py (top-level or constants module)

BORING_LIGANDS: set[str] = {
    # Solvents / buffers
    "HOH",
    "WAT",
    "DOD",
    "SO4",
    "PO4",
    "NO3",
    "CO3",
    "ACT",
    "ACE",
    "FMT",
    # Common ions
    "NA",
    "K",
    "CL",
    "BR",
    "IOD",
    "MG",
    "MN",
    "CA",
    "ZN",
    "CU",
    "FE",
    "NI",
    "CO",
    # Cryo / additives
    "PEG",
    "PEG2",
    "PEG3",
    "PEG4",
    "PG4",
    "GOL",
    "EDO",
    "MPD",
    "DMS",
    "IPA",
    # Known non-fragment residues
    "UNL",
    "UNK",
    # Nucleotides
    "ADP",
    "ANP",
    "NAD",
}


def make_dirs(dirlist: list[Path]):
    for dir_name in dirlist:
        if not dir_name.exists():
            dir_name.mkdir(parents=True, exist_ok=True)


class StructureFormat(Enum):
    PDB = "pdb"
    CIF = "cif"
    LIGAND = "ligand"

    @property
    def suffix(self) -> str:
        return f".{self.value}"


def get_ligand_info(ligand_id):
    url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{ligand_id}"
    r = requests.get(url)
    if r.status_code != 200:
        return None
    print(r.json())
    data = r.json()
    return {
        "id": ligand_id,
        "name": data["chem_comp"]["name"],
        "formula": data["chem_comp"]["formula"],
        "type": data["chem_comp"]["type"],
    }


@lru_cache(maxsize=10_000)
def smiles_from_chemcomp(ligand_id: str) -> str | None:
    ligand_id = ligand_id.upper()

    url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{ligand_id}"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return None

        data = r.json().get("chem_comp", {})
        desc = data.get("rcsb_chem_comp_descriptor", {})

        # Preferred order
        return (
            desc.get("canonical_smiles")
            or desc.get("smiles")
            or desc.get("inchi")  # last resort
        )

    except Exception:
        return None


def smiles_from_ccd(res_name: str) -> str | None:
    lib = load_ccd()
    cc = lib.monomers.get(res_name.upper())
    if cc is None:
        return None
    return cc.smiles or None


def rdkit_mol_from_cif_residue(res: gemmi.Residue) -> Chem.Mol | None:
    mol = Chem.RWMol()

    atom_indices = []
    for atom in res:
        if atom.element.atomic_number == 0:
            return None  # unknown element
        atom_indices.append(mol.AddAtom(Chem.Atom(atom.element.atomic_number)))

    conf = Chem.Conformer(len(atom_indices))
    for i, atom in enumerate(res):
        conf.SetAtomPosition(i, (atom.pos.x, atom.pos.y, atom.pos.z))

    mol.AddConformer(conf)
    mol = mol.GetMol()

    try:
        # 🔑 RDKit bond perception
        rdDetermineBonds.DetermineConnectivity(mol)
        rdDetermineBonds.DetermineBondOrders(mol)
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def rdkit_mol_from_coordinates(lig: PDBLigandInfo) -> Chem.Mol | None:
    mol = Chem.RWMol()

    atom_indices = []
    for elem in lig.element_symbols:
        atom_indices.append(mol.AddAtom(Chem.Atom(elem)))

    conf = Chem.Conformer(len(atom_indices))
    for i, (x, y, z) in enumerate(lig.coordinates):
        conf.SetAtomPosition(i, (x, y, z))

    mol.AddConformer(conf)
    mol = mol.GetMol()

    try:
        AllChem.EmbedMolecule(mol, randomSeed=0xF00D)
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def convert_cif_to_pdb(pdb_id: str) -> Path | None:
    try:
        cif_file = get_structure_file_path(pdb_id, structure_format=StructureFormat.CIF)
        pdb_file = get_structure_file_path(pdb_id, structure_format=StructureFormat.PDB)

        # Read CIF
        doc = gemmi.cif.read_file(str(cif_file))
        block = doc.sole_block()

        # Make structure
        structure = gemmi.make_structure_from_block(block)

        # Write to PDB
        structure.write_pdb(str(pdb_file))
        log.info(f"Wrote {pdb_file}")
    except Exception as ex:
        log.error(f"Failed to convert {pdb_id} to PDB")


def enrich_ligand_with_smiles(
    lig: PDBLigandInfo, residue: gemmi.Residue
) -> PDBLigandInfo:

    res_name = residue.name.upper()

    if is_boring_ligand(res_name):
        return lig  # silent skip

    smiles = smiles_from_chemcomp(res_name)
    if not smiles:
        log.debug(f"No ChemComp SMILES for {res_name}")
        return lig

    lig.smiles = smiles
    return lig


def extract_ligands_from_cif(cif_text: str, pdb_id: str):
    doc = gemmi.cif.read_string(cif_text)
    structure = gemmi.make_structure_from_block(doc.sole_block())

    ligands = []

    for model in structure:
        for chain in model:
            for res in chain:
                if res.is_water():
                    continue
                if res.entity_type != gemmi.EntityType.NonPolymer:
                    continue

                lig = PDBLigandInfo(
                    ligand_id=res.name,
                    ligand_name=f"{res.name} ({pdb_id}:{chain.name}:{res.seqid.num})",
                    chain_id=chain.name,
                    res_seq=res.seqid.num,
                    insertion_code=res.seqid.icode or "",
                    atom_count=len(res),
                    coordinates=[(a.pos.x, a.pos.y, a.pos.z) for a in res],
                    atom_names=[a.name for a in res],
                    element_symbols=[a.element.name for a in res],
                    smiles="",
                    molecular_weight=0.0,
                    formula="",
                    logp=0.0,
                    hbd=0,
                    hba=0,
                    tpsa=0.0,
                    rotatable_bonds=0,
                    aromatic_rings=0,
                    heavy_atoms=len(res),
                )

                lig = enrich_ligand_with_smiles(lig, residue=res)
                ligands.append(lig)

    return ligands


def search_fragment_screening_structures(rows: int = 500) -> list[PDBLigandInfo]:
    """search for fragment screening structures"""
    query = {
        "query": {
            "type": "terminal",
            "service": "full_text",
            "parameters": {"value": "XChem"},
        },
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": rows}},
    }
    response = requests.post(SEARCH_URL, json=query)
    response.raise_for_status()
    found_pdb_ids = [r["identifier"] for r in response.json()["result_set"]]
    log.info(f"Found {len(found_pdb_ids)} fragment-screening structures")
    return found_pdb_ids


pdb_ids = search_fragment_screening_structures()


def fetch_mmcif(pdb_id: str) -> tuple[str, Path]:
    url = f"https://files.rcsb.org/download/{pdb_id}{StructureFormat.CIF.suffix}"
    r = requests.get(url)
    r.raise_for_status()
    pdb_file = get_structure_file_path(pdb_id, StructureFormat.CIF)
    if not pdb_file.exists():  # don't re-download existing files
        with open(pdb_file, "wb") as f:
            f.write(r.content)
    return r.text, pdb_file


def extract_ligands_from_pdb(
    pdb_id: str, cif_text: str = None, structure_format=StructureFormat.PDB
) -> list[PDBLigandInfo] | None:
    try:
        pdb_file_path = get_structure_file_path(pdb_id, structure_format)
        parser = PDBLigandParser()
        ligand_list = parser.parse_pdb_file(pdb_file_path, deduplicate=True)
        return ligand_list
    except Exception as e:
        log.error(f"Error parsing {pdb_id}: {e}")


valid_structure_formats = Literal[
    StructureFormat.PDB, StructureFormat.CIF, StructureFormat.LIGAND
]


def get_structure_file_path(
    pdb_id: str, structure_format: valid_structure_formats = StructureFormat.PDB
) -> Path:
    if structure_format == StructureFormat.PDB:
        file_dir = PDB_CACHE_DIR
    elif structure_format == StructureFormat.CIF:
        file_dir = CIF_CACHE_DIR
    else:
        raise ValueError(f"Unsupported format: {structure_format}")
    pdb_file_path = Path(file_dir) / f"{pdb_id}{structure_format.suffix}"
    return pdb_file_path


def fetch_structure(
    pdb_id: str,
    structure_format: StructureFormat,
    retries: int = 2,
) -> tuple[str, Path] | None:
    pdb_id = pdb_id.upper()
    path = get_structure_file_path(pdb_id, structure_format)

    if path.exists():
        return path.read_text(), path

    url = f"{PDB_FTP_BASE}{pdb_id}{structure_format.suffix}"

    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            path.write_bytes(r.content)
            return r.text, path
        except Exception as e:
            log.warning(f"{pdb_id}: fetch attempt {attempt+1} failed ({e})")
            time.sleep(2**attempt)

    log.error(f"{pdb_id}: failed to download")
    return None


def parse_ligands(
    pdb_id: str,
    structure_text: str,
    structure_format: StructureFormat,
) -> list[PDBLigandInfo]:

    if structure_format == StructureFormat.PDB:
        parser = PDBLigandParser()
        return parser.parse_pdb_file(
            get_structure_file_path(pdb_id, StructureFormat.PDB),
            deduplicate=False,
        )

    if structure_format == StructureFormat.CIF:
        return extract_ligands_from_cif(structure_text, pdb_id)

    raise ValueError(f"Unsupported structure format: {structure_format}")


def accumulate_ligands(
    pdb_ids: list[str],
    structure_format: StructureFormat,
) -> dict[str, PDBLigandInfo]:

    ligand_registry: dict[str, PDBLigandInfo] = {}

    for pdb_id in pdb_ids:
        result = fetch_structure(pdb_id, StructureFormat.PDB)
        if not result:
            result = fetch_structure(pdb_id, StructureFormat.CIF)
            convert_cif_to_pdb(pdb_id)
            if not result:
                continue

        text, _ = result
        ligands = parse_ligands(pdb_id, text, structure_format)
        log.info(f"{pdb_id}: found {len(ligands)} ligands")

        for lig in ligands:
            ligand_registry.setdefault(lig.ligand_id, lig)

    log.info(f"🧬 Unique ligands: {len(ligand_registry)}")
    return ligand_registry


def filter_fragments(
    ligands: dict[str, PDBLigandInfo],
    exclude: set[str] = BORING_LIGANDS,
) -> dict[str, PDBLigandInfo]:

    fragments = {k: v for k, v in ligands.items() if v.ligand_id not in exclude}

    log.info(f"🧪 Fragment ligands after filtering: {len(fragments)}")
    return fragments


def ligands_to_dataframe(ligands: dict[str, PDBLigandInfo]) -> pd.DataFrame:

    df = pd.DataFrame(vars(lig) for lig in ligands.values())
    log.info(f"📊 DataFrame with {len(df)} rows and {len(df.columns)} columns")
    return df


def save_ligands_csv(
    df: pd.DataFrame,
    structure_format: StructureFormat,
    output_dir: Path = CACHE_DIR,
) -> Path:

    out = output_dir / f"ligands_{structure_format.value}.csv"
    df.to_csv(out, index=False)
    log.info(f"💾 Saved ligand table → {out}")
    return out


def run_fragment_ligand_pipeline(
    structure_format: StructureFormat = StructureFormat.PDB,
):
    pdb_ids = search_fragment_screening_structures()
    all_ligands = accumulate_ligands(pdb_ids, structure_format)
    fragments = filter_fragments(all_ligands)
    df = ligands_to_dataframe(fragments)
    save_ligands_csv(df, structure_format)
    return df


def is_boring_ligand(res_name: str) -> bool:
    return res_name.upper() in BORING_LIGANDS


def is_fragment(lig: PDBLigandInfo) -> bool:
    return lig.ligand_id not in BORING_LIGANDS


if __name__ == "__main__":
    setup_logging(project_name="elmo_inspect")
    make_dirs([CIF_CACHE_DIR, PDB_CACHE_DIR, LIGAND_CACHE_DIR])
    df = run_fragment_ligand_pipeline(structure_format=StructureFormat.PDB)
    print(df.head())

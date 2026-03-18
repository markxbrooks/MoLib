import numpy as np
import pandas as pd
from decologr import Decologr as log
from molib.entities.atom import Atom3D
from molib.entities.chain import Chain3D
from molib.entities.model import Model3D
from molib.entities.molecule import Molecule3D
from molib.entities.residue import Res3D
from molib.ligand.pdb.spec import PDBLineSpec
from molib.pdb.coordinate.data import CoordinateData
from picogl.mode import GLMode


def _safe_int(slice_str: str) -> int | None:
    """Convert to int safely."""
    try:
        return int(slice_str.strip())
    except (ValueError, TypeError):
        return None


def _safe_char(line: str, idx: int) -> str:
    """Safely extract a single character."""
    return line[idx].strip() if len(line) > idx else ""


class PDBRecordType:
    """PDB Record Type"""

    TITLE = "TITLE"
    ATOM = "ATOM"
    HETATM = "HETATM"
    END = "END"


class PDBSecondaryStruct:
    """PDB Secondary Struct"""

    HELIX = "HELIX"
    SHEET = "SHEET"


class HelixLayout:
    """Helix Layout"""

    chain_start = PDBLineSpec("chain_start", 19, 20)
    start = PDBLineSpec("start", 21, 25, int)
    chain_end = PDBLineSpec("chain_end", 31, 32)
    end = PDBLineSpec("end", 33, 37, int)

    @classmethod
    def fields(cls):
        return [
            cls.chain_start,
            cls.start,
            cls.chain_end,
            cls.end,
        ]


class SheetLayout:
    """Sheet layout"""

    chain_start = PDBLineSpec("chain_start", 21, 22)
    start = PDBLineSpec("start", 22, 26, int)
    chain_end = PDBLineSpec("chain_end", 32, 33)
    end = PDBLineSpec("end", 33, 37, int)

    @classmethod
    def fields(cls):
        return [
            cls.chain_start,
            cls.start,
            cls.chain_end,
            cls.end,
        ]


class PDBHeaderLayout:
    """PDB Layout class - Pure Python class for PDB-type and PDB-type"""

    title = PDBLineSpec("title", 0, 10)
    record = PDBLineSpec("record", 0, 6)

    @classmethod
    def fields(cls):
        return [
            cls.title,
            cls.record,
        ]


from typing import Dict, Tuple


def parse_pdb_text_sec_str(pdb_text: str) -> Dict[Tuple[str, int], str]:
    """
    Parse HELIX / SHEET records into a residue-level secondary structure map.

    Returns:
        dict { (chain_id, res_num): 'H' | 'E' }
    """
    sec_map: Dict[Tuple[str, int], str] = {}

    for line in pdb_text.splitlines():
        if len(line) < 40:  # guard against malformed lines
            continue

        record = PDBHeaderLayout.record.parse(line)
        if not record:
            continue

        record = record.strip().upper()

        # --- HELIX ---
        if record == PDBSecondaryStruct.HELIX:
            chain_start = HelixLayout.chain_start.parse(line)
            start = HelixLayout.start.parse(line)
            chain_end = HelixLayout.chain_end.parse(line)
            end = HelixLayout.end.parse(line)

            if (
                not chain_start
                or not chain_end
                or not isinstance(start, int)
                or not isinstance(end, int)
            ):
                continue

            if chain_start == chain_end:
                for r in range(start, end + 1):
                    sec_map[(chain_start, r)] = "H"

        # --- SHEET ---
        elif record == PDBSecondaryStruct.SHEET:
            chain_start = SheetLayout.chain_start.parse(line)
            start = SheetLayout.start.parse(line)
            chain_end = SheetLayout.chain_end.parse(line)
            end = SheetLayout.end.parse(line)

            if (
                not chain_start
                or not chain_end
                or not isinstance(start, int)
                or not isinstance(end, int)
            ):
                continue

            if chain_start == chain_end:
                for r in range(start, end + 1):
                    sec_map[(chain_start, r)] = "E"

    return sec_map


def extract_pdb_title(pdb_text: str) -> str:
    """
    Extract the TITLE line from PDB text.

    Args:
        pdb_text: Raw PDB file content as string

    Returns:
        str: The title text from the TITLE line, or empty string if not found
    """
    title_lines = []

    for line in pdb_text.splitlines():
        if line.startswith(PDBRecordType.TITLE):
            # TITLE lines can be continued with spaces in columns 1-6
            # Extract the title content (columns 11-80)
            title_content = PDBHeaderLayout.title.parse(line)
            if title_content:
                title_lines.append(title_content)
        elif line.startswith("     ") and title_lines:
            # Continuation line for TITLE
            title_content = PDBHeaderLayout.title.parse(line)
            if title_content:
                title_lines.append(title_content)
        elif line.startswith(
            (PDBRecordType.ATOM, PDBRecordType.HETATM, PDBRecordType.END)
        ):
            # Stop at other record types (but not HEADER, as it comes before TITLE)
            break

    # Join all title lines and clean up
    full_title = " ".join(title_lines).strip()
    return full_title


def _apply_atom_validation(
    atom,
    chain_id,
    res_num,
    atom_name,
    validation_data,
):

    if not validation_data:
        atom.atom_validated = None
        return

    missing_atoms = validation_data["missing_atoms"]
    residues_with_missing = validation_data["residues_with_missing"]

    atom_key = (chain_id, res_num, atom_name)
    residue_key = (chain_id, res_num)

    if atom_key in missing_atoms:
        atom.atom_validated = False
    elif residue_key in residues_with_missing:
        atom.atom_validated = False
    else:
        atom.atom_validated = True


def _apply_residue_validation(residue, res_key, validation_data):

    if not validation_data:
        residue.residue_validated = None
        residue.residue_validation_error = None
        return

    residues_with_missing = validation_data["residues_with_missing"]

    if res_key in residues_with_missing:
        residue.residue_validated = False
        residue.residue_validation_error = "Missing atoms"
    else:
        residue.residue_validated = True
        residue.residue_validation_error = None


def _prepare_validation_data(validation_report: dict) -> dict:
    """prepare validation data"""
    if not validation_report or "missing_atoms" not in validation_report:
        return None

    missing_atoms = set()
    residues_with_missing = set()

    for atom_address in validation_report["missing_atoms"]:

        res_num = atom_address.res_id.seqid.num
        chain_id = atom_address.chain_name
        atom_name = atom_address.atom_name

        missing_atoms.add((chain_id, res_num, atom_name))
        residues_with_missing.add((chain_id, res_num))

    return {
        "missing_atoms": missing_atoms,
        "residues_with_missing": residues_with_missing,
    }


class PDBPandaColumns:
    """PDB columns for pandas dataframe"""

    CHAIN_ID = "chain_id"
    RESIDUE_NUMBER = "residue_number"
    RESIDUE_NAME = "residue_name"
    X_COORD = "x_coord"
    Y_COORD = "y_coord"
    Z_COORD = "z_coord"
    ALT_LOC = "alt_loc"
    B_FACTOR = "b_factor"
    SEGMENT_ID = "segment_id"
    ELEMENT_SYMBOL = "element_symbol"


def parse_pdb_atoms_to_mol3d(
    atom_df: pd.DataFrame,
    coordinate_data: CoordinateData,
    pdb_text: str = "",
    validation_report: dict | None = None,
) -> Molecule3D:

    mol = Molecule3D(coordinate_data=coordinate_data)
    model = Model3D(name="model_1")
    mol.add_model(model)

    sec_lookup = parse_pdb_text_sec_str(pdb_text)

    validation_data = _prepare_validation_data(validation_report)

    chain_map: dict[str, Chain3D] = {}

    grouped = atom_df.groupby(
        [PDBPandaColumns.CHAIN_ID, PDBPandaColumns.RESIDUE_NUMBER], sort=False
    )

    for (chain_id, res_num), group in grouped:

        chain_id = chain_id.strip()
        res_num = int(res_num)

        # Ensure chain
        if chain_id not in chain_map:
            chain = Chain3D(name=chain_id, parent=model)
            chain_map[chain_id] = chain
            model.add_chain(chain_id, chain)
        else:
            chain = chain_map[chain_id]

        # Residue metadata (first row is enough)
        first = group.iloc[0]

        res_name = first[PDBPandaColumns.RESIDUE_NAME].strip()
        coords = (
            float(first[PDBPandaColumns.X_COORD]),
            float(first[PDBPandaColumns.Y_COORD]),
            float(first[PDBPandaColumns.Z_COORD]),
        )

        res_key = (chain_id, res_num)

        residue = Res3D(
            name=str(res_name),
            residue_number=res_num,
            chain_id=chain_id,
            type=str(res_name),
            coords=coords,
            secstruc=sec_lookup.get(res_key, "C"),
            parent=chain,
        )

        _apply_residue_validation(residue, res_key, validation_data)

        chain.add_residue(residue)

        # Process atoms within residue
        for row in group.itertuples(index=False):

            atom_name = row.atom_name.strip()

            coords = np.array(
                (row.x_coord, row.y_coord, row.z_coord),
                dtype=float,
            )

            element = (
                getattr(row, PDBPandaColumns.ELEMENT_SYMBOL, None) or atom_name[0]
            ).strip()

            atom = Atom3D(
                name=atom_name,
                alt_loc=getattr(row, PDBPandaColumns.ALT_LOC, "").strip(),
                b_factor=float(row.b_factor),
                segment_id=row.segment_id,
                chain_id=chain_id,
                element=element,
                coords=coords,
                parent=residue,
            )

            _apply_atom_validation(
                atom,
                chain_id,
                res_num,
                atom_name,
                validation_data,
            )

            residue.atoms[atom_name] = atom

    return mol

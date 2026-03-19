"""
Parser
"""
import numpy as np

from molib.entities.atom import Atom3D
from molib.ligand.pdb.spec import PDBLineSpec


class PDBLayout:
    """PDB Layout class - Pure Python class for PDB-type and PDB-type"""

    record_type = PDBLineSpec("record_type", 0, 6)
    atom_serial = PDBLineSpec("atom_serial", 6, 11, int)
    atom_name = PDBLineSpec("atom_name", 12, 16)
    res_name = PDBLineSpec("res_name", 17, 20)
    chain_id = PDBLineSpec("chain_id", 21, 22)
    res_seq = PDBLineSpec("res_seq", 22, 26, int)
    insertion_code = PDBLineSpec("insertion_code", 26, 27)
    x = PDBLineSpec("x", 30, 38, float)
    y = PDBLineSpec("y", 38, 46, float)
    z = PDBLineSpec("z", 46, 54, float)
    occupancy = PDBLineSpec("occupancy", 54, 60, float)
    temp_factor = PDBLineSpec("temp_factor", 60, 66, float)
    element = PDBLineSpec("element", 76, 78)

    @classmethod
    def fields(cls):
        return [
            cls.record_type,
            cls.atom_serial,
            cls.atom_name,
            cls.res_name,
            cls.chain_id,
            cls.res_seq,
            cls.insertion_code,
            cls.x,
            cls.y,
            cls.z,
            cls.occupancy,
            cls.temp_factor,
            cls.element,
        ]


class PDBSecStruct:
    """PDB Sec Struct"""

    ATOM = "ATOM"


def parse_pdb_coordinates_from_file(file_path: str):
    """
    parse_pdb_coordinates_from_file

    :param file_path: str
    :return:
    """
    coords = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.startswith(PDBSecStruct.ATOM):
                x = PDBLayout.x.parse(line)
                y = PDBLayout.y.parse(line)
                z = PDBLayout.z.parse(line)
                coords.append((x, y, z))

    return coords


def parse_pdb_atoms(file_path: str) -> list[Atom3D]:
    atoms: list[Atom3D] = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.startswith(PDBSecStruct.ATOM):
                continue

            x = PDBLayout.x.parse(line)
            y = PDBLayout.y.parse(line)
            z = PDBLayout.z.parse(line)

            atom = Atom3D(
                serial=PDBLayout.atom_serial.parse(line),
                name=PDBLayout.atom_name.parse(line),
                chain_id=PDBLayout.chain_id.parse(line),
                element=PDBLayout.element.parse(line),
                res_name=PDBLayout.res_name.parse(line),
                res_seq=PDBLayout.res_seq.parse(line),
                coords=np.array([x, y, z], dtype=np.float32),
                occupancy=PDBLayout.occupancy.parse(line) or 1.0,
                b_factor=PDBLayout.temp_factor.parse(line) or 0.0,
            )

            atoms.append(atom)

    return atoms

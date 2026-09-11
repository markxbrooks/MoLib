"""
PDBLayout
"""


from molib.ligand.pdb.spec import PDBLineSpec


class PDBLayout:
    """PDB Layout class - Pure Python class for PDB-type and PDB-type"""

    record_type = PDBLineSpec("record_type", 0, 6)
    atom_serial = PDBLineSpec("atom_serial", 6, 11, int)
    atom_name = PDBLineSpec("atom_name", 12, 16)
    alt_loc = PDBLineSpec("res_name", 16, 17)
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
    coords = (x, y, z)

    @classmethod
    def fields(cls):
        return [
            cls.record_type,
            cls.atom_serial,
            cls.atom_name,
            cls.alt_loc,
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
            cls.coords,
        ]


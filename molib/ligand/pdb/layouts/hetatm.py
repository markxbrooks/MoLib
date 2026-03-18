"""
Defines the layout for HETATM records in a Protein Data Bank (PDB) file.

This module encapsulates the specification for the structure and
attributes of HETATM lines in the PDB format. Each attribute defined
maps to a specific region of the line in the PDB file.

Classes:
    HETATMLayout: Represents the structure of a HETATM record in a PDB file.
"""

from molib.ligand.pdb.spec import PDBLineSpec


class HETATMLayout:
    """HETATMLayout"""

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

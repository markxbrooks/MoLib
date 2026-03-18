"""
Represents the layout specifications for a PDB file format line. The class
defines the positions and characteristics of various fields within a
Protein Data Bank (PDB) file line.

Attributes
----------
atom_serial : PDBLineSpec
    Field specification for the atom serial number.
atom_name : PDBLineSpec
    Field specification for the atom name.
res_name : PDBLineSpec
    Field specification for the residue name.
chain_id : PDBLineSpec
    Field specification for the chain identifier.
res_seq : PDBLineSpec
    Field specification for the residue sequence number.
insertion_code : PDBLineSpec
    Field specification for the insertion code.
x : PDBLineSpec
    Field specification for the x-coordinate of the atom.
y : PDBLineSpec
    Field specification for the y-coordinate of the atom.
z : PDBLineSpec
    Field specification for the z-coordinate of the atom.
occupancy : PDBLineSpec
    Field specification for the occupancy value.
temp_factor : PDBLineSpec
    Field specification for the temperature factor.
element : PDBLineSpec
    Field specification for the chemical element symbol.
"""


class PDBFileLayout:
    """PDBFileLayout"""

    atom_serial = PDBLineSpec(name="atom_serial", start_pos=6, stop_pos=11)
    atom_name = PDBLineSpec(name="atom_name", start_pos=12, stop_pos=16)
    res_name = PDBLineSpec(name="res_name", start_pos=17, stop_pos=20)
    chain_id = PDBLineSpec(name="chain_id", start_pos=21, stop_pos=22)
    res_seq = PDBLineSpec(name="res_seq", start_pos=22, stop_pos=26)
    insertion_code = PDBLineSpec(name="insertion_code", start_pos=26, stop_pos=27)
    x = PDBLineSpec(name="x", start_pos=30, stop_pos=38)
    y = PDBLineSpec(name="y", start_pos=38, stop_pos=46)
    z = PDBLineSpec(name="z", start_pos=46, stop_pos=54)
    occupancy = PDBLineSpec(name="occupancy", start_pos=54, stop_pos=60)
    temp_factor = PDBLineSpec(name="temp_factor", start_pos=60, stop_pos=66)
    element = PDBLineSpec(name="element", start_pos=76, stop_pos=78)

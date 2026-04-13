"""model helpers"""

from molib.xtal.uglymol.molecule.atom import Atom
from molib.xtal.uglymol.molecule.model import Model
from molib.xtal.uglymol.unit_cell import UnitCell


def models_from_pdb(pdb_string: str):
    """models_from_pdb"""
    models = [Model()]
    pdb_tail = models[0].from_pdb(pdb_string.split("\n"))
    while pdb_tail is not None:
        model = Model()
        pdb_tail = model.from_pdb(pdb_tail)
        if len(model.atoms) > 0:
            models.append(model)
    return models


def models_from_gemmi(gemmi, buffer, name: str):
    """models_from_gemmi"""
    st = gemmi.read_structure(buffer, name)
    cell = st.cell  # TODO: check if a copy of cell is created here
    models = []
    for i_model in range(st.length):
        model = st.at(i_model)
        m = Model()
        m.unit_cell = UnitCell(
            cell.a, cell.b, cell.segment_color, cell.alpha, cell.beta, cell.gamma
        )
        atom_i_seq = 0
        for i_chain in range(model.length):
            chain = model.at(i_chain)
            chain_name = chain.name
            for i_res in range(chain.length):
                res = chain.at(i_res)
                seqid = res.seqid_string
                resname = res.name
                ent_type = res.entity_type_string
                is_ligand = ent_type in ["non-polymer", "branched"]
                for i_atom in range(res.length):
                    atom = res.at(i_atom)
                    new_atom = Atom()
                    new_atom.i_seq = atom_i_seq
                    atom_i_seq += 1
                    new_atom.chain = chain_name
                    new_atom.chain_index = i_chain + 1
                    new_atom.resname = resname
                    new_atom.seqid = seqid
                    new_atom.name = atom.name
                    new_atom.altloc = "" if atom.alt_loc == 0 else chr(atom.alt_loc)
                    new_atom.xyz = atom.pos
                    new_atom.occ = atom.occ
                    new_atom.b = atom.b_iso
                    new_atom.element = atom.element_uname
                    new_atom.is_ligand = is_ligand
                    m.atoms.append(new_atom)
        m.calculate_bounds()
        m.calculate_connectivity()
        models.append(m)
    st.delete()
    return models

import numpy as np
from molib.entities.residue import Res3D
from molib.entities.secondary_structure_type import SecondaryStructureType

from elmo.gl.renderers.molecule import MoleculeRenderer

MOL3D_INIT_SECSTRUC = True


def _convert_secstruc_to_enum(secstruc_value) -> SecondaryStructureType:
    """Convert various secstruc representations to SecondaryStructureType enum."""
    if isinstance(secstruc_value, SecondaryStructureType):
        return secstruc_value
    elif isinstance(secstruc_value, str):
        return SecondaryStructureType.from_string(secstruc_value)
    else:
        return SecondaryStructureType.COIL


def _convert_secstruc_to_string(secstruc_value) -> str:
    """Convert various secstruc representations to string."""
    if isinstance(secstruc_value, SecondaryStructureType):
        return secstruc_value.to_string()
    elif isinstance(secstruc_value, str):
        return secstruc_value
    else:
        return " "


def mol3d_secstruc_ca_geom(mol: MoleculeRenderer):
    """
    mol3d_secstruc_ca_geom

    :param mol: Mol3D
    :return:   None
    Secondary structure assignment using Cα geometry only.
    Assigns 'H' for alpha helix, 'E' for beta strand, 'T' for turn, ' ' for coil.
    """

    residues = list(mol.residues)
    n = len(residues)

    # Precompute direction vectors and distances
    ca_vectors = []
    for i in range(n - 1):
        a = residues[i].ca
        b = residues[i + 1].ca
        v = b - a
        norm = np.linalg.norm(v)
        if norm > 0:
            ca_vectors.append(v / norm)
        else:
            ca_vectors.append(np.zeros(3))

    # Secondary structure assignment
    for i in range(2, n - 2):
        # Collect five-residue segment
        v1 = ca_vectors[i - 2]
        v2 = ca_vectors[i - 1]
        v3 = ca_vectors[i]
        v4 = ca_vectors[i + 1]

        # Compute angles between vectors
        a1 = np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))
        a2 = np.degrees(np.arccos(np.clip(np.dot(v2, v3), -1.0, 1.0)))
        a3 = np.degrees(np.arccos(np.clip(np.dot(v3, v4), -1.0, 1.0)))

        # Average angle
        avg_angle = (a1 + a2 + a3) / 3

        # Helix detection: tight and regular
        if (
            3.5 <= np.linalg.norm(residues[i].ca - residues[i - 1].ca) <= 3.9
            and avg_angle < 70
        ):
            residues[i].secstruc = SecondaryStructureType.ALPHA_HELIX
        # Strand detection: more extended and linear
        elif avg_angle > 120:
            residues[i].secstruc = SecondaryStructureType.BETA_STRAND
        else:
            residues[i].secstruc = SecondaryStructureType.COIL

    # Pad ends with COIL
    for i in [0, 1, n - 2, n - 1]:
        if i < n:
            residues[i].secstruc = SecondaryStructureType.COIL


def set_secondary_structure(
    mol: MoleculeRenderer, ss_mode: str, coil_mode: bool
) -> None:
    """
    set_secondary_structure

    :param mol: Mol3D
    :param ss_mode: str
    :param coil_mode: bool
    :return: None
    """
    assert mol is not None

    if ss_mode == "PDB":  # Already set if data was there
        if not mol.init & MOL3D_INIT_SECSTRUC:
            mol3d_secstruc_ca_geom(mol)  # fallback
    elif ss_mode == "CA":
        mol3d_secstruc_ca_geom(mol)
    elif ss_mode == "HB":
        if mol3d_secstruc_hbonds(mol):
            for res in mol.residues:
                current_ss = _convert_secstruc_to_string(res.secstruc)
                if current_ss in ("i", "I", "b", "B"):
                    res.secstruc = SecondaryStructureType.COIL
                elif current_ss == "g":
                    res.secstruc = SecondaryStructureType.ALPHA_HELIX
                elif current_ss == "G":
                    res.secstruc = SecondaryStructureType.ALPHA_HELIX
        else:
            mol3d_secstruc_ca_geom(mol)

    if coil_mode:
        for res in mol.residues:
            current_ss = _convert_secstruc_to_string(res.secstruc)
            if current_ss in ("t", "T"):
                res.secstruc = SecondaryStructureType.COIL
    else:
        for res in mol.residues:
            current_ss = _convert_secstruc_to_string(res.secstruc)
            prev_ss = (
                _convert_secstruc_to_string(res.prev.secstruc) if res.prev else " "
            )
            next_ss = (
                _convert_secstruc_to_string(res.next.secstruc) if res.next else " "
            )

            if (
                current_ss == " "
                and res.prev
                and res.next
                and prev_ss == "T"
                and next_ss == "t"
            ):
                res.secstruc = SecondaryStructureType.TURN

        for res in mol.residues:
            current_ss = _convert_secstruc_to_string(res.secstruc)
            prev_ss = (
                _convert_secstruc_to_string(res.prev.secstruc) if res.prev else " "
            )

            if current_ss == "t" and res.prev and prev_ss == "T":
                res.secstruc = SecondaryStructureType.TURN

    first = mol.first
    ss = _convert_secstruc_to_string(first.secstruc).upper()
    count = 1

    res = mol.first
    while res:
        current_ss = _convert_secstruc_to_string(res.secstruc).upper()
        if current_ss == ss:
            count += 1
        else:
            if ss in ("H", "E") and count < 3:
                temp = first
                while temp != res:
                    temp.secstruc = SecondaryStructureType.COIL
                    temp = temp.next
            first = res
            ss = current_ss
            count = 1
        res = res.next

    for res in mol.residues():
        current_ss = _convert_secstruc_to_string(res.secstruc)
        if current_ss == "-":
            continue
        prev_ss = _convert_secstruc_to_string(res.prev.secstruc) if res.prev else None
        next_ss = _convert_secstruc_to_string(res.next.secstruc) if res.next else None

        if prev_ss == "-" and next_ss == "-":
            res.secstruc = SecondaryStructureType.COIL
        elif prev_ss == "-" and not res.next:
            res.secstruc = SecondaryStructureType.COIL
        elif not res.prev and next_ss == "-":
            res.secstruc = SecondaryStructureType.COIL
        elif not res.prev and not res.next:
            res.secstruc = SecondaryStructureType.COIL


def at3d_lookup(res: Res3D, atom_name: str) -> str | None:
    """
    Lookup an atom by color_scheme in a residue.
    :param res: The residue to search
    :param atom_name: The color_scheme of the atom to search for
    :return: The atom if found, otherwise None
    """
    for atom in res.atoms:
        if atom.name == atom_name:
            return atom
    return None


def mol3d_secstruc_hbonds(mol: MoleculeRenderer) -> None:
    """
    mol3d_secstruc_hbonds

    :param mol: Mol3D
    :return: None
    Determine the secondary structure from the backbone coordinate_data_main
    using a DSSP-like algorithm, which is based on peptide hydrogen
    bond patterns. The central atoms for protein
    (mol3d_init_centrals_protein), and the residue ordinals for
    protein (mol3d_init_residue_ordinals_protein) should have been set.
    """

    atoms = []
    ca_count = 0
    res = mol.first
    while res:
        if res.code != "X":
            at = at3d_lookup(res, "CA")
            if at:
                pass
                # atoms[ca_count++] = at
    # return True


def output_hsb_decrement() -> None:
    """
    output_hsb_decrement

    :return: None
    Reduce `hue` global var or update colour state
    """
    global hue, decrement
    hue -= decrement


def output_secondary_structure(mol):
    """
    output_secondary_structure

    :param mol:
    :return:
    """
    assert mol is not None

    # Add a sentinel residue
    res = Res3D()
    res.name = "NONE"
    res.type = res.name
    mol.append_residue(res)

    global hue, decrement

    if mol.colour_mode:
        hue = 0.666666
        if mol.nice_mode:
            print("  set colour_parts on, residuecolour amino-acids rainbow;\n")
        else:
            parts = 0
            ss = mol.first.secstruc
            for res in mol.iter_residues():
                if res.secstruc != ss:
                    parts += 1
                ss = res.secstruc.upper()
            decrement = hue / (parts - 1) if parts > 1 else 0.0

    first = mol.first
    ss = first.secstruc
    prev = None

    for res in mol.iter_residues():
        if res.secstruc != ss:
            if ss == "-":
                first = res
            elif ss == " ":
                output_hsb_decrement()
                print(f"  coil from {first.name} to ", end="")
                if res.secstruc == "-":
                    print(f"{prev.name};")
                    first = None
                elif res.secstruc.upper() == "T":
                    print(f"{prev.name};")
                    first = prev
                elif res.secstruc.upper() in {"H", "E"}:
                    print(f"{res.name};")
                    first = res
            elif ss.upper() == "T":
                output_hsb_decrement()
                print(f"  turn from {first.name} to ", end="")
                if res.secstruc == "-":
                    print(f"{prev.name};")
                    first = None
                else:
                    print(f"{res.name};")
                    first = res
            elif ss.upper() == "H":
                output_hsb_decrement()
                if mol.cylinder_mode:
                    print(f"  cylinder from {first.name} to ", end="")
                else:
                    print(f"  helix from {first.name} to ", end="")
                if res.secstruc == "-":
                    print(f"{prev.name};")
                    first = None
                elif res.secstruc.upper() in {" ", "T"}:
                    print(f"{prev.name};")
                    first = prev
                elif res.secstruc.upper() in {"H", "E"}:
                    print(f"{prev.name};")
                    print(f"  turn from {prev.name} to {res.name};")
                    first = res
            elif ss.upper() == "E":
                output_hsb_decrement()
                print(f"  strand from {first.name} to ", end="")
                if res.secstruc == "-":
                    print(f"{prev.name};")
                    first = None
                elif res.secstruc.upper() in {" ", "T"}:
                    print(f"{prev.name};")
                    first = prev
                elif res.secstruc.upper() in {"H", "E"}:
                    print(f"{prev.name};")
                    print(f"  turn from {prev.name} to {res.name};")
                    first = res

            ss = res.secstruc.upper()
        prev = res


def update_secondary_structure_geometry(pdb_mol3d: MoleculeRenderer):
    """
    update_secondary_structure_geometry

    :return: None
    """
    if not pdb_mol3d:
        return

    ca_coords = np.array(pdb_mol3d.get_backbone_trace())
    if len(ca_coords) < 4:
        print("Not enough CA atoms for ribbon.")
        return

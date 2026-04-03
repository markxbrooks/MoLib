import numpy as np
from molib.entities.molecule import Molecule3D, STANDARD_POLYPEPTIDE_RESIDUES
from molib.entities.residue import Res3D
from molib.entities.secondary_structure_type import SecondaryStructureType

# Bit reserved for callers that build Molecule3D from PDB: OR into ``mol.init`` when
# HELIX/SHEET (or mmCIF equivalent) supplied secondary structure.
MOL3D_INIT_SECSTRUC = 1 << 0


def _convert_secstruc_to_enum(
    secstruc_value: str | SecondaryStructureType,
) -> SecondaryStructureType:
    """Convert various secstruc representations to SecondaryStructureType enum."""
    if isinstance(secstruc_value, SecondaryStructureType):
        return secstruc_value
    elif isinstance(secstruc_value, str):
        return SecondaryStructureType.from_string(secstruc_value)
    else:
        return SecondaryStructureType.COIL


def _convert_secstruc_to_string(secstruc_value: str | SecondaryStructureType) -> str:
    """Convert various secstruc representations to string."""
    if isinstance(secstruc_value, SecondaryStructureType):
        return secstruc_value.to_string()
    elif isinstance(secstruc_value, str):
        return secstruc_value
    else:
        return " "


def _link_residues_ordered(mol: Molecule3D):
    """
    PDB-built :class:`Molecule3D` keeps residues on chains without ``prev`` / ``next``.
    Link them in :meth:`Molecule3D.get_all_residues` order so molscript-style passes
    (short H/E runs, turns) match legacy single-list traversal.

    Returns the first residue, or ``None`` if the molecule has no residues.
    """
    first = None
    prev = None
    for res in mol.get_all_residues():
        if first is None:
            first = res
        res.prev = prev
        if prev is not None:
            prev.next = res
        prev = res
    if prev is not None:
        prev.next = None
    return first


def mol3d_protein_has_explicit_secondary_structure(mol: Molecule3D) -> bool:
    """
    True if any standard polypeptide residue with CA carries non-coil SS.

    When a PDB/mmCIF file has no HELIX/SHEET records, parsers typically leave every
    residue at the default :class:`SecondaryStructureType.COIL`; then geometry-based
    inference should run.
    """
    for res in mol.residues_with_ca_protein_only():
        ss = getattr(res, "secstruc", None)
        if ss is None:
            continue
        if isinstance(ss, SecondaryStructureType):
            if ss != SecondaryStructureType.COIL:
                return True
            continue
        if SecondaryStructureType.from_string(str(ss).strip()) != SecondaryStructureType.COIL:
            return True
    return False


def _polypeptide_residue_with_ca(res: Res3D) -> bool:
    if not res.has_ca():
        return False
    res_name = (
        (getattr(res, "name", "") or getattr(res, "type", "") or "")
        .strip()
        .upper()
    )
    return res_name in STANDARD_POLYPEPTIDE_RESIDUES


def _virtual_ca_bend_deg(ca_prev: np.ndarray, ca_mid: np.ndarray, ca_next: np.ndarray) -> float:
    """Angle ∠(Cα(i−1), Cα(i), Cα(i+1)); helices ~90°, extended strands ~120°+."""
    u = ca_prev - ca_mid
    v = ca_next - ca_mid
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return 180.0
    return float(
        np.degrees(np.arccos(np.clip(np.dot(u / nu, v / nv), -1.0, 1.0)))
    )


def _mol3d_secstruc_ca_geom_residue_run(residues: list[Res3D]) -> None:
    """
    Cα-only SS for one contiguous polypeptide run (single chain segment).

    Uses the virtual Cα angle and a short-window direction-change average.  Strand
    detection cannot rely on ``avg_angle > 120`` alone — on real structures that
    metric rarely exceeds ~100°; bend at Cα separates sheet (~120°+) from helix
    (~85–100°) much more cleanly than direction averages alone.
    """
    n = len(residues)
    if n < 5:
        for res in residues:
            res.secstruc = SecondaryStructureType.COIL
        return

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

    for i in range(2, n - 2):
        v1 = ca_vectors[i - 2]
        v2 = ca_vectors[i - 1]
        v3 = ca_vectors[i]
        v4 = ca_vectors[i + 1]

        a1 = np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))
        a2 = np.degrees(np.arccos(np.clip(np.dot(v2, v3), -1.0, 1.0)))
        a3 = np.degrees(np.arccos(np.clip(np.dot(v3, v4), -1.0, 1.0)))

        avg_angle = (a1 + a2 + a3) / 3
        step = float(np.linalg.norm(residues[i].ca - residues[i - 1].ca))
        bend = _virtual_ca_bend_deg(
            residues[i - 1].ca, residues[i].ca, residues[i + 1].ca
        )

        is_strand = (bend >= 110.0 and avg_angle < 60.0) or (
            bend >= 118.0 and avg_angle < 72.0
        )
        is_helix = (
            3.45 <= step <= 4.05
            and bend <= 118.0
            and avg_angle < 96.0
            and not is_strand
        )

        if is_strand and 3.2 <= step <= 4.2:
            residues[i].secstruc = SecondaryStructureType.BETA_STRAND
        elif is_helix:
            residues[i].secstruc = SecondaryStructureType.ALPHA_HELIX
        else:
            residues[i].secstruc = SecondaryStructureType.COIL

    for idx in (0, 1, n - 2, n - 1):
        if 0 <= idx < n:
            residues[idx].secstruc = SecondaryStructureType.COIL


def mol3d_secstruc_ca_geom(mol: Molecule3D):
    """
    mol3d_secstruc_ca_geom

    :param mol: Mol3D
    :return:   None
    Secondary structure assignment using Cα coords only.
    Assigns 'H' for alpha helix, 'E' for beta strand, 'T' for turn, ' ' for coil.

    Runs independently per chain so vectors do not cross chain breaks or
    non-protein residues (ligands between peptide links).
    """

    for model in mol.models:
        for chain in model.chains.values():
            seq = [r for r in chain.residues if _polypeptide_residue_with_ca(r)]
            seq.sort(key=lambda r: r.residue_number)
            _mol3d_secstruc_ca_geom_residue_run(seq)


def set_secondary_structure(mol: Molecule3D, ss_mode: str, coil_mode: bool) -> None:
    """
    set_secondary_structure

    :param mol: Mol3D
    :param ss_mode: str
    :param coil_mode: bool
    :return: None
    """
    assert mol is not None

    if ss_mode == "PDB":
        # No HELIX/SHEET in file → everything remains default coil; infer from Cα geometry.
        if not mol3d_protein_has_explicit_secondary_structure(mol):
            mol3d_secstruc_ca_geom(mol)
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

    first = _link_residues_ordered(mol)
    if first is None:
        return

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

    ss = _convert_secstruc_to_string(first.secstruc).upper()
    count = 1

    res = first
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

    for res in mol.residues:
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


def mol3d_secstruc_hbonds(mol: Molecule3D) -> None:
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


def update_secondary_structure_geometry(pdb_mol3d: Molecule3D):
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

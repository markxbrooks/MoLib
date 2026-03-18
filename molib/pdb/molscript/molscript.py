import numpy as np
from molib.pdb.molscript.colour.unequal import colour_unequal
from molib.pdb.molscript.smoothing import priestle_smoothing

from elmo.gl.backend.legacy.entities.strand.strand import (
    count_residue_selections,
    ext3d_update,
    hermite_get,
    mol3d_chain_delete,
)

PEPTIDE_CHAIN_ATOMNAME = "CA"
PEPTIDE_DISTANCE = 4.2
NUCLEOTIDE_CHAIN_ATOMNAME = "P"
NUCLEOTIDE_DISTANCE = 10.0

# Helix and strand constants
HELIX_HERMITE_FACTOR = 4.7
STRAND_HERMITE_FACTOR = 0.5
HELIX_ALPHA = np.radians(32.0)
HELIX_BETA = np.radians(-11.0)
ANGLE_PI = np.pi


def coil_segment_init():
    pass


def coil(is_peptide_chain, smoothing):
    import numpy as np

    atomname, coilname = None, None
    chain_distance, t = None, None
    first_ch, ch = None, None
    points = None
    res = None
    first, last = None, None
    vec1, vec2 = np.zeros(3), np.zeros(3)
    col = None
    slot, segment = None, None
    hermite_factor = 0.5 * current_state.splinefactor
    segments = current_state.segments

    assert count_residue_selections() == 1

    if current_state.colour_parts and (segments % 2):
        segments += 1

    if is_peptide_chain:
        atomname = PEPTIDE_CHAIN_ATOMNAME
        chain_distance = PEPTIDE_DISTANCE
        first_ch = get_peptide_chains()
        coilname = "coil" if smoothing else "turn"
    else:
        atomname = NUCLEOTIDE_CHAIN_ATOMNAME
        chain_distance = NUCLEOTIDE_DISTANCE
        coilname = "double-helix"
        first_ch = get_nucleotide_chains()

    for ch in first_ch:
        if ch.length < 2:
            continue

        msg_chain(coilname, ch)
        points = get_atom_positions(ch)

        first = None
        res = ch.residues[0].prev
        if res:
            first = at3d_lookup(res, atomname)
            if first and v3_distance(first.xyz, points) >= chain_distance:
                first = None
        if first is None:
            first = ch.atoms[0]

        last = None
        res = ch.residues[ch.length - 1].next
        if res:
            last = at3d_lookup(res, atomname)
            if last and v3_distance(last.xyz, points[ch.length - 1]) >= chain_distance:
                last = None
        if last is None:
            last = ch.atoms[ch.length - 1]

        if smoothing:
            priestle_smoothing(points, ch.length, current_state.smoothsteps)

        col = ch.residues[0].colour if current_state.colour_parts else None

        if current_state.coilradius < 0.01:
            ls = line_segment_init()
            ls.new = True
            ls.p = points[0]
            ls.c = col if col else current_state.plane_colour
            ext3d_update(points, 0.0)

            vec2 = points[1] - first.xyz
            if first == ch.atoms[0]:
                vec2 *= 2.0
            vec2 *= hermite_factor

            for slot in range(ch.length - 1):
                vec1 = vec2.copy()
                if slot == ch.length - 2:
                    vec2 = last.xyz - points[ch.length - 2]
                    if last == ch.atoms[ch.length - 1]:
                        vec2 *= 2.0
                else:
                    vec2 = points[slot + 2] - points[slot]
                vec2 *= hermite_factor

                hermite_set(points[slot], points[slot + 1], vec1, vec2)
                for segment in range(1, segments):
                    t = segment / segments
                    ls = line_segment_next()
                    hermite_get(ls.p, t)
                    if (
                        current_state.colour_parts
                        and (segment == segments // 2)
                        and colour_unequal(col, ch.residues[slot + 1].colour)
                    ):
                        ls.c = col
                        ls = line_segment_next()
                        ls.new = True
                        ls.p = ls[-1].p
                        col = ch.residues[slot + 1].colour
                    if col:
                        ls.c = col
                    ext3d_update(ls.p, 0.0)

                ls = line_segment_next()
                ls.p = points[slot + 1]
                if col:
                    ls.c = col
                ext3d_update(points[slot + 1], 0.0)

            output_line(True)

        else:
            cs = coil_segment_init()
            cs.p = points[0]
            cs.c = col if col else current_state.plane_colour
            ext3d_update(cs.p, current_state.coilradius / np.sqrt(2.0))

            vec2 = points[1] - first.xyz
            if first == ch.atoms[0]:
                vec2 *= 2.0
            vec2 *= hermite_factor

            for slot in range(ch.length - 1):
                vec1 = vec2.copy()
                if slot == ch.length - 2:
                    vec2 = last.xyz - points[ch.length - 2]
                    if last == ch.atoms[ch.length - 1]:
                        vec2 *= 2.0
                else:
                    vec2 = points[slot + 2] - points[slot]
                vec2 *= hermite_factor

                hermite_set(points[slot], points[slot + 1], vec1, vec2)
                for segment in range(1, segments):
                    t = segment / segments
                    cs = coil_segment_next()
                    hermite_get(cs.p, t)
                    if current_state.colour_parts and (segment == segments // 2):
                        col = ch.residues[slot + 1].colour
                    if col:
                        cs.c = col
                    ext3d_update(cs.p, current_state.coilradius / np.sqrt(2.0))

                cs = coil_segment_next()
                cs.p = points[slot + 1]
                if col:
                    cs.c = col
                ext3d_update(points[slot + 1], current_state.coilradius / np.sqrt(2.0))

            output_coil()

    if first_ch:
        mol3d_chain_delete(first_ch)

    assert count_residue_selections() == 0


def helix():
    """Render helix structures using the molscript algorithm."""

    # This is a placeholder implementation
    # The actual implementation would need access to the current_state and chain data structures
    # that are not available in this context
    # For now, we'll create a simple placeholder that can be called
    # without causing import errors
    # In a real implementation, this would:
    # 1. Get peptide chains from current_state
    # 2. Calculate helix axes and tangents
    # 3. Generate helix segments with proper geometry
    # 4. Call output_helix() to render

    pass


def strand():
    """Render strand structures using the molscript algorithm."""

    # This is a placeholder implementation
    # The actual implementation would need access to the current_state and chain data structures
    # that are not available in this context
    # For now, we'll create a simple placeholder that can be called
    # without causing import errors
    # In a real implementation, this would:
    # 1. Get peptide chains from current_state
    # 2. Calculate strand normals and directions
    # 3. Apply priestle smoothing
    # 4. Generate strand segments with proper geometry
    # 5. Call output_strand() to render

    pass

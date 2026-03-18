"""
Kabsch & Sander Secondary Structure Detection
==============================================

Port of Ribbons' pdb-pro-ss algorithm for ElMo.
Implements the Kabsch & Sander (1983) method for detecting secondary structure
from hydrogen bonding patterns.

Based on:
- ribbons/utils/pdb-pro-ss.C
- ribbons/utils/hb-calc.C
- ribbons/utils/pdb-hbio.C

Reference:
Kabsch, W. & Sander, C. (1983) Dictionary of protein secondary structure:
pattern recognition of hydrogen-bonded and geometrical features.
Biopolymers 22:2577-2637.
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb

# Constants from hb-calc.C
MISSING = -10000001.0
Q1 = 0.42  # Partial charge on C, O
Q2 = 0.20  # Partial charge on N, H
DC = 332.0  # Reciprocal dielectric constant for kcal
EC = -0.5  # Energy cutoff in kcal/mole

# Helix angle for H calculation (123 degrees in radians)
HELIX_ALPHA = math.radians(123.0)


class BackboneAtoms:
    """Store backbone atom coordinates for a residue"""

    def __init__(self):
        self.n = np.array([MISSING, MISSING, MISSING])  # N (amide nitrogen)
        self.h = np.array([MISSING, MISSING, MISSING])  # H (amide hydrogen)
        self.c = np.array([MISSING, MISSING, MISSING])  # C (carbonyl carbon)
        self.o = np.array([MISSING, MISSING, MISSING])  # O (carbonyl oxygen)
        self.ca = np.array([MISSING, MISSING, MISSING])  # CA (alpha carbon)


def calculate_hydrogen_positions(
    nres: int, xb: List[BackboneAtoms], seq: List[str]
) -> int:
    """
    Calculate positions of main chain hydrogens if missing.

    Based on H_calc() in hb-calc.C

    Args:
        nres: Number of residues (1-indexed)
        xb: List of BackboneAtoms (indexed 1..nres)
        seq: List of amino acid one-letter codes (indexed 1..nres)

    Returns:
        Number of hydrogens calculated
    """
    k = 0
    cosa = math.cos(HELIX_ALPHA)
    sina = math.sin(HELIX_ALPHA)

    for i in range(1, nres + 1):
        # Skip if H already present
        if xb[i].h[0] != MISSING:
            continue

        # Skip proline (no amide hydrogen)
        if seq[i] == "P":
            continue

        # Need previous C, current N, and current CA
        if xb[i - 1].c[0] == MISSING or xb[i].n[0] == MISSING or xb[i].ca[0] == MISSING:
            continue

        # Calculate hydrogen position
        # Vector from previous C to current N
        x = xb[i].n - xb[i - 1].c

        # Vector from CA to N
        y = xb[i].ca - xb[i].n

        # Cross products to establish coordinate system
        z = np.cross(y, x)
        y = np.cross(z, x)

        # Normalize
        x = x / np.linalg.norm(x)
        y = y / np.linalg.norm(y)

        # Rotate in the plane
        h_pos = x * cosa + y * sina
        xb[i].h = h_pos + xb[i].n

        k += 1

    return k


def dist_sq(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate squared distance between two points"""
    diff = a - b
    return np.dot(diff, diff)


def calculate_hydrogen_bonds(
    nres: int, xb: List[BackboneAtoms], max_hb: int
) -> List[Tuple[int, int]]:
    """
    Calculate hydrogen bonds using Kabsch-Sander energy formula.

    Based on DataHbond() in hb-calc.C

    Energy formula:
    E = q1*q2*(1/r(ON) + 1/r(CH) - 1/r(OH) - 1/r(CN)) * 332

    Args:
        nres: Number of residues (1-indexed)
        xb: List of BackboneAtoms (indexed 0..nres)
        max_hb: Maximum number of H-bonds to store

    Returns:
        List of (CO_index, NH_index) tuples for H-bonds
    """
    hbonds = []

    # Loop over all carbonyls (i) vs all N-H (j)
    for i in range(0, nres + 1):
        if xb[i].c[0] == MISSING or xb[i].o[0] == MISSING:
            continue

        for j in range(0, nres + 1):
            if xb[j].n[0] == MISSING or xb[j].h[0] == MISSING:
                continue

            # Skip adjacent residues (i==j, i==j-1, i==j-2, i==j+1, i==j+2)
            if i == j or i == j - 1 or i == j - 2:
                continue
            if i == j + 1 or i == j + 2:
                continue

            # Calculate H-bond energy
            r_on = math.sqrt(dist_sq(xb[i].o, xb[j].n))
            r_ch = math.sqrt(dist_sq(xb[i].c, xb[j].h))
            r_oh = math.sqrt(dist_sq(xb[i].o, xb[j].h))
            r_cn = math.sqrt(dist_sq(xb[i].c, xb[j].n))

            eb = Q1 * Q2 * DC * (1.0 / r_on + 1.0 / r_ch - 1.0 / r_oh - 1.0 / r_cn)

            if eb < EC:
                hbonds.append((i, j))
                if len(hbonds) >= max_hb:
                    raise RuntimeError(
                        f"Too many H-bonds! Increase max_hb (currently {max_hb})"
                    )

    return hbonds


def set_turn(
    nres: int, nhb: int, hbonds: List[Tuple[int, int]]
) -> Tuple[List[str], List[str], List[str]]:
    """
    Identify 3-turn, 4-turn, and 5-turn patterns from H-bonds.

    Based on SetTurn() in hb-calc.C

    Args:
        nres: Number of residues (1-indexed)
        nhb: Number of H-bonds
        hbonds: List of (CO_index, NH_index) tuples

    Returns:
        Tuple of (t3, t4, t5) arrays, each of length nres+1
        Symbols: '>' = CO makes H-bond, '<' = NH makes H-bond,
                 'X' = both, '3'/'4'/'5' = bracketed residues
    """
    t3 = [" "] * (nres + 1)
    t4 = [" "] * (nres + 1)
    t5 = [" "] * (nres + 1)

    for co_idx, nh_idx in hbonds:
        nt = nh_idx - co_idx  # Turn number
        no = co_idx

        if nt == 3:
            # 3-turn
            if t3[no] != "<":
                t3[no] = ">"
            else:
                t3[no] = "X"

            # Mark bracketed residues
            for j in range(no + 1, no + nt):
                if t3[j] == " ":
                    t3[j] = "3"

            # Mark NH end
            if t3[no + nt] != "X" and t3[no + nt] != ">":
                t3[no + nt] = "<"
            else:
                if t3[no + nt] == ">":
                    t3[no + nt] = "X"

        elif nt == 4:
            # 4-turn
            if t4[no] != "<":
                t4[no] = ">"
            else:
                t4[no] = "X"

            for j in range(no + 1, no + nt):
                if t4[j] == " ":
                    t4[j] = "4"

            if t4[no + nt] != "X" and t4[no + nt] != ">":
                t4[no + nt] = "<"
            else:
                if t4[no + nt] == ">":
                    t4[no + nt] = "X"

        elif nt == 5:
            # 5-turn
            if t5[no] != "<":
                t5[no] = ">"
            else:
                t5[no] = "X"

            for j in range(no + 1, no + nt):
                if t5[j] == " ":
                    t5[j] = "5"

            if t5[no + nt] != "X" and t5[no + nt] != ">":
                t5[no + nt] = "<"
            else:
                if t5[no + nt] == ">":
                    t5[no + nt] = "X"

    return t3, t4, t5


def set_helix(nres: int, t3: List[str], t4: List[str], t5: List[str]) -> List[str]:
    """
    Identify helices from turn patterns.

    Based on SetHelix() in hb-calc.C

    Summary codes:
    - 'H' = 4-helix (alpha-helix)
    - 'G' = 3-helix (3,10-helix)
    - 'I' = 5-helix (pi-helix)
    - 'T' = H-bonded turn
    - ' ' = coil

    Args:
        nres: Number of residues (1-indexed)
        t3, t4, t5: Turn arrays from set_turn()

    Returns:
        Summary array of length nres+1
    """
    sy = [" "] * (nres + 1)

    # Process 4-turns (alpha helices)
    count = 0
    for i in range(1, nres + 1):
        if t4[i] == ">" or t4[i] == "X":
            count += 1
            if count > 1:
                sy[i] = sy[i + 1] = sy[i + 2] = sy[i + 3] = "H"
        else:
            if t4[i] == "4" or t4[i] == "<":
                if count == 1:
                    for j in range(3):
                        if sy[i + j] != "H":
                            sy[i + j] = "T"
            count = 0

    # Process 3-turns (3,10 helices)
    count = 0
    for i in range(1, nres + 1):
        if t3[i] == ">" or t3[i] == "X":
            count += 1
            if count > 1:
                # Check if already alpha helix
                tag = 0
                for j in range(3):
                    if sy[i + j] == "H":
                        tag = 1
                        break

                if tag == 0:
                    sy[i] = sy[i + 1] = sy[i + 2] = "G"
                else:
                    for j in range(3):
                        if sy[i + j] != "H":
                            sy[i + j] = "T"
        else:
            if t3[i] == "3" or t3[i] == "<":
                if count == 1:
                    for j in range(2):
                        if sy[i + j] != "H":
                            sy[i + j] = "T"
            count = 0

    # Process 5-turns (pi helices)
    count = 0
    for i in range(1, nres + 1):
        if t5[i] == ">" or t5[i] == "X":
            count += 1
            if count > 1:
                # Check if already alpha or 3,10 helix
                tag = 0
                for j in range(5):
                    if sy[i + j] == "H" or sy[i + j] == "G":
                        tag = 1
                        break

                if tag == 0:
                    sy[i] = sy[i + 1] = sy[i + 2] = sy[i + 3] = sy[i + 4] = "I"
                else:
                    for j in range(5):
                        if sy[i + j] != "H" and sy[i + j] != "G":
                            sy[i + j] = "T"
        else:
            if t5[i] == "5" or t5[i] == "<":
                if count == 1:
                    for j in range(4):
                        if sy[i + j] != "H" and sy[i + j] != "G":
                            sy[i + j] = "T"
            count = 0

    return sy


def set_bridge(nres: int, nhb: int, hbonds: List[Tuple[int, int]]) -> np.ndarray:
    """
    Identify beta bridges from H-bonds.

    Based on SetBridge() in hb-calc.C

    Args:
        nres: Number of residues (1-indexed)
        nhb: Number of H-bonds
        hbonds: List of (CO_index, NH_index) tuples

    Returns:
        2D numpy array (nres+1 x nres+1) with bridge types:
        0 = no bridge, 1 = parallel bridge, 2 = antiparallel bridge
    """
    br = np.zeros((nres + 1, nres + 1), dtype=np.uint8)

    for i in range(1, nhb):
        no = hbonds[i][0]
        nn = hbonds[i][1]

        for j in range(1, nhb):
            if i == j:
                continue

            noj = hbonds[j][0]
            nnj = hbonds[j][1]

            # Parallel bridge pattern
            if nn == noj and no == nnj - 2:
                if abs((no + 1) - nn) >= 3:
                    br[no + 1, nn] = 1
                    br[nn, no + 1] = 1

            # Antiparallel bridge pattern 1
            if no == nnj and nn == noj:
                if abs(no - nn) >= 3:
                    br[no, nn] = 2
                    br[nn, no] = 2

            # Antiparallel bridge pattern 2
            if no == nnj - 2 and nn == noj + 2:
                if abs((no + 1) - (nn - 1)) >= 3:
                    br[no + 1, nn - 1] = 2
                    br[nn - 1, no + 1] = 2

    return br


def set_ladder(
    nres: int, br: np.ndarray
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Identify ladders from bridges.

    Based on SetLadder() in hb-calc.C

    Args:
        nres: Number of residues (1-indexed)
        br: Bridge matrix from set_bridge()

    Returns:
        Tuple of (br1, br2, sh1, sh2) arrays
    """
    br1 = [" "] * (nres + 1)
    br2 = [" "] * (nres + 1)
    sh1 = [" "] * (nres + 1)
    sh2 = [" "] * (nres + 1)
    tmp = [" "] * (nres + 1)

    low = [
        " ",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
    ]

    i = 1
    nlet = 1

    while i <= nres:
        ii = i
        j = i + 2
        tag = 0
        s0 = s1 = sn = 0
        b0 = b1 = bn = 0

        while j <= nres and ii <= nres:
            if br[ii, j] == 0:
                if tag == 0:
                    j += 1
                    continue
                else:
                    # Process parallel ladder (tag == 1)
                    if tag == 1:
                        l = s0 + sn
                        m = s1 + sn

                        for k in range(s0, l):
                            tmp[k] = "E"
                        for k in range(s1, m):
                            tmp[k] = "E"

                        e0 = l - 1
                        e1 = m - 1
                        c0 = c1 = 0

                        # Extend ladder
                        while (c0 < 5 and c1 < 2) or (c0 < 2 and c1 < 5):
                            if l + c0 > nres or m + c1 > nres:
                                break
                            if br[l + c0, m + c1] == 0:
                                c1 += 1
                                if c0 < 2:
                                    if c1 == 5:
                                        c0 += 1
                                        c1 = 0
                                else:
                                    if c1 == 2:
                                        c0 += 1
                                        c1 = 0
                            else:
                                e0 = l + c0
                                e1 = m + c1
                                tmp[e0] = tmp[e1] = "E"
                                br[e0, e1] = br[e1, e0] = 0
                                l = e0 + 1
                                m = e1 + 1
                                c0 = c1 = 0

                        # Assign to br1 or br2
                        tag0 = 0
                        for k in range(s0, e0 + 1):
                            if br1[k] != " ":
                                tag0 = 1
                                break

                        if tag0 == 0:
                            for k in range(s0, e0 + 1):
                                if tmp[k] == "E":
                                    br1[k] = low[nlet]
                                    sh1[k] = "E"
                        else:
                            for k in range(s0, e0 + 1):
                                if tmp[k] == "E":
                                    br2[k] = low[nlet]
                                    sh2[k] = "E"

                        tag1 = 0
                        for k in range(s1, e1 + 1):
                            if br1[k] != " ":
                                tag1 = 1
                                break

                        if tag1 == 0:
                            for k in range(s1, e1 + 1):
                                if tmp[k] == "E":
                                    br1[k] = low[nlet]
                                    sh1[k] = "E"
                        else:
                            for k in range(s1, e1 + 1):
                                if tmp[k] == "E":
                                    br2[k] = low[nlet]
                                    sh2[k] = "E"

                        # Clear tmp
                        for k in range(nres + 1):
                            tmp[k] = " "

                        j = s1
                        if j <= nres:
                            ii = i

                        s1 = s0 = sn = 0
                        nlet += 1
                        if nlet > 26:
                            nlet = 1

                    # Process antiparallel ladder (tag == 2)
                    else:
                        l = b0 + bn
                        m = b1 - bn

                        for k in range(b0, l):
                            tmp[k] = "E"
                        for k in range(b1, m, -1):
                            tmp[k] = "E"

                        e0 = l - 1
                        e1 = m + 1
                        c0 = c1 = 0

                        # Extend ladder
                        while (c0 < 5 and c1 < 2) or (c0 < 2 and c1 < 5):
                            if br[l + c0, m - c1] == 0:
                                c1 += 1
                                if c0 < 2:
                                    if c1 == 5:
                                        c0 += 1
                                        c1 = 0
                                else:
                                    if c1 == 2:
                                        c0 += 1
                                        c1 = 0
                            else:
                                e0 = l + c0
                                e1 = m - c1
                                tmp[e0] = tmp[e1] = "E"
                                br[e0, e1] = br[e1, e0] = 0
                                l = e0 + 1
                                m = e1 - 1
                                c0 = c1 = 0

                        # Assign to br1 or br2 (uppercase for antiparallel)
                        tag0 = 0
                        for k in range(b0, e0 + 1):
                            if br1[k] != " ":
                                tag0 = 1
                                break

                        if tag0 == 0:
                            for k in range(b0, e0 + 1):
                                if tmp[k] == "E":
                                    br1[k] = low[nlet].upper()
                                    sh1[k] = "E"
                        else:
                            for k in range(b0, e0 + 1):
                                if tmp[k] == "E":
                                    br2[k] = low[nlet].upper()
                                    sh2[k] = "E"

                        tag1 = 0
                        for k in range(b1, e1 - 1, -1):
                            if br1[k] != " ":
                                tag1 = 1
                                break

                        if tag1 == 0:
                            for k in range(b1, e1 - 1, -1):
                                if tmp[k] == "E":
                                    br1[k] = low[nlet].upper()
                                    sh1[k] = "E"
                        else:
                            for k in range(b1, e1 - 1, -1):
                                if tmp[k] == "E":
                                    br2[k] = low[nlet].upper()
                                    sh2[k] = "E"

                        # Clear tmp
                        for k in range(nres + 1):
                            tmp[k] = " "

                        j = b1
                        if j <= nres:
                            ii = i

                        b1 = b0 = bn = 0
                        nlet += 1
                        if nlet > 26:
                            nlet = 1

                    tag = 0
            else:
                # Parallel bridge
                if br[ii, j] == 1:
                    if sn == 0:
                        s0 = ii
                        s1 = j
                        tag = 1
                    br[ii, j] = br[j, ii] = 0
                    sn += 1
                    j += 1
                    ii += 1
                # Antiparallel bridge
                else:
                    if bn == 0:
                        b0 = ii
                        b1 = j
                        tag = 2
                    br[ii, j] = br[j, ii] = 0
                    bn += 1
                    j -= 1
                    ii += 1

        i += 1

    return br1, br2, sh1, sh2


def set_sheet(
    nres: int,
    br1: List[str],
    br2: List[str],
    sh1: List[str],
    sh2: List[str],
    sy: List[str],
) -> Tuple[List[str], List[str]]:
    """
    Identify sheets from ladders and update summary.

    Based on SetSheet() in hb-calc.C

    Args:
        nres: Number of residues (1-indexed)
        br1, br2: Bridge arrays from set_ladder()
        sh1, sh2: Sheet arrays from set_ladder()
        sy: Summary array (will be updated with sheet assignments)

    Returns:
        Tuple of (sheet, summary) arrays
    """
    sh = [" "] * (nres + 1)
    bag = [" "] * 100

    low = [
        " ",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
    ]

    bagi = 1
    nlet = 1
    end = 0
    count = 0

    while end == 0:
        begin = 1
        tag = 1

        while tag == 1:
            coun = count

            for i in range(1, nres + 1):
                if sh[i] != " ":
                    continue

                if br1[i] == " " and br2[i] == " ":
                    continue

                # Process br1
                if br1[i] != " ":
                    if begin == 1:
                        sh[i] = low[nlet].upper()
                        count += 1
                        bag[1] = br1[i]
                        begin = 0
                    else:
                        tag1 = 0
                        for j in range(1, bagi + 1):
                            if bag[j] == br1[i]:
                                tag1 = 1
                                break

                        if tag1 == 1:
                            if sh[i] == " ":
                                sh[i] = low[nlet].upper()
                                count += 1
                        else:
                            if br2[i] != " ":
                                tag11 = 0
                                for j in range(1, bagi + 1):
                                    if bag[j] == br2[i]:
                                        tag11 = 1
                                        break

                                if tag11 == 1:
                                    bagi += 1
                                    bag[bagi] = br1[i]
                                    if sh[i] == " ":
                                        sh[i] = low[nlet].upper()
                                        count += 1

                # Process br2
                if br2[i] != " ":
                    if begin == 1:
                        sh[i] = low[nlet].upper()
                        count += 1
                        bag[1] = br2[i]
                        begin = 0
                    else:
                        tag2 = 0
                        for j in range(1, bagi + 1):
                            if bag[j] == br2[i]:
                                tag2 = 1
                                break

                        if tag2 == 1:
                            if sh[i] == " ":
                                sh[i] = low[nlet].upper()
                                count += 1
                        else:
                            if br1[i] != " ":
                                tag22 = 0
                                for j in range(1, bagi + 1):
                                    if bag[j] == br1[i]:
                                        tag22 = 1
                                        break

                                if tag22 == 1:
                                    bagi += 1
                                    bag[bagi] = br2[i]
                                    if sh[i] == " ":
                                        sh[i] = low[nlet].upper()
                                        count += 1

            if coun == count:
                tag = 0
                nlet += 1

            # Clear bag
            for j in range(1, bagi + 1):
                bag[j] = " "

            bagi = 1
            tag = 1
            end = 1

            # Check if more sheets to process
            for j in range(1, nres + 1):
                if sh[j] == " ":
                    if br1[j] != " " or br2[j] != " ":
                        end = 0
                        break

    # Update summary with sheet assignments
    j = 0
    for i in range(1, nres + 1):
        if sh[i] == " ":
            if sh1[i] == "E" or sh2[i] == "E":
                sy[i] = "E"
                j += 1
            else:
                if j == 1:
                    sy[i - 1] = "B"
                j = 0
        else:
            if sh[i - 1] != sh[i]:
                j = 0
            j += 1
            sy[i] = "E"

    if j == 1:
        sy[i - 1] = "B"

    return sh, sy


def hbond_key(nres: int, nhb: int, hbonds: List[Tuple[int, int]]) -> List[str]:
    """
    Create H-bond key array.

    Based on hbond_key() in pdb-pro-ss.C

    Args:
        nres: Number of residues (1-indexed)
        nhb: Number of H-bonds
        hbonds: List of (CO_index, NH_index) tuples

    Returns:
        Array of length nres+1 with H-bond keys:
        'O' = CO makes H-bond, 'H' = NH makes H-bond,
        'B' = both, '.' = none
    """
    ho = [" "] * (nres + 1)

    for co_idx, nh_idx in hbonds:
        co = co_idx
        nh = nh_idx

        if ho[co] == "H":
            ho[co] = "B"
        else:
            if ho[co] == " ":
                ho[co] = "O"

        if ho[nh] == "O":
            ho[nh] = "B"
        else:
            if ho[nh] == " ":
                ho[nh] = "H"

    for i in range(1, nres + 1):
        if ho[i] == " ":
            ho[i] = "."

    return ho


def summ_out(c: str) -> str:
    """
    Convert DSSP summary code to Ribbons code.

    Based on summ_out() in pdb-pro-ss.C
    """
    if c == " ":
        return "c"
    elif c == "E":
        return "S"
    elif c == "H":
        return "H"
    elif c == "G":
        return "3"
    elif c == "I":
        return "5"
    elif c == "T":
        return "T"
    elif c == "B":
        return "s"
    else:
        return "x"


def detect_secondary_structure(
    backbone_atoms: List[BackboneAtoms],
    sequence: List[str],
    residue_numbers: Optional[List[str]] = None,
) -> Dict:
    """
    Main function to detect secondary structure using Kabsch-Sander method.

    Args:
        backbone_atoms: List of BackboneAtoms (indexed 0..nres, with [0] dummy)
        sequence: List of amino acid one-letter codes (indexed 0..nres, with [0] dummy)
        residue_numbers: Optional list of residue number strings

    Returns:
        Dictionary with:
        - 'summary': List of summary SS codes (H, E, G, I, T, B, ' ')
        - 'ribbons_ss': List of Ribbons SS codes (H, S, 3, 5, T, s, c)
        - 'hbond_key': List of H-bond keys (O, H, B, .)
        - 'sheet': List of sheet assignments
        - 'nres': Number of residues
        - 'nhb': Number of H-bonds
    """
    nres = len(backbone_atoms) - 1  # Subtract dummy [0]

    # Calculate missing hydrogens
    nh_calc = calculate_hydrogen_positions(nres, backbone_atoms, sequence)

    # Calculate H-bonds
    max_hb = 2 * nres
    hbonds = calculate_hydrogen_bonds(nres, backbone_atoms, max_hb)
    nhb = len(hbonds)

    # Identify turns
    t3, t4, t5 = set_turn(nres, nhb, hbonds)

    # Identify helices
    sy = set_helix(nres, t3, t4, t5)

    # Identify bridges
    br = set_bridge(nres, nhb, hbonds)

    # Identify ladders
    br1, br2, sh1, sh2 = set_ladder(nres, br)

    # Identify sheets
    sh, sy = set_sheet(nres, br1, br2, sh1, sh2, sy)

    # Create H-bond key
    ho = hbond_key(nres, nhb, hbonds)

    # Convert to Ribbons codes
    ribbons_ss = [summ_out(sy[i]) for i in range(nres + 1)]

    return {
        "summary": sy,
        "ribbons_ss": ribbons_ss,
        "hbond_key": ho,
        "sheet": sh,
        "nres": nres,
        "nhb": nhb,
        "nh_calculated": nh_calc,
    }


def extract_backbone_atoms_from_pdb(
    pdb_data: Union[PandasPdb, pd.DataFrame], chain_id: Optional[str] = None
) -> Tuple[List[BackboneAtoms], List[str], List[str]]:
    """
    Extract backbone atoms from a PDB structure for secondary structure detection.

    Args:
        pdb_data: Either a PandasPdb object or a pandas DataFrame with ATOM records
        chain_id: Optional chain ID to filter by (if None, uses first chain or all)

    Returns:
        Tuple of (backbone_atoms, sequence, residue_numbers)
        - backbone_atoms: List of BackboneAtoms (indexed 0..nres, with [0] as dummy)
        - sequence: List of amino acid one-letter codes (indexed 0..nres, with [0] as dummy)
        - residue_numbers: List of residue number strings (indexed 0..nres, with [0] as dummy)
    """
    # Extract DataFrame
    if isinstance(pdb_data, PandasPdb):
        atom_df = pdb_data.df.get("ATOM")
        if atom_df is None:
            raise ValueError("No ATOM records found in PDB data")
    elif isinstance(pdb_data, pd.DataFrame):
        atom_df = pdb_data
    else:
        raise TypeError("pdb_data must be PandasPdb or pandas DataFrame")

    # Filter by chain if specified
    if chain_id is not None:
        atom_df = atom_df[atom_df["chain_id"] == chain_id].copy()

    if atom_df.empty:
        raise ValueError("No atoms found in PDB data")

    # Sort by chain, residue number, and atom name
    atom_df = atom_df.sort_values(["chain_id", "residue_number", "atom_name"]).copy()

    # Initialize with dummy [0] entry
    backbone_atoms = [BackboneAtoms()]  # Index 0 is dummy
    sequence = [" "]  # Index 0 is dummy
    residue_numbers = [""]  # Index 0 is dummy

    # Map of three-letter to one-letter amino acid codes
    aa_three_to_one = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLN": "Q",
        "GLU": "E",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
        "SEC": "U",
        "PYL": "O",
        "ASX": "B",
        "GLX": "Z",
        "XAA": "X",
        "UNK": "X",
    }

    current_residue = None
    current_atoms = BackboneAtoms()

    for _, row in atom_df.iterrows():
        res_num = str(row["residue_number"])
        res_name = row["residue_name"].strip().upper()
        atom_name = row["atom_name"].strip().upper()
        chain = row["chain_id"]

        # Create residue key
        residue_key = (chain, res_num)

        # Check if new residue
        if current_residue != residue_key:
            # Save previous residue if exists
            if current_residue is not None:
                backbone_atoms.append(current_atoms)
                # Get one-letter code
                one_letter = aa_three_to_one.get(res_name, "X")
                sequence.append(one_letter)
                residue_numbers.append(f"{chain}{res_num}")

            # Start new residue
            current_residue = residue_key
            current_atoms = BackboneAtoms()

        # Extract coordinates
        coords = np.array(
            [float(row["x_coord"]), float(row["y_coord"]), float(row["z_coord"])]
        )

        # Store atom coordinates
        if atom_name == "N":
            current_atoms.n = coords
        elif atom_name in ("H", "HN", "H1"):
            current_atoms.h = coords
        elif atom_name == "C":
            current_atoms.c = coords
        elif atom_name in ("O", "OT1"):
            current_atoms.o = coords
        elif atom_name == "CA":
            current_atoms.ca = coords

    # Don't forget the last residue
    if current_residue is not None:
        backbone_atoms.append(current_atoms)
        res_name = (
            atom_df[atom_df["residue_number"] == int(current_residue[1])]
            .iloc[0]["residue_name"]
            .strip()
            .upper()
        )
        one_letter = aa_three_to_one.get(res_name, "X")
        sequence.append(one_letter)
        residue_numbers.append(f"{current_residue[0]}{current_residue[1]}")

    return backbone_atoms, sequence, residue_numbers


def detect_secondary_structure_from_pdb(
    pdb_data: Union[PandasPdb, pd.DataFrame], chain_id: Optional[str] = None
) -> Dict:
    """
    Convenience function to detect secondary structure directly from PDB data.

    Args:
        pdb_data: Either a PandasPdb object or a pandas DataFrame with ATOM records
        chain_id: Optional chain ID to filter by

    Returns:
        Dictionary with secondary structure assignments (see detect_secondary_structure())
    """
    backbone_atoms, sequence, residue_numbers = extract_backbone_atoms_from_pdb(
        pdb_data, chain_id
    )

    return detect_secondary_structure(backbone_atoms, sequence, residue_numbers)

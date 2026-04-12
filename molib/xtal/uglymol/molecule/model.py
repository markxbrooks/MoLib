"""Model"""

import math

import numpy as np

from molib.core.constants import MoLibConstant
from molib.xtal.uglymol.cubicles import Cubicles
from molib.xtal.uglymol.molecule.atom import Atom
from molib.xtal.uglymol.unit_cell import UnitCell


class Model:
    """Model"""

    def __init__(self):
        self.atoms = []
        self.unit_cell = None
        self.has_hydrogens = False
        self.lower_bound = [0, 0, 0]
        self.upper_bound = [0, 0, 0]
        self.residue_map = None
        self.cubes = None

    def from_pdb(self, pdb_lines):
        """from_pdb"""
        chain_index = 0  # will be incremented for the first atom
        last_chain = None
        atom_i_seq = 0
        continuation = None
        for i, line in enumerate(pdb_lines):
            rec_type = line[:6].upper()
            if rec_type in ["ATOM  ", "HETATM"]:
                new_atom = Atom()
                new_atom.from_pdb_line(line)
                new_atom.i_seq = atom_i_seq
                atom_i_seq += 1
                if not self.has_hydrogens and new_atom.element == "H":
                    self.has_hydrogens = True
                if new_atom.chain != last_chain:
                    chain_index += 1
                new_atom.chain_index = chain_index
                last_chain = new_atom.chain
                self.atoms.append(new_atom)
            elif rec_type == "ANISOU":
                pass
            elif rec_type == "CRYST1":
                a = float(line[6:15])
                b = float(line[15:24])
                c = float(line[24:33])
                alpha = float(line[33:40])
                beta = float(line[40:47])
                gamma = float(line[47:54])
                self.unit_cell = UnitCell(a, b, c, alpha, beta, gamma)
            elif rec_type[:3] == "TER":
                last_chain = None
            elif rec_type == "ENDMDL":
                for j in range(i, len(pdb_lines)):
                    if pdb_lines[j][:6].upper() == "MODEL ":
                        continuation = pdb_lines[j:]
                        break
                break
        if len(self.atoms) == 0:
            raise ValueError("No atom records found.")
        self.calculate_bounds()
        self.calculate_connectivity()
        return continuation

    def calculate_bounds(self):
        """calculate_bounds"""
        self.lower_bound = [float("inf")] * 3
        self.upper_bound = [-float("inf")] * 3
        for atom in self.atoms:
            for j in range(3):
                v = atom.xyz[j]
                if v < self.lower_bound[j]:
                    self.lower_bound[j] = v
                if v > self.upper_bound[j]:
                    self.upper_bound[j] = v
        # with a margin
        for k in range(3):
            self.lower_bound[k] -= 0.001
            self.upper_bound[k] += 0.001

    def next_residue(self, atom, backward):
        """next_residue"""
        length = len(self.atoms)
        start = (atom.i_seq if atom else 0) + length  # +length to avoid idx<0 below
        for i in range(1 if atom else 0, length):
            idx = (start - i if backward else start + i) % length
            a = self.atoms[idx]
            if not a.is_main_conformer():
                continue
            if (a.name == MoLibConstant.PEPTIDE_CHAIN_ATOMNAME and a.element == "C") or a.name == "P":
                return a

    def extract_trace(self):
        """extract_trace"""
        segments = []
        current_segment = []
        last_atom = None
        for atom in self.atoms:
            if atom.alt_loc not in ["", "A"]:
                continue
            if (atom.name == MoLibConstant.PEPTIDE_CHAIN_ATOMNAME and atom.element == "C") or atom.name == "P":
                start_new = True
                if last_atom is not None and last_atom.chain_index == atom.chain_index:
                    dxyz2 = atom.distance_sq(last_atom)
                    if (atom.name == MoLibConstant.PEPTIDE_CHAIN_ATOMNAME and dxyz2 <= 5.5 * 5.5) or (
                        atom.name == "P" and dxyz2 < 7.5 * 7.5
                    ):
                        current_segment.append(atom)
                        start_new = False
                if start_new:
                    if len(current_segment) > 2:
                        segments.append(current_segment)
                    current_segment = [atom]
                last_atom = atom
        if len(current_segment) > 2:
            segments.append(current_segment)
        return segments

    def get_residues(self):
        """get_residues"""
        if self.residue_map is not None:
            return self.residue_map
        residues = {}
        for atom in self.atoms:
            resid = atom.resid()
            if resid not in residues:
                residues[resid] = [atom]
            else:
                residues[resid].append(atom)
        self.residue_map = residues
        return residues

    def calculate_tangent_vector(self, residue):
        """calculate_tangent_vector"""
        a1 = None
        a2 = None
        peptide = len(residue[0].resname) == 3
        name1 = "C" if peptide else "C2'"
        name2 = "O" if peptide else "O4'"
        for atom in residue:
            if not atom.is_main_conformer():
                continue
            if atom.name == name1:
                a1 = atom.xyz
            elif atom.name == name2:
                a2 = atom.xyz
        if a1 is None or a2 is None:
            return [0, 0, 1]  # arbitrary value
        d = [a1[0] - a2[0], a1[1] - a2[1], a1[2] - a2[2]]
        length = math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
        return [d[0] / length, d[1] / length, d[2] / length]

    def get_center(self):
        """get_center"""
        xsum = ysum = zsum = 0
        n_atoms = len(self.atoms)
        for atom in self.atoms:
            xyz = atom.xyz
            xsum += xyz[0]
            ysum += xyz[1]
            zsum += xyz[2]
        return [xsum / n_atoms, ysum / n_atoms, zsum / n_atoms]

    def calculate_connectivity(self):
        """Calculate bond connectivity based on spatial partitioning."""
        atoms = self.atoms
        cubes = Cubicles(atoms, 3.0, self.lower_bound, self.upper_bound)

        for i, box in enumerate(cubes.boxes):
            if not box:
                continue

            nearby_atoms = cubes.get_nearby_atoms(i)
            if not nearby_atoms:
                continue

            box_coords = np.array([atoms[a].xyz for a in box])  # (M, 3)
            nearby_coords = np.array([atoms[a].xyz for a in nearby_atoms])  # (N, 3)

            for j, atom_id in enumerate(box):
                diffs = nearby_coords - box_coords[j]
                d2 = np.sum(diffs**2, axis=1)

                for k, nearby_atom in enumerate(nearby_atoms):
                    if nearby_atom > atom_id and atoms[atom_id].is_bonded_to(
                        atoms[nearby_atom], d2[k]
                    ):
                        atoms[atom_id].bonds.add(nearby_atom)
                        atoms[nearby_atom].bonds.add(atom_id)

    def get_nearest_atom(self, x: float, y: float, z: float, atom_name: str):
        """Find nearest atom to (x, y, z), optionally filtered by atom name."""
        if self.cubes is None:
            raise ValueError("Missing Cubicles")

        box_id = self.cubes.find_box_id(x, y, z)
        indices = self.cubes.get_nearby_atoms(box_id)

        if not indices:
            return None

        # Collect candidate atoms
        candidates = [self.atoms[i] for i in indices]

        if atom_name is not None:
            candidates = [a for a in candidates if a.name == atom_name]
            if not candidates:
                return None

        coords = np.array([a.xyz for a in candidates])  # (N, 3)
        target = np.array([x, y, z])

        d2 = np.sum((coords - target) ** 2, axis=1)  # vectorized squared distance
        nearest_idx = np.argmin(d2)
        return candidates[nearest_idx]

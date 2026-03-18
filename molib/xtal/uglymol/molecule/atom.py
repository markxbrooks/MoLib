"""Atom class"""

from molib.xtal.uglymol.molecule.definition import NOT_LIGANDS


class Atom:
    """Atom"""

    def __init__(self):
        self.name = ""
        self.altloc = ""
        self.resname = ""
        self.chain = ""
        self.chain_index = -1
        self.seqid = ""
        self.xyz = [0, 0, 0]
        self.occ = 1.0
        self.b = 0
        self.element = ""
        self.i_seq = -1
        self.is_ligand = None
        self.bonds = []

    def from_pdb_line(self, pdb_line):
        """from_pdb_line"""
        if len(pdb_line) < 66:
            raise ValueError(f"ATOM or HETATM record is too short: {pdb_line}")
        rec_type = pdb_line[0:6]
        if rec_type not in ["HETATM", "ATOM  "]:
            raise ValueError(f"Wrong record type: {rec_type}")
        self.name = pdb_line[12:16].strip()
        self.altloc = pdb_line[16:17].strip()
        self.resname = pdb_line[17:20].strip()
        self.chain = pdb_line[20:22].strip()
        self.seqid = pdb_line[22:27].strip()
        x = float(pdb_line[30:38])
        y = float(pdb_line[38:46])
        z = float(pdb_line[46:54])
        self.xyz = [x, y, z]
        self.occ = float(pdb_line[54:60])
        self.b = float(pdb_line[60:66])
        if len(pdb_line) >= 78:
            self.element = pdb_line[76:78].strip().upper()
        # if len(pdb_line) >= 80:
        #     self.charge = pdb_line[78:80].strip()
        self.is_ligand = self.resname not in NOT_LIGANDS

    def distance_sq(self, other):
        """distance_sq"""
        dx = self.xyz[0] - other.xyz[0]
        dy = self.xyz[1] - other.xyz[1]
        dz = self.xyz[2] - other.xyz[2]
        return dx * dx + dy * dy + dz * dz

    def distance(self, other):
        """distance"""
        return self.distance_sq(other) ** 0.5

    def midpoint(self, other):
        """midpoint"""
        return [
            (self.xyz[0] + other.xyz[0]) / 2,
            (self.xyz[1] + other.xyz[1]) / 2,
            (self.xyz[2] + other.xyz[2]) / 2,
        ]

    def is_hydrogen(self):
        """is_hydrogen"""
        return self.element in ["H", "D"]

    def is_ion(self):
        """is_ion"""
        return self.element == self.resname

    def is_water(self):
        """is_water"""
        return self.resname == "HOH"

    def is_same_conformer(self, other):
        """is_same_conformer"""
        return self

    def is_main_conformer(self):
        """is_main_conformer"""
        return self.altloc in ["", "A"]

    def bond_radius(self):
        """bond_radius"""
        # rather crude
        if self.element == "H":
            return 1.3
        if self.element in ["S", "P"]:
            return 2.43
        return 1.99

    def is_bonded_to(self, other):
        """is_bonded_to"""
        max_dist = 2.2 * 2.2
        if not self.is_same_conformer(other):
            return False
        dxyz2 = self.distance_sq(other)
        if dxyz2 > max_dist:
            return False
        if self.element == "H" and other.element == "H":
            return False
        return dxyz2 <= self.bond_radius() * other.bond_radius()

    def resid(self):
        """resid"""
        return f"{self.seqid}/{self.chain}"

    def long_label(self):
        """long_label"""
        a = self  # eslint-disable-line @typescript-eslint/no-this-alias
        return (
            f"{a.name} /{a.seqid} {a.resname}/{a.chain} - occ: "
            f"{a.occ:.2f} bf: {a.b:.2f} ele: {a.element} pos: "
            f"({a.xyz[0]:.2f},{a.xyz[1]:.2f},{a.xyz[2]:.2f})"
        )

    def short_label(self):
        """short_label"""
        a = self  # eslint-disable-line @typescript-eslint/no-this-alias
        return f"{a.name} /{a.seqid} {a.resname}/{a.chain}"

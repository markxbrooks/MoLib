"""
MoleculeSecondaryStructure
"""


class MoleculeSecondaryStructure:
    def __init__(self):
        self.helix_segments = []
        self.helix_segment_count = 0
        self.coil_segments = []
        self.coil_segment_count = 0
        self.strand_segments = []
        self.strand_segment_count = 0

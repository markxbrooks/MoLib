"""entity lists"""

from molib.core.amino_acids import AMINO_ACIDS

NUCLEIC_ACIDS = [
    "DA",
    "DC",
    "DG",
    "DT",
    "A",
    "C",
    "G",
    "U",
    "rA",
    "rC",
    "rG",
    "rU",
    "Ar",
    "Cr",
    "Gr",
    "Ur",
]
NOT_LIGANDS = ["HOH"] + list(AMINO_ACIDS) + NUCLEIC_ACIDS
SPOT_SEL = ["all", "unindexed", "#1"]
SHOW_AXES = ["two", "three", "none"]
SPOT_SHAPES = ["wheel", "square"]

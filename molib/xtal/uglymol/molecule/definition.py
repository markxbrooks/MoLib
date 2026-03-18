"""entity lists"""

AMINO_ACIDS = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "MSE",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "UNK",
]
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
NOT_LIGANDS = ["HOH"] + AMINO_ACIDS + NUCLEIC_ACIDS
SPOT_SEL = ["all", "unindexed", "#1"]
SHOW_AXES = ["two", "three", "none"]
SPOT_SHAPES = ["wheel", "square"]

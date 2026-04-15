"""
MoLib Constants
"""
import numpy as np


class MoLibConstant:
    """MoLib Constant"""
    EPSILON_SMALL = 1e-8
    EPSILON = 1e-6
    PEPTIDE_CHAIN_ATOMNAME = "CA" # MoLibConstant.PEPTIDE_CHAIN_ATOMNAME
    PEPTIDE_DISTANCE = 4.2
    NUCLEOTIDE_CHAIN_ATOMNAME = "P"
    NUCLEOTIDE_DISTANCE = 10.0

    # Helix and strand constants
    HELIX_HERMITE_FACTOR = 4.7
    STRAND_HERMITE_FACTOR = 0.5
    HELIX_ALPHA = np.radians(32.0)
    HELIX_BETA = np.radians(-11.0)
    ANGLE_PI = np.pi

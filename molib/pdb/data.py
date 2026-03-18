#!/usr/bin/env python
"""
Ribbons Program Data Structures
==============================

This file contains all the data structures extracted from the Ribbons molecular
visualization program, including color schemes, secondary structure codes,
material properties, and other configuration data.

Extracted from:
- data/protein.color
- data/nucleic.color
- data/atom.color
- data/res.color
- data/ribbons.matter
- data/*.rcolor files
"""

# =============================================================================
# HYDROGEN BONDING PATTERNS
# =============================================================================

hb_colors = {
    "B": 7,  # β-sheet hydrogen bonds
    "H": 1,  # α-helix hydrogen bonds
    "O": 4,  # 3₁₀ helix hydrogen bonds
    "P": 7,  # π-helix hydrogen bonds
    "Q": 4,  # left-handed helix hydrogen bonds
    "x": 12,  # unclassified hydrogen bonds
}

# =============================================================================
# SECONDARY STRUCTURE CODES
# =============================================================================

# Main secondary structure assignments
ss_colors = {
    "H": 6,  # α-helix
    "3": 4,  # 3₁₀ helix
    "S": 2,  # β-sheet (parallel)
    "A": 2,  # β-sheet (antiparallel)
    "s": 8,  # small sheet
    "c": 8,  # coil
    "T": 8,  # turn
    "X": 9,  # unknown/undefined
}

# Secondary structure from hydrogen bonding
sshb_colors = {
    "H": 6,  # α-helix
    "3": 4,  # 3₁₀ helix
    "S": 2,  # β-sheet
    "A": 2,  # β-sheet (antiparallel)
    "s": 8,  # small sheet
    "c": 8,  # coil
    "T": 8,  # turn
}

# Secondary structure from phi/psi values
sspp_colors = {
    "H": 6,  # α-helix
    "3": 4,  # 3₁₀ helix
    "S": 2,  # β-sheet
    "A": 2,  # β-sheet (antiparallel)
    "s": 8,  # small sheet
    "c": 8,  # coil
    "T": 8,  # turn
}

# =============================================================================
# AMINO ACID COLOR SCHEMES
# =============================================================================

# Protein amino acid sequence colors
protein_seq_colors = {
    "A": 2,
    "C": 3,
    "D": 1,
    "E": 1,
    "F": 2,
    "G": 6,
    "H": 4,
    "I": 2,
    "K": 4,
    "L": 2,
    "M": 3,
    "N": 5,
    "P": 6,
    "Q": 5,
    "R": 4,
    "S": 8,
    "T": 8,
    "V": 2,
    "W": 2,
    "Y": 8,
}

# Nucleic acid sequence colors
nucleic_seq_colors = {
    "A": 3,  # Adenine
    "T": 4,  # Thymine
    "G": 8,  # Guanine
    "C": 6,  # Cytosine
    "U": 4,  # Uracil
}

# Residue type colors for proteins
protein_res_colors = {
    "A": 2,
    "C": 3,
    "D": 1,
    "E": 1,
    "F": 2,
    "G": 6,
    "H": 4,
    "I": 2,
    "K": 4,
    "L": 2,
    "M": 3,
    "N": 5,
    "P": 6,
    "Q": 5,
    "R": 4,
    "S": 8,
    "T": 8,
    "V": 2,
    "W": 2,
    "Y": 8,
}

# Residue type colors for nucleic acids
nucleic_res_colors = {
    "A": 3,  # Adenine
    "T": 4,  # Thymine
    "G": 8,  # Guanine
    "C": 6,  # Cytosine
    "U": 4,  # Uracil
}

# =============================================================================
# ATOM COLORS
# =============================================================================

# Atom colors (based on first letter of atom name)
atom_colors = {
    "C": 7,  # Carbon
    "N": 4,  # Nitrogen
    "O": 1,  # Oxygen
    "H": 6,  # Hydrogen
    "S": 3,  # Sulfur
    "P": 5,  # Phosphorus
}

# =============================================================================
# RAMACHANDRAN PLOT COLORS
# =============================================================================

# Ramachandran plot classifications (rama)
rama_colors = {
    "E": 2,  # Extended β-sheet
    "e": 16,  # Extended β-sheet (alternate)
    "R": 4,  # Right-handed α-helix
    "r": 6,  # Right-handed α-helix (alternate)
    "c": 6,  # Collagen
    "L": 14,  # Left-handed α-helix
    "l": 3,  # Left-handed α-helix (alternate)
    "G": 7,  # Glycine
    "g": 9,  # Glycine (alternate)
    "n": 9,  # Not assigned
    "T": 5,  # Turn
    "t": 12,  # Turn (alternate)
    "x": 8,  # Unusual
    "X": 1,  # Unusual (alternate)
    "?": 1,  # Unknown
}

# Ramachandran plot defaults (rama1)
rama1_colors = {
    "E": 2,  # Extended β-sheet
    "e": 16,  # Extended β-sheet (alternate)
    "R": 2,  # Right-handed α-helix
    "r": 16,  # Right-handed α-helix (alternate)
    "c": 16,  # Collagen
    "L": 2,  # Left-handed α-helix
    "l": 16,  # Left-handed α-helix (alternate)
    "G": 4,  # Glycine
    "g": 6,  # Glycine (alternate)
    "n": 16,  # Not assigned
    "T": 2,  # Turn
    "t": 16,  # Turn (alternate)
    "x": 8,  # Unusual
    "X": 1,  # Unusual (alternate)
    "?": 1,  # Unknown
}

# Ramachandran plot alternate (rama2)
rama2_colors = {
    "E": 2,  # Extended β-sheet
    "e": 16,  # Extended β-sheet (alternate)
    "R": 4,  # Right-handed α-helix
    "r": 6,  # Right-handed α-helix (alternate)
    "c": 6,  # Collagen
    "L": 14,  # Left-handed α-helix
    "l": 3,  # Left-handed α-helix (alternate)
    "G": 7,  # Glycine
    "g": 9,  # Glycine (alternate)
    "n": 9,  # Not assigned
    "T": 5,  # Turn
    "t": 12,  # Turn (alternate)
    "x": 8,  # Unusual
    "X": 1,  # Unusual (alternate)
    "?": 1,  # Unknown
}

# =============================================================================
# OMEGA ANGLE COLORS
# =============================================================================

omega_colors = {
    "T": 2,  # Trans
    "s": 3,  # Small deviation
    "u": 6,  # Unusual
    "C": 4,  # Cis
    "b": 10,  # Bad
    "d": 5,  # Disallowed
    "X": 1,  # Unknown
}

# =============================================================================
# CHI ANGLE COLORS
# =============================================================================

# Chi angle colors (chi)
chi_colors = {
    "A": 4,
    "B": 6,
    "C": 2,
    "D": 16,
    "E": 16,
    "F": 16,
    "G": 16,
    "H": 16,
    "I": 16,
    "a": 9,
    "b": 9,
    "c": 9,
    "d": 9,
    "e": 9,
    "f": 9,
    "g": 9,
    "h": 9,
    "i": 9,
    "N": 7,
    "X": 1,
}

# Chi angle defaults (chi1)
chi1_colors = {
    "A": 2,
    "B": 2,
    "C": 2,
    "D": 2,
    "E": 2,
    "F": 2,
    "G": 2,
    "H": 2,
    "I": 2,
    "a": 6,
    "b": 6,
    "c": 6,
    "d": 6,
    "e": 6,
    "f": 6,
    "g": 6,
    "h": 6,
    "i": 6,
    "N": 7,
    "X": 1,
}

# Chi angle alternate (chi2)
chi2_colors = {
    "A": 4,
    "B": 6,
    "C": 2,
    "D": 16,
    "E": 16,
    "F": 16,
    "G": 16,
    "H": 16,
    "I": 16,
    "a": 9,
    "b": 9,
    "c": 9,
    "d": 9,
    "e": 9,
    "f": 9,
    "g": 9,
    "h": 9,
    "i": 9,
    "N": 7,
    "X": 1,
}

# Chi angle modified (mchi)
mchi_colors = {
    "A": 4,
    "B": 6,
    "C": 2,
    "D": 16,
    "E": 16,
    "F": 16,
    "G": 16,
    "H": 16,
    "I": 16,
    "a": 9,
    "b": 9,
    "c": 9,
    "d": 9,
    "e": 9,
    "f": 9,
    "g": 9,
    "h": 9,
    "i": 9,
    "N": 7,
    "X": 1,
}

# =============================================================================
# B-FACTOR (TEMPERATURE FACTOR) COLORS
# =============================================================================

# B-factor colors for entire residue
bfactor_res_colors = {
    "A": 10,  # Very high B-factor
    "B": 6,  # High B-factor
    "C": 2,  # Medium B-factor
    "D": 3,  # Low-medium B-factor
    "E": 8,  # Low B-factor
    "F": 1,  # Very low B-factor
}

# B-factor colors for main-chain
bfactor_mc_colors = {
    "A": 10,  # Very high B-factor
    "B": 6,  # High B-factor
    "C": 2,  # Medium B-factor
    "D": 3,  # Low-medium B-factor
    "E": 8,  # Low B-factor
    "F": 1,  # Very low B-factor
}

# B-factor colors for side-chain
bfactor_sc_colors = {
    "A": 10,  # Very high B-factor
    "B": 6,  # High B-factor
    "C": 2,  # Medium B-factor
    "D": 3,  # Low-medium B-factor
    "E": 8,  # Low B-factor
    "F": 1,  # Very low B-factor
}

# =============================================================================
# INFORMATION CONTENT COLORS
# =============================================================================

info_colors = {
    "A": 10,  # High information content
    "B": 6,  # Medium-high information content
    "C": 16,  # Medium information content
    "7": 7,  # Low information content
}

# =============================================================================
# NUCLEIC ACID DOMAIN COLORS
# =============================================================================

nucleic_domain_colors = {
    "0": 5,  # Default
    "A": 16,  # Adenine domain
    "D": 14,  # Domain D
    "C": 15,  # Cytosine domain
    "V": 13,  # Domain V
    "T": 12,  # Thymine domain
    "1": 7,  # Domain 1
}

# =============================================================================
# RANGE-BASED COLOR SCHEMES
# =============================================================================

# B-factor range colors (7 ranges, 8 colors)
bfactor_ranges = {
    "ranges": [12.0, 18.0, 24.0, 30.0, 36.0, 42.0, 48.0],
    "colors": [10, 4, 6, 2, 3, 8, 1, 5],
}

# Curvature range colors (4 ranges, 5 colors)
curvature_ranges = {"ranges": [0.25, 0.40, 0.60, 0.75], "colors": [8, 3, 9, 6, 10]}

# Electrostatic potential range colors (4 ranges, 5 colors)
electro_ranges = {"ranges": [-15.0, -5.0, 5.0, 15.0], "colors": [4, 6, 7, 8, 1]}

# Occupancy range colors (4 ranges, 5 colors)
occupancy_ranges = {"ranges": [-0.5, -0.15, 0.15, 0.5], "colors": [4, 6, 9, 3, 1]}

# Omega angle range colors (5 ranges, 6 colors)
omega_ranges = {"ranges": [3.0, 3.5, 4.0, 4.5, 5.0], "colors": [1, 8, 3, 2, 6, 10]}

# =============================================================================
# MATERIAL PROPERTIES AND LIGHTING
# =============================================================================

# Import comprehensive material and lighting system
from .materials import (
    convert_to_hex_color,
    convert_to_rgb_255,
    get_alpha,
    get_ambient_color,
    get_basic_colors,
    get_color_palette,
    get_combined_hex_color,
    get_combined_rgb,
    get_diffuse_color,
    get_emissive_color,
    get_light_configuration,
    get_light_direction,
    get_light_intensity,
    get_material_properties,
    get_rainbow_colors,
    get_shininess,
    get_specular_color,
    is_light_on,
)
from .materials import lighting_settings as ambient_lighting
from .materials import lights, materials

# =============================================================================
# SECONDARY STRUCTURE FILE FORMAT
# =============================================================================

# .ss file column definitions
ss_file_columns = {
    "res#": "Residue number",
    "seq": "Amino acid sequence (single letter code)",
    "ss": "Main secondary structure assignment (H, S, T, c, etc.)",
    "sshb": "Secondary structure from hydrogen bonding",
    "hb": "Hydrogen bonding pattern (B, H, O, P, Q, x)",
    "sheet": "Beta sheet strand information (a, A, B, c)",
    "run": "Run number for secondary structure elements",
    "range": "Range information",
    "sspp": "Secondary structure from phi/psi values",
    "rama": "Ramachandran plot classification",
    "om": "Omega angle classification",
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_color_for_amino_acid(aa_code, scheme="protein_seq"):
    """Get material index for an amino acid code."""
    if scheme == "protein_seq":
        return protein_seq_colors.get(aa_code, 1)
    elif scheme == "protein_res":
        return protein_res_colors.get(aa_code, 1)
    elif scheme == "nucleic_seq":
        return nucleic_seq_colors.get(aa_code, 1)
    elif scheme == "nucleic_res":
        return nucleic_res_colors.get(aa_code, 1)
    else:
        return 1  # Default material


def get_color_for_secondary_structure(ss_code, scheme="ss"):
    """Get material index for a secondary structure code."""
    if scheme == "ss":
        return ss_colors.get(ss_code, 1)
    elif scheme == "sshb":
        return sshb_colors.get(ss_code, 1)
    elif scheme == "sspp":
        return sspp_colors.get(ss_code, 1)
    else:
        return 1  # Default material


def get_color_for_hydrogen_bond(hb_code):
    """Get material index for a hydrogen bonding pattern."""
    return hb_colors.get(hb_code, 1)


def get_color_for_atom(atom_name):
    """Get material index for an atom (based on first letter)."""
    first_letter = atom_name[0].upper() if atom_name else "X"
    return atom_colors.get(first_letter, 1)


def get_ramachandran_color(rama_code, scheme="rama"):
    """Get material index for a Ramachandran plot classification."""
    if scheme == "rama":
        return rama_colors.get(rama_code, 1)
    elif scheme == "rama1":
        return rama1_colors.get(rama_code, 1)
    elif scheme == "rama2":
        return rama2_colors.get(rama_code, 1)
    else:
        return 1  # Default material


# =============================================================================
# ENHANCED MATERIAL UTILITY FUNCTIONS
# =============================================================================


def get_material_rgb(material_index):
    """Get combined RGB color for a material index (backward compatibility)."""
    return get_combined_rgb(material_index)


def get_material_hex(material_index):
    """Get hex color string for a material index."""
    return get_combined_hex_color(material_index)


def get_material_name(material_index):
    """Get material name for a material index."""
    return get_material_properties(material_index)["name"]


def get_secondary_structure_material(ss_code):
    """Get material index for secondary structure with enhanced mapping."""
    ss_material_map = {
        "H": 1,  # Red for helices
        "G": 1,  # Red for 3-10 helices
        "I": 1,  # Red for pi helices
        "E": 2,  # Green for sheets
        "B": 2,  # Green for beta bridges
        "T": 3,  # Yellow for turns
        "S": 3,  # Yellow for bends
        "C": 7,  # Gray for coils
        " ": 7,  # Gray for other
    }
    return ss_material_map.get(ss_code, 1)


def get_element_material(element):
    """Get material index for chemical element with enhanced mapping."""
    element_material_map = {
        "C": 7,  # Gray for carbon
        "N": 4,  # Blue for nitrogen
        "O": 1,  # Red for oxygen
        "H": 0,  # White for hydrogen
        "S": 3,  # Yellow for sulfur
        "P": 8,  # Orange for phosphorus
        "F": 2,  # Green for fluorine
        "Cl": 2,  # Green for chlorine
        "Br": 2,  # Green for bromine
        "I": 2,  # Green for iodine
    }
    return element_material_map.get(element, 7)  # Default to gray


def get_chain_material(chain_id, chain_index=0):
    """Get material index for protein chain with cycling through basic colors."""
    basic_colors = [
        1,
        2,
        4,
        3,
        5,
        6,
        8,
        7,
    ]  # Red, Green, Blue, Yellow, Magenta, Cyan, Orange, Gray
    return basic_colors[chain_index % len(basic_colors)]


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example usage
    print("Enhanced Material System")
    print("========================")
    print()

    # Test amino acid materials
    print("Amino acid materials:")
    for aa in ["A", "G", "P", "W"]:
        material_idx = get_color_for_amino_acid(aa)
        material_name = get_material_name(material_idx)
        hex_color = get_material_hex(material_idx)
        print(f"  {aa}: {material_idx} ({material_name}) {hex_color}")

    print()

    # Test secondary structure materials
    print("Secondary structure materials:")
    for ss in ["H", "S", "T", "c"]:
        material_idx = get_secondary_structure_material(ss)
        material_name = get_material_name(material_idx)
        hex_color = get_material_hex(material_idx)
        print(f"  {ss}: {material_idx} ({material_name}) {hex_color}")

    print()

    # Test element materials
    print("Element materials:")
    for element in ["C", "N", "O", "H", "S", "P"]:
        material_idx = get_element_material(element)
        material_name = get_material_name(material_idx)
        hex_color = get_material_hex(material_idx)
        print(f"  {element}: {material_idx} ({material_name}) {hex_color}")

    print()

    # Test chain materials
    print("Chain materials:")
    for i, chain_id in enumerate(["A", "B", "C", "D"]):
        material_idx = get_chain_material(chain_id, i)
        material_name = get_material_name(material_idx)
        hex_color = get_material_hex(material_idx)
        print(f"  Chain {chain_id}: {material_idx} ({material_name}) {hex_color}")

    print()

    # Show material properties example
    print("Material properties example (Red material, index 1):")
    red_props = get_material_properties(1)
    print(f"  Name: {red_props['name']}")
    print(f"  Ambient: {red_props['ambient']}")
    print(f"  Diffuse: {red_props['diffuse']}")
    print(f"  Specular: {red_props['specular']}")
    print(f"  Shininess: {red_props['shininess']}")
    print(f"  Combined RGB: {get_combined_rgb(1)}")
    print(f"  Hex Color: {get_combined_hex_color(1)}")

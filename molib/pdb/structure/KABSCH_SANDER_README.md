# Kabsch & Sander Secondary Structure Detection

This module provides a Python port of Ribbons' Kabsch & Sander (1983) secondary structure detection algorithm for use in ElMo.

## Overview

The Kabsch & Sander method detects secondary structure by analyzing hydrogen bonding patterns in protein structures. It identifies:
- **Helices**: α-helix (H), 3₁₀-helix (G/3), π-helix (I/5)
- **Sheets**: β-sheets (E/S), isolated β-bridges (B/s)
- **Turns**: H-bonded turns (T)
- **Coils**: Unstructured regions (c)

## Usage

### Basic Usage

```python
from biopandas.pdb import PandasPdb
from elmo.pdb.structure.kabsch_sander import detect_secondary_structure_from_pdb

# Load PDB file
ppdb = PandasPdb().read_pdb('protein.pdb')

# Detect secondary structure
result = detect_secondary_structure_from_pdb(ppdb)

# Access results
print(f"Number of residues: {result['nres']}")
print(f"Number of H-bonds: {result['nhb']}")
print(f"Secondary structure codes: {result['ribbons_ss'][1:]}")  # Skip dummy [0]
```

### Advanced Usage

```python
from elmo.pdb.structure.kabsch_sander import (
    extract_backbone_atoms_from_pdb,
    detect_secondary_structure
)

# Extract backbone atoms manually
backbone_atoms, sequence, residue_numbers = extract_backbone_atoms_from_pdb(ppdb, chain_id='A')

# Run detection
result = detect_secondary_structure(backbone_atoms, sequence, residue_numbers)

# Access detailed results
summary = result['summary']  # DSSP-style codes (H, E, G, I, T, B, ' ')
ribbons_ss = result['ribbons_ss']  # Ribbons codes (H, S, 3, 5, T, s, c)
hbond_key = result['hbond_key']  # H-bond patterns (O, H, B, .)
sheet = result['sheet']  # Sheet assignments
```

## Output Format

The function returns a dictionary with:

- **`summary`**: List of DSSP-style secondary structure codes
  - `'H'` = α-helix
  - `'E'` = extended strand (β-sheet)
  - `'G'` = 3₁₀-helix
  - `'I'` = π-helix
  - `'T'` = turn
  - `'B'` = isolated β-bridge
  - `' '` = coil

- **`ribbons_ss`**: List of Ribbons-style secondary structure codes
  - `'H'` = α-helix
  - `'S'` = β-sheet
  - `'3'` = 3₁₀-helix
  - `'5'` = π-helix
  - `'T'` = turn
  - `'s'` = small sheet/isolated bridge
  - `'c'` = coil

- **`hbond_key`**: List of H-bond participation codes
  - `'O'` = CO makes H-bond
  - `'H'` = NH makes H-bond
  - `'B'` = both CO and NH make H-bonds
  - `'.'` = no H-bonds

- **`sheet`**: List of sheet strand assignments (letters A-Z)

- **`nres`**: Number of residues

- **`nhb`**: Number of hydrogen bonds detected

- **`nh_calculated`**: Number of hydrogen positions calculated

## Algorithm Details

The algorithm follows these steps:

1. **Extract backbone atoms**: N, H, C, O, CA from PDB structure
2. **Calculate missing hydrogens**: If H atoms are missing, calculate positions using geometry
3. **Calculate H-bonds**: Use Kabsch-Sander energy formula:
   ```
   E = q1*q2*(1/r(ON) + 1/r(CH) - 1/r(OH) - 1/r(CN)) * 332
   ```
   Where q1=0.42, q2=0.20, cutoff = -0.5 kcal/mol
4. **Identify turns**: Find 3-turn, 4-turn, and 5-turn patterns
5. **Identify helices**: Extend turns into helices (α, 3₁₀, π)
6. **Identify bridges**: Find β-bridges from H-bond patterns
7. **Identify ladders**: Group bridges into extended patterns
8. **Identify sheets**: Extend ladders into β-sheets
9. **Create summary**: Combine all assignments

## Reference

Kabsch, W. & Sander, C. (1983) Dictionary of protein secondary structure:
pattern recognition of hydrogen-bonded and geometrical features.
Biopolymers 22:2577-2637.

## Implementation Notes

- Arrays are 1-indexed (with dummy [0] entry) to match the original C++ code
- Missing atoms are marked with `MISSING = -10000001.0`
- The algorithm handles missing hydrogen atoms by calculating positions
- Proline residues are skipped for hydrogen calculation (no amide H)

## Integration with ElMo

This module can be used as an alternative to BioPython's DSSP for secondary structure detection in ElMo. It provides the same Kabsch-Sander algorithm used by Ribbons, ensuring consistency between the two programs.


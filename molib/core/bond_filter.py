# bond_filter.py

from collections import defaultdict
from math import isfinite

# Example data (replace with your real tables)
# bond_lengths[(elem_i, elem_j)] = (average_bond_length, tolerance)
bond_lengths = {
    ('C', 'H'): (1.09, 0.15),
    ('H', 'C'): (1.09, 0.15),
    ('C', 'C'): (1.54, 0.20),
    ('C', 'O'): (1.43, 0.15),
    ('O', 'C'): (1.43, 0.15),
    # ... add more pairs
}

# Ensure symmetric lookups
def nominal_length(i, j):
    key = (i, j)
    if key in bond_lengths:
        return bond_lengths[key]
    key = (j, i)
    if key in bond_lengths:
        return bond_lengths[key]
    return None  # unknown

# Valence table (maximum number of bonds for a simple count)
valence = {
    'H': 1,
    'C': 4,
    'N': 3,
    'O': 2,
    'S': 2,  # simplistic; adapt as needed
    # ...
}

# Example atoms: list of dicts with id and element (and optionally coords)
atoms = [
    {'id': 0, 'element': 'C'},
    {'id': 1, 'element': 'H'},
    {'id': 2, 'element': 'H'},
    {'id': 3, 'element': 'H'},
    {'id': 4, 'element': 'H'},
]

# Example candidate bonds: (i, j, distance)
candidates = [
    (0, 1, 1.08),
    (0, 2, 1.10),
    (0, 3, 1.50),  # slightly long for C-H, should be filtered
    (0, 4, 1.09),
    (1, 2, 1.50),   # H-H (not realistic for this molecule)
]

# Helper: convert indices to element symbols
def elem(idx):
    return atoms[idx]['element']

# Step 1 & 2: length filter
filtered = []
for (i, j, dist) in candidates:
    ei, ej = elem(i), elem(j)
    L = nominal_length(ei, ej)
    if L is None:
        continue  # unknown pair; skip or apply broad rule
    avg, tol = L
    if not (isfinite(dist) and isfinite(avg) and isfinite(tol)):
        continue
    if abs(dist - avg) <= tol:
        filtered.append({'i': i, 'j': j, 'dist': dist, 'elem_i': ei, 'elem_j': ej})

# Step 3: valency check
# Build a running count of bonds per atom
bond_count = defaultdict(int)
for b in filtered:
    bond_count[b['i']] += 1
    bond_count[b['j']] += 1

# Enforce valence: drop bonds that would exceed valence
def within_valence(i, j):
    ei, ej = elem(i), elem(j)
    max_i = valence.get(ei, 0)
    max_j = valence.get(ej, 0)
    cur_i = bond_count[i]
    cur_j = bond_count[j]
    # If either atom already at max, this bond would exceed
    if cur_i >= max_i or cur_j >= max_j:
        return False
    return True

# Apply again to prune oversubscribed bonds
final_bonds = []
for b in filtered:
    if within_valence(b['i'], b['j']):
        final_bonds.append(b)
        # update counts
        bond_count[b['i']] += 1
        bond_count[b['j']] += 1

# Output
for b in final_bonds:
    print(f"{b['elem_i']}-{b['elem_j']} bond: atoms {b['i']}-{b['j']}, distance {b['dist']:.2f} Å")


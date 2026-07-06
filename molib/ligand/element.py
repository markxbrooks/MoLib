"""Chemical element properties used for distance-based bond detection."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Element:
    """Chemical element with covalent radius."""

    symbol: str
    covalent_radius: float

    @property
    def is_hydrogen(self) -> bool:
        return self.symbol == "H"


ELEMENTS: dict[str, Element] = {
    "H": Element("H", 0.31),
    "C": Element("C", 0.76),
    "N": Element("N", 0.71),
    "O": Element("O", 0.66),
    "F": Element("F", 0.57),
    "P": Element("P", 1.07),
    "S": Element("S", 1.05),
    "Cl": Element("Cl", 0.99),
    "Br": Element("Br", 1.20),
    "I": Element("I", 1.39),
}


def get_covalent_radii(covalent_radii, element_symbols, i, j):
    """Get covalent radii for a pair of atoms."""
    elem1 = element_symbols[i]
    elem2 = element_symbols[j]
    radius1 = covalent_radii.get(elem1, 1.0)
    radius2 = covalent_radii.get(elem2, 1.0)
    return elem1, elem2, radius1, radius2


def calculate_distance(coordinates, i, j):
    """Calculate distance between two atoms in 3D coordinates."""
    coord1 = coordinates[i]
    coord2 = coordinates[j]
    return (
        (coord1[0] - coord2[0]) ** 2
        + (coord1[1] - coord2[1]) ** 2
        + (coord1[2] - coord2[2]) ** 2
    ) ** 0.5

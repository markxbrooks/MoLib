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


def normalize_element_symbol(raw: str) -> str:
    """Normalize a PDB element column to an RDKit-style symbol.

    PDB stores magnesium as ``MG``; RDKit expects ``Mg``.

    Parameters
    ----------
    raw
        Element field from a PDB ATOM/HETATM record.

    Returns
    -------
    str
        Title-cased element symbol, or an empty string if ``raw`` is blank.
    """
    symbol = (raw or "").strip()
    if not symbol:
        return ""
    if len(symbol) == 1:
        return symbol.upper()
    return symbol[0].upper() + symbol[1:].lower()


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


# Bondi van der Waals radii (Å). Keys are uppercase element symbols.
DEFAULT_VDW_RADIUS = 1.50
VDW_RADII: dict[str, float] = {
    "H": 1.20,
    "D": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
    "NA": 2.27,
    "MG": 1.73,
    "K": 2.75,
    "CA": 2.31,
    "MN": 1.61,
    "FE": 2.00,
    "CU": 1.40,
    "ZN": 1.39,
    "SE": 1.90,
    "NI": 1.63,
}


def infer_element_symbol(element: str | None, atom_name: str = "") -> str:
    """Return a normalized element symbol from *element* or *atom_name*.

    :param element: PDB/mmCIF element column (may be blank)
    :param atom_name: Atom name used when *element* is missing (first letter)
    :return: Title-cased symbol, or ``""``
    """
    symbol = normalize_element_symbol(element or "")
    if symbol:
        return symbol
    for char in atom_name or "":
        if char.isalpha():
            return char.upper()
    return ""


def coerce_positive_radius(value: object) -> float | None:
    """Return *value* as a positive finite radius, otherwise ``None``.

    :param value: Explicit radius (PQR column, constructor argument, …)
    :return: Radius in Å, or ``None`` if unusable
    """
    try:
        radius = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if radius != radius or radius <= 0.0:
        return None
    return radius


def vdw_radius_for_element(element: str | None, atom_name: str = "") -> float:
    """Return the Bondi van der Waals radius for *element* (Å).

    :param element: Element symbol; inferred from *atom_name* when blank
    :param atom_name: Fallback atom name (e.g. ``CA`` → carbon)
    :return: Radius in Å; :data:`DEFAULT_VDW_RADIUS` for unknown elements
    """
    symbol = infer_element_symbol(element, atom_name).upper()
    return float(VDW_RADII.get(symbol, DEFAULT_VDW_RADIUS))


def resolve_atomic_radius(
    radius: object | None,
    element: str | None = None,
    atom_name: str = "",
) -> float:
    """Return an explicit radius when valid, otherwise the element VDW radius.

    :param radius: PQR / caller-supplied radius in Å
    :param element: Element symbol for the VDW fallback
    :param atom_name: Atom name used to infer element when it is missing
    :return: Positive radius in Å
    """
    explicit = coerce_positive_radius(radius)
    if explicit is not None:
        return explicit
    return vdw_radius_for_element(element, atom_name)

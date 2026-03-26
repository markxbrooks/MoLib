"""
Atom class
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
from decologr import Decologr as log
from molib.core.color.helpers import get_color_scheme
from molib.core.color.map import ColorMap
from molib.core.color.strategy import ColorScheme
from molib.entities.secondary_structure_type import SecondaryStructureType
from molib.entities.structure import Structure3D
from molib.parser.pdb import PDBLayout
from molib.xtal.uglymol.molecule.definition import NOT_LIGANDS

# Performance optimization: Cache color scheme mappings
_COLOR_SCHEME_CACHE = {}


class Atom3D(Structure3D):
    """3D Atom class representing a PDB ATOM/HETATM record - Pure Python class for performance."""

    def __init__(
        self,
        # PDB-style identifiers
        name: str = "",
        serial: Optional[int] = None,
        res_name: Optional[str] = None,
        res_seq: Optional[int] = None,
        alt_loc: Optional[str] = None,
        chain_id: Optional[str] = None,
        segment_id: Optional[str] = None,
        atom_validated: Optional[bool] = None,
        atom_validation_error: Optional[str] = None,
        atom_contact_id: Optional[int] = None,
        atom_contact_distance: Optional[float] = None,
        # Chemical info
        element: Optional[str] = None,
        pqr_charge: Optional[float] = None,
        radius: Optional[float] = None,
        # Crystallographic/occupancy info
        b_factor: float = 0.0,
        occupancy: float = 1.0,
        # Coordinates
        coords: Optional["np.ndarray"] = None, # np.asarray(coords) if coords is not None else np.zeros(3)
        # Hierarchy
        parent: Optional["Res3D"] = None,
        # Visualization
        chain_color: Optional["np.ndarray"] = None,
        _cached_colors: Optional[dict] = None,
        # Structure3D parameters
        type: str = "",
        selected: bool = False,
        visible: bool = True,
        secstruc: Optional[SecondaryStructureType] = None,
        next: Optional["Structure3D"] = None,
        **kwargs,  # For any additional parameters
    ):
        # Initialize Structure3D first
        super().__init__(
            name=name,
            type=type,
            selected=selected,
            visible=visible,
            secstruc=secstruc or SecondaryStructureType.COIL,
            chain_id=chain_id or "A",
            coords=coords if coords is not None else (0.0, 0.0, 0.0),
            next=next,
            **kwargs,
        )

        # Atom-specific attributes
        self.alt_loc = alt_loc
        self.segment_id = segment_id
        self.atom_validated = atom_validated
        self.atom_validation_error = atom_validation_error
        self.atom_contact_id = atom_contact_id
        self.atom_contact_distance = atom_contact_distance
        self.element = element
        self.pqr_charge = pqr_charge
        self.radius = radius
        self.b_factor = b_factor
        self.occupancy = occupancy
        self.parent = parent
        self.chain_color = chain_color
        self._cached_colors = _cached_colors

        # Initialize element color
        self.set_element_color()
        if self.selected:
            self.color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            
        # Further initializations
        self.segment_id = segment_id
        self.atom_validated = atom_validated
        self.atom_validation_error = atom_validation_error
        self.atom_contact_id = atom_contact_id
        self.atom_contact_distance = atom_contact_distance
        self.element = element
        self.pqr_charge = pqr_charge
        self.radius = radius
        self.b_factor = b_factor
        self.occupancy = occupancy
        self.coords = coords
        
        # For comparison with uglymol
        # self.name = ""
        # self.altloc = ""
        self.res_name = res_name
        self.res_seq = res_seq
        # self.chain = ""
        # self.chain_index = -1
        # self.seqid = ""
        # self.xyz = [0, 0, 0]
        # self.occ = 1.0
        # self.b = 0
        # self.element = ""
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
        
        x = PDBLayout.x.parse(line)
        y = PDBLayout.y.parse(line)
        z = PDBLayout.z.parse(line)
            
        serial=PDBLayout.atom_serial.parse(line)
        name=PDBLayout.atom_name.parse(line)
        alt_loc=PDBLayout.alt_loc.parse(line)
        chain_id=PDBLayout.chain_id.parse(line)
        element=PDBLayout.element.parse(line)
        res_name=PDBLayout.res_name.parse(line)
        res_seq=PDBLayout.res_seq.parse(line)
        coords=np.array([x, y, z], dtype=np.float32)
        occupancy=PDBLayout.occupancy.parse(line) or 1.0
        b_factor=PDBLayout.temp_factor.parse(line) or 0.0
        
        self.name = name
        self.alt_loc = alt_loc
        self.res_seq = res_seq
        self.chain = chain_id
        self.coords = (x, y, z)
        self.is_ligand = self.resname not in NOT_LIGANDS

    # ------------------------------------------------------------------
    # PyMOL-like convenience accessors
    # ------------------------------------------------------------------

    @property
    def resn(self) -> Optional[str]:
        """Residue name (PyMOL-style), derived from parent residue if available."""
        parent = self.parent
        return getattr(parent, "name", None) if parent is not None else None

    @property
    def resi(self) -> Optional[int]:
        """Residue number (PyMOL-style), derived from parent residue if available."""
        parent = self.parent
        return getattr(parent, "residue_number", None) if parent is not None else None

    @property
    def chain(self) -> str:
        """Chain identifier (PyMOL-style)."""
        return self.chain_id

    @property
    def segi(self) -> Optional[str]:
        """Segment identifier (PyMOL-style), mapped from segment_id."""
        return self.segment_id

    @property
    def alt(self) -> Optional[str]:
        """Alternate location identifier (PyMOL-style), mapped from alt_loc."""
        return self.alt_loc

    # ------------------------------------------------------------------
    # Geometry, predicates, labels (shared ergonomics with xtal UglyMol Atom).
    # Optional distance-based bond guesses live in ``atom_bond_heuristics`` (not ElMo GL bonds).
    # ------------------------------------------------------------------

    def _coords_float3(self) -> Tuple[float, float, float]:
        """Return this atom's position as three Python floats."""
        p = self.pos
        return float(p[0]), float(p[1]), float(p[2])

    def distance_sq(self, other: "Atom3D") -> float:
        """Squared Euclidean distance to ``other`` in angstroms."""
        ax, ay, az = self._coords_float3()
        bx, by, bz = other._coords_float3()
        dx, dy, dz = ax - bx, ay - by, az - bz
        return dx * dx + dy * dy + dz * dz

    def distance(self, other: "Atom3D") -> float:
        """Euclidean distance to ``other`` in angstroms."""
        return self.distance_sq(other) ** 0.5

    def midpoint(self, other: "Atom3D") -> Tuple[float, float, float]:
        """Midpoint between this atom and ``other``."""
        ax, ay, az = self._coords_float3()
        bx, by, bz = other._coords_float3()
        return (ax + bx) / 2, (ay + by) / 2, (az + bz) / 2

    def is_hydrogen(self) -> bool:
        """True for hydrogen isotopes used in PDB/mmCIF (H and D)."""
        el = (self.element or "").strip().upper()
        return el in ("H", "D")

    def is_water(self) -> bool:
        """True if the parent residue (or ``res_name``) is standard water HO4/HOH naming."""
        name = self.resn or self.res_name
        if name is None:
            return False
        return str(name).strip().upper() == "HOH"

    def _normalized_alt_loc(self) -> str:
        return (self.alt_loc or "").strip()

    def is_main_conformer(self) -> bool:
        """True for blank or 'A' alternate-location ID (common PDB main conformer)."""
        a = self._normalized_alt_loc()
        return a in ("", "A")

    def is_same_conformer(self, other: "Atom3D") -> bool:
        """
        True if alternate-location IDs are compatible for the same physical model.

        Blank altloc is treated as compatible with any ID (standard PDB convention).
        """
        a = self._normalized_alt_loc()
        b = other._normalized_alt_loc()
        if not a or not b:
            return True
        return a == b

    def _label_seq_str(self) -> str:
        """Residue sequence key for labels (parent number, else atom's res_seq)."""
        if self.resi is not None:
            return str(self.resi)
        if self.res_seq is not None:
            return str(self.res_seq)
        return "?"

    def _label_resname_str(self) -> str:
        rn = self.resn or self.res_name
        return (rn or "?").strip()

    def resid(self) -> str:
        """Compact ``{residue_number}/{chain_id}`` style key."""
        return f"{self._label_seq_str()}/{self.chain_id}"

    def short_label(self) -> str:
        """One-line atom identity: name, seq, residue name, chain."""
        return (
            f"{self.name} /{self._label_seq_str()} "
            f"{self._label_resname_str()}/{self.chain_id}"
        )

    def long_label(self) -> str:
        """Verbose string incl. occupancy, B-factor, element, and coordinates."""
        x, y, z = self._coords_float3()
        el = (self.element or "").strip()
        return (
            f"{self.name} /{self._label_seq_str()} "
            f"{self._label_resname_str()}/{self.chain_id} - occ: "
            f"{self.occupancy:.2f} bf: {self.b_factor:.2f} ele: {el} pos: "
            f"({x:.2f},{y:.2f},{z:.2f})"
        )

    def get_chain_color(self) -> "np.ndarray":
        """
        Retrieve this atom's chain colour from the top-level Molecule3D.
        Falls back gracefully if not found.
        """
        from molib.entities.molecule import Molecule3D

        try:
            if self.chain_color is not None:
                return self.chain_color

            node = self.parent
            while node is not None and not isinstance(node, Molecule3D):
                node = getattr(node, "parent", None)

            if isinstance(node, Molecule3D):
                color = node.chain_colors.get(self.chain_id, (1.0, 0.0, 0.0))
            else:
                color = (0.5, 0.5, 0.5)  # fallback

            self.chain_color = np.array(color, dtype=np.float32)
            return self.chain_color

        except Exception as ex:
            log.exception(f"Failed to get chain colour {ex}")
            return np.array((0.5, 0.5, 0.5), dtype=np.float32)

    def set_element_color(self) -> None:
        """Assign colour from element type, fallback to gray."""
        self.color = np.array(
            ColorMap.atom_colors.get(self.element, (0.9, 0.9, 0.9)), dtype=np.float32
        )

    def set_color(self, r: float, g: float, b: float) -> None:
        """Manually override atom colour (ignores element)."""
        self.color = np.array([r, g, b], dtype=np.float32)

    def apply_color_scheme(
        self, color_scheme: Union[str, ColorScheme] = ColorScheme.ELEMENT
    ) -> None:
        """
        Apply a coloring color_scheme to this atom.

        :param color_scheme: The coloring color_scheme to use, either as a ColorScheme enum or a string.
        """
        # Convert string to Enum if necessary
        color_scheme = get_color_scheme(color_scheme)

        # Delegate to ColorMap
        ColorMap.apply_strategy(self, color_scheme)

    @classmethod
    def _get_color_scheme_map(cls, scheme: ColorScheme):
        """Get cached color scheme mapping to avoid recreating dictionaries."""
        if scheme not in _COLOR_SCHEME_CACHE:
            _COLOR_SCHEME_CACHE[scheme] = {
                ColorScheme.ELEMENT: cls._color_by_element,
                ColorScheme.CHAIN: cls._color_by_chain,
                ColorScheme.SECONDARY_STRUCTURE: cls._color_by_secondary,
                ColorScheme.B_FACTOR: cls._color_by_b_factor,
                ColorScheme.VALIDATION: cls._color_by_validation,
                ColorScheme.CONTACT_DISTANCE: cls._color_by_contact_distance,
            }
        return _COLOR_SCHEME_CACHE[scheme]

    def apply_atom_color_scheme(
        self, color_scheme: Union[str, ColorScheme] = ColorScheme.ELEMENT
    ) -> None:
        """Apply a colour scheme to this atom - PERFORMANCE OPTIMIZED."""
        scheme = get_color_scheme(color_scheme)

        # Use cached color scheme mapping
        color_scheme_map = self._get_color_scheme_map(scheme)

        # Get the function or fallback to gray
        color_func = color_scheme_map.get(
            scheme, lambda: np.array((0.8, 0.8, 0.8), dtype=np.float32)
        )

        # Call the function and assign directly to avoid extra array creation
        self.color = color_func(self)

    @classmethod
    def apply_color_scheme_batch(
        cls, atoms: list["Atom3D"], color_scheme: ColorScheme
    ) -> None:
        """Apply color scheme to a batch of atoms - PERFORMANCE OPTIMIZED."""
        if not atoms:
            return

        # Get cached color scheme mapping once
        color_scheme_map = cls._get_color_scheme_map(color_scheme)
        color_func = color_scheme_map.get(
            color_scheme, lambda atom: np.array((0.8, 0.8, 0.8), dtype=np.float32)
        )

        # Apply to all atoms in batch
        for atom in atoms:
            atom.color = color_func(atom)

    def _color_by_element(self) -> "np.ndarray":
        """Approx PyMOL default element colors."""
        return np.array(
            ColorMap.ELEMENT_COLORS.get(self.element, (0.5, 0.5, 0.5)), dtype=np.float32
        )

    def _color_by_chain(self) -> "np.ndarray":
        """Per-chain color: same palette/order as ``generate_chain_colors`` / ribbon chain_colors."""
        from molib.pdb.color import rgb_for_chain_id_among

        res = self.parent
        chain = getattr(res, "parent", None) if res is not None else None
        model = getattr(chain, "parent", None) if chain is not None else None
        if model is not None and getattr(model, "chains", None):
            chain_ids = list(model.chains.keys())
            rgb = rgb_for_chain_id_among(self.chain_id, chain_ids)
        else:
            rgb = rgb_for_chain_id_among(self.chain_id, [self.chain_id])
        return np.array(rgb, dtype=np.float32)

    def _color_by_secondary(self) -> "np.ndarray":
        """
        Color atom based on secondary structure.
        Uses the secstruc attribute from the parent residue.
        """
        from molib.core.color.map import ColorMap

        # Get secondary structure from parent residue
        if self.parent and hasattr(self.parent, "secstruc"):
            secstruc = self.parent.secstruc
        else:
            # Fallback to atom's own secstruc if available
            secstruc = getattr(self, "secstruc", " ")

        # Handle both enum and string types
        if hasattr(secstruc, "to_string"):
            # It's a SecondaryStructureType enum
            secstruc_str = secstruc.to_string()
        else:
            # It's already a string
            secstruc_str = str(secstruc)

        # Get colour from secondary structure colour map
        color = ColorMap.secondary_structure_color_map.get(
            secstruc_str, ColorMap.secondary_structure_color_map[" "]
        )

        log.debug(
            f"Atom {self.name} secstruc {secstruc_str} -> colour {color}", silent=True, scope=self.__class__.__name__
        )
        return np.array(color, dtype=np.float32)

    def _color_by_b_factor(self) -> "np.ndarray":
        """Color atom based on B-factor using white-to-red scale."""
        from molib.core.color.map import ColorMap

        if self.b_factor is not None:
            # Convert B-factor to colour (white=0, red=200)
            color = ColorMap.b_factor_to_color(self.b_factor)
            log.debug(
                f"Atom {self.name} B-factor {self.b_factor} -> colour {color}",
                silent=True, scope=self.__class__.__name__
            )
            return np.array(color, dtype=np.float32)
        else:
            # Fallback to element colour if no B-factor
            log.debug(
                f"Atom {self.name} using element colour (B-factor: {self.b_factor})",
                silent=True, scope=self.__class__.__name__,
            )
            return self._color_by_element()

    def _color_by_validation(self) -> "np.ndarray":
        """
        Color atom based on validation status.
        Uses the atom_validated attribute to determine colour.
        """
        # Check if atom has validation status
        if hasattr(self, "atom_validated") and self.atom_validated is not None:
            if self.atom_validated:
                # Valid atom - green
                color = (0.0, 1.0, 0.0)  # Green
                log.debug(f"Atom {self.name} validated -> green", silent=True, scope=self.__class__.__name__ )
            else:
                # Invalid atom - red
                color = ColorMap.INVALID  # (1.0, 0.0, 0.0)  # Red
                log.debug(f"Atom {self.name} not validated -> red", silent=True, scope=self.__class__.__name__ )
        else:
            # No validation data - gray
            color = (0.5, 0.5, 0.5)  # Gray
            log.debug(f"Atom {self.name} no validation data -> gray", scope=self.__class__.__name__, silent=True)

        return np.array(color, dtype=np.float32)

    def _color_by_contact_distance(self) -> "np.ndarray":
        """Color atom based on contact distance using blue-to-red scale."""
        from molib.core.color.map import ColorMap

        if self.atom_contact_distance is not None:
            # Convert contact distance to colour (blue=close, red=far)
            color = ColorMap.contact_distance_to_color(self.atom_contact_distance)
            log.debug(
                f"Atom {self.name} contact distance {self.atom_contact_distance} -> colour {color}",
                silent=True, scope=self.__class__.__name__
            )
            return np.array(color, dtype=np.float32)
        else:
            # Fallback to element colour if no contact distance
            log.debug(
                f"Atom {self.name} using element colour (contact distance: {self.atom_contact_distance})",
                silent=True, scope=self.__class__.__name__
            )
            return self._color_by_element()

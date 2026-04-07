"""
Molecule 3D class
"""

from typing import Generator, Iterator, Optional

import numpy as np
from decologr import Decologr as log
from molib.core.color.manager import ChainColorManager
from molib.core.color.strategy import ColorScheme
from molib.entities.atom import Atom3D
from molib.entities.model import Model3D
from molib.entities.residue import Res3D
from molib.entities.selection import CoordinateSelection
from molib.pdb.coordinate.data import CoordinateData

# Standard polypeptide residue names (PDB 3-letter). Used to exclude ligands/HETATM
# from ribbon backbone so ribbons follow only the protein chain.
STANDARD_POLYPEPTIDE_RESIDUES = frozenset(
    {
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
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
        "SEC",
        "PYL",
        "MSE",  # Selenocysteine, Pyrrolysine, Selenomethionine
    }
)


class Molecule3D:
    """
    Mol3D class - Pure Python class for meshdata and colour

    Top-level Molecule
    """

    def __init__(
        self,
        name: str = "",
        models: Optional[list[Model3D]] = None,
        colour_mode: bool = False,
        _color_bonds: bool = True,
        chain_colors: Optional[dict] = None,
        coordinate_data: Optional[CoordinateData] = None,
        _color_scheme: ColorScheme = ColorScheme.ELEMENT,
        **kwargs,
    ):
        self.init = None
        self.name = name
        self.models = models or []
        self.colour_mode = colour_mode
        self._color_bonds = _color_bonds
        self.chain_colors = chain_colors or {}
        self.coordinate_data = coordinate_data
        self._color_scheme = _color_scheme

        # Post initialization
        self._post_init()

    def _post_init(self):
        """Post initialization"""
        # Copy chain colors from coordinate_data if available
        if self.coordinate_data and self.coordinate_data.chain_colors:
            self.chain_colors = dict(self.coordinate_data.chain_colors)
            log.info(
                f"🔄 Initialized chain colors: {list(self.chain_colors.keys())}",
                scope=self.__class__.__name__,
            )

        ChainColorManager().set_color_map(self.chain_colors)

    def add_model(self, model: "Model3D"):
        """add model"""
        model.parent = self
        self.models.append(model)

    def get_chain_coord_data(self, chain_id: str):
        """Get coordinates and colors for atoms in a chain."""
        for model in self.models:
            if chain_id in model.chains:  # safer check
                data = [
                    (atom.coords, atom.color) for res in self.residues for atom in res
                ]
                coords, colors = zip(*data)
                return np.array(coords, dtype=np.float32), np.array(
                    colors, dtype=np.float32
                )
        return None

    def set_chain_color(self, chain_id: str, r: float, g: float, b: float) -> None:
        """
        Set the colour of a single chain across all models (or just default model).
        """
        model = self.get_default_model()
        if chain_id in model.chains:
            model.set_chain_color(chain_id, r, g, b)

    def set_chain_coloring(self) -> None:
        """
        Apply stored chain colors across all models.
        """
        for model in self.models:
            model.apply_chain_coloring(self.coordinate_data.chain_colors)

    def set_atom_coloring_by_element(self) -> None:
        """
        Apply element-based coloring to all atoms in all models.
        """
        try:
            for model in self.models:
                model.apply_atom_coloring_by_element()
        except Exception as ex:
            log.error(
                f"Error {ex} occurred applying atom coloring",
                scope=self.__class__.__name__,
            )

    def set_atom_coloring_by_scheme(self, color_scheme: ColorScheme) -> None:
        """
        set_atom_coloring_by_scheme
        Example Usage:
        ==============
        >>># User picks coloring mode
        ...mol3d.set_atom_coloring_by_element()         # default
        ...mol3d.set_atom_coloring_by_scheme(ColorScheme.CHAIN) # chain-based rainbow
        ...mol3d.set_atom_coloring_by_scheme(ColorScheme.SECONDARY_STRUCTURE)
        """
        self.color_scheme = color_scheme
        try:
            model = self.get_default_model()
            if model:
                log.info(
                    f"🔄 Applying colour scheme {color_scheme} to model",
                    silent=True,
                    scope=self.__class__.__name__,
                )
                model.apply_atom_color_scheme(color_scheme)
            else:
                log.warning(
                    "No default model found for colour scheme application",
                    scope=self.__class__.__name__,
                )
        except Exception as ex:
            log.error(
                f"Error {ex} occurred applying atom coloring",
                scope=self.__class__.__name__,
            )

    def set_bond_coloring_state(self, color_bonds: bool) -> None:
        """
        Set whether bonds should be colored the same as atoms or a uniform colour.
        """
        self.color_bonds = color_bonds
        # log.info(f"Set bond coloring state to {color_bonds}", scope=self.__class__.__name__)

    @property
    def color_scheme(self) -> ColorScheme:
        """Current colour color_scheme."""
        return self._color_scheme

    @color_scheme.setter
    def color_scheme(self, new_scheme: ColorScheme) -> None:
        """Set a new colour color_scheme and reinitialize buffers if needed."""
        if not isinstance(new_scheme, ColorScheme):
            raise ValueError(f"Invalid ColorScheme: {new_scheme}")
        if self._color_scheme == new_scheme:
            return
        self._color_scheme = new_scheme
        self.set_atom_coloring_by_scheme(new_scheme)

    @property
    def color_bonds(self) -> bool:
        """Current colour color_scheme."""
        return self._color_bonds

    @color_bonds.setter
    def color_bonds(self, new_bonds_state: bool) -> None:
        """Set a new colour color_scheme and reinitialize buffers if needed."""
        if not isinstance(new_bonds_state, bool):
            raise ValueError(f"Invalid state: {new_bonds_state}")
        if self._color_bonds == new_bonds_state:
            return
        self._color_bonds = new_bonds_state

    def set_all_chains_color(self, r: float, g: float, b: float) -> None:
        """
        Set all chains in all models to the given colour.
        """
        for model in self.models:
            model.set_all_chains_color(r, g, b)

    def set_selection_color(
        self, selection: CoordinateSelection, r: float, g: float, b: float
    ) -> None:
        """
        Set the colour of all residues in the given selection.

        :param selection: CoordinateSelection
        :param r: float, red component (0–1)
        :param g: float, green component (0–1)
        :param b: float, blue component (0–1)
        """
        if selection.chain_id is None:
            raise ValueError("Selection must have a chain_id.")
        if selection.start_residue is None or selection.end_residue is None:
            raise ValueError("Selection must have start_residue and end_residue.")

        for model in self.models:
            chain = model.chains.get(selection.chain_id)
            if chain is None:
                continue  # Selection's chain not present in this model

            for res_num, residue in chain.residues.items():
                if selection.start_residue <= res_num <= selection.end_residue:
                    residue.set_color(r, g, b)

    def get_atom_count(self) -> int:
        """
        get_atom_count

        :return: int
        Returns count of atoms
        """
        return sum(len(res.atoms) for res in self.get_all_residues())

    def get_default_model(self) -> Model3D:
        """
        get_default_model

        :return: Model3D
        """
        if not self.models:
            self.models.append(Model3D(model_id=0))
        return self.models[0]

    def append_residue(self, res: Res3D):
        """
        append_residue

        :param res: Res3D
        :return:
        """
        model = self.get_default_model()
        chain = model.get_or_create_chain(res.chain_id)
        chain.append_residue(res)

    def get_all_residues(self) -> Iterator[Res3D]:
        """
        get_all_residues

        :return: Iterator[Res3D]
        All residues
        """
        for model in self.models:
            for chain in model.chains.values():
                yield from chain.residues

    def get_all_atoms(self):
        """
        get_all_atoms

        :return: Iterator[Atom3D]
        Yield all Atom objects from all models, chains, and residues.
        """
        for model in self.models:
            for chain in model.chains.values():
                for residue in chain.residues:
                    atoms = residue.atoms
                    if isinstance(atoms, dict):
                        yield from atoms.values()
                    else:
                        yield from atoms

    @property
    def residues(self) -> Iterator[Res3D]:
        """
        residues

        :return: Iterator[Res3D]
        """
        return self.get_all_residues()

    def get_backbone_trace(self) -> list[np.ndarray]:
        """List of CA coordinates."""
        residues = self.get_ca_residues()
        return [res.ca for res in residues]

    def get_backbone_trace_colors(self) -> list[np.ndarray]:
        """List of colors for each CA."""
        residues = self.get_ca_residues()
        return [res.color for res in residues]

    def get_ca_coords(self) -> np.ndarray:
        """Numpy array of all CA coordinates."""
        residues = self.get_ca_residues()
        return np.array([res.ca for res in residues], dtype=np.float32)

    def residues_with_ca_protein_only(self):
        """Yield residues that have CA and are standard polypeptide (excludes ligands)."""
        for res in self.get_all_residues():
            if not res.has_ca():
                continue
            res_name = (
                (getattr(res, "name", "") or getattr(res, "type", "") or "")
                .strip()
                .upper()
            )
            if res_name in STANDARD_POLYPEPTIDE_RESIDUES:
                yield res

    def get_ca_residues(self) -> list:
        """Return a list of residues that have CA atoms (backbone)."""
        if not hasattr(self, "_cached_ca_residues"):
            self._cached_ca_residues = [res for res in self.get_all_residues() if res.has_ca()]
        return self._cached_ca_residues

    def get_ca_residues_protein_only(self) -> list:
        """Single authoritative ordered CA residue list."""
        return list(self.residues_with_ca_protein_only())

    def get_ca_coords_protein_only(self) -> np.ndarray:
        residues = self.get_ca_residues_protein_only()
        return np.array([res.ca for res in residues], dtype=np.float32)

    def get_ca_colors_protein_only(self) -> np.ndarray:
        residues = self.get_ca_residues_protein_only()
        return np.array([res.color for res in residues], dtype=np.float32)

    def get_chain_ids_for_ca_protein_only(self) -> list[str]:
        residues = self.get_ca_residues_protein_only()
        return [res.chain_id for res in residues]

    def get_ca_colors(self) -> np.ndarray:
        """
        get_ca_colors

        :return: np.ndarray
        Get all CA colors
        """
        return np.array(self.get_backbone_trace_colors(), dtype=np.float32)

    def get_chain_ids(self, repeat_per_atom: bool = False) -> list[str]:
        """
        get_chain_ids

        :param repeat_per_atom: bool
        :return: list[str]
        Returns chain IDs for each residue (or for each atom if repeat_per_atom=True).
        """
        ids = []
        for res in self.residues:
            if repeat_per_atom:
                atom_count = len(res.atoms)
                ids.extend([res.chain_id] * (atom_count if atom_count else 1))
            else:
                ids.append(res.chain_id)
        return ids

    def get_chain_ids_for_ca(self) -> list[str]:
        """
        get_chain_ids_for_ca

        :return: list[str]
        Returns chain IDs for each residue (or for each atom if repeat_per_atom=True).
        """

        ids = []
        for res in self.residues:
            ids.append(res.chain_id)
        return ids

    def get_nucleotide_chains(self, atom_name: str = "O1A") -> list[str] | None:
        """
        get_nucleotide_chains

        :param atom_name: str
            Atom color_scheme to check (e.g. "O1A" for nucleotides)
        :return: list[st]
            List of chain IDs that contain the atom

        Identify chains that likely contain nucleotides by checking for a specific atom.
        """
        nucleotide_chains = []
        for model in self.models:
            for chain in model.chains.values():
                has_nucleotide_atoms = chain.get_atoms(atom_name)
                if has_nucleotide_atoms:
                    nucleotide_chains.append(chain.chain_id)
        return nucleotide_chains if nucleotide_chains else None

    def get_peptide_chains(self, atom_name: str = "CA") -> list[str] | None:
        """
        Identify chains that likely contain polypeptides by checking for a specific atom.

        :param atom_name: str
            Atom color_scheme to check (e.g. "CA" for peptides)
        :return: list[str]
            List of chain IDs that contain the atom
        """
        peptide_chains = []
        for model in self.models:
            for chain in model.chains.values():
                has_peptide_atoms = chain.get_atoms(atom_name)
                if has_peptide_atoms:
                    peptide_chains.append(chain.chain_id)
        return peptide_chains if peptide_chains else None

    def apply_colors(self):
        """Update colors for all atoms based on chain colors, selection, or colour scheme."""
        for model in self.models:
            for chain in model.chains.values():
                for residue in chain.residues:
                    for atom in residue.atoms.values():
                        # Example: use chain colour if defined, otherwise default
                        atom.color = (
                            atom.chain_color
                            if atom.chain_color is not None
                            else atom.color
                        )

    # ------------------------------------------------------------------
    # Segment- and alt-loc-level helpers
    # ------------------------------------------------------------------

    def get_segments(self) -> list[str]:
        """
        Return a sorted list of distinct segment IDs (segment_id) present in this molecule.
        """
        segments = set()
        for atom in self.get_all_atoms():
            if isinstance(atom, Atom3D) and atom.segment_id:
                segments.add(atom.segment_id)
        return sorted(segments)

    def set_segment_color(self, segment_id: str, r: float, g: float, b: float) -> None:
        """
        Set the colour for all atoms in the given segment across all models.
        """
        for atom in self.get_all_atoms():
            if isinstance(atom, Atom3D) and atom.segment_id == segment_id:
                atom.color[:] = (r, g, b)

    def hide_segment(self, segment_id: str) -> None:
        """
        Mark all atoms in the given segment as not visible.
        """
        for atom in self.get_all_atoms():
            if isinstance(atom, Atom3D) and atom.segment_id == segment_id:
                atom.visible = False

    def show_segment(self, segment_id: str) -> None:
        """
        Mark all atoms in the given segment as visible.
        """
        for atom in self.get_all_atoms():
            if isinstance(atom, Atom3D) and atom.segment_id == segment_id:
                atom.visible = True

    def set_altloc_visibility(
        self,
        alt_loc: str,
        visible: bool = True,
        chain_id: str | None = None,
    ) -> None:
        """
        Toggle visibility for all atoms with a given alt_loc identifier.

        Optionally restrict to a specific chain.
        """
        for atom in self.get_all_atoms():
            if not isinstance(atom, Atom3D):
                continue
            if atom.alt_loc != alt_loc:
                continue
            if chain_id is not None and atom.chain_id != chain_id:
                continue
            atom.visible = visible

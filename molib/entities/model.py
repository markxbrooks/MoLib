"""
Model3D class

Represents a single Model (there may be many) in a PDB file
"""

from typing import Dict, List, Optional, Union

from decologr import Decologr as log

from molib.entities.bond import Bond3D
from molib.entities.chain import Chain3D
from molib.core.color.strategy import ColorScheme


class Model3D:
    """
    Model3D class - Pure Python class for performance

    Optional, supports NMR ensembles
    """

    def __init__(
        self,
        name: str = "",
        model_id: int = 0,
        chains: Optional[Dict[str, Chain3D]] = None,
        **kwargs,
    ):
        self.name = name
        self.model_id = model_id
        self.chains = chains or {}
        # Optional connectivity layer: list of Bond3D objects belonging to this model.
        self.bonds: List[Bond3D] = []

    def add_chain(self, chain_id: str, chain: "Chain3D") -> None:
        chain.parent = self
        self.chains[chain_id] = chain

    def get_or_create_chain(self, chain_id: str) -> Chain3D:
        """
        get_or_create_chain

        :param chain_id: Chain ID
        :return: Chain3D
        """
        if chain_id not in self.chains:
            self.chains[chain_id] = Chain3D(chain_id=chain_id)
        return self.chains[chain_id]

    def set_chain_color(self, chain_id: str, r: float, g: float, b: float) -> None:
        """
        set_chain_color

        :param chain_id: The id of the chain
        :param r: The red colour
        :param g: The green colour
        :param b: The blue colour
        Set the colour for all residues and atoms in a specific chain.
        """
        chain = self.chains.get(chain_id)
        if chain is None:
            raise ValueError(f"Chain '{chain_id}' not found in model {self.model_id}.")

        for res in chain.residues:
            res.set_color(r, g, b)

    def set_all_chains_color(self, r: float, g: float, b: float) -> None:
        """
        set_all_chains_color

        :params: r, g, b RBG values
        Set the colour for all residues and atoms in all chains in this model.
        """
        for chain in self.chains:
            for res in chain.residues:
                res.set_color(r, g, b)

    def apply_chain_coloring(self, chain_colors: dict) -> None:
        """
        apply_chain_coloring

        :param chain_colors: dict
        """
        for chain_id, chain in self.chains.items():
            r, g, b = chain_colors.get(chain_id, (1.0, 1.0, 1.0))
            chain.set_color(r, g, b)

    def apply_atom_coloring_by_element(self) -> None:
        """
        apply_atom_coloring_by_element
        """
        for chain in self.chains.values():
            for residue in chain.residues:
                for atom in residue.atoms:
                    atom.set_atom_coloring()

    def set_atom_color_scheme(
        self, color_scheme: Union[str, ColorScheme] = ColorScheme.ELEMENT
    ) -> None:
        """
        set_atom_color_scheme

        :param color_scheme: str
        :return: None
        """
        self.apply_atom_color_scheme(color_scheme)

    def apply_atom_color_scheme(
        self, color_scheme: Union[str, ColorScheme] = ColorScheme.ELEMENT
    ) -> None:
        """
        apply_atom_color_scheme - PERFORMANCE OPTIMIZED

        :param color_scheme: ColorScheme to apply to the model
        :return: None
        """
        try:
            # Collect all atoms for batch processing
            all_atoms = []
            for chain in self.chains.values():
                for residue in chain.residues:
                    all_atoms.extend(residue.atoms.values())

            # Apply color scheme in batch
            from molib.entities.atom import Atom3D

            Atom3D.apply_color_scheme_batch(all_atoms, color_scheme)
        except Exception as ex:
            log.error(f"Error {ex} occurred applying atom coloring")

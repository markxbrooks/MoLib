"""
Module for accessing colors from color_array maps
"""

from decologr import Decologr as log
from molib.core.color import ColorMap


def get_atom_color(name: str) -> tuple[float, float, float]:
    """
    get_atom_color

    :param name: str
    :return: tuple[float, float, float] color_array
    """
    color = ColorMap.atom_colors.get(name, (0.5, 0.5, 0.5))
    return color


def get_ss_color(ss: str) -> tuple[float, float, float]:
    """
    get_ss_color

    :param ss: str
    :return: tuple[float, float, float] color_array
    """
    color = ColorMap.secondary_structure_color_map.get(
        ss, ColorMap.secondary_structure_color_map[" "]
    )
    return color


def generate_chain_colors(chain_ids: list) -> dict:
    """
    generate_chain_colors

    :param chain_ids: list
    :return: dict
    """
    unique_chains = set(chain_ids)
    rgb_colors = ColorMap.colors
    chain_colors = {
        chain: (
            rgb_colors[chain_number][0],
            rgb_colors[chain_number][1],
            rgb_colors[chain_number][2],
        )
        for chain_number, chain in enumerate(unique_chains)
    }
    log.message(f"self.chain_colors: {chain_colors}", silent=True)
    return chain_colors

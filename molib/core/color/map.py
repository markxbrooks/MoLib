"""
Color Palettes with strategies for coloring
"""

from typing import Callable, Union

import numpy as np

from molib.core.color.color import Color
from molib.core.color.strategy import ColorScheme


class ColorMap:
    CHAIN_COLORS = [
        (0.9, 0.1, 0.1),  # red
        (0.1, 0.9, 0.1),  # green
        (0.1, 0.1, 0.9),  # blue
        (0.9, 0.9, 0.1),  # yellow
        (0.9, 0.1, 0.9),  # magenta
        (0.1, 0.9, 0.9),  # cyan
    ]

    # Secondary structure color map registry
    _ss_color_maps: dict[str, dict] = {}
    _current_ss_map: str = "default"

    # Predefined secondary structure color maps
    @classmethod
    def _initialize_ss_maps(cls):
        """Initialize predefined secondary structure color maps."""
        if cls._ss_color_maps:
            return

        # Default map (current implementation)
        cls._ss_color_maps["default"] = {
            "H": Color(spec=0, x=1.0, y=0.0, z=0.0),  # Red for helix
            "E": Color(spec=0, x=1.0, y=1.0, z=0.0),  # Yellow for strand
            "T": Color(spec=0, x=1.0, y=1.0, z=0.0),  # Yellow for turn
            " ": Color(spec=0, x=0.7, y=0.7, z=0.7),  # Light gray for coil
            "-": Color(spec=0, x=0.7, y=0.7, z=0.7),  # Light gray for coil
        }

        # Rainbow map - distinct colors for each structure type
        cls._ss_color_maps["rainbow"] = {
            "H": Color(spec=0, x=1.0, y=0.0, z=0.0),  # Red for helix
            "G": Color(spec=0, x=1.0, y=0.5, z=0.0),  # Orange for 3-10 helix
            "I": Color(spec=0, x=1.0, y=1.0, z=0.0),  # Yellow for pi helix
            "E": Color(spec=0, x=0.0, y=1.0, z=0.0),  # Green for strand
            "B": Color(spec=0, x=0.0, y=0.8, z=0.0),  # Dark green for beta bridge
            "T": Color(spec=0, x=0.0, y=1.0, z=1.0),  # Cyan for turn
            "S": Color(spec=0, x=0.0, y=0.5, z=1.0),  # Light blue for bend
            "C": Color(spec=0, x=0.5, y=0.0, z=1.0),  # Purple for coil
            " ": Color(spec=0, x=0.6, y=0.6, z=0.6),  # Gray for other
            "-": Color(spec=0, x=0.6, y=0.6, z=0.6),  # Gray for other
        }

        # Monochrome map - grayscale based on structure type
        cls._ss_color_maps["monochrome"] = {
            "H": Color(spec=0, x=0.9, y=0.9, z=0.9),  # Light gray for helix
            "G": Color(
                spec=0, x=0.8, y=0.8, z=0.8
            ),  # Medium light gray for 3-10 helix
            "I": Color(spec=0, x=0.7, y=0.7, z=0.7),  # Medium gray for pi helix
            "E": Color(spec=0, x=0.6, y=0.6, z=0.6),  # Dark gray for strand
            "B": Color(spec=0, x=0.5, y=0.5, z=0.5),  # Darker gray for beta bridge
            "T": Color(spec=0, x=0.4, y=0.4, z=0.4),  # Dark gray for turn
            "S": Color(spec=0, x=0.3, y=0.3, z=0.3),  # Darker gray for bend
            "C": Color(spec=0, x=0.2, y=0.2, z=0.2),  # Very dark gray for coil
            " ": Color(spec=0, x=0.1, y=0.1, z=0.1),  # Black for other
            "-": Color(spec=0, x=0.1, y=0.1, z=0.1),  # Black for other
        }

        # Pastel map - soft, muted colors
        cls._ss_color_maps["pastel"] = {
            "H": Color(spec=0, x=1.0, y=0.7, z=0.7),  # Light red for helix
            "G": Color(spec=0, x=1.0, y=0.8, z=0.6),  # Light orange for 3-10 helix
            "I": Color(spec=0, x=1.0, y=1.0, z=0.7),  # Light yellow for pi helix
            "E": Color(spec=0, x=0.7, y=1.0, z=0.7),  # Light green for strand
            "B": Color(spec=0, x=0.6, y=0.9, z=0.6),  # Light green for beta bridge
            "T": Color(spec=0, x=0.7, y=1.0, z=1.0),  # Light cyan for turn
            "S": Color(spec=0, x=0.6, y=0.8, z=1.0),  # Light blue for bend
            "C": Color(spec=0, x=0.8, y=0.7, z=1.0),  # Light purple for coil
            " ": Color(spec=0, x=0.8, y=0.8, z=0.8),  # Light gray for other
            "-": Color(spec=0, x=0.8, y=0.8, z=0.8),  # Light gray for other
        }

        # High contrast map - maximum visual distinction
        cls._ss_color_maps["high_contrast"] = {
            "H": Color(spec=0, x=1.0, y=0.0, z=0.0),  # Bright red for helix
            "G": Color(spec=0, x=1.0, y=0.5, z=0.0),  # Bright orange for 3-10 helix
            "I": Color(spec=0, x=1.0, y=1.0, z=0.0),  # Bright yellow for pi helix
            "E": Color(spec=0, x=0.0, y=1.0, z=0.0),  # Bright green for strand
            "B": Color(spec=0, x=0.0, y=0.8, z=0.0),  # Dark green for beta bridge
            "T": Color(spec=0, x=0.0, y=1.0, z=1.0),  # Bright cyan for turn
            "S": Color(spec=0, x=0.0, y=0.0, z=1.0),  # Bright blue for bend
            "C": Color(spec=0, x=0.5, y=0.0, z=1.0),  # Bright purple for coil
            " ": Color(spec=0, x=1.0, y=1.0, z=1.0),  # White for other
            "-": Color(spec=0, x=1.0, y=1.0, z=1.0),  # White for other
        }

        # Nature-inspired map - colors found in nature
        cls._ss_color_maps["nature"] = {
            "H": Color(spec=0, x=0.8, y=0.2, z=0.2),  # Deep red for helix
            "G": Color(spec=0, x=0.9, y=0.5, z=0.1),  # Orange for 3-10 helix
            "I": Color(spec=0, x=0.9, y=0.8, z=0.1),  # Gold for pi helix
            "E": Color(spec=0, x=0.2, y=0.6, z=0.2),  # Forest green for strand
            "B": Color(spec=0, x=0.1, y=0.5, z=0.1),  # Dark green for beta bridge
            "T": Color(spec=0, x=0.1, y=0.7, z=0.8),  # Ocean blue for turn
            "S": Color(spec=0, x=0.2, y=0.4, z=0.8),  # Sky blue for bend
            "C": Color(spec=0, x=0.4, y=0.2, z=0.6),  # Purple for coil
            " ": Color(spec=0, x=0.5, y=0.5, z=0.5),  # Stone gray for other
            "-": Color(spec=0, x=0.5, y=0.5, z=0.5),  # Stone gray for other
        }

        # Sekulski map - element-based colors
        cls._ss_color_maps["sekulski"] = {
            "H": Color(spec=0, x=1.0, y=0.63, z=0.0),  # Selenium orange for helix
            "G": Color(spec=0, x=1.0, y=0.63, z=0.0),  # Selenium orange for 3-10 helix
            "I": Color(spec=0, x=1.0, y=0.63, z=0.0),  # Selenium orange for pi helix
            "E": Color(spec=0, x=0.3, y=0.76, z=1.0),  # Hafnium blue for strand
            "B": Color(spec=0, x=0.3, y=0.76, z=1.0),  # Hafnium blue for beta bridge
            "T": Color(spec=0, x=0.78, y=0.5, z=0.2),  # Copper brown for turn
            "S": Color(spec=0, x=0.78, y=0.5, z=0.2),  # Copper brown for bend
            "C": Color(spec=0, x=0.78, y=0.5, z=0.2),  # Copper brown for coil
            " ": Color(spec=0, x=0.7, y=0.7, z=0.7),  # Light gray for other
            "-": Color(spec=0, x=0.7, y=0.7, z=0.7),  # Light gray for other
        }

    ELEMENT_COLORS = {
        "H": (1.0, 1.0, 1.0),  # White
        "C": (0.8, 0.8, 0.8),  # Light gray
        "N": (0.0, 0.0, 1.0),  # Blue
        "O": (1.0, 0.0, 0.0),  # Red
        "S": (1.0, 1.0, 0.0),  # Yellow
        "P": (1.0, 0.5, 0.0),  # Orange
    }
    INVALID = (1.0, 0.0, 0.0)
    colors: list = [
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 0),
        (1, 1, 0),
        (1, 0, 1),
        (0, 1, 1),
        (0, 0.5, 0),
        (0, 1, 0.5),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 0),
        (0, 0.5, 0),
        (0, 1, 0.5),
    ]

    one_letter_atom_colors: dict = {
        "C": (0.0, 0.0, 0.0),
        "O": (1.0, 0.0, 0.0),
        "H": (1.0, 1.0, 1.0),
        "N": (0.0, 0.0, 1.0),
        "S": (1.0, 1.0, 0.0),
        " ": (0.7, 0.7, 0.7),
    }

    atom_colors: dict = {
        "C": (0.8, 0.8, 0.8),
        "O": (1.0, 0.0, 0.0),
        "H": (1.0, 1.0, 1.0),
        "N": (0.0, 0.0, 1.0),
        "S": (1.0, 1.0, 0.0),
        "P": (1.0, 0.0, 0.0),
        " ": (0.9, 0.9, 0.9),
    }

    secondary_structure_color_map: dict = {
        # Helices (H=alpha, G=3-10, I=pi)
        "H": (1.0, 0.0, 0.0),  # red
        "G": (1.0, 0.0, 0.0),  # red
        "I": (1.0, 0.0, 0.0),  # red
        # Sheets / strands (E) and beta-bridges (B)
        "E": (1.0, 1.0, 0.0),  # yellow
        "B": (1.0, 1.0, 0.0),  # yellow
        # Turns/bends
        "T": (0.0, 1.0, 1.0),  # cyan
        "S": (0.0, 1.0, 1.0),  # cyan
        # Coil/other
        "C": (0.6, 0.6, 0.6),  # gray
        " ": (0.6, 0.6, 0.6),  # gray
    }

    # Legacy SS_COLORS - now uses the registry system
    @classmethod
    def _get_ss_colors(cls):
        """Get the current secondary structure color map."""
        cls._initialize_ss_maps()
        return cls._ss_color_maps[cls._current_ss_map]

    # Back-compat: Tests expect ColorMap.SS_COLORS to be a subscriptable dict.
    # We'll maintain a class attribute alias that is refreshed when maps change.

    # Class property for backward compatibility
    @classmethod
    def get_ss_colors(cls):
        """Get the current secondary structure color map."""
        return cls._get_ss_colors()

    @staticmethod
    def b_factor_to_color(
        b_factor: float, min_b: float = 0.0, max_b: float = 200.0
    ) -> tuple[float, float, float]:
        """
        Convert B-factor to colour using white-to-red scale.

        Args:
            b_factor: B-factor value
            min_b: Minimum B-factor for white colour (default: 0.0)
            max_b: Maximum B-factor for red colour (default: 200.0)

        Returns:
            RGB tuple (r, g, b) with values 0.0-1.0
        """
        # Clamp B-factor to range
        b_factor = max(min_b, min(max_b, b_factor))

        # Handle edge case where min_b equals max_b
        if max_b == min_b:
            normalized = 0.0
        else:
            # Normalize to 0-1 range
            normalized = (b_factor - min_b) / (max_b - min_b)

        # White (1, 1, 1) to Red (1, 0, 0) interpolation
        # Red component stays at 1.0
        # Green and blue components decrease from 1.0 to 0.0
        red = 1.0
        green = 1.0 - normalized
        blue = 1.0 - normalized

        return (red, green, blue)

    @staticmethod
    def contact_distance_to_color(
        contact_distance: float, min_dist: float = 0.0, max_dist: float = 5.0
    ) -> tuple[float, float, float]:
        """
        Convert contact distance to colour using blue-to-red scale.
        Closer distances (more contacts) are blue, farther distances are red.

        Args:
            contact_distance: Contact distance value in Angstroms
            min_dist: Minimum distance for blue colour (default: 0.0)
            max_dist: Maximum distance for red colour (default: 5.0)

        Returns:
            RGB tuple (r, g, b) with values 0.0-1.0
        """
        # Clamp contact distance to range
        contact_distance = max(min_dist, min(max_dist, contact_distance))

        # Handle edge case where min_dist equals max_dist
        if max_dist == min_dist:
            normalized = 0.0
        else:
            # Normalize to 0-1 range
            normalized = (contact_distance - min_dist) / (max_dist - min_dist)

        # Blue (0, 0, 1) to Red (1, 0, 0) interpolation
        # Blue component decreases from 1.0 to 0.0
        # Red component increases from 0.0 to 1.0
        # Green stays at 0.0
        red = normalized
        green = 0.0
        blue = 1.0 - normalized

        return (red, green, blue)

    chain_colors: dict = {}

    # ✅ Strategy registry
    _strategies: dict[str, Callable] = {}

    @classmethod
    def register_strategy(cls, name: str, func: Callable) -> None:
        """Register a new coloring color_scheme by color_scheme."""
        cls._strategies[name] = func

    @classmethod
    def get_strategy(cls, name: str) -> Callable:
        """Retrieve a color_scheme function."""
        return cls._strategies.get(name)

    @classmethod
    def apply_strategy(
        cls, atom, strategy: Union[str, ColorScheme] = ColorScheme.ELEMENT
    ) -> None:
        """Apply a registered coloring color_scheme to an atom."""
        func = cls.get_strategy(strategy)
        if func:
            atom.color = np.array(func(atom), dtype=np.float32)

    # Secondary structure color map management methods
    @classmethod
    def register_ss_color_map(cls, name: str, color_map: dict) -> None:
        """
        Register a new secondary structure color map.

        Args:
            name: Name of the color map
            color_map: Dictionary mapping secondary structure codes to Color objects
        """
        cls._initialize_ss_maps()
        cls._ss_color_maps[name] = color_map

    @classmethod
    def get_ss_color_map(cls, name: str) -> dict:
        """
        Get a secondary structure color map by name.

        Args:
            name: Name of the color map

        Returns:
            Dictionary mapping secondary structure codes to Color objects
        """
        cls._initialize_ss_maps()
        return cls._ss_color_maps.get(name, cls._ss_color_maps["default"])

    @classmethod
    def set_ss_color_map(cls, name: str) -> bool:
        """
        Set the current secondary structure color map.

        Args:
            name: Name of the color map to set as current

        Returns:
            True if successful, False if color map not found
        """
        cls._initialize_ss_maps()
        if name in cls._ss_color_maps:
            cls._current_ss_map = name
            # Refresh alias used by tests
            try:
                cls.SS_COLORS = cls._get_ss_colors()
            except Exception:
                pass
            return True
        return False

    @classmethod
    def get_available_ss_color_maps(cls) -> list[str]:
        """
        Get list of available secondary structure color map names.

        Returns:
            List of color map names (excluding individual color registrations)
        """
        cls._initialize_ss_maps()

        # Filter out individual color registrations, keep only proper color maps
        proper_maps = []
        for map_name in cls._ss_color_maps.keys():
            # Check if this is a proper color map (has multiple secondary structure types)
            color_map = cls._ss_color_maps[map_name]
            if len(color_map) > 1:  # Proper color maps have multiple SS types
                proper_maps.append(map_name)

        return proper_maps

    @classmethod
    def get_current_ss_color_map_name(cls) -> str:
        """
        Get the name of the current secondary structure color map.

        Returns:
            Name of the current color map
        """
        return cls._current_ss_map

    @classmethod
    def get_ss_color(cls, ss_code: str) -> Color:
        """
        Get the color for a specific secondary structure code.

        Args:
            ss_code: Secondary structure code (H, E, T, etc.)

        Returns:
            Color object for the secondary structure code
        """
        colors = cls.get_ss_colors()
        return colors.get(ss_code, colors.get(" ", Color(spec=0, x=0.7, y=0.7, z=0.7)))

    @classmethod
    def create_custom_ss_color_map(cls, name: str, **kwargs) -> None:
        """
        Create a custom secondary structure color map from keyword arguments.

        Args:
            name: Name for the new color map
            **kwargs: Secondary structure codes as keys, RGB tuples as values
                     Example: H=(1.0, 0.0, 0.0), E=(0.0, 1.0, 0.0)
        """
        color_map = {}
        for ss_code, rgb in kwargs.items():
            if isinstance(rgb, (tuple, list)) and len(rgb) == 3:
                color_map[ss_code] = Color(spec=0, x=rgb[0], y=rgb[1], z=rgb[2])
            elif isinstance(rgb, Color):
                color_map[ss_code] = rgb

        cls.register_ss_color_map(name, color_map)


def reg_named_color(name: str, r, g, b):
    """Register a new color map."""
    color_map = {name: Color(spec=0, x=r, y=g, z=b)}
    ColorMap.register_ss_color_map(name, color_map)


reg_named_color("paleyellow", 1.0, 1.0, 0.5)
reg_named_color("aquamarine", 0.5, 1.0, 1.0)
reg_named_color("deepsalmon", 1.0, 0.5, 0.5)
reg_named_color("palegreen", 0.65, 0.9, 0.65)
reg_named_color("deepolive", 0.6, 0.6, 0.1)
reg_named_color("deeppurple", 0.6, 0.1, 0.6)
reg_named_color("deepteal", 0.1, 0.6, 0.6)
reg_named_color("lightblue", 0.75, 0.75, 1.0)
reg_named_color("lightorange", 1.0, 0.8, 0.5)
reg_named_color("palecyan", 0.8, 1.0, 1.0)
reg_named_color("lightteal", 0.4, 0.7, 0.7)
reg_named_color("splitpea", 0.52, 0.75, 0.0)
reg_named_color("raspberry", 0.7, 0.3, 0.4)
reg_named_color("sand", 0.72, 0.55, 0.3)
reg_named_color("smudge", 0.55, 0.7, 0.4)
reg_named_color("violetpurple", 0.55, 0.25, 0.6)
reg_named_color("dirtyviolet", 0.7, 0.5, 0.5)
reg_named_color("_deepsalmon", 1.0, 0.42, 0.42)
reg_named_color("lightpink", 1.0, 0.75, 0.87)
reg_named_color("greencyan", 0.25, 1.0, 0.75)
reg_named_color("limon", 0.75, 1.0, 0.25)
reg_named_color("skyblue", 0.2, 0.5, 0.8)
reg_named_color("bluewhite", 0.85, 0.85, 1.0)
reg_named_color("warmpink", 0.85, 0.2, 0.5)
reg_named_color("darksalmon", 0.73, 0.55, 0.52)
reg_named_color("helium", 0.850980392, 1.0, 1.0)
reg_named_color("lithium", 0.8, 0.501960784, 1.0)
reg_named_color("beryllium", 0.760784314, 1.0, 0.0)
reg_named_color("boron", 1.0, 0.709803922, 0.709803922)
reg_named_color("fluorine", 0.701960784, 1.0, 1.0)
reg_named_color("neon", 0.701960784, 0.890196078, 0.960784314)
reg_named_color("sodium", 0.670588235, 0.360784314, 0.949019608)
reg_named_color("magnesium", 0.541176471, 1.0, 0.0)
reg_named_color("aluminum", 0.749019608, 0.650980392, 0.650980392)
reg_named_color("silicon", 0.941176471, 0.784313725, 0.62745098)
reg_named_color("phosphorus", 1.0, 0.501960784, 0.0)
reg_named_color("phosphorus", 1.0, 0.501960784, 0.0)
reg_named_color("argon", 0.501960784, 0.819607843, 0.890196078)
reg_named_color("potassium", 0.560784314, 0.250980392, 0.831372549)
reg_named_color("calcium", 0.239215686, 1.0, 0.0)
reg_named_color("scandium", 0.901960784, 0.901960784, 0.901960784)
reg_named_color("titanium", 0.749019608, 0.760784314, 0.780392157)
reg_named_color("vanadium", 0.650980392, 0.650980392, 0.670588235)
reg_named_color("chromium", 0.541176471, 0.6, 0.780392157)
reg_named_color("manganese", 0.611764706, 0.478431373, 0.780392157)
reg_named_color("iron", 0.878431373, 0.4, 0.2)
reg_named_color("cobalt", 0.941176471, 0.564705882, 0.62745098)
reg_named_color("nickel", 0.31372549, 0.815686275, 0.31372549)
reg_named_color("copper", 0.784313725, 0.501960784, 0.2)
reg_named_color("zinc", 0.490196078, 0.501960784, 0.690196078)
reg_named_color("gallium", 0.760784314, 0.560784314, 0.560784314)
reg_named_color("germanium", 0.4, 0.560784314, 0.560784314)
reg_named_color("arsenic", 0.741176471, 0.501960784, 0.890196078)
reg_named_color("selenium", 1.0, 0.631372549, 0.0)
reg_named_color("bromine", 0.650980392, 0.160784314, 0.160784314)
reg_named_color("krypton", 0.360784314, 0.721568627, 0.819607843)

# Additional useful colors for custom selection
reg_named_color("coral", 1.0, 0.5, 0.31)
reg_named_color("turquoise", 0.25, 0.88, 0.82)
reg_named_color("lavender", 0.9, 0.9, 0.98)
reg_named_color("mint", 0.6, 1.0, 0.6)
reg_named_color("peach", 1.0, 0.9, 0.7)
reg_named_color("plum", 0.87, 0.63, 0.87)
reg_named_color("gold", 1.0, 0.84, 0.0)
reg_named_color("silver", 0.75, 0.75, 0.75)
reg_named_color("copper", 0.72, 0.45, 0.2)
reg_named_color("bronze", 0.8, 0.5, 0.2)
reg_named_color("rubidium", 0.439215686, 0.180392157, 0.690196078)
reg_named_color("strontium", 0.0, 1.0, 0.0)
reg_named_color("yttrium", 0.580392157, 1.0, 1.0)
reg_named_color("zirconium", 0.580392157, 0.878431373, 0.878431373)
reg_named_color("niobium", 0.450980392, 0.760784314, 0.788235294)
reg_named_color("molybdenum", 0.329411765, 0.709803922, 0.709803)
reg_named_color("technetium", 0.231372549, 0.619607843, 0.619607843)
reg_named_color("ruthenium", 0.141176471, 0.560784314, 0.560784314)
reg_named_color("rhodium", 0.039215686, 0.490196078, 0.549019608)
reg_named_color("palladium", 0.0, 0.411764706, 0.521568627)
reg_named_color("silver", 0.752941176, 0.752941176, 0.752941176)
reg_named_color("cadmium", 1.0, 0.850980392, 0.560784314)
reg_named_color("indium", 0.650980392, 0.458823529, 0.450980392)
reg_named_color("tin", 0.4, 0.501960784, 0.501960784)
reg_named_color("antimony", 0.619607843, 0.388235294, 0.709803922)
reg_named_color("tellurium", 0.831372549, 0.478431373, 0.0)
reg_named_color("iodine", 0.580392157, 0.0, 0.580392157)
reg_named_color("xenon", 0.258823529, 0.619607843, 0.690196078)
reg_named_color("cesium", 0.341176471, 0.090196078, 0.560784314)
reg_named_color("barium", 0.0, 0.788235294, 0.0)
reg_named_color("lanthanum", 0.439215686, 0.831372549, 1.0)
reg_named_color("cerium", 1.0, 1.0, 0.780392157)
reg_named_color("praseodymium", 0.850980392, 1.0, 0.780392157)
reg_named_color("neodymium", 0.780392157, 1.0, 0.780392157)
reg_named_color("promethium", 0.639215686, 1.0, 0.780392157)
reg_named_color("samarium", 0.560784314, 1.0, 0.780392157)
reg_named_color("europium", 0.380392157, 1.0, 0.780392157)
reg_named_color("gadolinium", 0.270588235, 1.0, 0.780392157)
reg_named_color("terbium", 0.188235294, 1.0, 0.780392157)
reg_named_color("dysprosium", 0.121568627, 1.0, 0.780392157)
reg_named_color("holmium", 0.0, 1.0, 0.611764706)
reg_named_color("erbium", 0.0, 0.901960784, 0.458823529)
reg_named_color("thulium", 0.0, 0.831372549, 0.321568627)
reg_named_color("ytterbium", 0.0, 0.749019608, 0.219607843)
reg_named_color("lutetium", 0.0, 0.670588235, 0.141176471)
reg_named_color("hafnium", 0.301960784, 0.760784314, 1.0)
reg_named_color("tantalum", 0.301960784, 0.650980392, 1.0)
reg_named_color("tungsten", 0.129411765, 0.580392157, 0.839215686)
reg_named_color("rhenium", 0.149019608, 0.490196078, 0.670588235)
reg_named_color("osmium", 0.149019608, 0.4, 0.588235294)
reg_named_color("iridium", 0.090196078, 0.329411765, 0.529411765)
reg_named_color("platinum", 0.815686275, 0.815686275, 0.878431373)
reg_named_color("gold", 1.0, 0.819607843, 0.137254902)
reg_named_color("mercury", 0.721568627, 0.721568627, 0.815686275)
reg_named_color("thallium", 0.650980392, 0.329411765, 0.301960784)
reg_named_color("lead", 0.341176471, 0.349019608, 0.380392157)
reg_named_color("bismuth", 0.619607843, 0.309803922, 0.709803922)
reg_named_color("polonium", 0.670588235, 0.360784314, 0.0)
reg_named_color("astatine", 0.458823529, 0.309803922, 0.270588235)
reg_named_color("radon", 0.258823529, 0.509803922, 0.588235294)
reg_named_color("francium", 0.258823529, 0.0, 0.4)
reg_named_color("radium", 0.0, 0.490196078, 0.0)
reg_named_color("actinium", 0.439215686, 0.670588235, 0.980392157)
reg_named_color("thorium", 0.0, 0.729411765, 1.0)
reg_named_color("protactinium", 0.0, 0.631372549, 1.0)
reg_named_color("uranium", 0.0, 0.560784314, 1.0)
reg_named_color("neptunium", 0.0, 0.501960784, 1.0)
reg_named_color("plutonium", 0.0, 0.419607843, 1.0)
reg_named_color("americium", 0.329411765, 0.360784314, 0.949019608)
reg_named_color("curium", 0.470588235, 0.360784314, 0.890196078)
reg_named_color("berkelium", 0.541176471, 0.309803922, 0.890196078)
reg_named_color("californium", 0.631372549, 0.211764706, 0.831372549)
reg_named_color("einsteinium", 0.701960784, 0.121568627, 0.831372549)
reg_named_color("fermium", 0.701960784, 0.121568627, 0.729411765)
reg_named_color("mendelevium", 0.701960784, 0.050980392, 0.650980392)
reg_named_color("nobelium", 0.741176471, 0.050980392, 0.529411765)
reg_named_color("lawrencium", 0.780392157, 0.0, 0.4)
reg_named_color("rutherfordium", 0.8, 0.0, 0.349019608)
reg_named_color("dubnium", 0.819607843, 0.0, 0.309803922)
reg_named_color("seaborgium", 0.850980392, 0.0, 0.270588235)
reg_named_color("bohrium", 0.878431373, 0.0, 0.219607843)
reg_named_color("hassium", 0.901960784, 0.0, 0.180392157)
reg_named_color("meitnerium", 0.921568627, 0.0, 0.149019608)
reg_named_color("deuterium", 0.9, 0.9, 0.9)
reg_named_color("lonepair", 0.5, 0.5, 0.5)
reg_named_color("pseudoatom", 0.9, 0.9, 0.9)

# Backward-compatibility: expose current SS color map as class attribute expected by tests
try:
    ColorMap._initialize_ss_maps()
    # Alias for tests expecting a dict-like map
    ColorMap.SS_COLORS = ColorMap._get_ss_colors()
except Exception:
    # Minimal fallback to avoid import-time failures
    ColorMap.SS_COLORS = {
        "H": Color(spec=0, x=1.0, y=0.0, z=0.0),
        "E": Color(spec=0, x=1.0, y=1.0, z=0.0),
        "T": Color(spec=0, x=1.0, y=1.0, z=0.0),
        " ": Color(spec=0, x=0.7, y=0.7, z=0.7),
        "-": Color(spec=0, x=0.7, y=0.7, z=0.7),
    }

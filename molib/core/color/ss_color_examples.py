"""
Examples of using the secondary structure color map system.

This module demonstrates how to use the new flexible color mapping system
for secondary structures in ElMo.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../ElMo/elmo/ui", "..", ".."))

from molib.core.color.color import Color
from molib.core.color.map import ColorMap


def example_basic_usage():
    """Basic usage examples."""
    print("=== Basic Usage Examples ===")

    # List available color maps
    print("Available color maps:", ColorMap.get_available_ss_color_maps())

    # Get current color map
    print("Current color map:", ColorMap.get_current_ss_color_map_name())

    # Switch to a different color map
    ColorMap.set_ss_color_map("rainbow")
    print("Switched to:", ColorMap.get_current_ss_color_map_name())

    # Get colors for specific secondary structure codes
    helix_color = ColorMap.get_ss_color("H")
    strand_color = ColorMap.get_ss_color("E")
    print(
        f"Helix color: RGB({helix_color.x:.1f}, {helix_color.y:.1f}, {helix_color.z:.1f})"
    )
    print(
        f"Strand color: RGB({strand_color.x:.1f}, {strand_color.y:.1f}, {strand_color.z:.1f})"
    )


def example_custom_color_map():
    """Example of creating a custom color map."""
    print("\n=== Custom Color Map Example ===")

    # Method 1: Using register_ss_color_map with a dictionary
    custom_map = {
        "H": Color(spec=0, x=1.0, y=0.0, z=0.0),  # Red helix
        "E": Color(spec=0, x=0.0, y=1.0, z=0.0),  # Green strand
        "T": Color(spec=0, x=0.0, y=0.0, z=1.0),  # Blue turn
        "C": Color(spec=0, x=1.0, y=1.0, z=0.0),  # Yellow coil
        " ": Color(spec=0, x=0.5, y=0.5, z=0.5),  # Gray other
    }
    ColorMap.register_ss_color_map("my_custom", custom_map)

    # Switch to the custom map
    ColorMap.set_ss_color_map("my_custom")
    print("Created and switched to custom color map")

    # Test the custom colors
    for code in ["H", "E", "T", "C", " "]:
        color = ColorMap.get_ss_color(code)
        print(f"  {code}: RGB({color.x:.1f}, {color.y:.1f}, {color.z:.1f})")


def example_protein_color_schemes():
    """Example of creating protein-specific color schemes."""
    print("\n=== Protein-Specific Color Schemes ===")

    # Create a "warm" color scheme
    warm_colors = {
        "H": Color(spec=0, x=1.0, y=0.3, z=0.0),  # Orange helix
        "E": Color(spec=0, x=1.0, y=0.6, z=0.0),  # Yellow-orange strand
        "T": Color(spec=0, x=1.0, y=0.8, z=0.2),  # Light orange turn
        "C": Color(spec=0, x=0.8, y=0.4, z=0.0),  # Dark orange coil
        " ": Color(spec=0, x=0.6, y=0.3, z=0.0),  # Brown other
    }
    ColorMap.register_ss_color_map("warm", warm_colors)

    # Create a "cool" color scheme
    cool_colors = {
        "H": Color(spec=0, x=0.0, y=0.3, z=1.0),  # Blue helix
        "E": Color(spec=0, x=0.0, y=0.6, z=1.0),  # Light blue strand
        "T": Color(spec=0, x=0.2, y=0.8, z=1.0),  # Cyan turn
        "C": Color(spec=0, x=0.0, y=0.0, z=0.8),  # Dark blue coil
        " ": Color(spec=0, x=0.3, y=0.3, z=0.6),  # Gray-blue other
    }
    ColorMap.register_ss_color_map("cool", cool_colors)

    # Test both schemes
    for scheme_name in ["warm", "cool"]:
        ColorMap.set_ss_color_map(scheme_name)
        print(f"\n{scheme_name.upper()} scheme:")
        for code in ["H", "E", "T", "C", " "]:
            color = ColorMap.get_ss_color(code)
            print(f"  {code}: RGB({color.x:.1f}, {color.y:.1f}, {color.z:.1f})")


def example_accessibility_colors():
    """Example of creating accessibility-friendly color schemes."""
    print("\n=== Accessibility-Friendly Color Schemes ===")

    # High contrast for colorblind users
    colorblind_friendly = {
        "H": Color(spec=0, x=1.0, y=0.0, z=0.0),  # Red helix
        "E": Color(spec=0, x=0.0, y=0.0, z=1.0),  # Blue strand
        "T": Color(spec=0, x=0.0, y=0.8, z=0.0),  # Green turn
        "C": Color(spec=0, x=0.8, y=0.0, z=0.8),  # Magenta coil
        " ": Color(spec=0, x=0.2, y=0.2, z=0.2),  # Dark gray other
    }
    ColorMap.register_ss_color_map("colorblind", colorblind_friendly)

    # Monochrome for print/grayscale
    ColorMap.set_ss_color_map("monochrome")
    print("Monochrome scheme (good for print):")
    for code in ["H", "E", "T", "C", " "]:
        color = ColorMap.get_ss_color(code)
        print(f"  {code}: RGB({color.x:.1f}, {color.y:.1f}, {color.z:.1f})")


def example_integration_with_ui():
    """Example of how this might be integrated with UI components."""
    print("\n=== UI Integration Example ===")

    # This would typically be called from a UI component
    def create_color_map_selector():
        """Create a UI selector for color maps."""
        available_maps = ColorMap.get_available_ss_color_maps()
        current_map = ColorMap.get_current_ss_color_map_name()

        print("Color Map Selector:")
        print(f"Current: {current_map}")
        print("Available options:")
        for i, map_name in enumerate(available_maps, 1):
            marker = "✓" if map_name == current_map else " "
            print(f"  {marker} {i}. {map_name}")

    create_color_map_selector()

    # Simulate user selection
    def select_color_map(map_name: str):
        """Simulate selecting a color map."""
        success = ColorMap.set_ss_color_map(map_name)
        if success:
            print(f"✓ Switched to '{map_name}' color map")
        else:
            print(f"✗ Failed to switch to '{map_name}' - map not found")

    select_color_map("nature")
    select_color_map("nonexistent")  # This should fail


if __name__ == "__main__":
    """Run all examples."""
    example_basic_usage()
    example_custom_color_map()
    example_protein_color_schemes()
    example_accessibility_colors()
    example_integration_with_ui()

    print("\n=== Summary ===")
    print("The new color map system provides:")
    print(
        "• 6 predefined color schemes (default, rainbow, monochrome, pastel, high_contrast, nature)"
    )
    print("• Easy creation of custom color maps")
    print("• Backward compatibility with existing code")
    print("• Flexible integration with UI components")
    print("• Support for accessibility requirements")

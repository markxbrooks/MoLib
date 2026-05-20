#!/usr/bin/env python
"""
Materials and Lighting Configuration
=====================================

This file contains the complete material properties and lighting configuration

"""

# =============================================================================
# MATERIAL PROPERTIES
# =============================================================================

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Material:
    name: str
    ambient: List[float]
    diffuse: List[float]
    specular: List[float]
    emissive: List[float]
    alpha: float
    shininess: float

    # ---- Derived properties ----

    @property
    def combined_rgb(self) -> List[float]:
        return [a + d for a, d in zip(self.ambient, self.diffuse)]

    @property
    def rgb255(self) -> List[int]:
        return [int(round(255 * max(0, min(1, v)))) for v in self.combined_rgb]

    @property
    def hex(self) -> str:
        r, g, b = self.rgb255
        return f"#{r:02x}{g:02x}{b:02x}"


class MaterialLibrary:
    """Material Library"""

    def __init__(self, materials: dict[int, Material]):
        self._materials = materials

    def get(self, index: int) -> Material:
        try:
            return self._materials[index]
        except KeyError:
            raise ValueError(f"Invalid material index: {index}")

    def __iter__(self):
        return iter(self._materials.items())

    def palette(self) -> list[str]:
        return [mat.hex for mat in self._materials.values()]

    def basic_colors(self) -> dict[int, Material]:
        return {i: self._materials[i] for i in range(8)}

    def rainbow(self) -> list[Material]:
        return [self._materials[i] for i in range(21, 41)]


@dataclass
class Light:
    direction: list[float]
    intensity: float
    on: bool = True

    def enable(self):
        self.on = True

    def disable(self):
        self.on = False


class LightingSystem:
    def __init__(self, lights: dict[int, Light], ambience: float, fog: dict):
        self.lights = lights
        self.ambience = ambience
        self.fog = fog

    def active_lights(self):
        return {i: l for i, l in self.lights.items() if l.on}


class GLMaterialAdapter:
    def __init__(self, material: Material):
        self.material = material

    def apply(self, gl):
        gl.set_ambient(self.material.ambient)
        gl.set_diffuse(self.material.diffuse)
        gl.set_specular(self.material.specular)
        gl.set_shininess(self.material.shininess)


# =============================================================================
# LIGHTING CONFIGURATION
# =============================================================================

# Light sources configuration
lights = {
    0: {"on": True, "direction": [-0.2000, 0.2000, 1.0000], "intensity": 1.0000},
    1: {"on": False, "direction": [0.0000, 0.7000, 0.7000], "intensity": 1.0000},
}

# Ambient lighting and fog settings
lighting_settings = {
    "ambience": 0.1000,
    "fog_on": False,
    "fog_density": 0.1500,
    "fog_mode": 0,
    "fog_depth": 0.0000,
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_material_properties(material_index):
    """Get complete material properties for a given index (0-40)."""
    if material_index not in materials:
        raise ValueError(
            f"Material index {material_index} not found. Valid range: 0-40"
        )
    return materials[material_index]


def get_ambient_color(material_index):
    """Get ambient RGB color for a material."""
    return get_material_properties(material_index)["ambient"]


def get_diffuse_color(material_index):
    """Get diffuse RGB color for a material."""
    return get_material_properties(material_index)["diffuse"]


def get_specular_color(material_index):
    """Get specular RGB color for a material."""
    return get_material_properties(material_index)["specular"]


def get_emissive_color(material_index):
    """Get emissive RGB color for a material."""
    return get_material_properties(material_index)["emissive"]


def get_shininess(material_index):
    """Get shininess value for a material."""
    return get_material_properties(material_index)["shininess"]


def get_alpha(material_index):
    """Get alpha (transparency) value for a material."""
    return get_material_properties(material_index)["alpha"]


def convert_to_rgb_255(rgb_float):
    """Convert RGB values from 0.0-1.0 range to 0-255 range."""
    return [int(round(255 * max(0, min(1, val)))) for val in rgb_float]


def convert_to_hex_color(rgb_float):
    """Convert RGB float values to hex color string."""
    rgb_255 = convert_to_rgb_255(rgb_float)
    return "#{:02x}{:02x}{:02x}".format(rgb_255[0], rgb_255[1], rgb_255[2])


def get_combined_rgb(material_index):
    """Get combined RGB color (ambient + diffuse) for a material."""
    mat = get_material_properties(material_index)
    ambient = mat["ambient"]
    diffuse = mat["diffuse"]
    return [ambient[i] + diffuse[i] for i in range(3)]


def get_combined_hex_color(material_index):
    """Get combined RGB color as hex string."""
    combined_rgb = get_combined_rgb(material_index)
    return convert_to_hex_color(combined_rgb)


def list_materials():
    """List all available materials with their names and indices."""
    print("Available Materials:")
    print("===================")
    for i in range(41):
        mat = materials[i]
        combined_hex = get_combined_hex_color(i)
        print(f"{i:2d}: {mat['name']:12s} {combined_hex}")


def get_light_configuration():
    """Get complete lighting configuration."""
    return {"lights": lights, "settings": lighting_settings}


def is_light_on(light_index):
    """Check if a specific light is on."""
    if light_index not in lights:
        return False
    return lights[light_index]["on"]


def get_light_direction(light_index):
    """Get direction vector for a specific light."""
    if light_index not in lights:
        return None
    return lights[light_index]["direction"]


def get_light_intensity(light_index):
    """Get intensity for a specific light."""
    if light_index not in lights:
        return None
    return lights[light_index]["intensity"]


# =============================================================================
# COLOR PALETTE UTILITIES
# =============================================================================


def get_color_palette():
    """Get a list of all material colors as hex strings."""
    return [get_combined_hex_color(i) for i in range(41)]


def get_basic_colors():
    """Get the basic 8 colors (0-7) as a dictionary."""
    basic_colors = {}
    for i in range(8):
        basic_colors[i] = {
            "name": materials[i]["name"],
            "hex": get_combined_hex_color(i),
            "rgb": get_combined_rgb(i),
        }
    return basic_colors


def get_rainbow_colors():
    """Get the rainbow spectrum colors (21-40) as a list."""
    rainbow = []
    for i in range(21, 41):
        rainbow.append(
            {
                "index": i,
                "name": materials[i]["name"],
                "hex": get_combined_hex_color(i),
                "rgb": get_combined_rgb(i),
            }
        )
    return rainbow


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("Ribbons Materials Configuration")
    print("===============================")
    print()

    # List all materials
    list_materials()
    print()

    # Show basic colors
    print("Basic Colors (0-7):")
    basic = get_basic_colors()
    for i, color in basic.items():
        print(f"  {i}: {color['name']:12s} {color['hex']}")
    print()

    # Show some specific material properties
    print("Material Properties Examples:")
    print("=============================")

    # Red material (index 1)
    red = get_material_properties(1)
    print(f"Red (index 1):")
    print(f"  Ambient:  {red['ambient']}")
    print(f"  Diffuse:  {red['diffuse']}")
    print(f"  Combined: {get_combined_rgb(1)}")
    print(f"  Hex:      {get_combined_hex_color(1)}")
    print(f"  Shininess: {red['shininess']}")
    print()

    # Blue material (index 4)
    blue = get_material_properties(4)
    print(f"Blue (index 4):")
    print(f"  Ambient:  {blue['ambient']}")
    print(f"  Diffuse:  {blue['diffuse']}")
    print(f"  Combined: {get_combined_rgb(4)}")
    print(f"  Hex:      {get_combined_hex_color(4)}")
    print(f"  Shininess: {blue['shininess']}")
    print()

    # Lighting configuration
    print("Lighting Configuration:")
    print("======================")
    light_config = get_light_configuration()
    for light_id, light in light_config["lights"].items():
        status = "ON" if light["on"] else "OFF"
        print(f"Light {light_id}: {status}")
        print(f"  Direction: {light['direction']}")
        print(f"  Intensity: {light['intensity']}")
    print()

    print(f"Ambience: {light_config['settings']['ambience']}")
    print(f"Fog: {'ON' if light_config['settings']['fog_on'] else 'OFF'}")
    print(f"Fog Density: {light_config['settings']['fog_density']}")

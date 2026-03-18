#!/usr/bin/env python
"""
Materials and Lighting Configuration
=====================================

This file contains the complete material properties and lighting configuration

"""

# =============================================================================
# MATERIAL PROPERTIES
# =============================================================================

# Complete material definitions for all 41 materials (0-40)
# Each material contains: ambient, diffuse, specular, emissive RGB values, alpha, and shininess
materials = {
    0: {
        "name": "White",
        "ambient": [1.0000, 1.0000, 1.0000],
        "diffuse": [0.1000, 0.1000, 0.1000],
        "alpha": 1.0000,
        "specular": [0.9000, 0.9000, 0.9000],
        "emissive": [0.0000, 0.0000, 0.0000],
        "shininess": 10.0000,
    },
    1: {
        "name": "Red",
        "ambient": [0.1200, 0.0600, 0.0600],
        "diffuse": [0.8800, 0.1200, 0.1200],
        "alpha": 1.0000,
        "specular": [0.8000, 0.7000, 0.7000],
        "emissive": [0.0000, 0.0000, 0.0000],
        "shininess": 48.0000,
    },
    2: {
        "name": "Green",
        "ambient": [0.0600, 0.1200, 0.0600],
        "diffuse": [0.1200, 0.8800, 0.1200],
        "alpha": 1.0000,
        "specular": [0.7000, 0.8000, 0.7000],
        "emissive": [0.0000, 0.0000, 0.0000],
        "shininess": 30.0000,
    },
    3: {
        "name": "Yellow",
        "ambient": [0.1200, 0.1200, 0.0600],
        "diffuse": [0.8800, 0.8800, 0.1200],
        "alpha": 1.0000,
        "specular": [0.8000, 0.8000, 0.7000],
        "emissive": [0.0000, 0.0000, 0.0000],
        "shininess": 30.0000,
    },
    4: {
        "name": "Blue",
        "ambient": [0.0600, 0.0600, 0.1200],
        "diffuse": [0.1200, 0.1200, 0.8800],
        "alpha": 1.0000,
        "specular": [0.7000, 0.7000, 0.8000],
        "emissive": [0.0000, 0.0000, 0.0000],
        "shininess": 48.0000,
    },
    5: {
        "name": "Magenta",
        "ambient": [0.1200, 0.0600, 0.1200],
        "diffuse": [0.8800, 0.1200, 0.8800],
        "alpha": 1.0000,
        "specular": [0.8000, 0.7000, 0.8000],
        "emissive": [0.0000, 0.0000, 0.0000],
        "shininess": 30.0000,
    },
    6: {
        "name": "Cyan",
        "ambient": [0.0600, 0.1200, 0.1200],
        "diffuse": [0.1200, 0.8800, 0.8800],
        "alpha": 1.0000,
        "specular": [0.7000, 0.8000, 0.8000],
        "emissive": [0.0000, 0.0000, 0.0000],
        "shininess": 30.0000,
    },
    7: {
        "name": "Gray",
        "ambient": [0.3200, 0.3200, 0.3200],
        "diffuse": [0.7500, 0.7500, 0.7500],
        "alpha": 1.0000,
        "specular": [0.3000, 0.3000, 0.3000],
        "emissive": [0.0000, 0.0000, 0.0000],
        "shininess": 10.0000,
    },
    8: {
        "name": "Orange",
        "ambient": [0.2000, 0.1000, 0.0000],
        "diffuse": [0.7500, 0.3500, 0.0000],
        "alpha": 1.0000,
        "specular": [0.3000, 0.2500, 0.2000],
        "emissive": [0.0000, 0.0000, 0.0000],
        "shininess": 20.0000,
    },
    9: {
        "name": "Dark Gray",
        "ambient": [0.1000, 0.1000, 0.1000],
        "diffuse": [0.3500, 0.3500, 0.3500],
        "alpha": 1.0000,
        "specular": [0.1000, 0.1000, 0.1000],
        "emissive": [0.0000, 0.0000, 0.0000],
        "shininess": 10.0000,
    },
    10: {
        "name": "Dark Blue",
        "ambient": [0.0000, 0.0000, 0.9000],
        "diffuse": [0.0100, 0.0100, 0.1800],
        "alpha": 1.0000,
        "specular": [0.5600, 0.5400, 0.7300],
        "emissive": [0.0000, 0.0000, 0.0000],
        "shininess": 44.0600,
    },
    11: {
        "name": "Light Gray",
        "ambient": [0.4000, 0.4000, 0.4000],
        "diffuse": [0.3000, 0.3000, 0.3000],
        "alpha": 1.0000,
        "specular": [0.9000, 0.9000, 0.9500],
        "emissive": [0.0000, 0.0000, 0.0000],
        "shininess": 30.0000,
    },
    12: {
        "name": "Purple",
        "ambient": [0.1000, 0.0000, 0.1200],
        "diffuse": [0.8000, 0.6000, 0.8600],
        "alpha": 1.0000,
        "specular": [0.3000, 0.2000, 0.3600],
        "emissive": [0.0000, 0.0000, 0.0000],
        "shininess": 6.5000,
    },
    13: {
        "name": "Black",
        "ambient": [0.0000, 0.0000, 0.0000],
        "diffuse": [0.0200, 0.0200, 0.0200],
        "alpha": 1.0000,
        "specular": [0.8800, 0.8800, 0.8800],
        "emissive": [0.0000, 0.0000, 0.0000],
        "shininess": 60.0000,
    },
    14: {
        "name": "Gold",
        "ambient": [0.4000, 0.2000, 0.0000],
        "diffuse": [0.9000, 0.5000, 0.0000],
        "alpha": 1.0000,
        "specular": [0.9000, 0.9000, 0.0000],
        "emissive": [0.0000, 0.0000, 0.0000],
        "shininess": 20.0000,
    },
    15: {
        "name": "Pink",
        "ambient": [0.2000, 0.1000, 0.1000],
        "diffuse": [0.7843, 0.3137, 0.4706],
        "alpha": 1.0000,
        "specular": [0.2000, 0.1000, 0.1000],
        "emissive": [0.0000, 0.0000, 0.0000],
        "shininess": 10.0000,
    },
    16: {
        "name": "Mint Green",
        "ambient": [0.0700, 0.1900, 0.1400],
        "diffuse": [0.6200, 0.9700, 0.8500],
        "alpha": 1.0000,
        "specular": [0.2000, 0.3000, 0.2000],
        "emissive": [0.0000, 0.1000, 0.0000],
        "shininess": 30.0000,
    },
    17: {
        "name": "Deep Purple",
        "ambient": [0.0825, 0.0000, 0.2500],
        "diffuse": [0.1320, 0.0000, 0.4000],
        "alpha": 1.0000,
        "specular": [0.3970, 0.1000, 1.0000],
        "emissive": [0.1490, 0.0500, 0.3500],
        "shininess": 30.0000,
    },
    18: {
        "name": "Purple 2",
        "ambient": [0.1675, 0.0000, 0.2500],
        "diffuse": [0.2680, 0.0000, 0.4000],
        "alpha": 1.0000,
        "specular": [0.7030, 0.1000, 1.0000],
        "emissive": [0.2510, 0.0500, 0.3500],
        "shininess": 30.0000,
    },
    19: {
        "name": "Purple 3",
        "ambient": [0.2500, 0.0000, 0.1675],
        "diffuse": [0.4000, 0.0000, 0.2680],
        "alpha": 1.0000,
        "specular": [1.0000, 0.1000, 0.7030],
        "emissive": [0.3500, 0.0500, 0.2510],
        "shininess": 30.0000,
    },
    20: {
        "name": "Purple 4",
        "ambient": [0.2500, 0.0000, 0.0825],
        "diffuse": [0.4000, 0.0000, 0.1320],
        "alpha": 1.0000,
        "specular": [1.0000, 0.1000, 0.3970],
        "emissive": [0.3500, 0.0500, 0.1490],
        "shininess": 30.0000,
    },
    21: {
        "name": "Deep Red",
        "ambient": [0.2500, 0.0000, 0.0000],
        "diffuse": [0.4000, 0.0000, 0.0000],
        "alpha": 1.0000,
        "specular": [1.0000, 0.1000, 0.1000],
        "emissive": [0.3500, 0.0500, 0.0500],
        "shininess": 30.0000,
    },
    22: {
        "name": "Red-Orange",
        "ambient": [0.2500, 0.0425, 0.0000],
        "diffuse": [0.4000, 0.0680, 0.0000],
        "alpha": 1.0000,
        "specular": [1.0000, 0.2530, 0.1000],
        "emissive": [0.3500, 0.1010, 0.0500],
        "shininess": 30.0000,
    },
    23: {
        "name": "Orange 2",
        "ambient": [0.2500, 0.0825, 0.0000],
        "diffuse": [0.4000, 0.1320, 0.0000],
        "alpha": 1.0000,
        "specular": [1.0000, 0.3970, 0.1000],
        "emissive": [0.3500, 0.1490, 0.0500],
        "shininess": 30.0000,
    },
    24: {
        "name": "Orange 3",
        "ambient": [0.2500, 0.1250, 0.0000],
        "diffuse": [0.4000, 0.2000, 0.0000],
        "alpha": 1.0000,
        "specular": [1.0000, 0.5500, 0.1000],
        "emissive": [0.3500, 0.2000, 0.0500],
        "shininess": 30.0000,
    },
    25: {
        "name": "Orange 4",
        "ambient": [0.2500, 0.1675, 0.0000],
        "diffuse": [0.4000, 0.2680, 0.0000],
        "alpha": 1.0000,
        "specular": [1.0000, 0.7030, 0.1000],
        "emissive": [0.3500, 0.2510, 0.0500],
        "shininess": 30.0000,
    },
    26: {
        "name": "Orange 5",
        "ambient": [0.2500, 0.2100, 0.0000],
        "diffuse": [0.4000, 0.3360, 0.0000],
        "alpha": 1.0000,
        "specular": [1.0000, 0.8560, 0.1000],
        "emissive": [0.3500, 0.3020, 0.0500],
        "shininess": 30.0000,
    },
    27: {
        "name": "Yellow 2",
        "ambient": [0.2500, 0.2500, 0.0000],
        "diffuse": [0.4000, 0.4000, 0.0000],
        "alpha": 1.0000,
        "specular": [1.0000, 1.0000, 0.1000],
        "emissive": [0.3500, 0.3500, 0.0500],
        "shininess": 30.0000,
    },
    28: {
        "name": "Yellow 3",
        "ambient": [0.2100, 0.2500, 0.0000],
        "diffuse": [0.3360, 0.4000, 0.0000],
        "alpha": 1.0000,
        "specular": [0.8560, 1.0000, 0.1000],
        "emissive": [0.3020, 0.3500, 0.0500],
        "shininess": 30.0000,
    },
    29: {
        "name": "Yellow 4",
        "ambient": [0.1675, 0.2500, 0.0000],
        "diffuse": [0.2680, 0.4000, 0.0000],
        "alpha": 1.0000,
        "specular": [0.7030, 1.0000, 0.1000],
        "emissive": [0.2510, 0.3500, 0.0500],
        "shininess": 30.0000,
    },
    30: {
        "name": "Yellow 5",
        "ambient": [0.1250, 0.2500, 0.0000],
        "diffuse": [0.2000, 0.4000, 0.0000],
        "alpha": 1.0000,
        "specular": [0.5500, 1.0000, 0.1000],
        "emissive": [0.2000, 0.3500, 0.0500],
        "shininess": 30.0000,
    },
    31: {
        "name": "Yellow 6",
        "ambient": [0.0825, 0.2500, 0.0000],
        "diffuse": [0.1320, 0.4000, 0.0000],
        "alpha": 1.0000,
        "specular": [0.3970, 1.0000, 0.1000],
        "emissive": [0.1490, 0.3500, 0.0500],
        "shininess": 30.0000,
    },
    32: {
        "name": "Yellow 7",
        "ambient": [0.0425, 0.2500, 0.0000],
        "diffuse": [0.0680, 0.4000, 0.0000],
        "alpha": 1.0000,
        "specular": [0.2530, 1.0000, 0.1000],
        "emissive": [0.1010, 0.3500, 0.0500],
        "shininess": 30.0000,
    },
    33: {
        "name": "Green 2",
        "ambient": [0.0000, 0.2500, 0.0000],
        "diffuse": [0.0000, 0.4000, 0.0000],
        "alpha": 1.0000,
        "specular": [0.1000, 1.0000, 0.1000],
        "emissive": [0.0500, 0.3500, 0.0500],
        "shininess": 30.0000,
    },
    34: {
        "name": "Green 3",
        "ambient": [0.0000, 0.2500, 0.0825],
        "diffuse": [0.0000, 0.4000, 0.1320],
        "alpha": 1.0000,
        "specular": [0.1000, 1.0000, 0.3970],
        "emissive": [0.0500, 0.3500, 0.1490],
        "shininess": 30.0000,
    },
    35: {
        "name": "Green 4",
        "ambient": [0.0000, 0.2500, 0.1250],
        "diffuse": [0.0000, 0.4000, 0.2000],
        "alpha": 1.0000,
        "specular": [0.1000, 1.0000, 0.5500],
        "emissive": [0.0500, 0.3500, 0.2000],
        "shininess": 30.0000,
    },
    36: {
        "name": "Green 5",
        "ambient": [0.0000, 0.2500, 0.1675],
        "diffuse": [0.0000, 0.4000, 0.2680],
        "alpha": 1.0000,
        "specular": [0.1000, 1.0000, 0.7030],
        "emissive": [0.0500, 0.3500, 0.2510],
        "shininess": 30.0000,
    },
    37: {
        "name": "Cyan 2",
        "ambient": [0.0000, 0.2500, 0.2500],
        "diffuse": [0.0000, 0.4000, 0.4000],
        "alpha": 1.0000,
        "specular": [0.1000, 1.0000, 1.0000],
        "emissive": [0.0500, 0.3500, 0.3500],
        "shininess": 30.0000,
    },
    38: {
        "name": "Cyan 3",
        "ambient": [0.0000, 0.1675, 0.2500],
        "diffuse": [0.0000, 0.2680, 0.4000],
        "alpha": 1.0000,
        "specular": [0.1000, 0.7030, 1.0000],
        "emissive": [0.0500, 0.2510, 0.3500],
        "shininess": 30.0000,
    },
    39: {
        "name": "Cyan 4",
        "ambient": [0.0000, 0.0825, 0.2500],
        "diffuse": [0.0000, 0.1320, 0.4000],
        "alpha": 1.0000,
        "specular": [0.1000, 0.3970, 1.0000],
        "emissive": [0.0500, 0.1490, 0.3500],
        "shininess": 30.0000,
    },
    40: {
        "name": "Blue 2",
        "ambient": [0.0000, 0.0000, 0.2500],
        "diffuse": [0.0000, 0.0000, 0.4000],
        "alpha": 1.0000,
        "specular": [0.1000, 0.1000, 1.0000],
        "emissive": [0.0500, 0.0500, 0.3500],
        "shininess": 30.0000,
    },
}

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

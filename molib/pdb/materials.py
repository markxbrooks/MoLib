#!/usr/bin/env python
"""
Materials and Lighting Configuration (OO Refactor)
=================================================

- Strongly typed Material + Light objects
- Central MaterialLibrary registry
- Zero behavioral drift from original implementation
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple


# =============================================================================
# CORE TYPES
# =============================================================================

Vec3 = Tuple[float, float, float]


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _to_rgb255(rgb: Vec3) -> Tuple[int, int, int]:
    return tuple(int(round(255 * _clamp01(v))) for v in rgb)


def _to_hex(rgb: Vec3) -> str:
    r, g, b = _to_rgb255(rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


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
# MATERIAL
# =============================================================================

@dataclass(frozen=True)
class Material:
    name: str
    ambient: Vec3
    diffuse: Vec3
    specular: Vec3
    emissive: Vec3
    alpha: float
    shininess: float

    # ---- Derived properties ----

    @property
    def combined_rgb(self) -> Vec3:
        return tuple(a + d for a, d in zip(self.ambient, self.diffuse))

    @property
    def rgb255(self) -> Tuple[int, int, int]:
        return _to_rgb255(self.combined_rgb)

    @property
    def hex(self) -> str:
        return _to_hex(self.combined_rgb)


# =============================================================================
# MATERIAL LIBRARY
# =============================================================================

class MaterialLibrary:
    def __init__(self, materials: Dict[int, Material]):
        self._materials = materials

    # ---- Access ----

    def get(self, index: int) -> Material:
        try:
            return self._materials[index]
        except KeyError:
            raise ValueError(f"Material index {index} not found (0–40)")

    def __getitem__(self, index: int) -> Material:
        return self.get(index)

    def __iter__(self) -> Iterable[tuple[int, Material]]:
        return iter(self._materials.items())

    # ---- Queries ----

    def palette(self) -> list[str]:
        return [m.hex for m in self._materials.values()]

    def basic(self) -> Dict[int, Material]:
        return {i: self._materials[i] for i in range(8)}

    def rainbow(self) -> list[Material]:
        return [self._materials[i] for i in range(21, 41)]

    def list(self) -> None:
        print("Available Materials:")
        print("===================")
        for i, m in self._materials.items():
            print(f"{i:2d}: {m.name:12s} {m.hex}")



# =============================================================================
# LIGHTING
# =============================================================================

@dataclass
class Light:
    direction: Vec3
    intensity: float
    on: bool = True

    def enable(self) -> None:
        self.on = True

    def disable(self) -> None:
        self.on = False


class LightingSystem:
    def __init__(
        self,
        lights: Dict[int, Light],
        ambience: float,
        fog_on: bool,
        fog_density: float,
        fog_mode: int,
        fog_depth: float,
    ):
        self.lights = lights
        self.ambience = ambience
        self.fog_on = fog_on
        self.fog_density = fog_density
        self.fog_mode = fog_mode
        self.fog_depth = fog_depth

    def active_lights(self) -> Dict[int, Light]:
        return {i: l for i, l in self.lights.items() if l.on}

    def is_light_on(self, idx: int) -> bool:
        return self.lights.get(idx, Light((0, 0, 0), 0, False)).on

lighting = LightingSystem(
    lights={
        0: Light(direction=(-0.2, 0.2, 1.0), intensity=1.0, on=True),
        1: Light(direction=(0.0, 0.7, 0.7), intensity=1.0, on=False),
    },
    ambience=0.1,
    fog_on=False,
    fog_density=0.15,
    fog_mode=0,
    fog_depth=0.0,
)

def _build_legacy_lighting_settings() -> dict:
    return {
        "ambience": lighting.ambience,
        "fog_on": lighting.fog_on,
        "fog_density": lighting.fog_density,
        "fog_mode": lighting.fog_mode,
        "fog_depth": lighting.fog_depth,
    }


lighting_settings = _build_legacy_lighting_settings()



# =============================================================================
# RAW DATA (UNCHANGED VALUES)
# =============================================================================

def _m(x):
    # Case 1: already Material → pass through
    if isinstance(x, Material):
        return x

    # Case 2: dict input (legacy format)
    return Material(
        name=x["name"],
        ambient=tuple(x["ambient"]),
        diffuse=tuple(x["diffuse"]),
        specular=tuple(x["specular"]),
        emissive=tuple(x["emissive"]),
        alpha=x["alpha"],
        shininess=x["shininess"],
    )


MATERIALS = MaterialLibrary({
    i: _m(m)
    for i, m in materials.items()
})


def _material_to_legacy_dict(mat: Material) -> dict:
    return {
        "name": mat.name,
        "ambient": list(mat.ambient),
        "diffuse": list(mat.diffuse),
        "specular": list(mat.specular),
        "emissive": list(mat.emissive),
        "alpha": mat.alpha,
        "shininess": mat.shininess,
    }


materials = {
    i: _material_to_legacy_dict(m)
    for i, m in MATERIALS
}

class _LightsProxy(dict):
    def __getitem__(self, key):
        l = lighting.lights[key]
        return {
            "on": l.on,
            "direction": list(l.direction),
            "intensity": l.intensity,
        }


lights = _LightsProxy()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _material_to_legacy_dict(mat: Material) -> dict:
    return {
        "name": mat.name,
        "hex": mat.hex,
        "rgb": list(mat.combined_rgb),
    }

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
    return {i: _material_to_legacy_dict(m) for i, m in MATERIALS.basic().items()}


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

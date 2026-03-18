# Secondary Structure Color Maps

This document describes the new flexible color mapping system for secondary structures in ElMo.

## Overview

The new system allows you to define and switch between different color schemes for rendering secondary structures (helices, strands, turns, coils, etc.). This provides better visualization options and supports various use cases including accessibility requirements.

## Available Color Maps

### Predefined Maps

1. **default** - The original ElMo color scheme
   - Helix (H): Red
   - Strand (E): Yellow
   - Turn (T): Yellow
   - Coil (C): Light gray

2. **rainbow** - Distinct colors for each structure type
   - Helix (H): Red
   - 3-10 Helix (G): Orange
   - Pi Helix (I): Yellow
   - Strand (E): Green
   - Beta Bridge (B): Dark green
   - Turn (T): Cyan
   - Bend (S): Light blue
   - Coil (C): Purple

3. **monochrome** - Grayscale scheme (good for print)
   - Different shades of gray for each structure type

4. **pastel** - Soft, muted colors
   - Gentle, easy-on-the-eyes colors

5. **high_contrast** - Maximum visual distinction
   - Bright, saturated colors for maximum visibility

6. **nature** - Earth-tone colors
   - Colors inspired by natural elements

## Usage

### Basic Usage

```python
from molib.core.color import ColorMap

# List available color maps
maps = ColorMap.get_available_ss_color_maps()
print(maps)  # ['default', 'rainbow', 'monochrome', 'pastel', 'high_contrast', 'nature']

# Get current color map
current = ColorMap.get_current_ss_color_map_name()
print(current)  # 'default'

# Switch to a different color map
ColorMap.set_ss_color_map('rainbow')

# Get color for a specific secondary structure code
helix_color = ColorMap.get_ss_color('H')
print(f"Helix color: RGB({helix_color.x}, {helix_color.y}, {helix_color.z})")
```

### Creating Custom Color Maps

```python
from molib.core.color import ColorMap
from molib.core.color.color import Color

# Method 1: Using register_ss_color_map
custom_map = {
    "H": Color(spec=0, x=1.0, y=0.0, z=0.0),  # Red helix
    "E": Color(spec=0, x=0.0, y=1.0, z=0.0),  # Green strand
    "T": Color(spec=0, x=0.0, y=0.0, z=1.0),  # Blue turn
    "C": Color(spec=0, x=1.0, y=1.0, z=0.0),  # Yellow coil
    " ": Color(spec=0, x=0.5, y=0.5, z=0.5)  # Gray other
}
ColorMap.register_ss_color_map("my_custom", custom_map)

# Switch to the custom map
ColorMap.set_ss_color_map("my_custom")
```

### Backward Compatibility

The new system maintains backward compatibility with existing code:

```python
# This still works
ss_colors = ColorMap().SS_COLORS
helix_color = ss_colors["H"]

# Or using the class method
ss_colors = ColorMap.get_ss_colors()
helix_color = ss_colors["H"]
```

## API Reference

### Class Methods

- `get_available_ss_color_maps()` → `list[str]`
  - Returns list of available color map names

- `get_current_ss_color_map_name()` → `str`
  - Returns name of current color map

- `set_ss_color_map(name: str)` → `bool`
  - Switch to specified color map
  - Returns True if successful, False if not found

- `get_ss_color_map(name: str)` → `dict`
  - Get color map by name
  - Returns default map if name not found

- `get_ss_color(ss_code: str)` → `Colour`
  - Get color for specific secondary structure code
  - Returns default color if code not found

- `register_ss_color_map(name: str, color_map: dict)` → `None`
  - Register a new color map

- `create_custom_ss_color_map(name: str, **kwargs)` → `None`
  - Create custom map from keyword arguments (RGB tuples)

## Secondary Structure Codes

The system supports standard DSSP secondary structure codes:

- **H** - Alpha helix
- **G** - 3-10 helix
- **I** - Pi helix
- **E** - Extended strand
- **B** - Beta bridge
- **T** - Turn
- **S** - Bend
- **C** - Coil
- **" "** - Other/unknown
- **"-"** - Gap

## Examples

See `elmo/ui/color/ss_color_examples.py` for comprehensive usage examples including:
- Basic usage
- Custom color map creation
- Protein-specific color schemes
- Accessibility-friendly colors
- UI integration patterns

## Integration with ElMo

The color map system integrates seamlessly with ElMo's rendering pipeline. When you change the color map, all secondary structure rendering will automatically use the new colors. This includes:

- Ribbon representations
- Cartoon representations
- Secondary structure highlighting
- Any other visualization that uses secondary structure colors

## Future Enhancements

Potential future enhancements could include:
- Color map persistence across sessions
- User-defined color map import/export
- Color map preview in UI
- Automatic color map suggestions based on content
- Integration with accessibility tools

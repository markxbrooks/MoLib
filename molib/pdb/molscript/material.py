# Import the new material system
from molib.pdb.materials import (
    get_alpha,
    get_ambient_color,
    get_diffuse_color,
    get_emissive_color,
    get_material_properties,
    get_shininess,
    get_specular_color,
)
from molib.pdb.molscript.colour.helpers import colour_copy_to_rgb
from molib.pdb.molscript.colour.unequal import colour_unequal
from molib.pdb.molscript.colour.values import black_colour
from OpenGL.GL import glMaterialfv
from OpenGL.raw.GL.VERSION.GL_1_0 import (
    GL_AMBIENT,
    GL_DIFFUSE,
    GL_EMISSION,
    GL_FRONT_AND_BACK,
    GL_SHININESS,
    GL_SPECULAR,
    glMaterialf,
)

from elmo.ui.state.opengl import GLState

current_shininess = 1.0
current_material_index = None


def set_material_properties(current_state: GLState):
    """Set OpenGL material properties from current state (legacy method)."""
    global current_shininess
    rgb = colour_copy_to_rgb(current_state.specularcolour)

    if (
        colour_unequal(rgb, black_colour)
        and current_shininess != current_state.shininess
    ):
        sh = 128.0 * current_state.shininess
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, sh)

        current_shininess = current_state.shininess

    glcol = [rgb.x, rgb.y, rgb.z, 1.0]
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, glcol)

    rgb = colour_copy_to_rgb(current_state.emissivecolour)
    glcol = [rgb.x, rgb.y, rgb.z, 1.0]
    glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, glcol)


def set_material_by_index(material_index: int):
    """Set OpenGL material properties using material index from the new system."""
    global current_material_index

    if current_material_index == material_index:
        return  # Already set

    try:
        material = get_material_properties(material_index)

        # Set ambient color
        ambient = get_ambient_color(material_index)
        glcol = [ambient[0], ambient[1], ambient[2], get_alpha(material_index)]
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, glcol)

        # Set diffuse color
        diffuse = get_diffuse_color(material_index)
        glcol = [diffuse[0], diffuse[1], diffuse[2], get_alpha(material_index)]
        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, glcol)

        # Set specular color
        specular = get_specular_color(material_index)
        glcol = [specular[0], specular[1], specular[2], get_alpha(material_index)]
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, glcol)

        # Set emissive color
        emissive = get_emissive_color(material_index)
        glcol = [emissive[0], emissive[1], emissive[2], get_alpha(material_index)]
        glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, glcol)

        # Set shininess
        shininess = get_shininess(material_index)
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, shininess)

        current_material_index = material_index

    except (ValueError, KeyError) as e:
        # Fallback to default material (index 0)
        set_material_by_index(0)


def set_material_properties_enhanced(
    current_state: GLState, material_index: int = None
):
    """Enhanced material setting that can use either state or material index."""
    if material_index is not None:
        set_material_by_index(material_index)
    else:
        set_material_properties(current_state)


def reset_material_state():
    """Reset the material state to force reapplication."""
    global current_material_index
    current_material_index = None

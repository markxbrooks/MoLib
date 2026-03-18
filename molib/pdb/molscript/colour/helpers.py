import os

# Headless-safe: prefer a stub when running tests/CI even if PyOpenGL is installed
_HEADLESS = bool(
    os.environ.get("ELMO_TEST_HEADLESS") or os.environ.get("PYTEST_CURRENT_TEST")
)

if _HEADLESS:  # pragma: no cover - tests run headless

    def glColor4f(r, g, b, a):  # type: ignore
        return None

else:
    # Fallback: try real OpenGL, else stub
    try:  # pragma: no cover - import side-effect only
        from OpenGL.raw.GL.VERSION.GL_1_0 import glColor4f  # type: ignore
    except Exception:  # pragma: no cover - tests run headless

        def glColor4f(r, g, b, a):  # type: ignore
            return None


from molib.core.color.color import Color

# from elmo.ui.state.opengl import GLState


def set_colour_property(rgb: Color, current_state: "GLState"):
    """Set the OpenGL colour property, safely in headless environments."""
    try:
        alpha = 1.0 - getattr(current_state, "transparency", 0.0)
        glColor4f(rgb.x, rgb.y, rgb.z, alpha)
    except Exception:
        # In headless mode or when GL context is unavailable, ignore colour set
        return None


def colour_copy_to_rgb(color):
    """Convert colour to Color object."""
    if isinstance(color, Color):
        return color
    elif isinstance(color, (tuple, list)) and len(color) >= 3:
        return Color(spec=0, x=float(color[0]), y=float(color[1]), z=float(color[2]))
    else:
        # Default to white
        return Color(spec=0, x=1.0, y=1.0, z=1.0)

"""
OpenGL rendering of protein secondary structures
"""

from typing import List

import numpy as np
from decologr import Decologr as log
from molib.calc.math.vector import Vector3
from molib.pdb.materials import (
    get_alpha,
    get_ambient_color,
    get_diffuse_color,
    get_emissive_color,
    get_material_properties,
    get_shininess,
    get_specular_color,
)
from molib.pdb.molscript.hermite import hermite_get, hermite_set
from molib.pdb.molscript.math import (
    v3_cross_product,
    v3_dot_product,
    v3_length,
    v3_middle,
    v3_normalize,
    v3_scale,
)
from molib.pdb.molscript.molscript import (
    HELIX_ALPHA,
    HELIX_BETA,
    HELIX_HERMITE_FACTOR,
    STRAND_HERMITE_FACTOR,
)
from molib.pdb.molscript.smoothing import priestle_smoothing
from OpenGL.GL import (
    glBegin,
    glColor4fv,
    glEnd,
    glGetInteger,
    glMaterialfv,
    glMultMatrixf,
    glVertexPointer,
)
from OpenGL.GLU import gluNewQuadric
from OpenGL.raw.GL._types import GL_FLOAT
from OpenGL.raw.GL.KHR.debug import GL_VERTEX_ARRAY
from OpenGL.raw.GL.VERSION.GL_1_0 import (
    GL_AMBIENT,
    GL_BLEND,
    GL_COMPILE,
    GL_DIFFUSE,
    GL_FILL,
    GL_FRONT_AND_BACK,
    GL_LIGHTING,
    GL_LINE,
    GL_LINE_STRIP,
    GL_MAX_MODELVIEW_STACK_DEPTH,
    GL_MODELVIEW_STACK_DEPTH,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_SHININESS,
    GL_SPECULAR,
    GL_SRC_ALPHA,
    GL_TRIANGLE_FAN,
    GL_TRIANGLE_STRIP,
    GL_TRIANGLES,
    glBlendFunc,
    glCallList,
    glDisable,
    glEnable,
    glEndList,
    glGenLists,
    glLineWidth,
    glMaterialf,
    glNewList,
    glNormal3f,
    glPolygonMode,
    glPopMatrix,
    glPushMatrix,
    glRotatef,
    glScalef,
    glTranslatef,
    glVertex3f,
)
from OpenGL.raw.GL.VERSION.GL_1_1 import (
    glDisableClientState,
    glDrawArrays,
    glEnableClientState,
)
from OpenGL.raw.GLU import (
    GLU_SMOOTH,
    gluCylinder,
    gluDeleteQuadric,
    gluDisk,
    gluQuadricNormals,
    gluSphere,
)


class SecondaryStructure:
    """Renders protein secondary structures in OpenGL"""

    def __init__(self):
        # MolScript parameters from state.c
        self.coil_radius = 0.3  # coilradius = 0.2
        self.helix_radius = 6.0  # cylinderradius = 2.3
        self.helix_thickness = 0.4  # helixthickness = 0.3
        self.helix_width = 2.7  # helixwidth = 2.4
        self.strand_thickness = 0.8  # strandthickness = 0.6
        self.strand_width = 2.5  # strandwidth = 2.0
        self.calpha_radius = 0.2  # For C-alpha trace

        # Segment parameters
        self.segments = 6  # segments = 6 (PostScript/Raster3D gl_mode)
        self.smooth_steps = 2  # smoothsteps = 2
        self.helix_segments = 8  # For helix cylinder segments
        self.helix_pitch = 1.5  # Helix pitch for ribbon generation

        # Sheet parameters
        self.sheet_arrow_pos = 0.7  # Position where arrow head starts (0-1)
        self.sheet_arrow_width = 1.5  # Width multiplier for arrow head

        # MolScript-style colors
        self.colors = {
            "helix": (1.0, 0.0, 0.0, 1.0),  # Red
            "sheet": (1.0, 1.0, 0.0, 1.0),  # Yellow
            "coil": (0.7, 0.7, 0.7, 1.0),  # Gray
            "calpha": (0.5, 0.5, 0.5, 1.0),  # Dark gray for C-alpha trace
            "outline": (0.0, 0.0, 0.0, 1.0),  # Black
        }

        # Line parameters
        self.line_width = 2.0  # linewidth = 2.0
        self.outline_width = 1.0
        self.outline_color = (0.0, 0.0, 0.0, 1.0)  # Black

        # Material indices for secondary structures (using new material system)
        self.material_indices = {
            "helix": 1,  # Red material
            "sheet": 2,  # Green material
            "coil": 7,  # Gray material
        }

        # Pre-calculate angles
        self.helix_angles = np.linspace(0, 2 * np.pi, self.segments)
        self.tube_angles = np.linspace(0, 2 * np.pi, self.segments)

        # Display lists
        self.cylinder_list = None
        self.sphere_list = None
        self.lists_initialized = False

    def ensure_display_lists(self):
        """Initialize display lists if not already done"""
        if not self.lists_initialized:
            try:
                # Create cylinder display list
                self.cylinder_list = glGenLists(1)
                glNewList(self.cylinder_list, GL_COMPILE)

                quadric = gluNewQuadric()
                gluQuadricNormals(quadric, GLU_SMOOTH)
                gluCylinder(quadric, 1.0, 1.0, 1.0, 8, 1)  # Unit cylinder
                gluDeleteQuadric(quadric)

                glEndList()

                # Create sphere display list
                self.sphere_list = glGenLists(1)
                glNewList(self.sphere_list, GL_COMPILE)

                quadric = gluNewQuadric()
                gluQuadricNormals(quadric, GLU_SMOOTH)
                gluSphere(quadric, 1.0, 8, 8)  # Unit sphere
                gluDeleteQuadric(quadric)

                glEndList()

                self.lists_initialized = True

            except Exception as ex:
                log.exception(f"Error initializing display lists: {ex}")

    def draw_helix(self, start: np.ndarray, end: np.ndarray, radius: float = 0.5):
        """
        Draw alpha helix as a cylinder
        """
        try:
            # Check matrix stack depth to prevent overflow
            stack_depth = glGetInteger(GL_MODELVIEW_STACK_DEPTH)
            max_depth = glGetInteger(GL_MAX_MODELVIEW_STACK_DEPTH)

            if stack_depth >= max_depth - 2:  # Leave some headroom
                log.warning("Matrix stack near overflow, skipping helix segment")
                return

            # Calculate helix axis and length
            axis = end - start
            length = np.linalg.norm(axis)

            if length < 0.001:  # Skip very short segments
                return

            # Create rotation matrix to align cylinder with helix axis
            z_axis = np.array([0, 0, 1])
            rotation_axis = np.cross(z_axis, axis)
            if np.linalg.norm(rotation_axis) > 0:
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                angle = np.arccos(np.clip(np.dot(z_axis, axis / length), -1.0, 1.0))
            else:
                rotation_axis = np.array([1, 0, 0])
                angle = 0

            glPushMatrix()
            try:
                glTranslatef(*start)
                glRotatef(np.degrees(angle), *rotation_axis)

                # Draw cylinder
                quadric = gluNewQuadric()
                gluQuadricNormals(quadric, GLU_SMOOTH)
                gluCylinder(quadric, radius, radius, length, self.helix_segments, 1)

                # Caps removed to eliminate fresnel effect
                # The cylinder ends are now open, preventing the fresnel effect
                # that was caused by the filled caps

                # # Draw caps (DISABLED)
                # gluDisk(quadric, 0, radius, self.helix_segments, 1)
                # glTranslatef(0, 0, length)
                # gluDisk(quadric, 0, radius, self.helix_segments, 1)

                gluDeleteQuadric(quadric)
            finally:
                glPopMatrix()  # Ensure matrix is popped even if error occurs

        except Exception as ex:
            log.exception(f"Error drawing helix: {ex}")

    def draw_continuous_helix(self, points: List[np.ndarray], radius: float = 0.5):
        """
        Draw a continuous helix through multiple points as a single cylinder.
        More efficient than drawing individual segments.
        """
        try:
            if len(points) < 2:
                return

            # Check matrix stack depth to prevent overflow
            stack_depth = glGetInteger(GL_MODELVIEW_STACK_DEPTH)
            max_depth = glGetInteger(GL_MAX_MODELVIEW_STACK_DEPTH)

            if stack_depth >= max_depth - 2:  # Leave some headroom
                log.warning("Matrix stack near overflow, skipping continuous helix")
                return

            # Calculate total length and direction
            start = points[0]
            end = points[-1]
            axis = end - start
            length = np.linalg.norm(axis)

            if length < 0.001:  # Skip very short segments
                return

            # Create rotation matrix to align cylinder with helix axis
            z_axis = np.array([0, 0, 1])
            rotation_axis = np.cross(z_axis, axis)
            if np.linalg.norm(rotation_axis) > 0:
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                angle = np.arccos(np.clip(np.dot(z_axis, axis / length), -1.0, 1.0))
            else:
                rotation_axis = np.array([1, 0, 0])
                angle = 0

            glPushMatrix()
            try:
                glTranslatef(*start)
                glRotatef(np.degrees(angle), *rotation_axis)

                # Draw cylinder
                quadric = gluNewQuadric()
                gluQuadricNormals(quadric, GLU_SMOOTH)
                gluCylinder(quadric, radius, radius, length, self.helix_segments, 1)

                # Caps removed to eliminate fresnel effect
                # The cylinder ends are now open, preventing the fresnel effect
                # that was caused by the filled caps

                # # Draw caps (DISABLED)
                # gluDisk(quadric, 0, radius, self.helix_segments, 1)
                # glTranslatef(0, 0, length)
                # gluDisk(quadric, 0, radius, self.helix_segments, 1)

                gluDeleteQuadric(quadric)
            finally:
                glPopMatrix()  # Ensure matrix is popped even if error occurs

        except Exception as ex:
            log.exception(f"Error drawing continuous helix: {ex}")

    def draw_sheet(self, points: List[np.ndarray], normal: np.ndarray):
        """
        Draw beta sheet as a flat ribbon with proper vertex generation
        """
        try:
            if len(points) < 2:
                return

            # Generate ribbon vertices
            vertices = []

            # Calculate sheet direction and normalize
            direction = points[-1] - points[0]
            length = np.linalg.norm(direction)
            if length == 0:
                return
            direction = direction / length

            # Calculate perpendicular vector for ribbon width
            side = np.cross(direction, normal)
            if np.linalg.norm(side) == 0:
                # If direction and normal are parallel, use a default perpendicular
                side = np.array([1.0, 0.0, 0.0])
            side = side / np.linalg.norm(side) * (self.strand_width / 2)

            # Generate vertices for ribbon
            for i, point in enumerate(points):
                # Calculate progress along the sheet (0 to 1)
                progress = i / (len(points) - 1) if len(points) > 1 else 0

                # Calculate width - make it narrower at the ends for arrow effect
                if progress < 0.1 or progress > 0.9:
                    # Taper the ends
                    width_factor = 0.3 + 0.7 * (1 - abs(progress - 0.5) * 2)
                else:
                    width_factor = 1.0

                current_side = side * width_factor

                # Add vertices for both edges of the ribbon
                vertices.append(point + current_side)
                vertices.append(point - current_side)

            # Convert to numpy array for OpenGL
            if len(vertices) == 0:
                return

            vertices_array = np.array(vertices, dtype=np.float32)

            # Ensure we have enough vertices for a triangle strip (at least 3)
            if len(vertices_array) < 3:
                return

            # Draw the ribbon using vertex arrays
            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, vertices_array)
            glDrawArrays(GL_TRIANGLE_STRIP, 0, len(vertices_array))
            glDisableClientState(GL_VERTEX_ARRAY)

        except Exception as ex:
            log.exception(f"Error drawing sheet: {ex}")

    def draw_turn(self, points: List[np.ndarray], radius: float = 0.2):
        """
        Draw turn/loop as a smooth tube
        """
        try:
            if len(points) < 2:
                return

            # Draw smooth tube segments
            for i in range(len(points) - 1):
                start = points[i]
                end = points[i + 1]

                # Draw cylinder segment
                self.draw_cylinder_segment(start, end, radius)

                # Draw sphere at joint
                if i < len(points) - 2:
                    self.draw_sphere_at_point(points[i + 1], radius)

        except Exception as ex:
            log.exception(f"Error drawing turn: {ex}")

    def draw_cylinder_segment(self, start: np.ndarray, end: np.ndarray, radius: float):
        """Draw a cylinder segment between two points"""
        try:
            # Calculate segment properties
            axis = end - start
            length = np.linalg.norm(axis)
            if length == 0:
                return

            # Create rotation matrix
            z_axis = np.array([0, 0, 1])
            rotation_axis = np.cross(z_axis, axis)
            if np.linalg.norm(rotation_axis) > 0:
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                angle = np.arccos(np.dot(z_axis, axis / length))
            else:
                rotation_axis = np.array([1, 0, 0])
                angle = 0

            # Draw cylinder
            glPushMatrix()
            glTranslatef(*start)
            glRotatef(np.degrees(angle), *rotation_axis)

            quadric = gluNewQuadric()
            gluQuadricNormals(quadric, GLU_SMOOTH)
            gluCylinder(quadric, radius, radius, length, 8, 1)
            gluDeleteQuadric(quadric)

            glPopMatrix()

        except Exception as ex:
            log.exception(f"Error drawing cylinder segment: {ex}")

    def draw_sphere_at_point(self, point: np.ndarray, radius: float):
        """Draw a sphere at a point"""
        try:
            glPushMatrix()
            glTranslatef(*point)

            quadric = gluNewQuadric()
            gluQuadricNormals(quadric, GLU_SMOOTH)
            gluSphere(quadric, radius, 8, 8)
            gluDeleteQuadric(quadric)

            glPopMatrix()

        except Exception as ex:
            log.exception(f"Error drawing sphere: {ex}")

    def _setup_material(self, material_type):
        """Set up OpenGL material properties using the new material system"""
        material_index = self.material_indices.get(material_type, 1)  # Default to red

        # Get material properties from the new system
        ambient = get_ambient_color(material_index)
        diffuse = get_diffuse_color(material_index)
        specular = get_specular_color(material_index)
        shininess = get_shininess(material_index)
        alpha = get_alpha(material_index)

        # Set OpenGL material properties
        glcol = [ambient[0], ambient[1], ambient[2], alpha]
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, glcol)

        glcol = [diffuse[0], diffuse[1], diffuse[2], alpha]
        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, glcol)

        glcol = [specular[0], specular[1], specular[2], alpha]
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, glcol)

        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, shininess)

    def _draw_with_outline(self, draw_func, *args):
        """Draw an object with outline"""
        # Draw filled object
        glEnable(GL_LIGHTING)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        draw_func(*args)

        # Draw outline
        glDisable(GL_LIGHTING)
        glLineWidth(self.outline_width)
        glColor4fv(self.outline_color)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        draw_func(*args)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glEnable(GL_LIGHTING)

    def draw_helix_cartoon(self, start: np.ndarray, end: np.ndarray):
        """Draw MolScript-style helix"""
        try:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

            # Set material properties
            self._setup_material("helix")

            # Draw helix with outline
            self._draw_with_outline(
                self._draw_helix_ribbon, start, end, self.helix_radius
            )

        except Exception as ex:
            log.exception(f"Error drawing MolScript helix: {ex}")

    def draw_sheet_cartoon(self, points: List[np.ndarray], normal: np.ndarray):
        """Draw MolScript-style beta sheet"""
        try:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

            # Set material properties
            self._setup_material("sheet")

            # Draw sheet with outline
            self._draw_with_outline(self._draw_sheet_arrow, points, normal)

        except Exception as ex:
            log.exception(f"Error drawing MolScript sheet: {ex}")

    def draw_coil_cartoon(self, points: List[np.ndarray]):
        """Draw MolScript-style coil"""
        try:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

            # Set material properties
            self._setup_material("coil")

            # Generate smooth spline
            spline_points = self._catmull_rom_spline(points, segments=self.smooth_steps)

            # Draw coil with outline
            self._draw_with_outline(
                self._draw_smooth_tube, spline_points, self.coil_radius
            )

        except Exception as ex:
            log.exception(f"Error drawing MolScript coil: {ex}")

    def _draw_helix_ribbon(self, start, end, radius):
        """Draw detailed helix ribbon"""
        try:
            axis = end - start
            length = np.linalg.norm(axis)
            if length == 0:
                return

            # Calculate helix parameters
            turns = length / self.helix_pitch
            points_per_turn = 16
            total_points = int(turns * points_per_turn)

            # Generate helix points
            t = np.linspace(0, turns * 2 * np.pi, total_points)
            vertices = np.zeros((total_points * 2, 3))

            # Calculate transforms
            z_axis = np.array([0, 0, 1])
            rotation_axis = np.cross(z_axis, axis)
            if np.linalg.norm(rotation_axis) > 0:
                angle = np.arccos(np.dot(z_axis, axis / length))
                rotation_matrix = self._rotation_matrix(rotation_axis, angle)
            else:
                rotation_matrix = np.eye(3)

            # Generate ribbon vertices
            radius_vector = np.array([radius, 0, 0])
            ribbon_width = radius * 0.4

            for i, theta in enumerate(t):
                # Calculate helix point
                rot_point = self._rotate_z(radius_vector, theta)
                rot_point[2] = (theta / (2 * np.pi)) * (length / turns)
                point = start + np.dot(rotation_matrix, rot_point)

                # Calculate ribbon edges
                if i > 0:
                    tangent = point - prev_point
                    normal = np.cross(tangent, axis)
                    if np.linalg.norm(normal) > 0:
                        normal = normal / np.linalg.norm(normal)
                        vertices[i * 2] = prev_point + ribbon_width * normal
                        vertices[i * 2 + 1] = prev_point - ribbon_width * normal

                prev_point: object = point

            # Draw ribbon
            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, vertices)
            glDrawArrays(GL_TRIANGLE_STRIP, 0, len(vertices))
            glDisableClientState(GL_VERTEX_ARRAY)

        except Exception as ex:
            log.exception(f"Error drawing detailed helix ribbon: {ex}")

    def _draw_sheet_arrow(self, points, normal):
        """Draw detailed sheet arrow"""
        try:
            if len(points) < 2:
                return

            # Calculate sheet direction and length
            direction = points[-1] - points[0]
            length = np.linalg.norm(direction)
            if length == 0:
                return

            direction = direction / length
            side = np.cross(direction, normal)
            if np.linalg.norm(side) == 0:
                return

            side = side / np.linalg.norm(side) * self.sheet_width

            # Generate vertices
            vertices = []
            for i, pos in enumerate(points):
                progress = np.linalg.norm(pos - points[0]) / length

                # Calculate width based on position
                if progress < self.sheet_arrow_pos:
                    width = side * 0.5  # Regular width
                else:
                    # Expand for arrow head
                    t = (progress - self.sheet_arrow_pos) / (1 - self.sheet_arrow_pos)
                    width = side * (0.5 + t * self.sheet_arrow_width)

                # Add vertices for both edges
                vertices.append(pos + width)
                vertices.append(pos - width)

            # Convert to numpy array for OpenGL
            if len(vertices) == 0:
                return

            vertices_array = np.array(vertices, dtype=np.float32)

            # Ensure we have enough vertices for a triangle strip (at least 3)
            if len(vertices_array) < 3:
                return

            # Draw sheet
            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, vertices_array)
            glDrawArrays(GL_TRIANGLE_STRIP, 0, len(vertices_array))
            glDisableClientState(GL_VERTEX_ARRAY)

        except Exception as ex:
            log.exception(f"Error drawing detailed sheet arrow: {ex}")

    def _catmull_rom_spline(
        self, points: List[np.ndarray], segments: int = 10
    ) -> List[np.ndarray]:
        """Generate smooth spline through points"""
        try:
            if len(points) < 2:
                return points

            # Add extra control points at ends
            control_points = [points[0]] + points + [points[-1]]

            # Generate spline points
            spline_points = []
            for i in range(1, len(control_points) - 2):
                for t in np.linspace(0, 1, segments):
                    p = self._catmull_rom_point(
                        control_points[i - 1],
                        control_points[i],
                        control_points[i + 1],
                        control_points[i + 2],
                        t,
                    )
                    spline_points.append(p)

            return spline_points

        except Exception as ex:
            log.warning(f"Error generating spline: {ex}")
            return points

    def _catmull_rom_point(self, p0, p1, p2, p3, t):
        """Calculate point on Catmull-Rom spline"""
        t2 = t * t
        t3 = t2 * t

        return 0.5 * (
            (2 * p1)
            + (-p0 + p2) * t
            + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
            + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
        )

    def _get_perpendicular(self, vector: np.ndarray) -> np.ndarray:
        """Get a vector perpendicular to the input vector"""
        try:
            # Try cross product with x-axis first
            perp = np.cross(vector, [1, 0, 0])
            if np.linalg.norm(perp) < 1e-6:
                # If parallel to x-axis, use y-axis
                perp = np.cross(vector, [0, 1, 0])
            return perp / np.linalg.norm(perp)
        except Exception as ex:
            log.warning(f"Error getting perpendicular vector: {ex}")
            return np.array([1, 0, 0])

    def _parallel_transport(
        self, normal: np.ndarray, new_direction: np.ndarray
    ) -> np.ndarray:
        """Transport normal vector along curve using parallel transport"""
        try:
            # Normalize new direction
            new_direction = new_direction / np.linalg.norm(new_direction)

            # Project normal onto plane perpendicular to new direction
            proj = normal - np.dot(normal, new_direction) * new_direction

            # Normalize result
            if np.linalg.norm(proj) > 1e-6:
                return proj / np.linalg.norm(proj)
            else:
                return self._get_perpendicular(new_direction)

        except Exception as ex:
            log.warning(f"Error in parallel transport: {ex}")
            return normal

    def _rotation_matrix(self, axis: np.ndarray, angle: float) -> np.ndarray:
        """Create rotation matrix for rotation around axis by angle (radians)"""
        try:
            # Normalize axis
            axis = axis / np.linalg.norm(axis)

            # Rodriguez rotation formula
            c = np.cos(angle)
            s = np.sin(angle)
            t = 1 - c
            x, y, z = axis

            return np.array(
                [
                    [t * x * x + c, t * x * y - z * s, t * x * z + y * s],
                    [t * x * y + z * s, t * y * y + c, t * y * z - x * s],
                    [t * x * z - y * s, t * y * z + x * s, t * z * z + c],
                ]
            )

        except Exception as ex:
            log.warning(f"Error creating rotation matrix: {ex}")
            return np.eye(3)

    def _rotate_z(self, point: np.ndarray, angle: float) -> np.ndarray:
        """Rotate point around Z axis by angle (radians)"""
        try:
            c = np.cos(angle)
            s = np.sin(angle)
            return np.array(
                [point[0] * c - point[1] * s, point[0] * s + point[1] * c, point[2]]
            )
        except Exception as ex:
            log.warning(f"Error rotating point: {ex}")
            return point

    def _draw_smooth_tube(self, points: List[np.ndarray], radius: float):
        """Draw smooth tube using display lists"""
        try:
            # Ensure display lists are _initialized
            self.ensure_display_lists()

            if not self.lists_initialized:
                # Fallback to direct drawing if lists failed to initialize
                self._draw_smooth_tube_direct(points, radius)
                return

            # Calculate transform
            direction = points[-1] - points[0]
            length = np.linalg.norm(direction)
            if length == 0:
                return

            # Check matrix stack depth
            stack_depth = glGetInteger(GL_MODELVIEW_STACK_DEPTH)
            max_depth = glGetInteger(GL_MAX_MODELVIEW_STACK_DEPTH)

            if stack_depth >= max_depth - 1:
                # If stack is nearly full, use direct drawing
                self._draw_smooth_tube_direct(points, radius)
                return

            # Use display list for cylinder
            glPushMatrix()
            try:
                # Position and scale
                glTranslatef(*points[0])

                # Create 4x4 transformation matrix
                transform_matrix = np.eye(4)  # Start with identity matrix

                # Set scale and direction in upper 3x3
                direction_normalized = direction / length
                transform_matrix[:3, :3] = (
                    np.array(
                        [
                            [direction_normalized[0], 0, 0],
                            [0, direction_normalized[1], 0],
                            [0, 0, direction_normalized[2]],
                        ]
                    )
                    * radius
                )

                # Set translation in last column
                transform_matrix[:3, 3] = points[0]

                # Apply transformation
                glMultMatrixf(transform_matrix.T)

                # Draw unit cylinder
                glScalef(1.0, 1.0, length)
                glCallList(self.cylinder_list)
            finally:
                glPopMatrix()  # Ensure matrix is popped even if error occurs

        except Exception as ex:
            log.exception(f"Error drawing smooth tube: {ex}")
            # Fallback to direct drawing
            self._draw_smooth_tube_direct(points, radius)

    def _draw_smooth_tube_direct(self, points, radius):
        """Fallback method for direct tube drawing without display lists"""
        try:
            if len(points) < 2:
                return

            # Calculate tube direction and orientation
            direction = points[-1] - points[0]
            length = np.linalg.norm(direction)
            if length == 0:
                return

            direction = direction / length

            # Get perpendicular vectors for tube cross-section
            normal = self._get_perpendicular(direction)
            binormal = np.cross(direction, normal)
            binormal = binormal / np.linalg.norm(binormal)

            # Calculate vertices for tube cross-section
            vertices = []
            for theta in self.tube_angles:
                # Create circle points in normal-binormal plane
                circle_point = (
                    normal * np.cos(theta) + binormal * np.sin(theta)
                ) * radius
                vertices.append(circle_point)

            # Draw tube segments
            glBegin(GL_TRIANGLE_STRIP)
            for i in range(len(points) - 1):
                start = points[i]
                end = points[i + 1]

                for j in range(len(vertices)):
                    v1 = vertices[j]
                    v2 = vertices[(j + 1) % len(vertices)]

                    # Add vertices for segment
                    glVertex3f(*(start + v1))
                    glVertex3f(*(end + v1))
                    glVertex3f(*(start + v2))
                    glVertex3f(*(end + v2))

            glEnd()

        except Exception as ex:
            log.exception(f"Error in direct tube drawing: {ex}")

    def draw_calpha_trace(self, points: List[np.ndarray]):
        """Draw MolScript-style C-alpha trace"""
        try:
            if len(points) < 2:
                return

            # Draw connecting lines
            glDisable(GL_LIGHTING)
            glLineWidth(1.0)  # MolScript uses thin lines
            glColor4fv(self.colors["calpha"])

            # Draw C-alpha backbone
            glBegin(GL_LINE_STRIP)
            for point in points:
                glVertex3f(*point)
            glEnd()

            # Draw small spheres at C-alpha positions
            glEnable(GL_LIGHTING)

            # Set material properties for spheres
            glMaterialfv(
                GL_FRONT_AND_BACK, GL_AMBIENT, [x * 0.2 for x in self.colors["calpha"]]
            )
            glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, self.colors["calpha"])
            glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (0.2, 0.2, 0.2, 1.0))
            glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 10.0)

            # Draw small sphere at each C-alpha position_array
            for point in points:
                self.draw_sphere_at_point(point, self.calpha_radius)

        except Exception as ex:
            log.exception(f"Error drawing MolScript C-alpha trace: {ex}")

    def draw_helix_graphics(self, points: List[np.ndarray]) -> None:
        """
        Draw helix using the graphics module algorithms with proper OpenGL rendering.

        Args:
            points: List of Cα coordinates for the helix
        """
        try:
            if len(points) < 3:
                return

            # Convert numpy arrays to Vector3 objects
            vector_points = []
            for point in points:
                v = Vector3()
                v.x = float(point[0])
                v.y = float(point[1])
                v.z = float(point[2])
                vector_points.append(v)

            # Calculate helix axes and tangents using graphics module algorithm
            axes = []
            tangents = []

            for i in range(1, len(vector_points) - 1):
                # Helix direction vector
                cvec = Vector3()
                cvec.x = vector_points[i + 1].x - vector_points[i - 1].x
                cvec.y = vector_points[i + 1].y - vector_points[i - 1].y
                cvec.z = vector_points[i + 1].z - vector_points[i - 1].z
                v3_normalize(cvec)

                # Normal vector for plane
                vec1 = Vector3()
                vec1.x = vector_points[i].x - vector_points[i - 1].x
                vec1.y = vector_points[i].y - vector_points[i - 1].y
                vec1.z = vector_points[i].z - vector_points[i - 1].z

                vec2 = Vector3()
                vec2.x = vector_points[i + 1].x - vector_points[i].x
                vec2.y = vector_points[i + 1].y - vector_points[i].y
                vec2.z = vector_points[i + 1].z - vector_points[i].z

                rvec = Vector3()
                v3_cross_product(rvec, vec1, vec2)
                v3_normalize(rvec)

                # Helix axis
                axis = Vector3()
                axis.x = np.cos(HELIX_ALPHA) * rvec.x + np.sin(HELIX_ALPHA) * cvec.x
                axis.y = np.cos(HELIX_ALPHA) * rvec.y + np.sin(HELIX_ALPHA) * cvec.y
                axis.z = np.cos(HELIX_ALPHA) * rvec.z + np.sin(HELIX_ALPHA) * cvec.z
                axes.append(axis)

                # Helix tangent
                tangent = Vector3()
                tangent.x = np.cos(HELIX_BETA) * cvec.x + np.sin(HELIX_BETA) * rvec.x
                tangent.y = np.cos(HELIX_BETA) * cvec.y + np.sin(HELIX_BETA) * rvec.y
                tangent.z = np.cos(HELIX_BETA) * cvec.z + np.sin(HELIX_BETA) * rvec.z
                v3_scale(tangent, HELIX_HERMITE_FACTOR)
                tangents.append(tangent)

            # Add terminal axes and tangents
            if axes:
                axes.insert(0, axes[0])
                axes.append(axes[-1])
                tangents.insert(0, tangents[0])
                tangents.append(tangents[-1])

            # Set up material properties
            glEnable(GL_LIGHTING)
            self._setup_material("helix")

            # Draw helix using Hermite curves
            segments = self.helix_segments
            radius = self.helix_radius

            for i in range(len(vector_points) - 1):
                # Set up Hermite curve
                hermite_set(
                    vector_points[i], vector_points[i + 1], tangents[i], tangents[i + 1]
                )

                # Generate points along the curve
                for segment in range(segments + 1):
                    t = segment / segments if segments > 0 else 0.0
                    point = hermite_get(t)

                    # Calculate helix width (tapered at ends)
                    if i == 0:
                        width_factor = 0.5 + 0.5 * t
                    elif i == len(vector_points) - 2:
                        width_factor = 0.5 + 0.5 * (1 - t)
                    else:
                        width_factor = 1.0

                    current_radius = radius * width_factor

                    # Interpolate axis
                    if i < len(axes) - 1:
                        axis = Vector3()
                        axis.x = (1 - t) * axes[i].x + t * axes[i + 1].x
                        axis.y = (1 - t) * axes[i].y + t * axes[i + 1].y
                        axis.z = (1 - t) * axes[i].z + t * axes[i + 1].z
                        v3_normalize(axis)
                    else:
                        axis = axes[i] if i < len(axes) else axes[-1]

                    # Draw helix segment as cylinder
                    self._draw_helix_segment(point, axis, current_radius)

        except Exception as ex:
            log.exception(f"Error drawing helix with graphics module: {ex}")

    def draw_strand_graphics(self, points: List[np.ndarray]) -> None:
        """
        Draw beta sheet using the graphics module algorithms with proper OpenGL rendering.

        Args:
            points: List of Cα coordinates for the strand
        """
        try:
            if len(points) < 3:
                return

            # Convert numpy arrays to Vector3 objects
            vector_points = []
            for point in points:
                v = Vector3()
                v.x = float(point[0])
                v.y = float(point[1])
                v.z = float(point[2])
                vector_points.append(v)

            # Calculate normals for the strand using graphics module algorithm
            normals = []
            for i in range(1, len(vector_points) - 1):
                # Middle point
                middle = Vector3()
                v3_middle(middle, vector_points[i - 1], vector_points[i + 1])

                # Normal vector
                normal = Vector3()
                normal.x = vector_points[i].x - middle.x
                normal.y = vector_points[i].y - middle.y
                normal.z = vector_points[i].z - middle.z
                v3_normalize(normal)
                normals.append(normal)

            # Add terminal normals
            if normals:
                normals.insert(0, normals[0])
                normals.append(normals[-1])

            # Apply smoothing
            smoothed_points = priestle_smoothing(vector_points)

            # Set up material properties
            glEnable(GL_LIGHTING)
            self._setup_material("sheet")

            # Draw strand using Hermite curves
            segments = self.segments
            width = self.strand_width / 2.0
            thickness = self.strand_thickness / 2.0

            for i in range(len(smoothed_points) - 1):
                # Calculate direction vectors
                if i == 0:
                    dir1 = Vector3()
                    dir1.x = smoothed_points[1].x - smoothed_points[0].x
                    dir1.y = smoothed_points[1].y - smoothed_points[0].y
                    dir1.z = smoothed_points[1].z - smoothed_points[0].z
                else:
                    dir1 = Vector3()
                    dir1.x = smoothed_points[i].x - smoothed_points[i - 1].x
                    dir1.y = smoothed_points[i].y - smoothed_points[i - 1].y
                    dir1.z = smoothed_points[i].z - smoothed_points[i - 1].z

                if i == len(smoothed_points) - 2:
                    dir2 = Vector3()
                    dir2.x = smoothed_points[i + 1].x - smoothed_points[i].x
                    dir2.y = smoothed_points[i + 1].y - smoothed_points[i].y
                    dir2.z = smoothed_points[i + 1].z - smoothed_points[i].z
                else:
                    dir2 = Vector3()
                    dir2.x = smoothed_points[i + 2].x - smoothed_points[i].x
                    dir2.y = smoothed_points[i + 2].y - smoothed_points[i].y
                    dir2.z = smoothed_points[i + 2].z - smoothed_points[i].z

                v3_normalize(dir1)
                v3_normalize(dir2)

                # Calculate Hermite tangent vectors
                vec1 = Vector3()
                vec1.x = dir1.x * STRAND_HERMITE_FACTOR
                vec1.y = dir1.y * STRAND_HERMITE_FACTOR
                vec1.z = dir1.z * STRAND_HERMITE_FACTOR

                vec2 = Vector3()
                vec2.x = dir2.x * STRAND_HERMITE_FACTOR
                vec2.y = dir2.y * STRAND_HERMITE_FACTOR
                vec2.z = dir2.z * STRAND_HERMITE_FACTOR

                # Set up Hermite curve
                hermite_set(smoothed_points[i], smoothed_points[i + 1], vec1, vec2)

                # Generate points along the curve
                for segment in range(segments + 1):
                    t = segment / segments if segments > 0 else 0.0
                    point = hermite_get(t)

                    # Interpolate direction and normal
                    dir = Vector3()
                    dir.x = (1 - t) * dir1.x + t * dir2.x
                    dir.y = (1 - t) * dir1.y + t * dir2.y
                    dir.z = (1 - t) * dir1.z + t * dir2.z
                    v3_normalize(dir)

                    normal = Vector3()
                    if i < len(normals) - 1:
                        normal.x = (1 - t) * normals[i].x + t * normals[i + 1].x
                        normal.y = (1 - t) * normals[i].y + t * normals[i + 1].y
                        normal.z = (1 - t) * normals[i].z + t * normals[i + 1].z
                    else:
                        normal = normals[i] if i < len(normals) else normals[-1]
                    v3_normalize(normal)

                    # Calculate side vector
                    side = Vector3()
                    v3_cross_product(side, normal, dir)
                    v3_normalize(side)

                    # Draw strand segment as ribbon
                    self._draw_strand_segment(point, side, normal, width, thickness)

        except Exception as ex:
            log.exception(f"Error drawing strand with graphics module: {ex}")

    def _draw_helix_segment(
        self, center: Vector3, axis: Vector3, radius: float
    ) -> None:
        """Draw a single helix segment as a triangle-based cylinder"""
        try:
            # Calculate segment length (approximate)
            length = 0.1  # Small segment length for smooth appearance

            # Create rotation matrix to align cylinder with axis
            z_axis = Vector3()
            z_axis.x = 0.0
            z_axis.y = 0.0
            z_axis.z = 1.0

            rotation_axis = Vector3()
            v3_cross_product(rotation_axis, z_axis, axis)
            if v3_length(rotation_axis) > 0:
                v3_normalize(rotation_axis)
                angle = np.arccos(np.clip(v3_dot_product(z_axis, axis), -1.0, 1.0))
            else:
                rotation_axis.x = 1.0
                rotation_axis.y = 0.0
                rotation_axis.z = 0.0
                angle = 0.0

            glPushMatrix()
            try:
                glTranslatef(center.x, center.y, center.z)
                glRotatef(
                    np.degrees(angle), rotation_axis.x, rotation_axis.y, rotation_axis.z
                )

                # Draw cylinder using triangles
                self._draw_cylinder_triangles(radius, length, self.helix_segments)
            finally:
                glPopMatrix()

        except Exception as ex:
            log.exception(f"Error drawing helix segment: {ex}")

    def _draw_cylinder_triangles(
        self, radius: float, height: float, segments: int
    ) -> None:
        """Draw a cylinder using GL_TRIANGLES for better rendering"""
        try:
            # Generate vertices for top and bottom circles
            top_vertices = []
            bottom_vertices = []
            normals = []

            for i in range(segments + 1):
                angle = 2.0 * np.pi * i / segments
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)

                # Top circle
                top_vertices.append([radius * cos_a, radius * sin_a, height])
                # Bottom circle
                bottom_vertices.append([radius * cos_a, radius * sin_a, 0.0])
                # Normal (pointing outward)
                normals.append([cos_a, sin_a, 0.0])

            # Draw cylinder sides using triangle strips
            glBegin(GL_TRIANGLE_STRIP)
            for i in range(segments + 1):
                # Normal for this segment
                glNormal3f(normals[i][0], normals[i][1], normals[i][2])
                # Bottom vertex
                glVertex3f(
                    bottom_vertices[i][0], bottom_vertices[i][1], bottom_vertices[i][2]
                )
                # Top vertex
                glVertex3f(top_vertices[i][0], top_vertices[i][1], top_vertices[i][2])
            glEnd()

            # Caps removed to eliminate fresnel effect
            # The cylinder ends are now open, preventing the fresnel effect
            # that was caused by the filled caps

            # # Draw top cap using triangle fan (DISABLED)
            # glBegin(GL_TRIANGLE_FAN)
            # glNormal3f(0.0, 0.0, 1.0)  # Normal pointing up
            # glVertex3f(0.0, 0.0, height)  # Center point
            # for i in range(segments + 1):
            #     glVertex3f(top_vertices[i][0], top_vertices[i][1], top_vertices[i][2])
            # glEnd()

            # # Draw bottom cap using triangle fan (DISABLED)
            # glBegin(GL_TRIANGLE_FAN)
            # glNormal3f(0.0, 0.0, -1.0)  # Normal pointing down
            # glVertex3f(0.0, 0.0, 0.0)  # Center point
            # for i in range(segments + 1):
            #     glVertex3f(
            #         bottom_vertices[segments - i][0],
            #         bottom_vertices[segments - i][1],
            #         bottom_vertices[segments - i][2],
            #     )
            # glEnd()

        except Exception as ex:
            log.exception(f"Error drawing cylinder triangles: {ex}")

    def _draw_strand_segment(
        self,
        center: Vector3,
        side: Vector3,
        normal: Vector3,
        width: float,
        thickness: float,
    ) -> None:
        """Draw a single strand segment as a triangle-based ribbon"""
        try:
            # Calculate ribbon corners
            p1 = Vector3()
            p1.x = center.x + width * side.x + thickness * normal.x
            p1.y = center.y + width * side.y + thickness * normal.y
            p1.z = center.z + width * side.z + thickness * normal.z

            p2 = Vector3()
            p2.x = center.x + width * side.x - thickness * normal.x
            p2.y = center.y + width * side.y - thickness * normal.y
            p2.z = center.z + width * side.z - thickness * normal.z

            p3 = Vector3()
            p3.x = center.x - width * side.x - thickness * normal.x
            p3.y = center.y - width * side.y - thickness * normal.y
            p3.z = center.z - width * side.z - thickness * normal.z

            p4 = Vector3()
            p4.x = center.x - width * side.x + thickness * normal.x
            p4.y = center.y - width * side.y + thickness * normal.y
            p4.z = center.z - width * side.z + thickness * normal.z

            # Draw ribbon as two triangles for better rendering
            glBegin(GL_TRIANGLES)
            glNormal3f(normal.x, normal.y, normal.z)

            # First triangle: p1, p2, p3
            glVertex3f(p1.x, p1.y, p1.z)
            glVertex3f(p2.x, p2.y, p2.z)
            glVertex3f(p3.x, p3.y, p3.z)

            # Second triangle: p1, p3, p4
            glVertex3f(p1.x, p1.y, p1.z)
            glVertex3f(p3.x, p3.y, p3.z)
            glVertex3f(p4.x, p4.y, p4.z)

            glEnd()

        except Exception as ex:
            log.exception(f"Error drawing strand segment: {ex}")

    def draw_coil_triangles(self, points: List[np.ndarray]) -> None:
        """Draw coil as a triangle strip for better rendering"""
        try:
            if len(points) < 2:
                return

            # Set up material properties for coil
            glEnable(GL_LIGHTING)
            self._setup_material("coil")

            # Generate vertices for tube along the path
            vertices = []
            normals = []

            radius = self.coil_radius
            segments = 8  # Number of segments around the tube

            for i, point in enumerate(points):
                # Calculate direction vector
                if i == 0:
                    # First point - use direction to next point
                    direction = points[1] - points[0]
                elif i == len(points) - 1:
                    # Last point - use direction from previous point
                    direction = points[i] - points[i - 1]
                else:
                    # Middle point - average direction
                    direction = (points[i + 1] - points[i - 1]) / 2.0

                # Normalize direction
                length = np.linalg.norm(direction)
                if length > 0:
                    direction = direction / length
                else:
                    direction = np.array([1.0, 0.0, 0.0])

                # Calculate perpendicular vectors for tube cross-section
                if abs(direction[2]) < 0.9:
                    # Not parallel to Z axis
                    up = np.array([0.0, 0.0, 1.0])
                else:
                    # Parallel to Z axis, use X axis
                    up = np.array([1.0, 0.0, 0.0])

                # Cross product to get perpendicular
                right = np.cross(direction, up)
                right = right / np.linalg.norm(right)

                # Cross product again to get true perpendicular
                up = np.cross(right, direction)
                up = up / np.linalg.norm(up)

                # Generate circle of vertices around the point
                for j in range(segments + 1):
                    angle = 2.0 * np.pi * j / segments
                    cos_a = np.cos(angle)
                    sin_a = np.sin(angle)

                    # Calculate vertex position
                    vertex = point + radius * (cos_a * right + sin_a * up)
                    vertices.append(vertex)

                    # Calculate normal (pointing outward from tube)
                    normal = cos_a * right + sin_a * up
                    normals.append(normal)

            # Draw tube using triangle strips
            glBegin(GL_TRIANGLE_STRIP)
            for i in range(len(points) - 1):
                for j in range(segments + 1):
                    # Current segment
                    idx1 = i * (segments + 1) + j
                    # Next segment
                    idx2 = (i + 1) * (segments + 1) + j

                    # Normal for this vertex
                    glNormal3f(normals[idx1][0], normals[idx1][1], normals[idx1][2])
                    # Vertex
                    glVertex3f(vertices[idx1][0], vertices[idx1][1], vertices[idx1][2])

                    # Normal for next vertex
                    glNormal3f(normals[idx2][0], normals[idx2][1], normals[idx2][2])
                    # Next vertex
                    glVertex3f(vertices[idx2][0], vertices[idx2][1], vertices[idx2][2])
            glEnd()

        except Exception as ex:
            log.exception(f"Error drawing coil triangles: {ex}")

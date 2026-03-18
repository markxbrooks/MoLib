point_vert = """
attribute vec3 colour;
attribute float group;
uniform float show_only;
uniform float r2_max;
uniform float r2_min;
uniform float size;
varying vec3 vcolor;
void main() {
  vcolor = colour;
  float r2 = dot(position, position);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  if (r2 < r2_min || r2 >= r2_max || (show_only != -2.0 && show_only != group))
    gl_Position.x = 2.0;
  gl_PointSize = size;
}
"""
fog_pars_fragment = ""
fog_end_fragment = ""
round_point_frag = (
    """
"""
    + fog_pars_fragment
    + """
varying vec3 vcolor;
void main() {
  // not sure how reliable is such rounding of points
  vec2 diff = gl_PointCoord - vec2(0.5, 0.5);
  float dist_sq = 4.0 * dot(diff, diff);
  if (dist_sq >= 1.0) discard;
  float alpha = 1.0 - dist_sq * dist_sq * dist_sq;
  gl_FragColor = vec4(vcolor, alpha);
"""
    + fog_end_fragment
    + """
}
"""
)
square_point_frag = (
    """
"""
    + fog_pars_fragment
    + """
varying vec3 vcolor;
void main() {
  gl_FragColor = vec4(vcolor, 1.0);
"""
    + fog_end_fragment
    + """
}
"""
)

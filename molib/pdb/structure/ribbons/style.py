from dataclasses import dataclass
# B-spline ribbon effective half-width is 0.5 * get_width(ss) * width (guide-point factor).
# Legacy ribbons use constant half-width 0.5. To match, use width so 0.5*0.6*width ≈ 0.5 → width ≈ 1.67.
RIBBON_WIDTH_SCALE = 2.7


@dataclass
class RibbonStyleConfig:
    """Ribbon cross-section style and B-spline width scale (context-based API)."""

    style: str
    width_scale: float = RIBBON_WIDTH_SCALE
    use_ribbons_style: bool = True
    #: Append a Ribbons-style arrowhead at the C-terminus when geometry exposes ribbon edges.
    has_arrow: bool = False

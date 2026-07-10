from math import fabs

from molib.core.color.color import Color
from molib.pdb.molscript.colour.values import COLOUR_GREY


def colour_unequal(c1: Color, c2: Color) -> bool:
    """
    Are the two colours unequal? Any difference in colour specification
    or components is tested.
    """
    assert c1 is not None
    assert c2 is not None

    if c1.spec != c2.spec:
        return True
    if fabs(c1.r - c2.r) >= 0.0005:
        return True
    if c1.spec != COLOUR_GREY:
        if fabs(c1.g - c2.g) >= 0.0005:
            return True
        if fabs(c1.b - c2.b) >= 0.0005:
            return True
    return False

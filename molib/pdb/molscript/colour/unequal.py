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
    if fabs(c1.x - c2.x) >= 0.0005:
        return True
    if c1.spec != COLOUR_GREY:
        if fabs(c1.y - c2.y) >= 0.0005:
            return True
        if fabs(c1.z - c2.z) >= 0.0005:
            return True
    return False

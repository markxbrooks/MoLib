from elmo.gl.molscript.segment.sided import SidedSegment
from molib.calc.math.vector import Vector3
from molib.pdb.molscript.math import v3_difference, v3_normalize


def calculate_side_vector(ss_last: SidedSegment) -> Vector3:
    """Side vector: from p1 to p4 in last segment"""
    side = v3_difference(
        Vector3(ss_last.p4.x, ss_last.p4.y, ss_last.p4.z),
        Vector3(ss_last.p1.x, ss_last.p1.y, ss_last.p1.z),
    )
    side = v3_normalize(side)
    return side

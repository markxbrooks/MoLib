"""Parse crystallographic symmetry-operator terms (e.g. ``-x+1/2``)."""

from __future__ import annotations

import re
from typing import Any, MutableSequence

_SYMOP_COORDINATE_RE = re.compile(r"^(?P<sign>[+-]?)(?P<axis>[xyz])$")
_SYMOP_TRANSLATION_RE = re.compile(
    r"^(?P<sign>[+-]?)(?P<numerator>\d)/(?P<denominator>\d)$"
)


def parse_symop_term(
    row: MutableSequence[float],
    symop: str,
    term: str | Any,
) -> None:
    """Parse one signed term of a symmetry operator into *row*.

    :param row: Mutable length-4 row ``[rx, ry, rz, translation]``
    :param symop: Full operator string (for error messages)
    :param term: Single term such as ``x``, ``-y``, or ``+1/2``
    :raises ValueError: If *term* is not a coordinate or translation
    """
    m = _SYMOP_COORDINATE_RE.match(term)
    if m:
        sign = -1 if m.group("sign") == "-" else 1
        pos = {"x": 0, "y": 1, "z": 2}[m.group("axis")]
        row[pos] = sign
        return

    m = _SYMOP_TRANSLATION_RE.match(term)
    if not m:
        raise ValueError("What is " + term + " in " + symop)

    sign = -1 if m.group("sign") == "-" else 1
    row[3] += sign * int(m.group("numerator")) / int(m.group("denominator"))

import re
from typing import Any

_SYMOP_COORDINATE_RE = re.compile(r"^[+-]?([xyz])$")
_SYMOP_TRANSLATION_RE = re.compile(r"^[+-]?(\d)/(\d)$")


def parse_symop_term(row: list[int], symop, term: str | Any):
    """parse symop term"""
    m = _SYMOP_COORDINATE_RE.match(term)
    if m:
        pos = {"x": 0, "y": 1, "z": 2}[m["axis"]]
        row[pos] = sign
    else:
        m = _SYMOP_TRANSLATION_RE.match(term)
        if not m:
            raise ValueError("What is " + term + " in " + symop)

        row[3] = (
                sign
                * int(m["numerator"])
                / int(m["denominator"])
        )

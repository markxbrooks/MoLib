import random

from molib.xtal.uglymol.data import _lut


def generate_uuid():
    d0 = int(random.random() * 0xFFFFFFFF)
    d1 = int(random.random() * 0xFFFFFFFF)
    d2 = int(random.random() * 0xFFFFFFFF)
    d3 = int(random.random() * 0xFFFFFFFF)

    uuid = (
        _lut[d0 & 0xFF]
        + _lut[(d0 >> 8) & 0xFF]
        + _lut[(d0 >> 16) & 0xFF]
        + _lut[(d0 >> 24) & 0xFF]
        + "-"
        + _lut[d1 & 0xFF]
        + _lut[(d1 >> 8) & 0xFF]
        + "-"
        + _lut[((d1 >> 16) & 0x0F) | 0x40]
        + _lut[(d1 >> 24) & 0xFF]
        + "-"
        + _lut[(d2 & 0x3F) | 0x80]
        + _lut[(d2 >> 8) & 0xFF]
        + "-"
        + _lut[(d2 >> 16) & 0xFF]
        + _lut[(d2 >> 24) & 0xFF]
        + _lut[d3 & 0xFF]
        + _lut[(d3 >> 8) & 0xFF]
        + _lut[(d3 >> 16) & 0xFF]
        + _lut[(d3 >> 24) & 0xFF]
    )
    return uuid.lower()

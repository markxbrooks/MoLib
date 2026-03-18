class Color:
    def __init__(self, r, g=None, b=None):
        self.isColor = True
        self.r = 1
        self.g = 1
        self.b = 1
        self.set(r, g, b)

    def set(self, r, g=None, b=None):
        if g is None and b is None:
            value = r
            if hasattr(value, "isColor") and value.isColor:
                self.copy(value)
            elif isinstance(value, (int, float)):
                self.setHex(value)
        else:
            self.setRGB(r, g, b)
        return self

    def setHex(self, hex):
        hex = int(hex)
        self.r = ((hex >> 16) & 255) / 255
        self.g = ((hex >> 8) & 255) / 255
        self.b = (hex & 255) / 255
        return self

    def setRGB(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b
        return self

    def setHSL(self, h, s, l):
        h = h % 1
        s = max(0, min(s, 1))
        l = max(0, min(l, 1))

        if s == 0:
            self.r = self.g = self.b = l
        else:
            p = l * (1 + s) if l <= 0.5 else l + s - l * s
            q = 2 * l - p

            self.r = hue2rgb(q, p, h + 1 / 3)
            self.g = hue2rgb(q, p, h)
            self.b = hue2rgb(q, p, h - 1 / 3)
        return self

    def getHSL(self, target):
        r, g, b = self.r, self.g, self.b

        max_val = max(r, g, b)
        min_val = min(r, g, b)

        hue, saturation = 0, 0
        lightness = (min_val + max_val) / 2.0

        if min_val == max_val:
            hue = 0
            saturation = 0
        else:
            delta = max_val - min_val
            saturation = (
                delta / (max_val + min_val)
                if lightness <= 0.5
                else delta / (2 - max_val - min_val)
            )

            if max_val == r:
                hue = (g - b) / delta + (6 if g < b else 0)
            elif max_val == g:
                hue = (b - r) / delta + 2
            elif max_val == b:
                hue = (r - g) / delta + 4

            hue /= 6

        target.h = hue
        target.s = saturation
        target.l = lightness

        return target

    def clone(self):
        return Color(self.r, self.g, self.b)

    def copy(self, color):
        self.r = color.r
        self.g = color.g
        self.b = color.b
        return self

    def getHex(self):
        return (
            (int(self.r * 255) << 16) ^ (int(self.g * 255) << 8) ^ (int(self.b * 255))
        )

    def getHexString(self):
        return ("000000" + hex(self.getHex())[2:]).zfill(6)[-6:]


def to_col(num):
    return Color(num)


ColorSchemes = {
    "solarized dark": {
        "bg": Color(0x002B36),
        "fg": Color(0xFDF6E3),
        "map_den": Color(0xEEE8D5),
        "center": Color(0xFDF6E3),
        "lattices": list(
            map(
                to_col,
                [
                    0xDC322F,
                    0x2AA198,
                    0x268BD2,
                    0x859900,
                    0xD33682,
                    0xB58900,
                    0x6C71C4,
                    0xCB4B16,
                ],
            )
        ),
        "axes": list(map(to_col, [0xFFAAAA, 0xAAFFAA, 0xAAAAFF])),
    },
    "solarized light": {
        "bg": Color(0xFDF6E3),
        "fg": Color(0x002B36),
        "map_den": Color(0x073642),
        "center": Color(0x002B36),
        "lattices": list(
            map(
                to_col,
                [
                    0xDC322F,
                    0x2AA198,
                    0x268BD2,
                    0x859900,
                    0xD33682,
                    0xB58900,
                    0x6C71C4,
                    0xCB4B16,
                ],
            )
        ),
        "axes": list(map(to_col, [0xFFAAAA, 0xAAFFAA, 0xAAAAFF])),
    },
}


def hue2rgb(p, q, t):
    if t < 0:
        t += 1
    if t > 1:
        t -= 1
    if t < 1 / 6:
        return p + (q - p) * 6 * t
    if t < 1 / 2:
        return q
    if t < 2 / 3:
        return p + (q - p) * 6 * (2 / 3 - t)
    return p

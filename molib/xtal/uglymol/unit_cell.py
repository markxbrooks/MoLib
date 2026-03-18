import math

from molib.xtal.uglymol.math.helpers import multiply


class UnitCell:
    def __init__(self, a, b, c, alpha, beta, gamma):
        if a <= 0 or b <= 0 or c <= 0 or alpha <= 0 or beta <= 0 or gamma <= 0:
            raise ValueError("Zero or negative unit cell parameter(s).")
        self.parameters = [a, b, c, alpha, beta, gamma]
        deg2rad = math.pi / 180.0
        cos_alpha = math.cos(deg2rad * alpha)
        cos_beta = math.cos(deg2rad * beta)
        cos_gamma = math.cos(deg2rad * gamma)
        sin_alpha = math.sin(deg2rad * alpha)
        sin_beta = math.sin(deg2rad * beta)
        sin_gamma = math.sin(deg2rad * gamma)
        if sin_alpha == 0 or sin_beta == 0 or sin_gamma == 0:
            raise ValueError("Impossible angle - N*180deg.")
        cos_alpha_star_sin_beta = (cos_beta * cos_gamma - cos_alpha) / sin_gamma
        cos_alpha_star = cos_alpha_star_sin_beta / sin_beta
        s1rca2 = math.sqrt(1.0 - cos_alpha_star * cos_alpha_star)
        self.orth = [
            a,
            b * cos_gamma,
            c * cos_beta,
            0.0,
            b * sin_gamma,
            -c * cos_alpha_star_sin_beta,
            0.0,
            0.0,
            c * sin_beta * s1rca2,
        ]
        self.frac = [
            1.0 / a,
            -cos_gamma / (sin_gamma * a),
            -(cos_gamma * cos_alpha_star_sin_beta + cos_beta * sin_gamma)
            / (sin_beta * s1rca2 * sin_gamma * a),
            0.0,
            1.0 / (sin_gamma * b),
            cos_alpha_star / (s1rca2 * sin_gamma * b),
            0.0,
            0.0,
            1.0 / (sin_beta * s1rca2 * c),
        ]

    def fractionalize(self, xyz):
        return multiply(xyz, self.frac)

    def orthogonalize(self, xyz):
        return multiply(xyz, self.orth)

"""Math helpers"""

from molib.xtal.uglymol.data import cubeVerts


def multiply(xyz, mat):
    """multiply xyz with mat"""
    return [
        mat[0] * xyz[0] + mat[1] * xyz[1] + mat[2] * xyz[2],
        mat[4] * xyz[1] + mat[5] * xyz[2],
        mat[8] * xyz[2],
    ]


def calculate_vert_offsets(dims):
    """calculate vertical offsets for given dimensions"""
    vert_offsets = []
    for i in range(8):
        v = cubeVerts[i]
        vert_offsets.append(v[0] + dims[2] * (v[1] + dims[1] * v[2]))
    return vert_offsets


def modulo(a, b):
    """modulo"""
    reminder = a % b
    return reminder if reminder >= 0 else reminder + b


def calculate_stddev(a, offset):
    """calculate standard deviation"""
    sum_val = 0
    sq_sum = 0
    alen = len(a)
    for i in range(offset, alen):
        sum_val += a[i]
        sq_sum += a[i] * a[i]
    mean = sum_val / (alen - offset)
    variance = sq_sum / (alen - offset) - mean * mean
    return {"mean": mean, "rms": variance**0.5}


def clamp(value, min_value, max_value):
    """clamp a value to min_value and max_value"""
    if value < min_value:
        return max(min_value, min(max_value, value))
    return None


def euclidean_modulo(n, m):
    """euclidean_modulo"""
    return ((n % m) + m) % m


def find_max_dist(pos):
    """find_max_dist(pos)"""
    max_sq = 0
    for i in range(0, len(pos), 3):
        n = 3 * i
        sq = pos[n] * pos[n] + pos[n + 1] * pos[n + 1] + pos[n + 2] * pos[n + 2]
        if sq > max_sq:
            max_sq = sq
    return max_sq**0.5


def max_val(arr):
    """max_val(arr) -> int"""
    max_val = float("-inf")
    for value in arr:
        if value > max_val:
            max_val = value
    return max_val


def minus_ones(n):
    """minus_ones(n) -> int"""
    return [-1] * n

import json

from molib.xtal.uglymol.math.helpers import minus_ones


def parse_csv(text):
    lines = [line for line in text.split("\n") if len(line) > 0 and line[0] != "#"]
    pos = [0.0] * (len(lines) * 3)
    lattice_ids = []
    for i, line in enumerate(lines):
        nums = list(map(float, line.split(",")))
        for j in range(3):
            pos[3 * i + j] = nums[j]
        lattice_ids.append(nums[3])
    return {"pos": pos, "lattice_ids": lattice_ids}


def parse_json(text):
    d = json.loads(text)
    n = len(d["rlp"])
    if n > 0 and isinstance(d["rlp"][0], list):  # deprecated format
        pos = [0.0] * (3 * n)
        for i in range(n):
            for j in range(3):
                pos[3 * i + j] = d["rlp"][i][j]
    else:  # flat array - new format
        pos = d["rlp"]
    lattice_ids = d.get("experiment_id", minus_ones(n))
    return {"pos": pos, "lattice_ids": lattice_ids}

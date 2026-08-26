from itertools import chain


class Cubicles:
    def __init__(self, atoms, box_length, lower_bound, upper_bound):
        self.boxes = []
        self.box_length = box_length
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.xdim = -(-((upper_bound[0] - lower_bound[0]) / box_length) // 1)  # ceil
        self.ydim = -(-((upper_bound[1] - lower_bound[1]) / box_length) // 1)  # ceil
        self.zdim = -(-((upper_bound[2] - lower_bound[2]) / box_length) // 1)  # ceil
        nxyz = self.xdim * self.ydim * self.zdim
        for _ in range(nxyz):
            self.boxes.append([])
        for i in range(len(atoms)):
            xyz = atoms[i].xyz
            box_id = self.find_box_id(xyz[0], xyz[1], xyz[2])
            if box_id is None:
                raise ValueError("wrong cubicle")
            self.boxes[box_id].append(i)

    def find_box_id(self, x, y, z):
        xstep = (x - self.lower_bound[0]) // self.box_length
        ystep = (y - self.lower_bound[1]) // self.box_length
        zstep = (z - self.lower_bound[2]) // self.box_length
        box_id = (zstep * self.ydim + ystep) * self.xdim + xstep
        if box_id < 0 or box_id >= len(self.boxes):
            raise ValueError("Ups!")
        return box_id

    from itertools import chain

    def get_nearby_atoms(self, box_id):
        indices = []
        xydim = self.xdim * self.ydim
        u = (box_id % xydim) % self.xdim
        v = (box_id % xydim) // self.xdim
        w = box_id // xydim

        boxes = []
        for iu in range(max(0, u - 1), min(self.xdim, u + 2)):
            for iv in range(max(0, v - 1), min(self.ydim, v + 2)):
                for iw in range(max(0, w - 1), min(self.zdim, w + 2)):
                    other_box_id = (iw * xydim) + (iv * self.xdim) + iu
                    boxes.append(self.boxes[other_box_id])
        return list(chain.from_iterable(boxes))

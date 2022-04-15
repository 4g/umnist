import json
import random

import numpy as np
from tqdm import tqdm

def get_largest_empty_rect(a, skip=1):
    """
    https://stackoverflow.com/questions/2478447/find-largest-rectangle-containing-only-zeros-in-an-n%c3%97n-binary-matrix/30418912#30418912
    a = matrix made of zeros and ones
    skip = which of 0/1 are to be considered empty
    """
    area_max = (0, [])
    nrows, ncols = a.shape
    w = np.zeros(dtype=int, shape=a.shape)
    h = np.zeros(dtype=int, shape=a.shape)
    for r in range(nrows):
        for c in range(ncols):
            if a[r][c] == skip:
                continue
            if r == 0:
                h[r][c] = 1
            else:
                h[r][c] = h[r-1][c]+1
            if c == 0:
                w[r][c] = 1
            else:
                w[r][c] = w[r][c-1]+1
            minw = w[r][c]
            for dh in range(h[r][c]):
                minw = min(minw, w[r-dh][c])
                area = (dh+1)*minw
                if area > area_max[0]:
                    area_max = (area, [(r-dh, c-minw+1, r, c)])

    return area_max

class LocationGenerator:
    def __init__(self, size=1000, ratio=5, minsize=.1, maxsize=.5):
        self.size = size
        self.ratio = ratio
        self.small_size = self.size // self.ratio
        self.matrix = np.zeros((self.small_size, self.small_size))
        self.minsize = max(int(self.small_size * minsize), 1)
        self.maxsize = int(self.small_size * maxsize)

    def random_box(self, rect):
        w = rect[2] - rect[0]
        h = rect[3] - rect[1]
        size = random.randint(self.minsize, self.maxsize)
        size = min(w, size)
        size = min(h, size)
        rx = random.randint(rect[0], rect[2] - size)
        ry = random.randint(rect[1], rect[3] - size)
        return rx, ry, rx+size, ry+size

    def insert(self):
        area, coords = get_largest_empty_rect(self.matrix, skip=1)
        coords = self.random_box(coords[0])
        x1, y1, x2, y2 = coords
        self.matrix[x1:x2, y1:y2] = 1
        rescaled_coords = [i*self.ratio for i in coords]
        return rescaled_coords

    def reset(self):
        self.matrix = np.zeros((self.small_size, self.small_size))

class CachedLocationGenerator:
    def __init__(self, size=512, ratio=20, minsize=.1, maxsize=.5, numlocs=5, iters=100000):
        self.locgen = LocationGenerator(size, ratio, minsize, maxsize)
        self.iters = iters
        self.numlocs = numlocs

        self.locations = []
        self.index = 0
        self.index_loc = 0


    def create(self):
        for _ in tqdm(range(self.iters), desc="Creating locations"):
            iter_locations = [self.locgen.insert() for j in range(self.numlocs)]
            self.locations.append(iter_locations)
            self.locgen.reset()

        self.locations = np.asarray(self.locations, dtype=np.int32)

    def shuffle(self):
        np.random.shuffle(self.locations)
        self.index = 0
        self.index_loc = 0

    def save(self, fname):
        np.save(fname, self.locations)

    def load(self, fname):
        self.locations = np.load(fname)

    def insert(self):
        val = self.locations[self.index][self.index_loc]
        self.index_loc += 1
        return val

    def reset(self):
        self.index += 1
        self.index = self.index % len(self.locations)
        self.index_loc = 0

# locgen = CachedLocationGenerator(size=1024, ratio=20, minsize=0.03, maxsize=0.5, iters=30000)
# locgen.create()
# locgen.save(fname="locgen_1024.npy")
# locgen.load(fname="locgen_1024.npy")
# for i in range(1000000):
#     coords = locgen.insert()
#     if (i + 1) % 5 == 0:
#         locgen.reset()


# locgen = LocationGenerator(size=512, ratio=50, minsize=0.1, maxsize=0.5)
# from tqdm import tqdm
# for i in tqdm(range(10000)):
#     locgen.insert()
#     if i % 5 == 0:
#         locgen = LocationGenerator(size=512, ratio=50, minsize=0.1, maxsize=0.5)

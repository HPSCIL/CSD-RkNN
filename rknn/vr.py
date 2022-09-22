from shapely.geometry import Point, Polygon, LineString
from math import cos, pi, sin

from common.data_structure import MinHeap, NSmallestHolder


class DistanceCalculator:
    def __init__(self):
        self.cache = dict()

    def dist(self, p1, p2):
        if (p1, p2) not in self.cache:
            d = p1.geom.distance(p2.geom)
            self.cache[(p1, p2)] = d
            self.cache[(p2, p1)] = d
            return d
        else:
            return self.cache[(p1, p2)]


class Circle(Polygon):
    def __init__(self, o, r):
        if r > 0:
            Polygon.__init__(self, o.buffer(r).exterior.coords)
        else:
            Polygon.__init__(self)
        self.o = o
        self.r = r


class Sector(Polygon):
    def __init__(self, o, r, angles):
        if r > 0:
            circle = o.buffer(r)
            if abs(angles[0] - angles[1]) >= pi:
                raise ValueError('abs(angles[0] - angles[1]) must be less than Pi')
            l = r / cos(abs(angles[0] - angles[1]) / 2)
            triangle = Polygon(
                [(o.x, o.y), (o.x + cos(angles[0]) * l, o.y + sin(angles[0]) * l),
                 (o.x + cos(angles[1]) * l, o.y + sin(angles[1]) * l)])
            sector = triangle.intersection(circle)
            Polygon.__init__(self, sector.exterior.coords)
            self.arcVertices = list(triangle.exterior.intersection(circle.exterior))
        else:
            Polygon.__init__(self)
            self.arcVertices = [o, o]
        self.o = o
        self.r = r
        self.angles = angles


class Partition(Sector):
    def __init__(self, o, r, angles, k):
        Sector.__init__(self, o, r, angles)
        self.candidates = NSmallestHolder(k)


def getPartitions(o, index, n, k):
    bounds = index.root.geom.bounds
    r = Point((bounds[0], bounds[1])).distance(Point((bounds[2], bounds[3])))
    return [Partition(o, r, [2 * pi / n * i, 2 * pi / n * (i + 1)], k) for i in range(n)]


def kNNRadius(q, k, index):
    if k == 0:
        return 0
    knn = list(index.nearest(q, k))
    return knn[-1][1]


def isRkNN(e, q, k, facility_index):
    if q.geom.distance(e.geom) <= kNNRadius(e, k, facility_index):
        return True
    else:
        return False


def MonoPruning(q, k, index):
    partitions = list(getPartitions(q.geom, index, 6, k))
    for partition in partitions:
        partition.candidates = NSmallestHolder(k)
    visited = {q}
    h = MinHeap()
    for neighbor in q.neighbors:
        h.push((1, neighbor))
        visited.add(neighbor)
    while len(h) > 0:
        gd_p, p = h.pop()
        for partition in partitions:
            if partition.intersects(p.geom):
                if partition.candidates.is_full():
                    dist_pn, pn = partition.candidates.largest()
                else:
                    dist_pn = float('inf')
                dist_p = p.geom.distance(q.geom)
                if gd_p <= k and dist_p <= dist_pn:
                    partition.candidates.push((dist_p, p))
                    for neighbor in p.neighbors:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            h.push((gd_p + 1, neighbor))
        for neighbor in p.neighbors:
            if gd_p <= k:
                edge = LineString(list(neighbor.geom.coords)+list(p.geom.coords))
                for partition in partitions:
                    if not partition.candidates.is_full() and partition.intersects(edge):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            h.push((gd_p + 1, neighbor))
    visited = set()
    for partition in partitions:
        for dist_c, c in partition.candidates:
            if c not in visited:
                visited.add(c)
                yield c


def BiPruning(q, k, facility_index, user_index):
    partitions = list(getPartitions(q.geom, facility_index, 6, k))
    for partition in partitions:
        partition.candidates = list()
    visited = {q}
    h = MinHeap()
    for neighbor in q.neighbors:
        h.push((neighbor.geom.distance(q.geom), neighbor))
        visited.add(neighbor)
    while len(h) > 0:
        dist_p, p = h.pop()
        for partition in partitions:
            if partition.intersects(p.geom):
                if len(partition.candidates) < k:
                    partition.candidates.append((dist_p, p))
                    for neighbor in p.neighbors:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            h.push((neighbor.geom.distance(q.geom), neighbor))
        for neighbor in p.neighbors:
            edge = LineString(list(neighbor.geom.coords)+list(p.geom.coords))
            for partition in partitions:
                if len(partition.candidates) < k and partition.intersects(edge):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        h.push((neighbor.geom.distance(q.geom), neighbor))

    for partition in partitions:
        if len(partition.candidates) >= k:
            r, p = partition.candidates[-1]
            unprunedArea = Sector(partition.o, r, partition.angles)
        else:
            unprunedArea = Sector(partition.o, partition.r, partition.angles)
        for e in user_index.intersects(unprunedArea):
            yield e


def verification(candidates, q, k, facility_index):
    for c in candidates:
        if isRkNN(c, q, k, facility_index):
            yield c


def MonoRkNN(q, k, index):
    candidates = MonoPruning(q, k, index)

    for e in verification(candidates, q, k, index):
        yield e


def BiRkNN(q, k, facility_index, user_index):
    candidates = BiPruning(q, k, facility_index, user_index)
    for e in verification(candidates, q, k, facility_index):
        yield e

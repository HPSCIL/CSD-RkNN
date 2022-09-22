from math import acos

from numpy.linalg import norm
from shapely.geometry import Point, Polygon
import numpy as np
from shapely.ops import unary_union
from math import cos, pi, sin

from common.data_structure import MinHeap


def angle(o, x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    o = np.asarray(o)
    vector_ox = x - o
    vector_oy = y - o
    norm_ox = norm(vector_ox)
    norm_oy = norm(vector_oy)
    if norm_ox == 0 or norm_oy == 0:
        return 0
    return acos(vector_ox.dot(vector_oy) / (norm_ox * norm_oy))


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
        self.k = k
        self.upperRadiusList = list()
        self.sigList = list()
        self.boundingUpperRadius = self.r
        self.userUnprunedArea, self.facilityUnprunedArea = self.calculateUnprunedArea(self.boundingUpperRadius)
        self.updateUnprunedArea()

    def updateUnprunedArea(self):
        if len(self.upperRadiusList) >= self.k and self.upperRadiusList[-1] < self.boundingUpperRadius:
            self.boundingUpperRadius = self.upperRadiusList[-1]
            self.userUnprunedArea = Sector(self.o, self.boundingUpperRadius, self.angles)
            self.userUnprunedArea, self.facilityUnprunedArea = self.calculateUnprunedArea(self.boundingUpperRadius)

    def calculateUnprunedArea(self, boundingUpperRadius):
        userUnprunedArea = Sector(self.o, boundingUpperRadius, self.angles)
        facilityUnprunedArea = unary_union(
            [Circle(userUnprunedArea.arcVertices[0], boundingUpperRadius),
             Circle(userUnprunedArea.arcVertices[1], boundingUpperRadius),
             Sector(self.o, boundingUpperRadius * 2, self.angles)])
        return userUnprunedArea, facilityUnprunedArea

    def addUpperRadius(self, upperRadius):
        if len(self.upperRadiusList) < self.k:
            self.upperRadiusList.append(upperRadius)
            if len(self.upperRadiusList) == self.k:
                self.upperRadiusList.sort()
                self.updateUnprunedArea()
        elif upperRadius < self.upperRadiusList[-1]:
            self.upperRadiusList.append(upperRadius)
            self.upperRadiusList.sort()
            self.upperRadiusList.pop()
            self.updateUnprunedArea()

    def getMinMaxAngle(self, p):
        o = (self.o.x, self.o.y)
        p1 = [o[0] + cos(self.angles[0]), o[1] + sin(self.angles[0])]
        p2 = [o[0] + cos(self.angles[1]), o[1] + sin(self.angles[1])]
        angles = [angle(o, p1, p), angle(o, p2, p)]
        return sorted(angles)


def getPartitions(o, index, n, k):
    bounds = index.root.geom.bounds
    r = Point((bounds[0], bounds[1])).distance(Point((bounds[2], bounds[3])))
    return [Partition(o, r, [2 * pi / n * i, 2 * pi / n * (i + 1)], k) for i in range(n)]


def isPruned(e, facilityUnprunedAreas):
    for area in facilityUnprunedAreas:
        if e.geom.intersects(area):
            return False
    return True


def pruneSpace(f, partitions):
    for partition in partitions:
        minAngle, maxAngle = partition.getMinMaxAngle(f.geom)
        if minAngle < pi / 2:
            dist = f.geom.distance(partition.o)
            if maxAngle >= pi / 2:
                partition.addUpperRadius(float('inf'))
            else:
                partition.addUpperRadius(dist / (2 * cos(maxAngle)))
            if f.geom.intersects(partition.facilityUnprunedArea):
                lowerRadius = dist / (2 * cos(minAngle))
                partition.sigList.append([lowerRadius, f])


def filtering(q, k, index, partition_num):
    partitions = getPartitions(q.geom, index, partition_num, k)
    h = MinHeap()
    h.push((0, index.root))
    while len(h) > 0:
        e_dist, e = h.pop()
        if not isPruned(e, [p.facilityUnprunedArea for p in partitions]):
            if not e.is_data_node:
                for child in e.children:
                    h.push((child.geom.distance(q.geom), child))
            else:
                if e != q:
                    pruneSpace(e, partitions)
    return partitions


def verification(q, k, index, partitions):
    for partition in partitions:
        partition.sigList.sort()
    h = list()
    h.append(index.root)
    while len(h) > 0:
        e = h.pop()
        if not isPruned(e, [p.userUnprunedArea for p in partitions]):
            if not e.is_data_node:
                for child in e.children:
                    h.append(child)
            else:
                if q != e and isRkNN(e, k, partitions):
                    yield e


def isRkNN(u, k, partitions):
    for p in partitions:
        if u.geom.intersects(p):
            partition = p
            break
    count = 0
    for lowerRadius, f in partition.sigList:
        if f == u:
            continue
        dist = u.geom.distance(partition.o)
        if dist <= lowerRadius:
            return True
        if u.geom.distance(f.geom) < dist:
            count += 1
            if count >= k:
                return False
    return True


def BiRkNN(q, k, facility_index, user_index):
    partitions = filtering(q, k, facility_index, 12)
    return verification(q, k, user_index, partitions)


def MonoRkNN(q, k, index):
    partitions = filtering(q, k + 1, index, 12)
    return verification(q, k, index, partitions)

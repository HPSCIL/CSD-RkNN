# -*- coding:utf-8 -*-

from scipy.spatial.qhull import Voronoi
from shapely.geometry import Point

from common.data_structure import MinHeap
from index.rtree import RtreeIndex, RtreeNode


class VoRtreeIndex(RtreeIndex):
    def __init__(self, **kwargs):
        RtreeIndex.__init__(self, **kwargs)
        data = kwargs.get('data', [])
        if len(data) > 0:
            voronoi = Voronoi([(geom.x, geom.y) for uuid, geom in data])
            for ids in voronoi.ridge_points:
                nodes = [self.nodes[data[i][0]] for i in ids]
                nodes[0].add_neighbor(nodes[1])
                nodes[1].add_neighbor(nodes[0])
            for uuid, geom in data:
                self.nodes[uuid].dumps()

    def new_node(self, uuid, geom, child_ids, level):
        node = VoRtreeNode(self, uuid, geom, child_ids, None, level)
        return node

    def loads(self, uuid, data):
        node = self.new_node(uuid, data[0], data[1], data[2])
        node.neighbor_ids = data[3]
        return node

    def nearest(self, q, k=1):
        h = MinHeap()
        visited = set()
        if type(q) == type(self.root):
            point = q.geom
        elif type(q) == Point:
            point = q
        if type(q) == type(self.root) and q.uuid in self.nodes:
            visited.add(q)
            for neighbor in q.neighbors:
                visited.add(neighbor)
                h.push((point.distance(neighbor.geom), neighbor))
        else:
            nn, dist_nn = list(RtreeIndex.nearest(self, point, 1))[0]
            h.push((dist_nn, nn))
            visited.add(nn)
        for i in range(k):
            if len(h) > 0:
                dist_p, p = h.pop()
                yield p, dist_p
                if i == k - 1:
                    break
                for neighbor in p.neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        h.push((point.distance(neighbor.geom), neighbor))
            else:
                break


class VoRtreeNode(RtreeNode):
    def __init__(self, tree, uuid, geom, child_ids, neighbor_ids, level):
        RtreeNode.__init__(self, tree, uuid, geom, child_ids, level)
        self.neighbor_ids = neighbor_ids

    def to_data(self):
        return self.geom, self.child_ids, self.level, self.neighbor_ids

    def destruct(self):
        RtreeNode.destruct(self)
        del self.neighbor_ids

    @property
    def neighbors(self):
        for neighbor_id in self.neighbor_ids:
            yield self.tree.nodes[neighbor_id]

    def add_neighbor(self, neighbor):
        if self.neighbor_ids is None:
            self.neighbor_ids = []
        self.neighbor_ids.append(neighbor.uuid)

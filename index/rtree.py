# -*- coding:utf-8 -*-
import itertools
from uuid import uuid1 as generate_uuid

from shapely.geometry import Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from common.data_structure import MinHeap, NSmallestHolder

# from rtree import Rtree
from common.persistence import PersistentDict


def bounding_box(geoms):
    return unary_union(geoms).envelope


class NodePool(object):
    def __init__(self, tree, database):
        self.database = database
        self.cache = dict()
        self.tree = tree

    def __delitem__(self, key):
        del self.database[key]
        del self.cache[key]

    def __setitem__(self, key, node):
        self.database[key] = node.to_data()
        self.cache[key] = node

    def __getitem__(self, key):
        if key in self.cache:
            return self.cache[key]
        else:
            data = self.database[key]
            node = self.tree.loads(key, data)
            self.cache[key] = node
            return node

    def __contains__(self, item):
        if item in self.cache:
            return True
        if item in self.database:
            return True
        return False

    def reset_cache(self):
        self.database.reset()
        self.cache.clear()

    # def close(self):
    #     self.database.close()


class RtreeIndex(object):
    class Properties:
        def __init__(self, MAX_CHILDREN_NUM, MIN_CHILDREN_NUM, ROOT_ID):
            self.MAX_CHILDREN_NUM = MAX_CHILDREN_NUM
            self.MIN_CHILDREN_NUM = MIN_CHILDREN_NUM
            self.ROOT_ID = ROOT_ID

    def __init__(self, **kwargs):
        path = kwargs.get('path', None)
        self.MAX_CHILDREN_NUM = kwargs.get('max_children_num', 10)
        self.MIN_CHILDREN_NUM = kwargs.get('min_children_num', self.MAX_CHILDREN_NUM / 2)
        data = kwargs.get('data', [])
        self._database = None
        self._nodes = None
        self._properties = None
        self.path = path
        for uuid, geom in data:
            self.insert(uuid, geom)

    def close(self):
        self.database.close()

    def drop_file(self):
        PersistentDict.drop(self.path)

    @property
    def database(self):
        if self._database is None:
            if self.path is not None:
                self._database = PersistentDict(self.path)
            else:
                return None
        return self._database

    @database.setter
    def database(self, v):
        self._database = v

    @property
    def nodes(self):
        if self._nodes is None:
            if self.database is not None:
                self._nodes = NodePool(self, self.database)
            else:
                self._nodes = dict()
        return self._nodes

    @nodes.setter
    def nodes(self, v):
        self._nodes = v

    @property
    def properties(self):
        if self._properties is None:
            if self.database is not None:
                if 'properties' in self.database:
                    self._properties = self.database['properties']
                else:
                    root = self.new_node('a', Polygon(), [], 1)
                    root.dumps()
                    self._properties = RtreeIndex.Properties(self.MAX_CHILDREN_NUM, self.MIN_CHILDREN_NUM, root.uuid)
                    self.database['properties'] = self._properties
            else:
                root = self.new_node('a', Polygon(), [], 1)
                root.dumps()
                self._properties = RtreeIndex.Properties(self.MAX_CHILDREN_NUM, self.MIN_CHILDREN_NUM, root.uuid)
        return self._properties

    @properties.setter
    def properties(self, v):
        self._properties = v

    def __contains__(self, item):
        if type(item) != type(self.root):
            return False
        return item.uuid in self.nodes

    def reset_cache(self):
        self.nodes.reset_cache()

    @property
    def root(self):
        return self.nodes[self.properties.ROOT_ID]

    @root.setter
    def root(self, root_node):
        self.properties.ROOT_ID = root_node.uuid
        if self.database is not None:
            self.database['properties'] = self.properties

    def new_node(self, uuid, geom, child_ids, level):
        node = RtreeNode(self, uuid, geom, child_ids, level)
        return node

    def loads(self, uuid, data):
        return self.new_node(uuid, data[0], data[1], data[2])

    def create_data_node(self, uuid, geom):
        node = self.new_node(uuid, geom, None, 0)
        assert node.is_data_node
        node.dumps()
        return node

    def create_with_children(self, children):
        geom = bounding_box([c.geom for c in children])
        uuid = str(generate_uuid())
        child_ids = [c.uuid for c in children]
        level = children[0].level + 1
        node = self.new_node(uuid, geom, child_ids, level)
        assert (not node.is_data_node)
        node.dumps()
        return node

    def insert(self, uuid, geom):
        inserting_result = self.root.insert(self.create_data_node(uuid, geom))
        if type(inserting_result) is list:
            self.root = self.create_with_children(inserting_result)

    def delete(self, node):
        nodes = self.root.find_leaf(node)
        nodes[-1].remove_child(node.uuid)
        node.destruct()
        self.condense_tree(nodes)

    def plot(self, ax):
        height = self.root.level
        for level in range(height, -1, -1):
            self.plot_subtree(ax, self.root, level)

    def plot_subtree(self, ax, node, level):
        if node.level == level:
            plot_node(ax, node)
        elif node.level > level:
            for c in node.children:
                self.plot_subtree(ax, c, level)

    def intersects(self, geom: BaseGeometry):
        nodes = {self.root}
        while len(nodes) > 0:
            e = nodes.pop()
            if e.is_data_node:
                if geom.intersects(e.geom):
                    yield e
            else:
                if geom.intersects(e.geom):
                    for child in e.children:
                        nodes.add(child)

    def contains(self, geom: BaseGeometry):
        nodes = {self.root}
        while len(nodes) > 0:
            e = nodes.pop()
            if e.is_data_node:
                if geom.contains(e.geom):
                    yield e
            else:
                if geom.intersects(e.geom):
                    for child in e.children:
                        nodes.add(child)

    def nearest(self, q, k=1):
        if type(q) == type(self.root):
            point = q.geom
            uuid_q = q.uuid
        elif type(q) == Point:
            point = q
            uuid_q = str(generate_uuid())
        h = MinHeap()
        knn = NSmallestHolder(k)
        h.push((0, self.root))
        while len(h) > 0:
            e_min_dist, e = h.pop()
            if knn.is_full() and e_min_dist > knn.largest()[0]:
                break
            if e.is_leaf_node:
                for c in e.children:
                    if c.uuid != uuid_q:
                        c_min_dist = c.geom.distance(point)
                        knn.push((c_min_dist, c))
            else:
                for c in e.children:
                    c_min_dist = c.geom.distance(point)
                    if len(knn) < k or c_min_dist < knn.first()[0]:
                        h.push((c_min_dist, c))
        for dist, e in knn:
            yield e, dist

    def condense_tree(self, nodes):
        eliminated_nodes = []
        while len(nodes) > 0:
            n = nodes.pop()
            if not n.is_root and n.children_num < self.properties.MIN_CHILDREN_NUM:
                for child in n.children:
                    eliminated_nodes.append(child)
                nodes[-1].remove_child(n.uuid)
                n.destruct()
            else:
                n.geom = bounding_box([c.geom for c in n.children])
                n.dumps()
        while len(eliminated_nodes) > 0:
            self.root.insert(eliminated_nodes.pop())


class RtreeNode(object):
    def __init__(self, tree, uuid, geom, child_ids, level):
        self.tree = tree
        self.uuid = uuid
        self.geom = geom
        self.child_ids = child_ids
        self.level = level

    def to_data(self):
        return self.geom, self.child_ids, self.level

    def __cmp__(self, other):
        return 0

    def __lt__(self, other):
        return False

    def __gt__(self, other):
        return False

    def __le__(self, other):
        return True

    def __ge__(self, other):
        return True

    def __eq__(self, other):
        if type(other) != type(self):
            return False
        if other.uuid == self.uuid:
            return True

    def __hash__(self):
        return hash(self.uuid)

    def dumps(self):
        self.tree.nodes[self.uuid] = self

    def destruct(self):
        del self.tree.nodes[self.uuid]
        del self.tree
        del self.uuid
        del self.geom
        del self.child_ids
        del self.level

    @property
    def children(self):
        for child_id in self.child_ids:
            yield self.tree.nodes[child_id]

    def add_child(self, child):
        self.child_ids.append(child.uuid)

    def remove_child(self, child_id):
        self.child_ids.remove(child_id)

    @property
    def children_num(self):
        return len(self.child_ids)

    @property
    def is_data_node(self):
        return self.level == 0

    @property
    def is_leaf_node(self):
        return self.level == 1

    @property
    def is_intermediate_node(self):
        return (not self.is_data_node) and (not self.is_leaf_node)

    @property
    def is_root(self):
        return self.tree.root is self

    def insert(self, node):
        new_bounding_box = bounding_box([self.geom, node.geom])
        if self.level == node.level + 1:
            self.add_child(node)
            if self.children_num > self.tree.properties.MAX_CHILDREN_NUM:
                return self.split()
            else:
                if new_bounding_box.area > self.geom.area:
                    self.geom = new_bounding_box
                self.dumps()
                return self
        else:
            inserting_child = self.find_inserting_child(node)
            inserting_child_id = inserting_child.uuid
            inserting_result = inserting_child.insert(node)
            if type(inserting_result) is list:
                self.remove_child(inserting_child_id)
                for c in inserting_result:
                    self.add_child(c)
                if self.children_num <= self.tree.properties.MAX_CHILDREN_NUM:
                    if new_bounding_box.area > self.geom.area:
                        self.geom = new_bounding_box
                    self.dumps()
                    return self
                else:
                    return self.split()

            else:
                if new_bounding_box.area > self.geom.area:
                    self.geom = new_bounding_box
                    self.dumps()
                return self

    def find_inserting_child(self, node):
        min_enlargement = float('inf')
        min_area = float('inf')
        inserting_child = None
        for e in self.children:
            area = bounding_box([e.geom, node.geom]).area
            enlargement = area - e.geom.area
            if enlargement < min_enlargement:
                inserting_child = e
                min_enlargement = enlargement
                min_area = area
            elif enlargement == min_enlargement:
                if area < min_area:
                    inserting_child = e
                    min_area = area
        return inserting_child

    def _quadratic_split(self):
        seeds, remain_entries = self._pick_seeds()
        groups = [[seeds[0]], [seeds[1]]]
        group_bounding_boxes = [seeds[0].geom, seeds[1].geom]
        while len(remain_entries) > 0:
            if len(groups[0]) > self.tree.properties.MIN_CHILDREN_NUM:
                groups[1] += remain_entries
                break
            if len(groups[1]) > self.tree.properties.MIN_CHILDREN_NUM:
                groups[0] += remain_entries
                break
            e, bounding_boxes = self._pick_next(remain_entries, group_bounding_boxes)
            areas = [bounding_boxes[0].area, bounding_boxes[1].area]
            area_differences = [areas[0] - group_bounding_boxes[0].area, areas[1] - group_bounding_boxes[1].area]
            if area_differences[0] < area_differences[1]:
                target_group_id = 0
            elif area_differences[0] > area_differences[1]:
                target_group_id = 1
            else:
                if areas[0] < areas[1]:
                    target_group_id = 0
                elif areas[0] > areas[1]:
                    target_group_id = 1
                else:
                    if len(groups[0]) <= len(groups[1]):
                        target_group_id = 0
                    else:
                        target_group_id = 1
            groups[target_group_id].append(e)
            group_bounding_boxes[target_group_id] = bounding_boxes[target_group_id]
        try:
            return [self.tree.create_with_children(group) for group in groups]
        finally:
            self.destruct()

    def _pick_seeds(self):
        seeds = max(itertools.combinations(self.children, 2),
                    key=lambda x: bounding_box([x[0].geom, x[1].geom]).area - x[0].geom.area - x[1].geom.area)
        remain_entries = []
        for e in self.children:
            if e not in seeds:
                remain_entries.append(e)
        return seeds, remain_entries

    @staticmethod
    def _pick_next(remain_entries, group_bounding_boxes):
        difference = -1
        for i in range(len(remain_entries)):
            bbox1 = bounding_box([group_bounding_boxes[0], remain_entries[i].geom])
            bbox2 = bounding_box([group_bounding_boxes[1], remain_entries[i].geom])
            d1 = bbox1.area - group_bounding_boxes[0].area
            d2 = bbox2.area - group_bounding_boxes[1].area
            if abs(d1 - d2) > difference:
                difference = abs(d1 - d2)
                next_entry_id = i
                next_bounding_boxes = [bbox1, bbox2]
        next_entry = remain_entries.pop(next_entry_id)
        return next_entry, next_bounding_boxes

    split = _quadratic_split

    def find_leaf(self, node):
        if self.is_leaf_node:
            for child in self.children:
                if node == child:
                    return [self]
        else:
            for child in self.children:
                if child.geom.intersects(node.geom):
                    result = child.find_leaf(node)
                    if result is not None:
                        return [self] + result
            return None


def plot_node(ax, node, color=['red', 'green', 'yellow', 'blue', 'deeppink', 'orange', 'gray']):
    if isinstance(node.geom, Point):
        ax.plot(node.geom.x, node.geom.y, '.', color=color[node.level], markersize=1)
    else:
        coordinates = list(node.geom.exterior.coords)
        ax.plot([c[0] for c in coordinates], [c[1] for c in coordinates], '-', linewidth=node.level,
                color=color[node.level])

from collections import Iterable
from heapq import heapify, heappushpop, _siftdown_max, _siftup_max, heappop, heappush, _heapify_max, _heappop_max


def _heappush_max(heap, item):
    heap.append(item)
    _siftdown_max(heap, 0, len(heap) - 1)


def _heappushpop_max(heap, item):
    if heap and item < heap[0]:
        item, heap[0] = heap[0], item
        _siftup_max(heap, 0)
    return item


class MinHeap(Iterable):
    def __init__(self):
        self.items = []

    def pop(self):
        return heappop(self.items)

    def push(self, item):
        heappush(self.items, item)

    def first(self):
        if len(self.items) > 0:
            return self.items[0]
        else:
            return None

    def last(self):
        if len(self.items) > 0:
            return self.items[-1]
        else:
            return None

    smallest = first

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        for i in self.items:
            yield i


class MaxHeap(Iterable):
    def __init__(self):
        self.items = []

    def pop(self):
        return _heappop_max(self.items)

    def push(self, item):
        _heappush_max(self.items, item)

    def first(self):
        if len(self.items) > 0:
            return self.items[0]
        else:
            return None

    largest = first

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        for i in self.items:
            yield i


class NSmallestHolder:
    def __init__(self, n):
        self.items = []
        self.n = n

    def push(self, item):
        if len(self.items) < self.n:
            self.items.append(item)
            if len(self.items) == self.n:
                _heapify_max(self.items)
        else:
            _heappushpop_max(self.items, item)

    def pop(self):
        return _heappop_max(self.items)

    def first(self):
        if len(self.items) > 0:
            return self.items[0]
        else:
            return None

    largest = first

    def is_full(self):
        return len(self.items) >= self.n

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        for i in self.items:
            yield i


class NLargestHolder:
    def __init__(self, n):
        self.items = []
        self.n = n

    def push(self, item):
        if len(self.items) < self.n:
            self.items.append(item)
            if len(self.items) == self.n:
                heapify(self.items)
        else:
            heappushpop(self.items, item)

    def first(self):
        if len(self.items) > 0:
            return self.items[0]
        else:
            return None

    smallest = first

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        for i in self.items:
            yield i

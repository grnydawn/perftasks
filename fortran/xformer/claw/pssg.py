# coding: utf-8

from __future__ import (unicode_literals, print_function,
        division)

import sys
import copy

import seqgentools as sgt

_PY3 = sys.version_info >= (3, 0)

if _PY3:
#    Object = abc.ABCMeta("Object", (object,), {})
    from functools import reduce
    long = int
else:
    pass
#    Object = abc.ABCMeta("Object".encode("utf-8"),
#            (object,), {})

Object = abc.ABCMeta("Object", (object,), {})

INF = float("inf")
NAN = float("nan")

# provides fixed index for each points in space
# execute function per selected point
# generate failure immediately if indexed point is not allowed

class Axis(sgt.Sequence):

    def __init__(self, sequence):

        assert isinstance(sequence, sgt.Sequence)
        assert sequence.length() < INF

        self.sequence = sequence

    def getitem(self, index):
        return self.sequence[index]

    def copy(self, memo={}):
        return Axis(copy.deepcopy(self.sequence, memo=memo))

    def length(self):
        return self.sequence.length()

class ForbiddenSpace(object):

    def __init__(self, *intervals):
        self.intervals = intervals

    def lower_bound(self):
        return reduce(lambda x,y:(x[0]+1)*(y[0]+1), self.intervals) - 1

    def upper_bound(self):
        return reduce(lambda x,y:x[1]*y[1], self.intervals)

    def __contains__(self, indices):

        for index, interval in zip(indices, self.intervals):
            if index < interval[0] or index >= interval[1]:
                return False
        return True

 
class Space(Object):

    def __init__(self, evaluator, *axes):

        assert callable(evaluator)
        assert all(isinstance(a, Axis) for a in axes))

        self.evaluator = evaluator
        self.axes = axes
        self.space = sgt.Product(axes)

        self.max_size = reduce(lambda x, y: len(x)*len(y), axes) if axes else 0
        self.effective_size = self.max_size

        self._iter_index = -1

        self._forbidden_spaces = []

    def __iter__(self):

        self._iter_index = -1
        return self

    def __next__(self):

        self._iter_index = self.getnext()

        if isinstance(self._iter_index, int):
            val = self.space[self._iter_index]
        else:
            raise StopIteration

    def next(self):
        return self.__next__()

    def getnext(self, index=None):

        if index is index None:
            index = self._iter_index + 1

    def decompose(self, index):

        indices = []

        for length in self._axis_lengths:
            indices.append(index % length)
            index = index // length

        return indieces
        
    def evaluate(self, index):

        point = self.space[index]

        # try combine. if not allowed, then mark forbidden region
        import pdb; pdb.set_trace()


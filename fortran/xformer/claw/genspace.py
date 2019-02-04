# -*- coding: utf-8 -*-

"""generate search space module."""

from __future__ import unicode_literals

import bisect
from itertools import product, permutations

import pyloco
import seqgentools as sgt

from claw_helper import clawxforms

from .pssg import Axis, Space

## TODO: parameterized search space
#
#class XformSpace(sgt.Sequence):
#
#    def __init__(self, loopspace, xforms):
#
#        self._lspace = loopspace # permutations of loops
#        self._xforms = xforms # list of xformation sequence generators
#
#        self._laccumlens = [] # accumulated lengths per each loop permutations
#        self._lxmatch = {} # map between index of _lspace and index of self._xspacecache
#        self._xspacecache = {} # database of xform generators
#
#        for lidx, lpoint in enumerate(self._lspace):
#            nloops = len(lpoint)
#            looptypeid = nloops # For now, characteristic of loop seq is defined by the # of loops, but it can be changed.
#            if looptypeid in self._lxmatch:
#                xformid = self._lxmatch[looptypeid]
#            else:
#                xformid = len(self._xspacecache)
#                self._lxmatch[looptypeid] = xformid
#
#            if xformid in self._xspacecache:
#                xformseq = self._xspacecache[xformid]
#            else:
#                #xformseq = sgt.Product(*[xforms]*nloops) # For now, xformseq is paramterized by the # of loops, but it can be changed
#                import pdb; pdb.set_trace()
#                xformseq = sgt.Product(*[sgt.Product(xforms)]*nloops) # For now, xformseq is paramterized by the # of loops, but it can be changed
#                self._xspacecache[xformid] = xformseq
#
#            nxformseq = xformseq.length() # TODO: apply xform's search length
#
#            if self._laccumlens:
#                self._laccumlens.append(nxformseq+ self._laccumlens[-1])
#            else:
#                self._laccumlens.append(nxformseq)
#    
#    def getitem(self, index):
#
#        # TODO: find idx of self._laccumlens
#        loopidx = bisect.bisect_right(self._laccumlens, index)
#        nlastloop = self._laccumlens[loopidx-1]
#
#        loops = self._lspace[loopidx]
#        looptypeid = len(loops)
#
#        # TODO: get xspace idx from self._xmatch using the found idx
#        xformid = self._lxmatch[looptypeid]
#
#        xformseq = self._xspacecache[xformid]
#        nxformseq = xformid.length()
#
#        # TODO: subtract accumindex from index to get idx within a xspace
#        xformidx = index - nlastloop
#
#        # TODO: from chosen xspace item found in self._xspacecache, get xform item
#        xforms = xformseq[xformidx]
#
#
#        # TODO: return two tuple of (item from self._lspace, item from xfrom item from self._xspacecache)
#        return (loops, xforms)
#
#    def copy(self, memo={}):
#
#        return XformSpace(copy.deepcopy(self._lspace, memo),
#            copy.deepcopy(self._xforms, memo))
#
#    def length(self):
#
#        if self._laccumlens:
#            return self._laccumlens[-1]
#        else:
#            0

class GenSpace(pyloco.PylocoTask):

    def perform(self, targs):

        loops = []

        for filepath, loops in self.env["loops"].items():
            loops.append(loops)

        return 0, Space(Axis(loops), Axis(clawxforms))

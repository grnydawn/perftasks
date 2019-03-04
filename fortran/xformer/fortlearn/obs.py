# coding: utf-8
  
from __future__ import unicode_literals, print_function


import pyloco


class Observation(pyloco.PylocoTask):

    def __init__(self, parent):
        pass

    def perform(self, targs):

        # generate source code observation
        # per time slot: a list of (fileid, linenum, count, stmttype, stmt[0-K]params, upper[0-L]stmttype, before[0-M]stmttype, after[0-N]stmttype)
        # TODO: we need X number of features that can be represented as counts

        # construct objservations 


        # write obs

        import pdb; pdb.set_trace()

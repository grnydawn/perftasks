# -*- coding: utf-8 -*-

"""claw transformation module."""

from __future__ import unicode_literals

import sys
import os

import seqgentools as sgt

from fparser.two.Fortran2003 import Label_Do_Stmt, Nonlabel_Do_Stmt
from fparser.two.utils import walk_ast, FortranSyntaxError

here = os.path.dirname(__file__)
parser_path = os.path.join(here, "..", "..", "parser")
sys.path.append(os.path.abspath(os.path.realpath(parser_path)))

from fparser_analyzer import collect_nested_loops

# claw parameters
max_nestedloops_interchange = 3

class ClawXForm(sgt.Sequence):

    def __init__(self):
        pass

    @classmethod
    def analyze(cls, stmt):
        # return what xformer needes
        raise Exception("Subclass should implement 'analyze' method.")

class LoopInterchange(ClawXForm):

    @classmethod
    def analyze(cls, stmt):
        
        nested_loops = collect_nested_loops(stmt) 

        depth = 0
        nested = nested_loops
        ivars = []
        while nested:
            if len(nested) != 1:
                break
            dostmt = nested.keys()[0]
            loopctrl = dostmt.items[1]
            name = loopctrl.items[1][0].string
            ivars.append(name)
            nested = nested[dostmt]
            depth += 1

        return {"induction_vars": ivars} if len(ivars) > 1 else None
            
#    @classmethod
#    def xformcase(cls, stmt, xformdata):
#
#        induction_vars = xformdata["induction_vars"]
#
#        if len(induction_vars) < 2:
#            return []
#
#        # select N vars
#        # permutate vars
#
#        for num_vars in range(2, len(induction_vars)+1):
#                for selected_vars in permutations(induction_vars, num_vars):
#                    for selected_loop in selected_loops:
#                        xforms = analyses[selected_loop]
#                        for num_xforms in range(len(xforms)):
#                            for xformnames in permutations(xforms, num_xforms+1):
#                                for xformname in xformnames:
#                                    xformdata = xforms[xformname]
#                                    if xformdata:
#                                        for case in clawxforms[xformname].xformcase(selected_loop, xformdata):
#                                            yield case


        import pdb; pdb.set_trace()


clawxforms = [
    LoopInterchange,
]

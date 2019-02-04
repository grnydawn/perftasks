import sys
import os

from collections import OrderedDict as odict

#from fparser.two.utils import walk_ast, FortranSyntaxError
from fparser.two.Fortran2003 import (Label_Do_Stmt, Nonlabel_Do_Stmt,
    Block_Label_Do_Construct, Block_Nonlabel_Do_Construct, Action_Term_Do_Construct,
    Outer_Shared_Do_Construct)

from util import traverse

do_stmts = (Label_Do_Stmt, Nonlabel_Do_Stmt)
do_constructs = (Block_Label_Do_Construct, Block_Nonlabel_Do_Construct,
    Action_Term_Do_Construct, Outer_Shared_Do_Construct)

#def search_nested_loops(node, bag, depth):
#    if isinstance(node, do_stmts):
#        if node.parent not in bag["dopair"] or not bag["dopair"][node.parent]:
#            bag["dopair"][node.parent] = node
#        if node not in bag["donest"]:
#            bag["donest"][node] = []
#    elif isinstance(node, do_constructs):
#
#        bag[node] = []
#
#        try:
#            dopair = bag["dopair"][node]
#        except:
#            bag["dopair"][node] = None
#        else:
#            bag["donest"][dopair] = []
#
#        parent = getattr(node, "parent", None)
#        while parent:
#            if isinstance(parent, do_constructs):
#                bag[parent].append(node)
#                try:
#                    bag["donest"][bag["dopair"][parent]].append(bag["dopair"][node])
#                except:
#                    import pdb; pdb.set_trace()
#                parent = None
#            elif hasattr(parent, "parent"):
#                parent = parent.parent
#            else:
#                parent = None

def search_nested_loops(node, bag, depth):
    if isinstance(node, do_stmts):
        bag["dopair"][node.parent] = node
    elif isinstance(node, do_constructs):
        bag[node] = []
        if node is bag["top"]:
            parent = None
        else:
            parent = getattr(node, "parent", None)
        while parent:
            if isinstance(parent, do_constructs):
                bag[parent].append(node)
                parent = None
            elif hasattr(parent, "parent"):
                parent = parent.parent
            else:
                parent = None

def collect_nested_loops(dostmt):

    bag = odict({"top": dostmt.parent, "dopair": {dostmt.parent: dostmt}})
    ret = traverse(dostmt.parent, search_nested_loops, bag, subnode="content", prerun=True)

    nested = odict()
    queue = [(dostmt.parent, nested)]
    while queue:
        parent, nestdict = queue.pop(0)
        nesteddict = odict()
        nestdict[bag["dopair"][parent]] = nesteddict
        if parent in bag:
            queue.extend([(p, nesteddict) for p in bag[parent]])
    return nested

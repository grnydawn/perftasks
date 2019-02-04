# -*- coding: utf-8 -*-

"""parser module."""

from __future__ import unicode_literals

import pyloco
from fparser.two.Fortran2003 import Label_Do_Stmt, Nonlabel_Do_Stmt
from fparser.two.utils import walk_ast, FortranSyntaxError, Base


class LoopFinder(pyloco.PylocoTask):

    def perform(self, targs):

        loops = {}

        for path, tree in self.env['astlist'].items():
            l = []
            loops[path]= l
            for node in walk_ast(tree.content):

                for x in getattr(node, "content", []):
                    if isinstance(x, Base):
                        x.parent = node
                for x in getattr(node, "items", []):
                    if isinstance(x, Base):
                        x.parent = node

                if isinstance(node, (Label_Do_Stmt, Nonlabel_Do_Stmt)):
                    l.append(node)

        return 0, {"astlist": self.env['astlist'], "loops": loops}

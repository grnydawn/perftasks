# -*- coding: utf-8 -*-

"""parser module."""

from __future__ import unicode_literals

import pyloco
from fparser.two.Fortran2003 import Label_Do_Stmt, Nonlabel_Do_Stmt
from fparser.two.utils import walk_ast, FortranSyntaxError
from claw_helper import clawxforms

class LoopAnalyzer(pyloco.PylocoTask):

    def perform(self, targs):

        analyses = {}

        for path, dostmts in self.env["loops"].items():
            path_analyses = {}
            analyses[path] = path_analyses
            for dostmt in dostmts:
                a = {} 
                path_analyses[dostmt] = a
                for xform in clawxforms:
                    name = xform.__class__.__name__
                    a[name] = xform.analyze(dostmt)

        return 0,   {   "astlist": self.env["astlist"],
                        "loops": self.env["loops"],
                        "analyses": analyses
                    }

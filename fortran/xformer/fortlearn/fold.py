# coding: utf-8
  
from __future__ import unicode_literals, print_function

import os
import shutil

import pyloco

from util import run_shcmd

class FoldApp(pyloco.PylocoTask):

    def __init__(self, parent):

        self.add_option_argument("path", help="folding tool path.")

    def perform(self, targs):

        destdir = os.path.join(self.env['temp'], 'folding')
        os.makedirs(destdir)

        folds = []

        for nth, trace in enumerate(self.env['traces']):

            run_shcmd("%s %s.prv \"User function\""%(targs.path, self.env['exename']), cwd=trace)

            srcfile = os.path.join(self.env['temp'], "traces", str(nth), "%s.folding"%self.env['exename'],
                "%s.codeblocks.fused.any.any.any.slope.csv"%self.env['exename'])

            dstfile = os.path.join(destdir, "%d.csv"%nth)
            shutil.copy2(srcfile, dstfile)

            folds.append(dstfile)

        return 0, { "folds": folds }

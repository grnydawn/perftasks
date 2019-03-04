# coding: utf-8
  
from __future__ import unicode_literals, print_function

import os
import shutil

import pyloco

from util import run_shcmd

class RunApp(pyloco.PylocoTask):

    def __init__(self, parent):

        self.add_option_argument("-n", "--ntimes", metavar="ntimes", evaluate=True,
                help="(E) the number of repeated runs.")

    def perform(self, targs):


        ntimes = targs.ntimes.vargs[0] if targs.ntimes else 1

        datadirs = []

        for nth in range(ntimes):

            # clean
            run_shcmd(self.env['clean_cmd'], cwd=self.env['workdir'])

            # build
            run_shcmd(self.env['build_cmd'], cwd=self.env['workdir'])

            # run
            run_shcmd(self.env['run_cmd'], cwd=self.env['workdir'])

            # copy to temp dir
            srcprefix = os.path.join(self.env['workdir'], self.env['exename'])

            destdir = os.path.join(self.env['temp'], 'traces', str(nth))
            os.makedirs(destdir)

            shutil.copy2(srcprefix+".prv", destdir)
            shutil.copy2(srcprefix+".pcf", destdir)
            shutil.copy2(srcprefix+".row", destdir)

            datadirs.append(destdir)

        return 0, { "traces": datadirs }

# -*- coding: utf-8 -*-

"""kernel mg2 timing collection task module."""

from __future__ import unicode_literals

import os
import glob
import perftask


class KernelMG2Task(perftask.Task):
    """kernel mg2 timing collection task

    """
    def __init__(self, parent, tid, path, fragment, argv):

        self.add_data_argument('kernelpath', metavar='path', help='kernel directory')

        self.add_option_argument('-n', '--nproc', type=int, default=1, help='number of simultaneous kernel runs.')

    def perform(self, targs):

        # rank : thread : []
        timing = {}

        forward = {}
        promote = { "timing": timing}
        retval = 0

        # locate kernel executable
        kernel = os.path.join(targs.kernelpath, "kernel.exe")
        if not os.path.isfile(kernel):
            self.parent.error_exit("Kernel does not exist in %s."%targs.kernelpath)

        nproc = targs.nproc.vargs[0]

        if nproc not in timing:
            etimes = []
            timing[nproc] = etimes
        else:
            etimes = timing[nproc]



        # collection output
        for filepath in glob.glob(os.path.join(targs.kernelpath, "mg2_etime_nrank_*.out")):
            root, ext = os.path.splitext(filepath)
            nproc = int(root.split("_")[-1])

            if nproc not in timing:
                etimes = []
                timing[nproc] = etimes
            else:
                etimes = timing[nproc]

            with open(filepath) as f:
                for line in f:
                    if line.startswith(" micro_mg_tend2_0 :"):
                        etime = line.split()[-1]
                        try:
                            etimes.append(float(etime)*1E-6)
                        except:
                            pass

        return retval, forward, promote



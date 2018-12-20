# -*- coding: utf-8 -*-

"""kernel mg2 timing collection task module."""

from __future__ import unicode_literals

import os
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

        # run the kernel
        #retval, stdout, stderr = perftask.perftask_shellcmd("mpirun -n %d ./kernel.exe"%nproc, cwd=targs.kernelpath)
        retval, stdout, stderr = perftask.perftask_shellcmd("./kernel.exe", cwd=targs.kernelpath)

        for line in stdout.split("\n"):
            if line.startswith(" micro_mg_tend2_0 :"):
                etime = line.split()[-1]
                try:
                    etimes.append(float(etime)*1E-6)
                except:
                    pass


        # collection output

#        for filename in os.listdir(targs.datapath):
#
#            rank, thread = [int(v) for v in filename.split(".")]
#
#            if rank not in timing:
#                rtiming = {}
#                timing[rank] = rtiming
#            else:
#                rtiming = timing[rank]
#
#            if thread not in rtiming:
#                etimes = []
#                rtiming[thread] = etimes
#            else:
#                etimes = rtiming[thread]
#
#            with open(os.path.join(targs.datapath, filename)) as f:
#                for line in f:
#                    idx, start, stop, res = line.split()
#                    etimes.append(float(stop) - float(start))

        return retval, forward, promote



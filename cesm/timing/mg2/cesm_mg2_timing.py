# -*- coding: utf-8 -*-

"""cesm mg2 timing collection task module."""

from __future__ import unicode_literals

import os
import perftask


class CESMMG2Task(perftask.Task):
    """cesm mg2 timing collection task

    """
    def __init__(self, parent, tid, path, fragment, argv):

        self.add_data_argument('datapath', metavar='path', help='data file directory')

        #self.add_option_argument('-t', '--title', metavar='title',
        #                         help='title  plotting.')

    def perform(self, targs):

        # rank : thread : []
        timing = {}

        forward = {}
        promote = { "timing": timing}
        retval = 0

        for filename in os.listdir(targs.datapath):

            rank, thread = [int(v) for v in filename.split(".")]

            if rank not in timing:
                rtiming = {}
                timing[rank] = rtiming
            else:
                rtiming = timing[rank]

            if thread not in rtiming:
                etimes = []
                rtiming[thread] = etimes
            else:
                etimes = rtiming[thread]

            with open(os.path.join(targs.datapath, filename)) as f:
                for line in f:
                    idx, start, stop, res = line.split()
                    etimes.append(float(stop) - float(start))

        return retval, forward, promote



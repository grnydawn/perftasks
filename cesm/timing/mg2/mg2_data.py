# -*- coding: utf-8 -*-

"""cesm mg2 data aggregation task module."""

from __future__ import unicode_literals

import os
import perftask


class MG2DataTask(perftask.Task):
    """mg2 data aggregation task

    """
    def __init__(self, parent, tid, path, fragment, argv):

        self.add_data_argument('datapath', metavar='path', nargs="?", help='data file')

        self.add_option_argument('-a', '--allnum', evaluate=True, help='list all values')

    def collect_values(self, data, values):

        if isinstance(data, (int, float)):
            values.append(data*1000)
        elif isinstance(data, dict):
            for d in data.values():
                self.collect_values(d, values)
        elif isinstance(data, (list, tuple)):
            for d in data:
                self.collect_values(d, values)

    def perform(self, targs):

        # rank : thread : []
        values = []


        forward = {}
        promote = { "values": values}
        retval = 0

        if targs.allnum:

            self.collect_values(targs.allnum.vargs[0], values)
        else:

            self.collect_values(targs.datapath.vargs[0], values)

        return retval, forward, promote



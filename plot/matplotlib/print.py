# -*- coding: utf-8 -*-

"""print standard task module."""

from __future__ import unicode_literals

import pyloco

class PrintTask(pyloco.PylocoTask):
    """show content of data
    """

    def __init__(self):

        self.add_data_argument("data", nargs="*", help="input data.")
        self.add_option_argument("--version", action="version", version="print task version 0.1.0")

    def perform(self, targs):

        if targs.data:
            out = " ".join([str(d) for d in targs.data])
            pyloco.pyloco_print(out)

        return 0

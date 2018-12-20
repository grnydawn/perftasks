# -*- coding: utf-8 -*-

"""print task module."""

from __future__ import unicode_literals

from perftask import Task
import docopt

class PrintTask(Task):

    def __init__(self, parent, tid, path, fragment, argv):

        self.add_data_argument('data', nargs="*", evaluate=True, autoimport=True, help='input data.')

        self.add_option_argument('-s', '--str', metavar='object', action='append', help='run str() function.')
        self.add_option_argument('--version', action='version', version='print example task version 0.1.0')

    def perform(self, targs):

        printed = False

        if targs.str:
            for option in targs.str:
                for varg in option.vargs:
                    print(str(self.teval(varg)))
                    printed = True

        if not printed:
            if targs.data:
                print(str(targs.data))
            elif self.env["D"]:
                print(str(self.env["D"]))
            else:
                print("No data to print.")

        return 0


# -*- coding: utf-8 -*-

"""Linux perf-stat driver task"""

import os
import shlex
import pyloco


class PerfStatTask(pyloco.PylocoTask):

    def __init__(self):

        self.add_data_argument("target", metavar="path", help="path to an executable.")

        self.add_option_argument("-e", "--event", action="append", help="event to sample")
        self.add_option_argument("--target-args", help="target arguments")

        self.eventnames = {}
        self.eventvalues = {}

    def perform(self, targs):

        # run program under perf with arguments
        target = os.path.abspath(targs.target)

        if targs.target_args:
            target_args = targs.target_args
        else:
            target_args = ""

        ecodes = []
        if targs.event:
            for event in targs.event:
                for ecode in event.vargs:
                    ecodes.append(ecode)
                    self.eventnames[ecode] = ecode
                for ename, ecode in event.kwargs.items():
                    ecodes.append(ecode)
                    self.eventnames[ecode] = ename
                
        if not ecodes:
            print ("ERROR: No event code is given.")
            return -1

        events = ",".join(ecodes)

        cmd = "perf stat -e {events} -- {target} {target_args}".format(
                events=events, target=target, target_args=target_args)

        out, stdout, stderr = pyloco.util.runcmd(shlex.split(cmd))

        # parse results into dictionary and forward
        if out == 0:
            for line in stderr.split("\n"):
                items = line.split()
                if len(items) >= 2 and items[1] in self.eventnames:
                    self.eventvalues[items[1]] = int(items[0].replace(",", ""))

        else:
            print("Return value is not '0'.")
            print("STDOUT: ", stdout)
            print("STDERR: ", stderr)
            return out

        return 0, {"eventnames": self.eventnames, "eventvalues": self.eventvalues}

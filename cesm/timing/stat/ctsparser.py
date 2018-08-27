# -*- coding: utf-8 -*-

"""cesm timing parser task module."""

# TODO: split stat from timing
# TODO: add caller/callee viewer

import os
import re
import gzip
import perftask
import StringIO

_re_cesm_stat1 = re.compile(r"\s*Total\sranks\sin\scommunicator")
_re_cesm_stat2 = re.compile(r"\s*\*+\sGLOBAL\sSTATISTICS\s+\(")
_re_stat_table1 = re.compile(r"name\s+ncalls\s+nranks\s+mean_time")
_re_stat_table2 = re.compile(r"name\s+processes\s+threads\s+count")

class CesmTimingPaserTask(perftask.TaskFrame):

    def __init__(self, parent, url, argv):

        self.set_data_argument("timingfile", metavar="path", evaluate=False, nargs="+", help="CESM stat timing file.")

        try:
            import pandas
            self.env["pandas"] = self.env["pd"] = pandas
        except ImportError as err:
            self.error_exit("pandas module is not found.")

    def perform(self):

        def cesm_stat(c):
            if _re_cesm_stat1.match(c):
                return True
            elif _re_cesm_stat2.match(c):
                return True

        contents = {}

        # read timing file
        for path in self.targs.timingfile:
            path = os.path.abspath(os.path.realpath(path))
            try:
                with gzip.open(path) as zf:
                    contents[path] = zf.read()
            except IOError as err:
                with open(path) as f:
                    contents[path] = f.read()

        # handle other options

        # default action

        for path, content in contents.items():
            if cesm_stat(content):
                self.env["D"].append(self._cesm_stat(content))
            else:
                self.error_exit("Unknown cesm timing file: %s"%path)

        if not self.env["D"]:
            self.error_exit("No cesm timing input is found.")

        self.add_forward("D", self.env["D"])
   
        return 0     
    
    def _cesm_stat(self, content):
        def _read_table(c, start):
            table_str = c[start:].replace("(", "").replace(")", "").strip()
            return self.env["pd"].read_csv(StringIO.StringIO(table_str), sep="\s+",
                error_bad_lines=False, index_col=0) 

        stat_df = None
        match = _re_stat_table1.search(content)
        if match:
            stat_df = _read_table(content, match.start())
        else:
            match = _re_stat_table2.search(content)
            if match:
                stat_df = _read_table(content, match.start())
        return stat_df

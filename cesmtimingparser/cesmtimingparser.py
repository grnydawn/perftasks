# -*- coding: utf-8 -*-

"""cesm timing parser task module."""

import os
import re
import gzip
import perftask
import StringIO

_re_cesm_stat = re.compile(r"\s*Total\sranks\sin\scommunicator")
_re_cesm_timing = re.compile(r"\s*-+\sCCSM\sTIMING\sPROFILE")
_re_stat_table = re.compile(r"name\s+ncalls\s+nranks\s+mean_time")

class CesmTimingPaserTask(perftask.TaskFrameUnit):

    def __init__(self, ctr, parent, url, argv, env):

        self.targs = self.parser.parse_args(argv)

        self.timing_files = self.targs.data
        self.targs.data = []


        try:
            import pandas
            self.env["pandas"] = self.env["pd"] = pandas
        except ImportError as err:
            self.error_exit("pandas module is not found.")

    def perform(self):

        def cesm_stat(c):
            return _re_cesm_stat.match(c)

        def cesm_timing(c):
            return _re_cesm_timing.match(c)

        contents = {}

        self.env["D"] = [None, None]

        # read timing file
        for path in self.timing_files:
            path = os.path.abspath(os.path.realpath(path))
            try:
                with gzip.open(path) as zf:
                    contents[path] = zf.read()
            except IOError as err:
                with open(path) as f:
                    contents[path] = f.read()

        for path, content in contents.items():
            if cesm_stat(content):
                self.env["D"][0] = self._cesm_stat(content)
            elif cesm_timing(content):
                self.env["D"][1] = self._cesm_timing(content)
            else:
                self.error_exit("Unknown cesm timing file: %s"%path)

        if all(d is None for d in self.env["D"]):
            self.error_exit("No cesm timing input is found.")

        self.targs.forward = ["D=D"]
   
        return 0     
    
    def _cesm_stat(self, content):
        stat_df = None
        match = _re_stat_table.search(content)
        if match:
            table_str = content[match.start():].replace("(", "").replace(")", "").strip()
            stat_df = self.env["pd"].read_csv(StringIO.StringIO(table_str), sep="\s+",
                error_bad_lines=False, index_col=0) 
        return stat_df

    def _cesm_timing(self, content):
        self.error_exit("Under development")

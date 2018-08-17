# -*- coding: utf-8 -*-

"""cesm timing parser task module."""

# TODO: split stat from timing
# TODO: add caller/callee viewer

import os
import re
import gzip
import perftask
import StringIO

_re_cesm_timing1 = re.compile(r"[\s\r\n]*-+\sCCSM\sTIMING\sPROFILE\s-+[\r\n]+")
_re_cesm_timing2 = re.compile(r"[\s\r\n]*-+\sDRIVER\sTIMING\sFLOWCHART")
_re_cesm_timing3 = re.compile(r"\s*(?P<name>[^:\n]+):(?P<value>[^\n]+)")

class CesmTimingPaserTask(perftask.TaskFrameUnit):

    def __init__(self, ctr, parent, url, argv, env):

        try:
            import pandas
            self.env["pandas"] = self.env["pd"] = pandas
        except ImportError as err:
            self.error_exit("pandas module is not found.")

    def pop_inputdata(self, data):

        self.timing_files = []
        while data:
            self.timing_files.append(data.pop(0))

    def perform(self):

        def cesm_timing(c):
            match = _re_cesm_timing1.match(c)
            search = _re_cesm_timing2.search(c)
            if match and search:
                return match.end(), search.start()

        contents = {}

        if not self.timing_files and self.env["D"]:
            self.timing_files = self.env["D"]

        self.env["D"] = []

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
            block = cesm_timing(content)
            if block:
                self.env["D"].append(self._cesm_timing(content[block[0]:block[1]]))
            else:
                self.error_exit("Unknown cesm timing file: %s"%path)

        if not self.env["D"]:
            self.error_exit("No cesm timing input is found.")

        self.targs.forward = ["D=D"]
   
        return 0     
    
    def _cesm_stat(self, content):
        def _read_table(c, start):
            table_str = c[start:].replace("(", "").replace(")", "").strip()
            return self.env["pd"].read_csv(StringIO.StringIO(table_str), sep="\s+",
                error_bad_lines=False, index_col=0) 

        stat_df = None
        match = _re_stat_table1.search(content)
        if match:
            #table_str = content[match.start():].replace("(", "").replace(")", "").strip()
            #stat_df = self.env["pd"].read_csv(StringIO.StringIO(table_str), sep="\s+",
            #    error_bad_lines=False, index_col=0) 
            stat_df = _read_table(content, match.start())
        else:
            match = _re_stat_table2.search(content)
            if match:
                stat_df = _read_table(content, match.start())
        return stat_df

    def _cesm_timing(self, content):
        maps = {}
        for key, value in _re_cesm_timing3.findall(content):
            maps[key.replace(" ", "")] = value.split()
        return maps

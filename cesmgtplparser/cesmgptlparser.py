# -*- coding: utf-8 -*-

"""cesm gptl parser task module."""

import os
import re
import perftask

FORTRAN_EXTS = [ ".F", ".F90", ".f", ".f90"]

re_startf = re.compile("call\\s+t_startf\\s*\\(\\s*[\"'](?P<name>[^\"']+)[\"']\\s*\\)")
re_stopf = re.compile("call\\s+t_stopf\\s*\\(\\s*[\"'](?P<name>[^\"']+)[\"']\\s*\\)")

class CesmGptlParserTask(perftask.TaskFrameUnit):

    def __init__(self, ctr, parent, url, argv, env):

        self.targs = self.parser.parse_args(argv)

        self.cesm_srcpaths = self.targs.data
        self.targs.data = []
        self.gptl_blocks = {} # name: (path, start, end)

    def _parse_gptl(self, fpath):

        temp = {}

        base, ext = os.path.splitext(fpath)
        if ext in FORTRAN_EXTS:
            with open(fpath) as f:
                for lnum, line in enumerate(f):
                    search_startf = re_startf.search(line)
                    if search_startf:
                        tname = search_startf.group("name")
                        if tname:
                            temp[tname] = lnum
                    else:
                        search_stopf = re_stopf.search(line)
                        if search_stopf:
                            tname = search_stopf.group("name")
                            if tname and tname in temp:
                                self.gptl_blocks[tname] = [fpath, temp[tname], lnum] 
                                del temp[tname]

    def perform(self):

        for path in self.cesm_srcpaths:
            if os.path.exists(path):
                if os.path.isfile(path):
                    self._parse_gptl(path)
                elif os.path.isdir(path):
                    for root, subdirs, files in os.walk(path):
                        subdirs[:] = [d for d in subdirs if not d.startswith(".")]
                        #for subdir in subdirs:
                        #    dirpath = os.path.join(root, subdir)
                        for fname in files:
                            fpath = os.path.join(root, fname)
                            self._parse_gptl(fpath)
            else:
                self.error_exit("'%s' does not exist."%path)

        self.env['gptl'] = self.gptl_blocks
        self.targs.forward = ["D=[gptl]"]

#import os
#import re
#import gzip
#import StringIO
#
#_re_cesm_stat = re.compile(r"\s*Total\sranks\sin\scommunicator")
#_re_cesm_timing = re.compile(r"\s*-+\sCCSM\sTIMING\sPROFILE")
#_re_stat_table = re.compile(r"name\s+ncalls\s+nranks\s+mean_time")
#
#grep -R phys_grid_init . | grep t_st
#
#text search 
#show head and tail
#upward search for subroutine(function) and module and filepath
#
#class CesmGptlPaserTask(perftask.TaskFrameUnit):
#
#    def __init__(self, ctr, parent, url, argv, env):
#
#        self.targs = self.parser.parse_args(argv)
#
#        self.timing_files = self.targs.data
#        self.targs.data = []
#
#
#        try:
#            import pandas
#            self.env["pandas"] = self.env["pd"] = pandas
#        except ImportError as err:
#            self.error_exit("pandas module is not found.")
#
#    def perform(self):
#
#        def cesm_stat(c):
#            return _re_cesm_stat.match(c)
#
#        def cesm_timing(c):
#            return _re_cesm_timing.match(c)
#
#        contents = {}
#
#        self.env["D"] = [None, None]
#
#        # read timing file
#        for path in self.timing_files:
#            path = os.path.abspath(os.path.realpath(path))
#            try:
#                with gzip.open(path) as zf:
#                    contents[path] = zf.read()
#            except IOError as err:
#                with open(path) as f:
#                    contents[path] = f.read()
#
#        for path, content in contents.items():
#            if cesm_stat(content):
#                self.env["D"][0] = self._cesm_stat(content)
#            elif cesm_timing(content):
#                self.env["D"][1] = self._cesm_timing(content)
#            else:
#                self.error_exit("Unknown cesm timing file: %s"%path)
#
#        self.targs.forward = ["D=D"]
#   
#        return 0     
#    
#    def _cesm_stat(self, content):
#        stat_df = None
#        match = _re_stat_table.search(content)
#        if match:
#            table_str = content[match.start():].replace("(", "").replace(")", "").strip()
#            stat_df = self.env["pd"].read_csv(StringIO.StringIO(table_str), sep="\s+", error_bad_lines=False) 
#        return stat_df
#
#    def _cesm_timing(self, content):
#        self.error_exit("Under development")

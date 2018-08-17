# -*- coding: utf-8 -*-

"""cesm gptl parser task module."""

import os
import re
import perftask

FORTRAN_EXTS = [ ".F", ".F90", ".f", ".f90"]

re_startf = re.compile("call\\s+t_startf\\s*\\(\\s*[\"'](?P<name>[^\"']+)[\"']\\s*\\)")
re_stopf = re.compile("call\\s+t_stopf\\s*\\(\\s*[\"'](?P<name>[^\"']+)[\"']\\s*\\)")

# TODO: upward search for subroutine(function) and module and filepath

class CesmGptlParserTask(perftask.TaskFrameUnit):

    def __init__(self, ctr, parent, url, argv, env):

        self.gptl_blocks = {} # name: (path, start, end)

    def pop_inputdata(self, data):

        self.cesm_srcpaths = []
        while data:
            self.cesm_srcpaths.append(data.pop(0))

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

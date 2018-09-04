# -*- coding: utf-8 -*-

"""cesm gptl parser task module."""

import os
import re
import perftask

FORTRAN_EXTS = [ ".F", ".F90", ".f", ".f90"]

re_startf = re.compile("call\\s+t_startf\\s*\\(\\s*[\"'](?P<name>[^\"']+)[\"']\\s*[\\),]", re.I)
re_stopf = re.compile("call\\s+t_stopf\\s*\\(\\s*[\"'](?P<name>[^\"']+)[\"']\\s*[\\),]", re.I)
re_subp = re.compile("[\\s\\w]*(subroutine|function)\s*\w+\\(", re.I)

# TODO: upward search for subroutine(function) and module and filepath

def parent(loc):
    with open(loc[0], 'r') as f:
        lines = f.readlines()
        idx = loc[1]
        end = idx
        while idx >= 0:
            match = re_subp.search(lines[idx])
            if match:
                return lines[idx].strip()
            end = idx
            idx -= 1
   
    return "NOT FOUND"
 
class CesmGptlParserTask(perftask.TaskFrame):

    def __init__(self, parent, url, argv):

        self.add_data_argument("path", metavar="path", nargs="+", help="Fortran source tree")

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

        self.cesm_srcpaths = self.targs.path

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

        self.add_forward('D', self.gptl_blocks)

        return 0

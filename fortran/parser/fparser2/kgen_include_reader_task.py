# coding: utf-8

from __future__ import unicode_literals, print_function

import sys 
import os
import pyloco
import configparser

INTERNAL_NAMELEVEL_SEPERATOR = '__kgen__' # lower-case only

class Fparser2Task(pyloco.Task):

    def __init__(self, parent, tid, path, fragment, argv):

        self.add_data_argument("inifile", metavar="path", help="KGen include file.")

        self.add_option_argument("--only", metavar="path", action="append", help="select a source file.")
        self.add_option_argument("--macro", action="append", help="create a new macro to apply.")
        self.add_option_argument("--include", action="append", help="create a include to apply.")

    def perform(self, targs):

        srclist = {}

        if targs.inifile:

            print("Reading KGen include file: {}...".format(targs.inifile), end=" ")
            sys.stdout.flush()

            ini = KGenConfigParser()
            ini.read(targs.inifile)

            onlylist = []
            if targs.only:
                for only in targs.only:
                    for varg in only.vargs:
                        onlylist.append(varg) 

            for path, sec in ini.items():

                if path in ("DEFAULT",):
                    continue

                if targs.only:
                    skip = True
                    for only in onlylist:
                        basename = os.path.basename(path)
                        if only in (path, basename):
                            skip = False
                            break
                    if skip:
                        continue
                        
                macros = {}
                includes = []

                if targs.include:
                    for varg in targs.include.vargs:
                        includes.extend([s.strip() for s in varg.split(":")])

                for key, opt in sec.items():

                    if key in ("compiler", "compiler_options"):
                        continue

                    if key == "include":
                        includes.extend([s.strip() for s in opt.split(":")])
                    elif opt:
                        macros[key] = opt 
                    else:
                        macros[key] = None 

                if targs.macro:
                    for macro in targs.macro:
                        for varg in macro.vargs:
                            macros[varg] = None
                        for var, val in macro.kwargs.items():
                            macros[var] = val
                            
                srclist[path] = (macros, includes)

            print("DONE")
        else:
            print("ERROR: no input file.")

        return 0, {"srclist": srclist}, {}


class KGenConfigParser(configparser.ConfigParser):

    def __init__(self, *args, **kwargs):

        super(KGenConfigParser, self).__init__(*args, **kwargs)

        self.optionxform = lambda option: option

    def _optname_colon_to_dot(self, line):

        newline = line.strip()

        if len(newline)>0:
            if newline[0]==';': # comment
                return line
            elif newline[0]=='[' and newline[-1]==']': # filepath
                return line.replace(':', INTERNAL_NAMELEVEL_SEPERATOR)
            else: # else
                pos = line.find('=')
                if pos>0:
                    return line[:pos].replace(':', INTERNAL_NAMELEVEL_SEPERATOR) + line[pos:]
                else:
                    raise UserException('KGEN requires an equal symbol at each option line')
        else:
            return line

    def read(self, filename):

        from StringIO import StringIO

        fp = open(filename)

        lines = []
        for line in fp.readlines():
            lines.append(self._optname_colon_to_dot(line))
        fp.close()

        buf = StringIO(''.join(lines))
        self._read(buf, filename)

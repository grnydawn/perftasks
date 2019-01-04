# coding: utf-8

from __future__ import unicode_literals, print_function

import sys
import os
import pyloco
import subprocess

from fparser.two.parser import ParserFactory
from fparser.two.utils import walk_ast, FortranSyntaxError
from fparser.common.readfortran import FortranFileReader, FortranStringReader

def run_shcmd(cmd, input=None, **kwargs):

    show_error_msg = None
    if kwargs.has_key('show_error_msg'):
        show_error_msg = kwargs['show_error_msg']
        del kwargs['show_error_msg']

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, \
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, **kwargs)
    out, err = proc.communicate(input=input)

    if proc.returncode != 0 and show_error_msg:
        print('>> %s' % cmd)
        print('returned non-zero code from shell('+str(ret_code)+')\n OUTPUT: '+str(out)+'\n ERROR: '+str(err)+'\n')

    return out, err, proc.returncode

class Fparser2Task(pyloco.PylocoTask):

    def __init__(self):

        self.add_data_argument("src", metavar="path", nargs="*", help="Fortran source file input.")

        self.add_option_argument("-D", "--macro", nargs="*", help="Fortran macro definition.")
        self.add_option_argument("-I", "--include", nargs="*", help="Fortran source include paths.")
        self.add_option_argument("-S", "--save", action="store_true", help="Save preprocesed file.")
        self.add_option_argument("-a", "--alias", metavar="prefix", action="append", help="path alias.")

    def gen_aliases(self, paths, aliases):

        aliased_paths = []
        for path in paths:
            for old, new in aliases.items():
                if path.startswith(old):
                    aliased_paths.append(new+"/"+path[len(old):])
                    break
        return aliased_paths

    def perform(self, targs):

        astlist = {}

        if len(targs.src) == 0 and "srclist" not in self.env:
            print("ERROR: no input source file.")
        else:

            srclist = {}

            if "srclist" in self.env:
                srclist.update(self.env["srclist"])

            aliases = {}
            if targs.alias:
                for alias in targs.alias:
                    for varg in alias.vargs:
                        adef = varg.split("=", 1)
                        if len(adef)==2:
                            aliases[adef[1].strip()] = adef[0].strip()
                        else:
                            raise pyloco.UsageError("Wrong alias syntax: %s"%c)

            macros = {}
            if targs.macro:
                for m in targs.macro:
                    for c in m.split(","):
                        mdef = c.split("=", 1)
                        if len(mdef)==2:
                            macros[mdef[0].strip()] = mdef[1].strip()
                        else:
                            macros[mdef[0].strip()] = None
            includes = []
            if targs.include:
                for i in targs.include:
                    paths = i.split(":")
                    apaths = self.gen_aliases(paths, aliases)
                    includes.extend([s.strip() for s in paths])
                    includes.extend([s.strip() for s in apaths])

            if targs.src:
                for path in targs.src:
                    srclist[path] = (macros, includes)

            for path, (macros, includes) in srclist.items():

                print("parsing '{}'...".format(path), end=" ")
                sys.stdout.flush()

                try:

                    pp = "cpp"
                    flags = "-w -traditional -P"

                    pack_macros = []
                    for k, v in macros.items():
                        if v is None:
                            pack_macros.append("-D{}".format(k))
                        else:
                            pack_macros.append("-D{0}={1}".format(k,v))

                    pack_includes = []
                    aincludes = self.gen_aliases(includes, aliases)
                    for p in includes+aincludes:
                        pack_includes.append("-I{}".format(p))
                   
                    if not os.path.isfile(path):
                        apaths = self.gen_aliases([path], aliases)
                        if apaths and os.path.isfile(apaths[0]):
                            path = apaths[0]
                        else: 
                            print("'%s' does not exist.".format(path))
                            continue

                    with open(path) as fr:
                        output, err, retcode = run_shcmd('%s %s %s %s' % (pp, flags, " ".join(pack_includes), " ".join(pack_macros)), input=fr.read())

                        if targs.save:
                            root, ext = os.path.splitext(os.path.basename(path))
                            savepath = root+".pre"
                            with open(savepath, 'w') as fw:
                                fw.write(output)
                        
                        reader = FortranStringReader(output, ignore_comments=False)
                        f2008_parser = ParserFactory().create(std="f2008")
                        ast = f2008_parser(reader)

                        astlist[path] = ast
                        print("PASSED")
                except FortranSyntaxError as err:
                    print("FAILED Syntax with '{}'.".format(str(err)))
                except NameError as err:
                    print("FAILED Name with '{}'.".format(str(err)))
                except IOError as err:
                    print("FAILED I/O with '{}'.".format(str(err)))
                except Exception as err:
                    print("FAILED with '{}'.".format(str(err)))

        return 0, {"astlist": astlist}, {}

    def handle_include(self, lines):
        import re
        import os

        insert_lines = []
        for i, line in enumerate(lines):
            match = re.match(r'^\s*include\s*("[^"]+"|\'[^\']+\')', line, re.I)
            if match:
                if Config.include['file'].has_key(self.realpath):
                    include_dirs = Config.include['file'][self.realpath]['path']+Config.include['path']
                else:
                    include_dirs = Config.include['path']

                if os.path.isfile(Config.mpi['header']):
                    include_dirs.insert(0, os.path.dirname(Config.mpi['header']))

                filename = match.group(1)[1:-1].strip()
                path = filename
                for incl_dir in include_dirs+[os.path.dirname(self.realpath)]:
                    path = os.path.join(incl_dir, filename)
                    if os.path.exists(path):
                        break
                if os.path.isfile(path):
                    with open(path, 'r') as f:
                        included_lines = f.read()
                        insert_lines.extend(self.handle_include(included_lines.split('\n')))
                else:
                    raise UserException('Can not find %s in include paths of %s.'%(filename, self.realpath))
            else:
                insert_lines.append(line)

        return insert_lines


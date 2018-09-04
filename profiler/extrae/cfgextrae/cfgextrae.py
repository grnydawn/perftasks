# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import os
import fnmatch
import xml.etree.ElementTree as ET

import perftask

here = os.path.dirname(__file__)
template_dir = os.path.join(here, "template")

builtin_templates = {}

class CfgExtraeTask(perftask.TaskFrame):

    def __init__(self, parent, url, argv):

        # either this or EXTRAE_TEMPLATE env. should be set. This could be builtin template name
        self.add_data_argument("template", metavar="path", nargs="?", help="template file of extrae configuration file")

        self.add_option_argument("-o", "--output", metavar="path", help="path to save extrae configuration file")
        self.add_option_argument("--extrae", metavar="path", help="path to extrae top directory")
        self.add_option_argument("--tag", metavar="element", help="show tag of an element") # syntax: elementpath@....
        self.add_option_argument("--text", metavar="element", help="show text of an element") # syntax: elementpath@....
        self.add_option_argument("--attrib", metavar="element", help="show attributes of an element") # syntax: elementpath@....
        self.add_option_argument("--clear", metavar="element", help="reset an element") # syntax: elementpath@....
        self.add_option_argument("--get", metavar="element", help="reset an element") # syntax: elementpath@....


--tag
--text
--attrib
--clear
--get
--set
get(key, default=None)
items()
keys()
set(key, value)
append(subelement)
extend(subelements)
find(match)
findall(match)
findtext(match, default=None)
insert(index, element)
iter(tag=None)
iterfind(match)
itertext()
makeelement(tag, attrib)
remove(subelement)



        for fname in os.listdir(template_dir):
            fpath = os.path.join(template_dir, fname)
            try:
                builtin_templates[fname] =  ET.parse(fpath).getroot()
            except ET.ParseError:
                pass

    def perform(self):


        extrae_template = None
        if self.targs.template:
            extrae_template =  ET.parse(self.targs.template).getroot()
        elif "EXTRAE_TEMPLATE" in os.environ:
            extrae_template =  ET.parse(os.environ["EXTRAE_TEMPLATE"]).getroot()
        else:
            import socket
            fqdn = socket.getfqdn()
            for builtin_template in builtin_templates:
                if fnmatch.fnmatch(fqdn, builtin_template):
                    extrae_template = builtin_templates[builtin_template]
                    break
            if extrae_template is None:
                self.parent.error_exit("Neither '--template' option is provided nor 'EXTRAE_TEMPLATE' env. variable is set.")

        extrae_home = None
        if self.targs.extrae:
            extrae_home = self.targs.extrae
            extrae_template.set('home', extrae_home)
        elif "EXTRAE_HOME" in os.environ:
            extrae_home = os.environ["EXTRAE_HOME"]
            extrae_template.set('home', extrae_home)
        elif extrae_template is not None:
            extrae_home = extrae_template.get('home')
        else:
            self.parent.error_exit("Neither '--extrae' option is provided nor 'EXTRAE_HOME' env. variable is set.")

        import pdb; pdb.set_trace()
        return 0

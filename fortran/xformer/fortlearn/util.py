# coding: utf-8
  
from __future__ import unicode_literals, print_function

import subprocess

def run_shcmd(cmd, input=None, **kwargs):

    show_error_msg = None
    if 'show_error_msg' in kwargs:
        show_error_msg = kwargs['show_error_msg']
        del kwargs['show_error_msg']

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, \
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, **kwargs)
    out, err = proc.communicate(input=input)

    if proc.returncode != 0 and show_error_msg:
        print('>> %s' % cmd)
        print('returned non-zero code from shell('+str(ret_code)+')\n OUTPUT: '+str(out)+'\n ERROR: '+str(err)+'\n')

    return out, err, proc.returncode

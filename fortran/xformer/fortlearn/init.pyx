import os
import tempfile
import shutil

here = os.getcwd()

fparser@text = ${here}/../../parser/fparser2/fparser2_task.py

temp = tempfile.mkdtemp()

# backup original target
org = os.path.join(temp, "org")

if os.path.isfile(target):
    os.makedirs(org)
    shutil.copy2(target, org)
elif os.path.isdir(target):
    shutil.copytree(target, org)

result@pyloco = ${fparser} "${target}" \
                -- run.py -n 2 \
                -- fold.py "${folding_exe}" \ # include multiple run and folding
                -- obs.py # generate reward and observations of folding + code structure

# save observation as reference performance


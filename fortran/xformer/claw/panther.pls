target@arg = help="target source file or root directory."
clean@arg  = help="command to clean compilation intermittent files including object files."
build@arg  = help="command to build an executable."
run@arg    = help="command to run an executable."

[import*]
os
re

[init]
cheyenne@py = os.uname()[1].startswith("cheyenne") or re.match(r"r\d+i\d+n\d+", os.uname()[1]) is not None
mymacpro@py = os.uname()[1].startswith("cisl-blaine")

[setup@cheyenne]
fparser@text= /glade/u/home/youngsun/repos/github/perftasks/fortran/parser/fparser2/fparser2_task.py

result     = {fparser} {target:arg}
			-- loopfinder
			-- clawoptimizer {clean} {build} {run}

clean@arg  = help="command to clean compilation intermittent files including object files."
build@arg  = help="command to build an executable."
run@arg    = help="command to run an executable."

[import*]
os
re

[init]
cheyenne@py = os.uname()[1].startswith("cheyenne") or re.match(r"r\d+i\d+n\d+", os.uname()[1]) is not None
mymacpro@py = os.uname()[1].startswith("cisl-blaine")
fully_optimized@py = False

[setup@cheyenne]
#fparser@text= /glade/u/home/youngsun/repos/github/perftasks/fortran/parser/fparser2/fparser2_task.py

[genspace]
space       = genspace.py

#[while not fully_optimized and not space[1]["all_searched"]]
#[for case in space["space"]]
[for case in space]

    [if fully_optimized]
        [break]
    [end]

	[annotation]
        pr@py = print(case)

	[claw]

	[clean]

	[build]

	[run]
        fully_optimized@py = True
[end]

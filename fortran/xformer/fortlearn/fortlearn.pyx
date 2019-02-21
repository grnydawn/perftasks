target = "dgkernel/dg_kernel.F90"
clean_cmd = "make clean"
build_cmd = "make build"
run_cmd = "make run"

init@pyloco = init.pyx

finished = False

while not finished:

    step@pyloco = step.pyx

    finished = step.get("done", False)

fini@pyloco = fini.pyx



#            -- loopfinder.py \
#            -- clawoptimizer.pyx "{clean:arg}" "{build:arg} FC=gfortran" "{run:arg}"
# "${clean_cmd}" "${build_cmd}" "${run_cmd}"
#/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib

bdw_events@py = ["cycles", "instructions", "vec_ins=r533cc7", "branches",
                 "L1-dcache-loads", "L1-dcache-load-misses", "L1-dcache-stores",
                 "l1-icache-hits=r530180", "l1-icache-misses=r530280"]
bdw_event_arg@py = ",".join(bdw_events)
perfstat.py fortran/basic/sum.exe  -e {bdw_event_arg} -- print {{eventvalues}},{{eventnames}} --assert-stdout "False"

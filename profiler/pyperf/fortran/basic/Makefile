
LIBPFM4_DIR=/glade/p/cisl/asap/youngsun/opt/libpfm4/4.10.1
CHECK_EVENTS=${LIBPFM4_DIR}/examples/check_events
SHOWEVTINFO=${LIBPFM4_DIR}/examples/showevtinfo

# event codes
CYCLES=cycles
ALL_INS=instructions#r5300c0#INSTRUCTION_RETIRED
VEC_INS=r533cc7#FP_ARITH:PACKED
BR_INS=branches#r5300c4#BRANCH_INSTRUCTIONS_RETIRED
LOAD_INS=r5381d0#MEM_UOPS_RETIRED:ALL_LOADS
STORE_INS=r5382d0#MEM_UOPS_RETIRED:ALL_STORES
L1D_LOADS=L1-dcache-loads
L1D_LOAD_MISSES=L1-dcache-load-misses
L1D_STORES=L1-dcache-stores
L1I_HIT=r530180#ICACHE:HIT
L1I_MISSES=r530280#ICACHE:MISSES

arg ?= 

check:
	${CHECK_EVENTS} ${arg}

show:
	${SHOWEVTINFO} ${arg}

intel: clean
	ifort -O3 -qopt-report=5 -o sum.exe sum.f90
	#perf stat -e ${ALL_INS},${VEC_INS},${BR_INS},${LOAD_INS},${STORE_INS},${L1D_LOADS},${L1D_STORES},${L1D_LOAD_MISSES},${L1D_STORE_MISSESS} -- ./sum.exe
	perf stat -e ${CYCLES},${ALL_INS},${VEC_INS},${BR_INS},${L1D_LOADS},${L1D_LOAD_MISSES},${L1D_STORES},${L1I_HIT},${L1I_MISSES} -- ./sum.exe
	#perf stat -e ${CYCLES},${ALL_INS},${VEC_INS},${BR_INS},${LOAD_INS},${L1D_LOAD_MISSES},${STORE_INS},${L1I_HIT},${L1I_MISSES} -- ./sum.exe
	#perf stat -d ./sum.exe

pgi:


gnu:


clean:
	rm -f *.o a.out *.exe *.mod *.optrpt

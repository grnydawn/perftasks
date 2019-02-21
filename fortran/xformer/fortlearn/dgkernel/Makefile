
FC ?= ifort
SRC := ./dg_kernel.F90
EXE := ${SRC:.F90=.exe}
FLAGS := -O3

build: ${SRC} clean
	${FC} ${FLAGS} -o ${EXE} ${SRC}

run:
	${EXE}

clean:
	rm -f *.exe *.o *.mod

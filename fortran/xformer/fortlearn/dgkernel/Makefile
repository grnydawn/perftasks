EXTRAE_HOME ?= /glade/p/cisl/asap/youngsun/opt/extrae/3.5.4
FOLDING_HOME ?= /glade/p/cisl/asap/youngsun/opt/folding/1.3.2-Linux_x86_64

EXTRAE_CONFIG_FILE=extrae.xml

PAPI_HOME := /glade/p/cisl/asap/youngsun/opt/papi/5.6.0
XML_HOME := /usr
UNWIND_HOME := /glade/p/cisl/asap/youngsun/opt/libunwind/1.2.1

PAPILIB := -L${PAPI_HOME}/lib -lpapi
XMLLIB := -L${XML_HOME}/lib -lxml2
UNWINDLIB := -L${XML_HOME}/lib -lunwind
LIBERTYLIB := /usr/lib64/libiberty.a
EXTRAELIB := -L${EXTRAE_HOME}/lib -lseqtrace

ALLLIBS := ${LAPACKLIBS} ${PAPILIB} ${XMLLILB} ${UNWINDLIB} ${LIBERTYLIB} ${EXTRAELIB}

LIBPATHS := ${EXTRAE_HOME}/lib:${XML_HOME}/lib:${XML_HOME}/lib:${PAPI_HOME}/lib

PRERUN := export EXTRAE_HOME=${EXTRAE_HOME}; export EXTRAE_CONFIG_FILE=${EXTRAE_CONFIG_FILE}; export LD_LIBRARY_PATH=${LIBPATHS}:${LD_LIBRARY_PATH}

FC ?= ifort
SRC := ./dg_kernel.F90
EXE := ${SRC:.F90=.exe}
FLAGS := -g -O3

build: ${SRC}
	${FC} ${FLAGS} -o ${EXE} ${SRC} ${ALLLIBS}

run:
	${PRERUN}; ${EXE}

fold:
	${FOLDING_HOME}/bin/folding ./${EXE}.prv "User function"

clean:
	rm -rf ${EXE} ${EXE}.* *.o *.mod set-* TRACE.*

[test]
#run = kgen_include_reader_task.py include.ini --only "PhotosynthesisMod.F90,linear_1d_operators.F90,grid_class.F90,geopk.F90,zonal_mean.F90, modal_aero_newnuc.F90" --macro "USE_CONTIGUOUS=''" -- fparser2_task.py
run = kgen_include_reader_task.py include.ini --only "geopk.F90" --macro "USE_CONTIGUOUS=''" --macro "SPMD" -- fparser2_task.py
#run = kgen_include_reader_task.py include.ini --only "zonal_mean.F90" --macro "USE_CONTIGUOUS=''" -- fparser2_task.py
#run = kgen_include_reader_task.py include.ini --only "modal_aero_newnuc.F90" --macro "USE_CONTIGUOUS=''" -- fparser2_task.py --save 
#run = kgen_include_reader_task.py include.ini --macro "USE_CONTIGUOUS=''" -- fparser2_task.py
#run = kgen_include_reader_task.py include.ini --only "glam_strs2.F90" --macro "USE_CONTIGUOUS=''" -- fparser2_task.py
#run = kgen_include_reader_task.py include.ini --only "msise00.F90" --macro "USE_CONTIGUOUS=''" -- fparser2_task.py
#run = kgen_include_reader_task.py include.ini --only "glam_strs2.F90" --macro "USE_CONTIGUOUS=''" -- fparser2_task.py


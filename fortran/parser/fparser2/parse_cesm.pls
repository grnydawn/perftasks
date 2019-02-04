[import*]
os
re

[init]
cheyenne@py = os.uname()[1].startswith("cheyenne") or re.match(r"r\d+i\d+n\d+", os.uname()[1]) is not None
mymacpro@py = os.uname()[1].startswith("cisl-blaine")

[fparser@cheyenne]
run =   kgen_include_reader_task.py include.ini
            --only "namelist_from_str_mod.F90"
            --except "modal_aero_newnuc.F90"
            --macro "USE_CONTIGUOUS=''"
        -- fparser2_task.py 
            #--only "PhotosynthesisMod.F90,linear_1d_operators.F90,grid_class.F90,geopk.F90,zonal_mean.F90, modal_aero_newnuc.F90"
#run = kgen_include_reader_task.py include.ini --only "geopk.F90" --macro "USE_CONTIGUOUS=''" --macro "SPMD" -- fparser2_task.py 
#run = kgen_include_reader_task.py include.ini --only "zonal_mean.F90" --macro "USE_CONTIGUOUS=''" -- fparser2_task.py 
#run = kgen_include_reader_task.py include.ini --only "modal_aero_newnuc.F90" --macro "USE_CONTIGUOUS=''" -- fparser2_task.py --save  
#run = kgen_include_reader_task.py include.ini --macro "USE_CONTIGUOUS=''" -- fparser2_task.py 
#run = kgen_include_reader_task.py include.ini --only "glam_strs2.F90" --macro "USE_CONTIGUOUS=''" -- fparser2_task.py 
#run = kgen_include_reader_task.py include.ini --only "msise00.F90" --macro "USE_CONTIGUOUS=''" -- fparser2_task.py 
#run = kgen_include_reader_task.py include.ini --only "glam_strs2.F90" --macro "USE_CONTIGUOUS=''" -- fparser2_task.py 


[fparser@mymacpro]

run =   kgen_include_reader_task.py include.ini
            --only "PhotosynthesisMod.F90,linear_1d_operators.F90,grid_class.F90,geopk.F90,zonal_mean.F90, modal_aero_newnuc.F90"
            --except "modal_aero_newnuc.F90"
            --macro "USE_CONTIGUOUS=''"
        -- fparser2_task.py
            -a /Users/youngsun/data/cesmwork/bld=/gpfs/fs1/scratch/youngsun/KINTCESM/bld
            -a /Users/youngsun/data/cesmwork/cesm_work=/gpfs/fs1/scratch/youngsun/kgensystest/cesm_work

#run = kgen_include_reader_task.py include.ini --only "geopk.F90" --macro "USE_CONTIGUOUS=''" --macro "SPMD" -- fparser2_task.py -a /Users/youngsun/data/cesmwork/bld=/gpfs/fs1/scratch/youngsun/KINTCESM/bld -a /Users/youngsun/data/cesmwork/cesm_work=/gpfs/fs1/scratch/youngsun/kgensystest/cesm_work
#run = kgen_include_reader_task.py include.ini --only "zonal_mean.F90" --macro "USE_CONTIGUOUS=''" -- fparser2_task.py -a /Users/youngsun/data/cesmwork/bld=/gpfs/fs1/scratch/youngsun/KINTCESM/bld -a /Users/youngsun/data/cesmwork/cesm_work=/gpfs/fs1/scratch/youngsun/kgensystest/cesm_work
#run = kgen_include_reader_task.py include.ini --only "modal_aero_newnuc.F90" --macro "USE_CONTIGUOUS=''" -- fparser2_task.py --save  -a /Users/youngsun/data/cesmwork/bld=/gpfs/fs1/scratch/youngsun/KINTCESM/bld -a /Users/youngsun/data/cesmwork/cesm_work=/gpfs/fs1/scratch/youngsun/kgensystest/cesm_work
#run = kgen_include_reader_task.py include.ini --macro "USE_CONTIGUOUS=''" -- fparser2_task.py -a /Users/youngsun/data/cesmwork/bld=/gpfs/fs1/scratch/youngsun/KINTCESM/bld -a /Users/youngsun/data/cesmwork/cesm_work=/gpfs/fs1/scratch/youngsun/kgensystest/cesm_work
#run = kgen_include_reader_task.py include.ini --only "glam_strs2.F90" --macro "USE_CONTIGUOUS=''" -- fparser2_task.py -a /Users/youngsun/data/cesmwork/bld=/gpfs/fs1/scratch/youngsun/KINTCESM/bld -a /Users/youngsun/data/cesmwork/cesm_work=/gpfs/fs1/scratch/youngsun/kgensystest/cesm_work
#run = kgen_include_reader_task.py include.ini --only "msise00.F90" --macro "USE_CONTIGUOUS=''" -- fparser2_task.py -a /Users/youngsun/data/cesmwork/bld=/gpfs/fs1/scratch/youngsun/KINTCESM/bld -a /Users/youngsun/data/cesmwork/cesm_work=/gpfs/fs1/scratch/youngsun/kgensystest/cesm_work
#run = kgen_include_reader_task.py include.ini --only "glam_strs2.F90" --macro "USE_CONTIGUOUS=''" -- fparser2_task.py -a /Users/youngsun/data/cesmwork/bld=/gpfs/fs1/scratch/youngsun/KINTCESM/bld -a /Users/youngsun/data/cesmwork/cesm_work=/gpfs/fs1/scratch/youngsun/kgensystest/cesm_work


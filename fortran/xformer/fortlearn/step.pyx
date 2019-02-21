


result@pyloco = ${fparser} "${target}" \
                -- extrae.py \
                -- run.py \
                -- fold.py \ # include multiple run and folding
                -- observation.py \ # generate reward and observations of folding + code structure
                -- action.py \ # select a code line, select opt type and set opt parameter
                -- claw.py \ # add claw directives and run claw to generate new source


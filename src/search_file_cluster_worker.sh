#!/usr/bin/env bash
CUR_DIR=/work/pi_brun_umass_edu/avarghese/proverbot9001/src

read-opam.sh
eval $(opam env)

python3 $CUR_DIR/search_file_cluster_worker.py $@

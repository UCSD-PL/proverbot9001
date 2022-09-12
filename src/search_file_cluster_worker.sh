#!/usr/bin/env bash
CUR_DIR=$HOME/work/proverbot9001/src/

#read-opam.sh
eval $(opam env)

python3 $CUR_DIR/search_file_cluster_worker.py $@

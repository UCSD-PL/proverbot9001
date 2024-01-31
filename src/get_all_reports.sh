#!/usr/bin/env bash


module load opam/2.1.2 graphviz/2.49.0+py3.8.12 openmpi/4.1.3+cuda11.6.2
eval $(opam env)

python ./src/search_file_cluster.py -p cpu --hard-depth-limit=20 --prelude=./CompCert --num-workers=16 --weightsfile=data/polyarg-weights.dat -j1 compcert_projs_splits.json --search-type=dfs --output=output/rl_results/DFS-search-report

python ./src/search_file_cluster.py -p cpu --hard-depth-limit=20 --prelude=./CompCert --num-workers=16 --weightsfile=data/polyarg-weights.dat -j1 compcert_projs_splits.json --search-type=astar --output=output/rl_results/Astar-search-report

python ./src/search_file_cluster.py -p cpu --hard-depth-limit=20 --prelude=./CompCert --num-workers=16 --weightsfile=data/polyarg-weights-develop.dat -j1 lib/Parmov.v --scoring-function=pickled --search-type=best-first --pickled-estimator=data/rl_estimator.dat --output=output/rl_results/Astar-search-report --max-steps=128
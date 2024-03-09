#!/usr/bin/env bash

job_name=$1
eval $(opam env)

echo "Running rl searches"

python ./src/search_file_cluster.py -p cpu --mem=8G --hard-depth-limit=20 --prelude=./CompCert --num-workers=32 --weightsfile=data/polyarg-weights-develop.dat -j1 compcert_projs_splits.json --scoring-function=pickled --search-type=best-first --pickled-estimator=output/rl_$job_name/final_trained_weights_rl_pickle.pkl --output=output/rl_$job_name/results/Astar_rl-search-report --max-steps=128

python ./src/search_file_cluster.py -p cpu --mem=8G --hard-depth-limit=20 --prelude=./CompCert --num-workers=32 --weightsfile=data/polyarg-weights-develop.dat -j1 compcert_projs_splits.json --scoring-function=pickled --search-type=astar --pickled-estimator=output/rl_$job_name/final_trained_weights_rl_pickle.pkl --output=output/rl_$job_name/results/BestFirst_rl-search-report --max-steps=128

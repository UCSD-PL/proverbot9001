#!/usr/bin/env bash

gamma=0.9
train_path=data/rl_debug_jobs.json
coq2vecweightspath=data/term2vec-weights-59.dat

eval $(opam env)

echo "Running RL training"

python src/distributed_rl.py --mem=16G --num-actors=32 --supervised-weights=data/polyarg-weights-develop.dat \
             --coq2vec-weights=$coq2vecweightspath compcert_projs_splits.json --prelude=./CompCert  \
              --backend=serapi --gamma=$gamma -s5 -p5 --learning-rate=0.000005 -n7 -o output/rl/final_trained_weights_rl.pkl \
              --tasks-file=$train_path --resume=yes -b 1024 --allow-partial-batches --sync-target-every=128 \
               --state-dir=output/rl/state_dirs/drl --partition gpu -v --start-from output/rl/pretrained_weights.pkl


srun python src/rl_to_pickle.py output/rl/final_trained_weights_rl.pkl output/rl/final_trained_weights_rl_pickle.pkl
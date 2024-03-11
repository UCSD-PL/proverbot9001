#!/usr/bin/env bash

train_path=$2
if [ $1 == "CompCert" ]; then
    proj_split_title="compcert_projs_splits.json"
    prelude="CompCert"
elif [ $1 == "CoqGym" ]; then
    proj_split_title=coqgym_projs_splits.json
    prelude=coq-projects
fi

if [ -z "$3" ]; then
    resume="no"
else
    if [ $3 == "resume" ]; then
        resume="yes"
    else 
        resume="no"
    fi
fi

gamma=0.9
coq2vecweightspath=data/term2vec-weights-59.dat

eval $(opam env)

echo "Running RL training"

python src/distributed_rl.py --mem=16G --num-actors=32 --supervised-weights=data/polyarg-weights-develop.dat \
             --coq2vec-weights=$coq2vecweightspath $proj_split_title --prelude=./$prelude  \
              --backend=serapi --gamma=$gamma -s5 -p5 --learning-rate=0.000005 -n7 -o output/rl/final_trained_weights_rl.pkl \
              --tasks-file=$train_path --resume=$resume -b 1024 --allow-partial-batches --sync-target-every=128 \
               --state-dir=output/rl/state_dirs/drl --partition gpu -v --start-from output/rl/pretrained_weights.pkl


srun python src/rl_to_pickle.py output/rl/final_trained_weights_rl.pkl output/rl/final_trained_weights_rl_pickle.pkl
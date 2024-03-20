#!/usr/bin/env bash

set -x
set -e

job_name=$1
tasks_to_train_path=$3
if [ $2 == "CompCert" ]; then 
    proj_split_title="compcert_projs_splits.json"
    prelude="CompCert"
elif [ $2 == "CoqGym" ]; then
    proj_split_title=coqgym_projs_splits.json
    prelude=coq-projects
fi

gamma=0.85
coq2vecweightspath=data/term2vec-weights-59.dat


eval $(opam env)

echo "Running Pretraining"

python src/distributed_get_task_initial_states.py $tasks_to_train_path output/rl_$job_name/initial_states.json --state-dir output/rl_$job_name/state_dirs/prepare_init_state --num-workers 32 --resume --prelude=./$prelude

srun --pty -p gpu --mem=8G --time=8:00:00 --gres=gpu:1 python src/supervised_v_values.py \
        --encoder-weights=$coq2vecweightspath -o output/rl_$job_name/pretrained_weights.pkl output/rl_$job_name/initial_states.json \
        --prelude=CompCert --mode=train --supervised-weights=data/polyarg-weights-develop.dat -l 7e-7 --learning-rate-decay=.57 \
        --learning-rate-step=80 --gamma=$gamma --num-epochs=400


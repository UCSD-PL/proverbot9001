#!/usr/bin/env bash

gamma=0.9
train_path=data/rl_debug_jobs.json
coq2vecweightspath=data/term2vec-weights-59.dat


eval $(opam env)

echo "Running Pretraining"

python src/distributed_get_task_initial_states.py $train_path output/rl/initial_states.json --state-dir output/rl/state_dirs/prepare_init_state --num-workers 32 --resume 

srun --pty -p gpu --mem=8G --time=1:00:00 --gres=gpu:1 python src/supervised_v_values.py \
        --encoder-weights=$coq2vecweightspath -o output/rl/pretrained_weights.pkl output/rl/initial_states.json \
        --prelude=CompCert --mode=train --supervised-weights=data/polyarg-weights-develop.dat -l 7e-7 --learning-rate-decay=.57 \
        --learning-rate-step=80 --gamma=$gamma --num-epochs=200


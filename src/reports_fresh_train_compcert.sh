#!/usr/bin/env bash

# set -x
# set -e

module load opam/2.1.2 graphviz/2.49.0+py3.8.12 openmpi/4.1.3+cuda11.6.2


proj_split_title="compcert_projs_splits.json"
prelude="CompCert"
lr="1.8e-4"
job_name="CompCert_full"
gamma=0.85
coq2vecweightspath=data/term2vec-weights-59.dat
width=5
train_path=data/rl_train_jobs_new.json
resume=yes

# srun --pty -p gpu --mem=8G --time=8:00:00 --gres=gpu:1 python src/supervised_v_values.py \
#         --encoder-weights=$coq2vecweightspath -o output/rl_$job_name/pretrained_weights.pkl data/task_target.json \
#         --prelude=CompCert --mode=train --supervised-weights=data/polyarg-weights-develop.dat -l $lr --learning-rate-decay=.86 \
#         --learning-rate-step=972   --gamma=$gamma --num-epochs=25 --hidden-size 128 --num-layers 2



python src/distributed_rl.py --mem=16G --num-actors=32 --supervised-weights=data/polyarg-weights-develop.dat \
             --coq2vec-weights=$coq2vecweightspath $proj_split_title --prelude=./$prelude  \
              --backend=serapi --gamma=$gamma -s5 -p$width --learning-rate=0.000005 -n7 -o output/rl_$job_name/final_trained_weights_rl.pkl \
              --tasks-file=$train_path --resume=$resume -b 1024 --allow-partial-batches --sync-target-every=128 \
               --state-dir=output/rl_$job_name/state_dirs/drl --partition gpu -v --start-from output/rl_$job_name/pretrained_weights.pkl --hidden-size 128 --num-layers 2 \
               --partition gpu-preempt


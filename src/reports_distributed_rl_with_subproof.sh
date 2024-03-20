#!/usr/bin/env bash


module load opam/2.1.2 graphviz/2.49.0+py3.8.12 openmpi/4.1.3+cuda11.6.2


proj_split_title="compcert_projs_splits.json"
prelude="CompCert"
lr="1.8e-6"
job_name="with_subproof_filtered"
gamma=0.85
coq2vecweightspath=data/term2vec-weights-59.dat
width=5
train_path=data/compcert_train_tasks_paper_filtered.json
resume=no
ESTIMATOR_WEIGHTS=output/rl_$job_name/final_trained_weights_rl_pickle.pkl

# python src/distributed_get_task_initial_states.py $train_path output/rl_$job_name/initial_states.json --state-dir output/rl_$job_name/state_dirs/prepare_init_state --num-workers 32 --resume --prelude=./$prelude

# srun --pty -p gpu --mem=8G --time=8:00:00 --gres=gpu:1 python src/supervised_v_values.py \
#         --encoder-weights=$coq2vecweightspath -o output/rl_$job_name/pretrained_weights.pkl output/rl_$job_name/initial_states.json \
#         --prelude=CompCert --mode=train --supervised-weights=data/polyarg-weights-develop.dat -l $lr --learning-rate-decay=.86 \
#         --learning-rate-step=972   --gamma=$gamma --num-epochs=1000 --hidden-size 128 --num-layers 2 --batch-size=8



# python src/distributed_rl.py --mem=16G --num-actors=8 --supervised-weights=data/polyarg-weights-develop.dat \
#              --coq2vec-weights=$coq2vecweightspath $proj_split_title --prelude=./$prelude  \
#               --backend=serapi --gamma=$gamma -s5 -p$width --learning-rate=0.000005 -n7 -o output/rl_$job_name/final_trained_weights_rl.pkl \
#               --tasks-file=$train_path --resume=$resume -b 1024 --allow-partial-batches --sync-target-every=128 \
#                --state-dir=output/rl_$job_name/state_dirs/drl --partition gpu-preempt -v --start-from output/rl_$job_name/pretrained_weights.pkl --hidden-size 128 --num-layers 2 

# srun python src/rl_to_pickle.py output/rl_$job_name/final_trained_weights_rl.pkl output/rl_$job_name/final_trained_weights_rl_pickle.pkl

python ./src/search_file_cluster.py -p cpu --mem=8G --hard-depth-limit=20 --prelude=./CompCert --num-workers=32 --weightsfile=data/polyarg-weights-develop.dat -j1 compcert_projs_splits.json --scoring-function=pickled --search-type=astar --pickled-estimator=$ESTIMATOR_WEIGHTS --output=output/rl_$job_name/results/Astar_rl-search-report --max-steps=128 --search-width=$width
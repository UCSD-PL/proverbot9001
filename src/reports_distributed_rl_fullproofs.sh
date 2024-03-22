#!/usr/bin/env bash


module load opam/2.1.2 graphviz/2.49.0+py3.8.12 openmpi/4.1.3+cuda11.6.2


proj_split_title="compcert_projs_splits.json"
prelude="CompCert"
lr="7.8e-4"
job_name="without_subproof_filtered_ns2"
gamma=0.66
coq2vecweightspath=data/term2vec-weights-59.dat
width=5
train_path=data/rl_train_jobs_new.json
resume=no
ESTIMATOR_WEIGHTS=output/rl_$job_name/final_trained_weights_rl_pickle.pkl
nlayers=6
lr_decay=0.999
lr_step=1
hidden_size=26
sync_target=79
nepisodes=2
stepspereps=2

# mkdir output/rl_$job_name

# jq -c "[inputs|select(.target_length <= 5 and .target_length >= 3 and .largest_prediction_idx < 5) ]|.[]" $train_path > output/rl_$job_name/compcert_train_tasks_paper_filtered.json
# jq -c "[inputs|select((.tactic_prefix|length) <= 1 )]|.[]" output/rl_$job_name/compcert_train_tasks_paper_filtered.json > output/rl_$job_name/compcert_train_tasks_paper_filtered_no_subproof.json

# python src/distributed_get_task_initial_states.py output/rl_$job_name/compcert_train_tasks_paper_filtered_no_subproof.json output/rl_$job_name/initial_states.json --state-dir output/rl_$job_name/state_dirs/prepare_init_state --num-workers 32 --resume --prelude=./$prelude

# srun --pty -p gpu --mem=8G --time=8:00:00 --gres=gpu:1 python src/supervised_v_values.py \
#         --encoder-weights=$coq2vecweightspath -o output/rl_$job_name/pretrained_weights.pkl output/rl_$job_name/initial_states.json \
#         --prelude=CompCert --mode=train --supervised-weights=data/polyarg-weights-develop.dat -l $lr --learning-rate-decay=$lr_decay \
#         --learning-rate-step=$lr_step   --gamma=$gamma --num-epochs=1000 --hidden-size $hidden_size --num-layers $nlayers --batch-size=2



python src/distributed_rl.py --mem=16G --num-actors=8 --supervised-weights=data/polyarg-weights-develop.dat \
             --coq2vec-weights=$coq2vecweightspath $proj_split_title --prelude=./$prelude  \
              --backend=serapi --gamma=$gamma -s$stepspereps -p$width --learning-rate=$lr -n$nepisodes -o output/rl_$job_name/final_trained_weights_rl.pkl \
              --tasks-file=output/rl_$job_name/compcert_train_tasks_paper_filtered_no_subproof.json \
               --resume=$resume -b 1024 --allow-partial-batches --sync-target-every=$sync_target \
               --state-dir=output/rl_$job_name/state_dirs/drl --partition gpu-preempt -v --start-from output/rl_$job_name/pretrained_weights.pkl --hidden-size $hidden_size \
            --num-layers $nlayers --learning-rate-step=$lr_step  --learning-rate-decay=$lr_decay

srun python src/rl_to_pickle.py output/rl_$job_name/final_trained_weights_rl.pkl output/rl_$job_name/final_trained_weights_rl_pickle.pkl

python ./src/search_file_cluster.py -p cpu --mem=8G --hard-depth-limit=20 --prelude=./CompCert --num-workers=32 --weightsfile=data/polyarg-weights-develop.dat -j1 compcert_projs_splits.json --scoring-function=pickled --search-type=astar --pickled-estimator=$ESTIMATOR_WEIGHTS --output=output/rl_$job_name/results/Astar_rl-search-report --max-steps=256 --search-width=$width
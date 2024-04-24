#!/usr/bin/env bash


module load opam/2.1.2 graphviz/2.49.0+py3.8.12 openmpi/4.1.3+cuda11.6.2

filename=$(basename "$1")
job_name="with_subproof_hypfile_$filename"

proj_split_title=$(jq -r '.proj_splits_title' $1)
prelude=$(jq -r '.prelude' $1)
lr=$(jq -r '.lr' $1)
gamma=$(jq -r '.gamma' $1)
coq2vecweightspath=$(jq -r '.coq2vecweightspath' $1)
width=$(jq -r '.width' $1)
train_path=$(jq -r '.train_path' $1)
resume=$(jq -r '.resume' $1)
nlayers=$(jq -r '.nlayers' $1)
lr_decay=$(jq -r '.lr_decay' $1)
lr_step=$(jq -r '.lr_step' $1)
hidden_size=$(jq -r '.hidden_size' $1)
sync_target=$(jq -r '.sync_target' $1)
nepisodes=$(jq -r '.nepisodes' $1)
stepspereps=$(jq -r '.stepspereps' $1)
predictor_width=$(jq -r '.predictor_width' $1)


if jq 'has("filter_tasks")' $1; then
    filter_tasks=$(jq -r '.filter_tasks' $1)
else
    filter_tasks=false
fi


mkdir output/rl_$job_name


if $filter_tasks; then
    jq -c "[inputs|select(.target_length <= 5 and .target_length >= 3 and .largest_prediction_idx < 5) ]|.[]" $train_path > output/rl_$job_name/compcert_train_tasks_paper_filtered.json
else 
    cp $train_path output/rl_$job_name/compcert_train_tasks_paper_filtered.json
fi 


# python src/distributed_get_task_initial_states.py output/rl_$job_name/compcert_train_tasks_paper_filtered.json output/rl_$job_name/initial_states.json --state-dir output/rl_$job_name/state_dirs/prepare_init_state --num-workers 32 --resume --prelude=./$prelude

# srun --pty -p gpu --mem=8G --time=8:00:00 --gres=gpu:1 python src/supervised_v_values.py \
#         --encoder-weights=$coq2vecweightspath -o output/rl_$job_name/pretrained_weights.pkl output/rl_$job_name/initial_states.json \
#         --prelude=CompCert --mode=train --supervised-weights=data/polyarg-weights-develop.dat -l $lr --learning-rate-decay=$lr_decay \
#         --learning-rate-step=$lr_step   --gamma=$gamma --num-epochs=1000 --hidden-size $hidden_size --num-layers $nlayers --batch-size=20



python src/distributed_rl.py --mem=16G --num-actors=16 --supervised-weights=data/polyarg-weights-develop.dat \
             --coq2vec-weights=$coq2vecweightspath $proj_split_title --prelude=./$prelude  \
              --backend=serapi --gamma=$gamma -s$stepspereps -p$predictor_width --learning-rate=$lr -n$nepisodes -o output/rl_$job_name/final_trained_weights_rl.pkl \
              --tasks-file=output/rl_$job_name/compcert_train_tasks_paper_filtered.json \
               --resume=$resume -b 1024 --allow-partial-batches --sync-target-every=$sync_target \
               --state-dir=output/rl_$job_name/state_dirs/drl --partition gpu-preempt -v --hidden-size $hidden_size \
            --num-layers $nlayers --learning-rate-step=$lr_step  --learning-rate-decay=$lr_decay --verifyv-every=128 \
            # --start-from output/rl_$job_name/pretrained_weights.pkl 

# python src/distributed_rl_eval.py --mem=16G --num-actors=8 --supervised-weights=data/polyarg-weights-develop.dat \
#         --coq2vec-weights=$coq2vecweightspath $proj_split_title --prelude=./$prelude  \
#         --backend=serapi --gamma=$gamma --rl-weights=$ESTIMATOR_WEIGHTS --tasks-file=data/rl_test_jobs.json -p16 --partition=gpu \
#         --evaluate --state-dir=output/rl_$job_name/state_dirs/drl_eval


# srun python src/rl_to_pickle.py output/rl_$job_name/final_trained_weights_rl.pkl output/rl_$job_name/final_trained_weights_rl_pickle.pkl

# ESTIMATOR_WEIGHTS="output/rl_$job_name/final_trained_weights_rl_pickle.pkl"
# python ./src/search_file_cluster.py -p cpu --mem=8G --hard-depth-limit=20 --prelude=./CompCert --num-workers=32 --weightsfile=data/polyarg-weights-develop.dat -j1 compcert_projs_splits.json --scoring-function=pickled --search-type=astar --pickled-estimator=$ESTIMATOR_WEIGHTS --output=output/rl_$job_name/results/Astar_rl-search-report --max-steps=128 --search-width=$width
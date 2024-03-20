# Generate rl training jobs from coqgym
python src/gen_rl_tasks_cluster.py --prelude=coq-projects --supervised-weights=data/polyarg-weights-develop.dat -o coqgym_train_jobs.json coqgym_projs_splits.json

# Filter for proof length between 3 and 5, and proof width below 5
jq -c "[inputs|select(.target_length <= 5 and .target_length >= 3 and .largest_prediction_idx < 5)]|.[]" coqgym_train_jobs.json > coqgym_train_jobs_filtered.json

# Fill in task curriculum and subgoals
python src/fill_in_task_curriculum.py -s coqgym_train_jobs_filtered.json coqgym_train_jobs_filtered_with_curriculum

# Run distributed learning
python src/distributed_rl.py --partition=gpu-preempt --mem=8G --num-actors=30 --supervised-weights=data/polyarg-weights.dat --coq2vec-weights=./coq2vec/term2vec-weights-59.dat --prelude=./coq-projects --backend=serapi --gamma=0.85 -s7 -p5 --learning-rate=0.00018 -n1 --num-layers=2 --learning-rate-decay=0.86 --learning-rate-step=972 --worker-timeout="7:59:59" -o data/full_coqgym_with_curriculum.dat --state-dir=drl_state_filtered_with_curriculum --tasks-file=coqgym_train_jobs_filtered_with_curriculum.json --resume=no coqgym_projs_splits.json



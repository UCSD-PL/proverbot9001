cd ..
ROOT=/home/shizhuo2_illinois_edu/proverbot9001
ulimit -u 127590
module load opam/2.1.2
srun python src/rl_tuning_distributed_main.py --weightsfile=$ROOT/data/polyarg-weights-develop.dat --coq2vec_weights=$ROOT/data/term2vec-weights-59.dat \
         --tasks_file=/home/shizhuo2_illinois_edu/proverbot-dat/rl_train_jobs_curriculum5_width5_50lines.json --prelude=$ROOT/CompCert --backend=serapi --allow_partial_batches \
         --learning_rate=0.0001 -n1 -o $ROOT/data/rl_weights-compcert-5.dat -s7 --output $ROOT/data/rl_weights.dat --test_file /home/shizhuo2_illinois_edu/proverbot-dat/rl_train_jobs_len5_width5_25lines.json \
         --num_trails 5 --log_root /home/shizhuo2_illinois_edu/proverbot9001_logs --result_root /home/shizhuo2_illinois_edu/proverbot9001_results --exp_name rl_distributed4 \
         --curriculum

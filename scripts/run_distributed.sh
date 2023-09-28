cd ..
ROOT=/work/pi_brun_umass_edu/dylan/hyperparameter_search
EXP_NAME=rl_distributed_1000—1epoch
ulimit -u 127590
module load opam/2.1.2

srun --time=3:00:00 python src/rl_tuning_distributed_main.py --weightsfile=$ROOT/data/polyarg-weights-develop.dat --coq2vec_weights=$ROOT/data/term2vec-weights-59.dat \
         --tasks_file=/home/shizhuo2_illinois_edu/proverbot-dat/rl_train_jobs_len5_wid5-hyperpara-train-curriculum-1000.json --prelude=$ROOT/CompCert --backend=serapi --allow_partial_batches \
         --learning_rate=0.0001 -n5 -o $ROOT/data/rl_weights-compcert-5.dat -s7 --output $ROOT/data/rl_weights.dat --test_file /home/shizhuo2_illinois_edu/proverbot-dat/rl_train_jobs_len5_wid5-hyperpara-test.json \
         --num_trails 1000 --log_root /home/shizhuo2_illinois_edu/proverbot9001_logs --result_root /home/shizhuo2_illinois_edu/proverbot9001_results --exp_name $EXP_NAME \
         --curriculum

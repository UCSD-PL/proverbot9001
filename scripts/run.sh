cd ..
ROOT=/home/shizhuo2_illinois_edu/proverbot9001
python src/rl.py --supervised-weights=$ROOT/data/polyarg-weights-develop.dat --coq2vec-weights=$ROOT/data/term2vec-weights-59.dat $ROOT/compcert_projs_splits.json \
         --tasks-file=/home/dylan/proverbot_dat/rl_train_jobs_len5_wid5_200lines_curriculum.json --prelude=$ROOT/CompCert --backend=serapi --allow-partial-batches \
         --learning-rate=0.0001 -n1 -o $ROOT/data/rl_weights-compcert-5.dat -s7 --hyperparameter_search --output $ROOT/data/rl_weights.dat --test-file /home/dylan/proverbot_dat/rl_test_jobs_len5_wid5_200lines.json \
         --num-trails 1000 --num-cpus 10 --num-gpus 2

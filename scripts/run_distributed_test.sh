SHARED_DIR=/work/pi_brun_umass_edu/share
TRAIN_JOBS=$SHARED_DIR/rl_train_jobs.json
TEST_JOBS=$SHARED_DIR/rl_test_jobs.json
SUPERVISED_WEIGHTS=data/polyarg-weights-develop.dat
cd ..
python -u src/distributed_rl_eval.py \
    --supervised-weights=$SUPERVISED_WEIGHTS \
    --coq2vec-weights=data/term2vec-weights-59.dat compcert_projs_splits.json \
    --prelude=./CompCert --backend=serapi -p 16 \
    --rl-weights data/rl_weights_train_curriculum_distributed.dat --tasks-file=$TEST_JOBS \
    --evaluate
#!/bin/bash
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=8192  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 01:00:00  # Job time limit
#SBATCH -o slurm-%j.out  # %j = job ID


# Start the head node
suffix='6666'
ip_head=`hostname`:$suffix
export ip_head # Exporting for latter access by trainer.py

cd ..
ROOT=/home/shizhuo2_illinois_edu/proverbot9001
srun python src/rl.py --supervised-weights=$ROOT/data/polyarg-weights-develop.dat --coq2vec-weights=$ROOT/data/term2vec-weights-59.dat $ROOT/compcert_projs_splits.json \
         --tasks-file=/home/dylan/proverbot_dat/rl_train_jobs_len5_wid5_200lines_curriculum.json --prelude=$ROOT/CompCert --backend=serapi --allow-partial-batches \
         --learning-rate=0.0001 -n1 -o $ROOT/data/rl_weights-compcert-5.dat -s7 --hyperparameter_search --output $ROOT/data/rl_weights.dat --test-file /home/dylan/proverbot_dat/rl_test_jobs_len5_wid5_200lines.json \
         --num-trails 1000

#!/bin/bash
#
#SBATCH --job-name=RL_train
#SBATCH --output=submit/rl_train_results.txt  # output file
#SBATCH -e submit/rl_train_error.txt        # File to which STDERR will be written
#SBATCH --partition=gpu    # Partition to submit to 
#SBATCH --gpus=1
#
#SBATCH --time=24:00:00         # Maximum runtime in D-HH:MM
#SBATCH --mem-per-cpu=8000    # Memory in MB per cpu allocated

module add opam
module load opam

python -u src/train_rl_memgraph_vval_coq2vec_obligation.py --prelude CompCert --proof_file CompCert/common/Globalenvs.v --use_fast_check --max_attempts 10 --max_proof_len 50 --update_every 1000 --batch_size 100 --wandb_log --resume
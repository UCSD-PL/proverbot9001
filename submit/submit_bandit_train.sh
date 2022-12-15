#!/bin/bash
#
#SBATCH --job-name=Bandit_train
#SBATCH --output=submit/bandit_rl_train_results.txt  # output file
#SBATCH -e submit/bandit_rl_train_error.txt        # File to which STDERR will be written
#SBATCH --partition=gpu    # Partition to submit to 
#
#SBATCH --time=5:00:00         # Maximum runtime in D-HH:MM
#SBATCH --mem-per-cpu=2000    # Memory in MB per cpu allocated

module add opam
module load opam

python -u src/train_rl_bandit_nn_discrete_tactic.py --prelude CompCert --proof_file CompCert/common/Globalenvs.v --wandb_log
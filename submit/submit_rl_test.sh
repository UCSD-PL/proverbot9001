#!/bin/bash
#
#SBATCH --job-name=RL_search
#SBATCH --output=submit/rl_test_results.txt  # output file
#SBATCH -e submit/rl_test_error.txt        # File to which STDERR will be written
#SBATCH --partition=gpu    # Partition to submit to 
#
#SBATCH --time=01:00:00         # Maximum runtime in D-HH:MM
#SBATCH --mem-per-cpu=2000    # Memory in MB per cpu allocated


python -u src/train_rl_bandit.py --prelude CompCert --proof_file CompCert/common/Globalenvs.v --wandb_log

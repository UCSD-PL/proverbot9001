#!/bin/bash
#
#SBATCH --job-name=RL_search
#SBATCH --output=submit/rl_search_result.txt  # output file
#SBATCH -e submit/rl_search_error.txt        # File to which STDERR will be written
#SBATCH --partition=cpu    # Partition to submit to 
#
#SBATCH --time=12:00:00         # Maximum runtime in D-HH:MM
#SBATCH --mem-per-cpu=2000    # Memory in MB per cpu allocated

module add opam
module load opam
python -u src/train_rl_onestep_rf.py --prelude CompCert --proof_file CompCert/common/Globalenvs.v --wandb_log
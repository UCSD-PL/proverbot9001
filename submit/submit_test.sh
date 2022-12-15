#!/bin/bash
#
#SBATCH --job-name=test
#SBATCH --mem=4000
#SBATCH -p gpu
#SBATCH -o submit/test_results.txt
#SBATCH -e submit/test_errors.txt


python -u src/test_state_space_discrete_tactic.py
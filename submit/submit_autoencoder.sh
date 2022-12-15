#!/bin/bash
#
#SBATCH --job-name=proverbot_encoder_symbols
#SBATCH -p gpu-long
#SBATCH -G 1
#SBATCH --mem=12000
#SBATCH -o submit/autoencoder_results_symbols.txt
#SBATCH -e submit/autoencoder_error_symbols.txt
#SBATCH -t 2-00:00:00

module load cuda/10
python -u src/train_encoder.py --tokens symbols




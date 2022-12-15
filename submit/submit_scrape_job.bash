#!/bin/bash
#
#SBATCH --job-name=scrape_data
#SBATCH --output=submit/scrape_results.txt  # output file
#SBATCH -e submit/scrape_error.txt        # File to which STDERR will be written
#SBATCH --partition=cpu    # Partition to submit to 
#
#SBATCH --ntasks-per-node=8
#SBATCH --time=1:10:00         # Maximum runtime in D-HH:MM
#SBATCH --mem-per-cpu=8000    # Memory in MB per cpu allocated


module add opam
module load opam
eval $(opam env)
python src/scrape.py --prelude=CompCert common/Globalenvs.v
exit

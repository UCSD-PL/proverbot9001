#!/usr/bin/env bash

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/work/pi_brun_umass_edu/zhannakaufma/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
	    eval "$__conda_setup"
    else
	        if [ -f "/work/pi_brun_umass_edu/zhannakaufma/anaconda3/etc/profile.d/conda.sh" ]; then
			        . "/work/pi_brun_umass_edu/zhannakaufma/anaconda3/etc/profile.d/conda.sh"
				    else
					            export PATH="/work/pi_brun_umass_edu/zhannakaufma/anaconda3/bin:$PATH"
						        fi
fi
unset __conda_setup
# <<< conda initialize <<<

CUR_DIR=$HOME/work/search-diversity-proverbot/proverbot9001
cd $CUR_DIR
conda activate prover
pip3 install --no-input -e coq_serapy
python ./src/proverbot9001.py predictor-data --weightsfile data/polyarg-weights.dat --weightsfiles data/no-goal-head-weights.dat --scrapefile data/compcert-scrape.txt --dest ./comb_pred_4.csv

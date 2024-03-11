#!/usr/bin/env bash


task_path=$2
project=$1

if [ -z "$3" ]; then
    resume=""
else
    if [ $3 == "resume" ]; then
        resume="resume"
    else 
        resume="no-resume"
    fi
fi

module load opam/2.1.2 graphviz/2.49.0+py3.8.12 openmpi/4.1.3+cuda11.6.2
eval $(opam env)


bash src/reports_prepare_rl_pretrain.sh $1 $2
bash src/reports_train_rl.sh $1 $2 $resume

bash src/reports_makereport_search_rlweights.sh

bash src/reports_makereport_search_baselines.sh
bash src/reports_makereport_search_baselines.sh

bash src/reports_post_report_analysis.sh
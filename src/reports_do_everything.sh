#!/usr/bin/env bash

module load opam/2.1.2 graphviz/2.49.0+py3.8.12 openmpi/4.1.3+cuda11.6.2
eval $(opam env)


bash src/reports_prepare_rl_pretrain.sh
bash src/reports_train_rl.sh
bash src/reports_makereport_search_rlweights.sh
bash src/reports_makereport_search_baselines.sh
bash src/reports_post_report_analysis.sh
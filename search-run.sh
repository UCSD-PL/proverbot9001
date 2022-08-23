#!/bin/env bash
set -x
width=${width:-3}
depth=${depth:-6}
model=$1
shift
cat data/compcert-test-files.txt | xargs ./src/search_file_cluster.py --weightsfile=data/$model-weights.dat --search-width=$width --search-depth=$depth --prelude=./CompCert -o search-report-$model-$width-$depth $@
grep -o "Proofs Completed: *[0-9]\+\.[0-9]\+% ([0-9]\+/[0-9]\+)" \
     search-report-$model-$width-$depth/report.html

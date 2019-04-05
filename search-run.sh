#!/bin/env bash
set -x
width=${width:-3}
depth=${depth:-6}
cat data/compcert-test-files.txt | xargs ./src/proverbot9001.py search-report --weightsfile=data/$1-weights.dat --search-width=$width --search-depth=$depth --prelude=./CompCert -o search-report-$1-$width-$depth
grep -o "Proofs Completed: *[0-9]\+\.[0-9]\+% ([0-9]\+/[0-9]\+)" \
     search-report-$1-$width-$depth/report.html

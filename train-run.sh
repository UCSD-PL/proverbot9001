#!/bin/env bash
width=${width:-3}
depth=${depth:-6}
./src/proverbot9001.py train $1 data/scrape.txt data/$1-weights.dat ${@:2}
width=$width depth=$depth ./search-run.sh hypfeatures

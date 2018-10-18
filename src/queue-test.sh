#!/usr/bin/env bash

# exit on error
set -e

# determine physical directory of this script
src="${BASH_SOURCE[0]}"
while [ -L "$src" ]; do
  dir="$(cd -P "$(dirname "$src")" && pwd)"
  src="$(readlink "$src")"
  [[ $src != /* ]] && src="$dir/$src"
done
MYDIR="$(cd -P "$(dirname "$src")" && pwd)"
cd $MYDIR

if [ $# -eq 0 ]
then
    echo "Not enough arguments! Must supply predictor"
    exit 1
fi

weights=$(mktemp /tmp/proverbot-weights.XXX)
outdir=$(mktemp -d $PWD/../report-XXX)

echo "Saving to weights $weights"
echo "Saving report to $outdir"

export TS_SOCKET=/tmp/graphicscard

tsp -fn ./proverbot9001.py train "$@" ../data/scrape.txt "$weights" && \
  make -C .. FLAGS="--predictor=$1 -o $outdir --weightsfile=$weights" report

echo "$weights"
echo "$outdir"

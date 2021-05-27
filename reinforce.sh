#!/usr/bin/env bash

set -e

# determine physical directory of this script
src="${BASH_SOURCE[0]}"
while [ -L "$src" ]; do
  dir="$(cd -P "$(dirname "$src")" && pwd)"
  src="$(readlink "$src")"
  [[ $src != /* ]] && src="$dir/$src"
done
MYDIR="$(cd -P "$(dirname "$src")" && pwd)"
[ "$#" -eq 4 ] || (echo "4 argument required, $# provided" ; exit 1)

cd $MYDIR

GPU=$1
MODEL=$2
PROOFS=$3
EPISODES=$4

cat data/compcert-test-files.txt|xargs python3 src/reinforce.py \
  --predictor-weights=data/polyarg-weights.dat \
  --start-from=data/"$MODEL"-q-pretrain.dat \
  --prelude=./CompCert \
  --estimator="$MODEL" \
  --demonstrate-from=search-report-minimal \
  --proofs-file=cc-proofs-$PROOFS.txt \
  --progress \
  -j5 \
  --num-episodes=$EPISODES \
  --gpu=$GPU \
  data/compcert-scrape.txt \
  data/$MODEL-q-$PROOFS-$EPISODES.dat || true

python3 src/mk_reinforced_weights.py \
  data/polyarg-weights.dat \
  data/$MODEL-q-$PROOFS-$EPISODES.dat \
  data/re-$MODEL-$PROOFS-$EPISODES.dat

cat data/compcert-test-files.txt|xargs ./src/search_file.py \
  --weightsfile=data/re-$MODEL-$PROOFS-$EPISODES.dat \
  --prelude=./CompCert \
  --progress \
  -o search-report-re-$MODEL-$PROOFS-$EPISODES \
  -j8 \
  --proofs-file=cc-proofs-$PROOFS.txt \
  --gpu=$GPU

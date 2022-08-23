#!/usr/bin/env bash

set -ex

# determine physical directory of this script
src="${BASH_SOURCE[0]}"
while [ -L "$src" ]; do
  dir="$(cd -P "$(dirname "$src")" && pwd)"
  src="$(readlink "$src")"
  [[ $src != /* ]] && src="$dir/$src"
done
MYDIR="$(cd -P "$(dirname "$src")" && pwd)"
[ "$#" -ge 4 ] || (echo "4 argument required, $# provided" ; exit 1)

cd $MYDIR

GPU=$1
MODEL=$2
PROOFS=$3
EPISODES=$4
TAG=$5

shift 5
while (( "$#" )); do
    case "$1" in
        *) # preserve all other arguments
            PARAMS="$PARAMS $1"
            shift
            ;;
    esac
done

if test ! -f "data/$MODEL-q-$PROOFS-$EPISODES-$TAG.dat" || test -f "data/$MODEL-q-$PROOFS-$EPISODES-$TAG.tmp"; then
    cat data/compcert-test-files.txt|xargs python3 src/reinforce.py \
      --predictor-weights=data/polyarg-weights.dat \
      --start-from=data/"$MODEL"-q-pretrain.dat \
      --prelude=./CompCert \
      --estimator="$MODEL" \
      --demonstrate-from=search-report-minimal \
      --proofs-file=cc-proofs-$PROOFS.txt \
      --progress \
      --train-every-min=8 \
      --train-every-max=16 \
      --batch-size=512 \
      -j4 \
      --num-episodes=$EPISODES \
      --gpu=$GPU \
      data/compcert-scrape.txt \
      data/$MODEL-q-$PROOFS-$EPISODES-$TAG.dat $PARAMS || true

else
    echo "Resuming from existing reinforced weights"
fi

python3 src/mk_reinforced_weights.py \
  data/polyarg-weights.dat \
  data/$MODEL-q-$PROOFS-$EPISODES-$TAG.dat \
  data/re-$MODEL-$PROOFS-$EPISODES-$TAG.dat


cat data/compcert-test-files.txt|xargs ./src/search_file.py \
  --weightsfile=data/re-$MODEL-$PROOFS-$EPISODES-$TAG.dat \
  --prelude=./CompCert \
  --progress \
  -o search-report-re-$MODEL-$PROOFS-$EPISODES-$TAG \
  -j8 \
  --proofs-file=cc-proofs-$PROOFS.txt \
  --gpu=$GPU

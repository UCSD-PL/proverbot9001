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

[ "$#" -ge 2 ] || (echo "at least 2 argument required, $# provided" && exit 1)
PROJECT=$1
WEIGHTS_ID=$2
shift 2
NUM_THREADS=1
while (( "$#" )); do
    case "$1" in
        -j|--num-threads)
            NUM_THREADS=$2
            shift 2
            ;;
        *) # preserve all other arguments
            PARAMS="$PARAMS $1"
            shift
            ;;
    esac
done

make -j$NUM_THREADS -C $MYDIR/$PROJECT/
cp $MYDIR/$PROJECT/{Make,_CoqProject} || true
$MYDIR/get_bench_files.sh $PROJECT
[[ -f $MYDIR/$PROJECT/scrape.txt ]] || $MYDIR/scrape_bench.sh $PROJECT -P -j$NUM_THREADS

TEST_FILES_FILE=test-files.txt
[[ -f $MYDIR/$PROJECT/$TEST_FILES_FILE ]] || TEST_FILES_FILE=files.txt
TEST_FILES=$MYDIR/$PROJECT/$TEST_FILES_FILE
[[ -f $TEST_FILES ]] || (echo "Cannot find test file list" && exit 1)

cd $MYDIR/$PROJECT

cat $TEST_FILES_FILE | xargs $MYDIR/../src/search_file.py -o search-report-$WEIGHTS_ID -P --weightsfile=$MYDIR/../data/polyarg-weights-$WEIGHTS_ID.dat $PARAMS

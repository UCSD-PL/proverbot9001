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
TEST_FILES_FILE=test-files.txt
[[ -f $MYDIR/$PROJECT/$TEST_FILES_FILE ]] || TEST_FILES_FILE=files.txt
TEST_FILES=$MYDIR/$PROJECT/$TEST_FILES_FILE
[[ -f $TEST_FILES ]] || (echo "Cannot find test file list" && exit 1)
shift 2

cd $MYDIR/$PROJECT

cat $TEST_FILES_FILE | xargs $MYDIR/../src/search_file.py -o search-report-$WEIGHTS_ID -P --weightsfile=$MYDIR/../data/polyarg-weights-$WEIGHTS_ID.dat $@

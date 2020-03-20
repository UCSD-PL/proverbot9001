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
TEST_FILES=$MYDIR/$PROJECT/test-files.txt
[[ -f $TEST_FILES ]] || (echo "Cannot find test file list at $TEST_FILES" && exit 1)
shift 2

cd $MYDIR/$PROJECT

JOBS_FILE=test-files.txt ../../parallel-search-report.sh -o ../../$PROJECT-$WEIGHTS_ID-report -P --weightsfile=../../data/polyarg-weights-$WEIGHTS_ID.dat

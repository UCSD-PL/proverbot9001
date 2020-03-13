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

[ "$#" -eq 2 ] || (echo "2 arguments required, $# provided" && exit 1)
PROJECT=$1
PERCENT_TEST=$2

NUM_FILES_TOTAL=$(wc -l "$MYDIR/$PROJECT/files.txt" | cut -f1 -d' ')
NUM_FILES_TEST=$(echo "($NUM_FILES_TOTAL * $PERCENT_TEST) / 100" | bc)

shuf -n "$NUM_FILES_TEST" "$MYDIR/$PROJECT/files.txt" > "$MYDIR/$PROJECT/test-files.txt"
grep -v -f "$MYDIR/$PROJECT/test-files.txt" "$MYDIR/$PROJECT/files.txt" > "$MYDIR/$PROJECT/train-files.txt"

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

[ "$#" -ge 1 ] || (echo "at least 1 argument required, $# provided" && exit 1)
PROJECT=$1
TRAIN_FILES=$MYDIR/$PROJECT/train-files.txt
[[ -f $TRAIN_FILES ]] || (echo "Cannot find training file list at $TRAIN_FILES" && exit 1)
shift

cd $MYDIR/$PROJECT

cat "$TRAIN_FILES" | xargs python3 ../../src/scrape.py -c -j3 -o /dev/null $@
find -name "*.scrape" | xargs cat > scrape.txt

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

cd $MYDIR
git clone $1 $2
cd $2
if [[ -f configure ]]; then
    ./configure
fi
make -j3
find -name "*.vo" -not -path "./_opam/*" | sed 's/.vo/.v/' > files.txt

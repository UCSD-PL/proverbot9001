#!/usr/bin/env bash

set -e

[ "$#" -ne 1 ] || cd $1

find -name "*.vo" -not -path "./_opam/*" | sed 's/.vo/.v/' > files.txt

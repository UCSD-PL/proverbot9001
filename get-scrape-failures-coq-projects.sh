#!/usr/bin/env bash

shopt -s globstar
for project in $(jq -r '.[].project_name' coqgym_projs_splits.json); do
    ls coq-projects/$project/**/*.scrape > /dev/null
    grep -l "[Ee]rr" coq-projects/$project/scrape-output.out
done

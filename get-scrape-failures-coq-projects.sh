#!/usr/bin/env bash

shopt -s globstar

for project in $(jq -r '.[55:]|.[].project_name' coqgym_test_split.json); do
    ls coq-projects/$project/**/*.scrape > /dev/null
    grep "[Ee]rr" coq-projects/$project/scrape-test-output.out
done

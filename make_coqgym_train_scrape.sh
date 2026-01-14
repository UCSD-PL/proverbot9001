#!/bin/bash

> coqgym_scrapes.txt

for f in $(jq -r '.[].project_name' coqgym_projs_splits.json); do
    find "coq-projects/$f" -type f -name "*.scrape" \
      | grep -F -f <(jq -r --arg p "$f" '.[] | select(.project_name==$p) | .train_files[] | sub("\\.v$"; ".v.scrape")' coqgym_projs_splits.json) \
      | while read -r s; do cat "$s"; done >> coqgym_scrapes.txt
    echo $f
done

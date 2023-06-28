#!/usr/bin/env bash

set -e

FILES=$(echo $(jq -r ".[]  | (.test_files[], .train_files[])" compcert_projs_splits.json))
for file in $FILES; do
    echo "Fixing "$file
    python src/add_proof_using.py --prelude="./CompCert/" $file CompCert/$file.tmp
    mv CompCert/{$file.tmp,$file}
done

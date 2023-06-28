#!/usr/bin/env bash

set -e

FILES=$(echo $(jq -r ".[]  | (.test_files[], .train_files[])" compcert_projs_splits.json))
for file in $FILES; do
    echo "Fixing "$file.lin
    HASH=$(head -n 1 CompCert/$file.lin)
    python src/add_proof_using.py --prelude="./CompCert/" <(tail -n +2 CompCert/$file.lin) CompCert/$file.tmp
    echo $HASH > CompCert/$file.lin
    cat CompCert/$file.tmp >> CompCert/$file.lin
done

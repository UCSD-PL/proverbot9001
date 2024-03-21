#!/usr/bin/env bash
echo "This script will OVERWRITE the proofs files in the report you give it. Please interrupt if this isn't what you want; otherwise, hit enter"
read throwaway
for proofsfile in $(find $1 -name "*-proofs.txt"); do
    cat $proofsfile | jq -c ".[1].status = if .[1].steps_taken > 512 and .[1].status == \"SUCCESS\" then \"INCOMPLETE\" else .[1].status end" > $proofsfile.fixed
    mv $proofsfile{.fixed,}
done
